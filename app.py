"""
app.py - Production-Ready Customer Support RAG FastAPI Service.

Features:
- Pydantic input validation (reject empty/whitespace queries, max 1000 characters)
- Controlled error handling (safe client errors and generic server errors without leaking secrets or stack traces)
- Operational logging (latencies, success/failure status, intent, no PII/secrets)
- Response-time measurement in milliseconds (returned in payload and X-Response-Time-Ms header)
- Reusable components initialized once (embedding model, vector store, graph)
- Interactive API documentation via Swagger UI (/docs)
- Endpoints:
    GET  /health
    POST /ask
"""

import os
import time
import logging
from typing import List
from dotenv import load_dotenv

# FastAPI framework and HTTP utility imports
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator

# Import the RAG pipeline module and custom exception
import support_rag
from support_rag import LLMBackendError, ask_support

# Load environment configuration (.env)
load_dotenv()

# Configure operational logger to output timestamps, log levels, and messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("rag_api")

# Initialize FastAPI application instance with title and metadata
app = FastAPI(
    title="Customer Support RAG API",
    description="Production-ready FastAPI service exposing the Customer Support RAG workflow with input validation, latency measurement, and secure error handling.",
    version="1.0.0"
)


# ---------------------------------------------------------------------------
# Middleware: Global Response Time Measurement
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    HTTP Middleware that measures execution duration for EVERY incoming request.
    Appends the total latency in milliseconds to the 'X-Response-Time-Ms' response header.
    """
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(process_time_ms)
    return response


# ---------------------------------------------------------------------------
# Request & Response Data Schemas (Pydantic Models)
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """
    Input validation schema for customer questions.
    Enforces minimum length of 1 character and maximum length of 1,000 characters.
    """
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Customer question (1-1000 characters). Must not be empty or whitespace.",
        examples=["How do I return an item and get a refund?"]
    )

    @field_validator("question")
    @classmethod
    def validate_not_whitespace(cls, v: str) -> str:
        """Custom validator to reject inputs that consist solely of whitespace spaces."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or purely whitespace.")
        return cleaned


class AskResponse(BaseModel):
    """
    Output schema returned for successful /ask requests.
    Includes the grounded answer, reference sources, intent, backend used, and latency.
    """
    question: str       # Echoes the submitted question
    answer: str         # Grounded response generated from FAQ
    sources: List[str]  # Document sources used for context
    intent: str         # Detected intent category
    backend: str        # Active model backend ('ollama' or 'cloud')
    latency_ms: float   # Execution time in milliseconds


class HealthResponse(BaseModel):
    """
    Output schema for the GET /health monitoring endpoint.
    """
    status: str             # Service status ('running')
    backend: str            # Configured LLM backend ('ollama' / 'cloud')
    components_ready: bool  # True if embedding model and FAISS store are loaded
    timestamp: float        # Current server UNIX timestamp


# ---------------------------------------------------------------------------
# Controlled Error Handlers (Security & Production Readiness)
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles invalid client input (e.g. empty question, >1000 chars) gracefully.
    Returns HTTP 422 Unprocessable Entity with a clean, user-friendly error message.
    """
    logger.warning("Input validation error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": "Invalid question input. The question must be a non-empty string with at most 1,000 characters."
        }
    )


@app.exception_handler(LLMBackendError)
async def llm_backend_exception_handler(request: Request, exc: LLMBackendError):
    """
    Handles LLM provider/backend failures (e.g. cloud quota exhausted, Ollama unreachable).
    Returns HTTP 503 Service Unavailable without exposing secret keys, tokens, or raw tracebacks.
    """
    logger.error("Controlled LLM backend failure: %s", str(exc))
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Backend Unavailable",
            "detail": "The LLM inference backend is currently unavailable or quota-limited. Please check provider configuration."
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all fallback handler for unexpected server exceptions.
    Returns a generic HTTP 500 error preventing internal stack traces from leaking to clients.
    """
    logger.error("Unhandled server exception: %s", type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred while processing the request."
        }
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health check endpoint"
)
def health_check():
    """
    GET /health:
    Liveness probe reporting whether the API process is running and whether
    reusable RAG components (FAISS vector store) are initialized in memory.
    """
    configured_backend = os.getenv("LLM_BACKEND", "ollama")
    is_ready = bool(support_rag.vector_store is not None)
    return HealthResponse(
        status="running",
        backend=configured_backend,
        components_ready=is_ready,
        timestamp=time.time()
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["Support"],
    summary="Submit customer support question"
)
def ask_question(req: AskRequest, response: Response):
    """
    POST /ask:
    1. Validates the question via AskRequest Pydantic model.
    2. Measures start time with high-resolution timer (time.perf_counter).
    3. Calls RAG workflow (ask_support).
    4. Records latency in milliseconds.
    5. Returns grounded answer, sources, and latency metrics.
    """
    start_time = time.perf_counter()
    logger.info("Processing /ask request [length=%d chars]", len(req.question))

    try:
        # Invoke the LangGraph RAG pipeline
        result = ask_support(req.question)

        # Calculate processing latency in milliseconds
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Response-Time-Ms"] = str(latency_ms)

        backend_used = os.getenv("LLM_BACKEND", "ollama")
        logger.info(
            "Completed /ask request in %.2f ms [backend=%s, intent=%s]",
            latency_ms,
            backend_used,
            result.get("intent", "general_support")
        )

        # Return structured JSON response
        return AskResponse(
            question=req.question,
            answer=result.get("final_answer", ""),
            sources=result.get("sources", []),
            intent=result.get("intent", ""),
            backend=backend_used,
            latency_ms=latency_ms
        )

    except LLMBackendError:
        # Calculate latency even on backend failure
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Response-Time-Ms"] = str(latency_ms)
        logger.warning("Aborted /ask request after %.2f ms due to LLMBackendError", latency_ms)
        raise

    except Exception as e:
        # Calculate latency on unexpected error
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Response-Time-Ms"] = str(latency_ms)
        logger.error("Aborted /ask request after %.2f ms due to unexpected error: %s", latency_ms, type(e).__name__)
        raise
