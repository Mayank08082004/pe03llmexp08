"""
support_rag.py - RAG pipeline for Customer Support with modular LLM backends.
Supports Cloud-Hosted API (Hugging Face Inference Providers), Local LLM (Ollama),
and Mock mode for controlled testing.
"""

import os
import logging
from typing import TypedDict, List
import requests
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END

# Load environment variables from .env file (API keys, backend configuration, URLs)
load_dotenv()

# Set up a logger for recording pipeline events
logger = logging.getLogger("support_rag")


class LLMBackendError(Exception):
    """
    Custom exception raised when an error occurs during LLM backend invocation.
    Helps FastAPI catch backend errors without exposing sensitive tracebacks or API keys.
    """
    pass


# ---------------------------------------------------------------------------
# 1. Knowledge Base Definition & Document Preprocessing
# ---------------------------------------------------------------------------

# Raw FAQ text containing official customer support policies
support_faq = """
Order Tracking
Customers can view shipment updates in the Orders section of their account. A tracking link is shown after
an order is dispatched. If tracking has not updated for 48 hours, contact support with the order reference.
The assistant cannot access live order status.
Returns and Refunds
Unused items in their original packaging may be requested for return within 14 days of delivery. Some items
may be excluded. To start a return, open Orders, choose the item, and select Return Request. After an
approved return is received, a refund is usually processed to the original payment method within 5 to 7
business days. The assistant cannot approve a refund.
Account Access
If you cannot sign in, use the Forgot Password option on the sign-in page. Check your email for the password
reset link. If the email does not arrive, check the spam folder and confirm that you entered the registered
email address. Never share your password or one-time verification code.
Billing and Payments
Payment status is shown in the order details. If a payment appears to have been charged twice, do not submit
repeated payments. Contact the billing support team with the transaction date and order reference. Do not
share full card numbers or security codes.
Technical Support
If the website is not loading, refresh the page, check your internet connection, and try an updated browser.
If the problem continues, clear the browser cache or try a private window. When contacting support, describe
the error message and the steps that led to it. Do not share passwords or verification codes.
Contact Support
For an issue that is not covered by these FAQs, contact the support team through the Help section of the
website. Include a brief description and relevant order reference, but do not send passwords, one-time
codes, or full payment-card details.
"""

# Save the raw FAQ text to a local file for persistent reference
with open("support_faq.txt", "w", encoding="utf-8") as f:
    f.write(support_faq)

# Wrap the text into a standard LangChain Document object with metadata
faq_doc = Document(
    page_content=support_faq,
    metadata={"source": "support_faq.txt"}
)

# Configure text splitter: break large text into smaller overlapping chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,    # Maximum characters per chunk
    chunk_overlap=60   # Characters of overlap between chunks to retain context
)
chunks = splitter.split_documents([faq_doc])

# ---------------------------------------------------------------------------
# 2. Embedding Model & Vector Index Initialization (Startup / Non-blocking)
# ---------------------------------------------------------------------------

embedding_model = None
vector_store = None
retriever = None


def init_components():
    """
    Initializes the HuggingFace embedding model and FAISS vector index.
    Run at application startup or lazily upon retrieval requests so that the
    web service process can immediately bind its port without blocking Uvicorn.
    """
    global embedding_model, vector_store, retriever
    if vector_store is not None:
        return vector_store

    logger.info("Initializing HuggingFace embedding model and FAISS vector store...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embedding_model)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    logger.info("RAG vector store and retriever successfully initialized.")
    return vector_store


# ---------------------------------------------------------------------------
# 3. Decoupled LLM Backend Invocation
# ---------------------------------------------------------------------------

def call_llm_backend(prompt: str) -> str:
    """
    Invokes the configured LLM backend based on environment settings.
    Keeps the LLM backend decoupled from the RAG retrieval pipeline so backends
    can be swapped (Cloud Gemini/HF vs. Local Ollama) without modifying pipeline logic.
    Never exposes API keys or private tokens in errors or logs.
    """
    # Read backend choice: 'cloud' (default), 'gemini', 'ollama', or 'mock'
    backend = os.getenv("LLM_BACKEND", "cloud").lower()
    fallback_on_failure = os.getenv("FALLBACK_ON_CLOUD_FAILURE", "false").lower() == "true"

    # Option A: Cloud-hosted API (Google Gemini or Hugging Face)
    if backend in ("cloud", "gemini"):
        gemini_key = (os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY") or "").strip("'\"")
        cloud_provider = os.getenv("CLOUD_PROVIDER", "gemini" if gemini_key else "huggingface").lower()

        try:
            if cloud_provider == "gemini" or (gemini_key and not os.getenv("HF_TOKEN")):
                return _call_gemini(prompt)
            elif os.getenv("HF_TOKEN"):
                return _call_huggingface(prompt)
            elif gemini_key:
                return _call_gemini(prompt)
            else:
                raise LLMBackendError("Cloud LLM configuration error: GEMINI_KEY (or HF_TOKEN) is not configured.")
        except LLMBackendError:
            if fallback_on_failure:
                logger.info("Falling back to local Ollama backend...")
                return _call_ollama(prompt)
            raise

    # Option B: Local LLM served via Ollama (llama3:8b)
    elif backend == "ollama":
        return _call_ollama(prompt)

    # Option C: Mock mode for offline unit testing without network or local GPU
    elif backend == "mock":
        if "INSUFFICIENT_CONTEXT" in prompt or "student discount" in prompt.lower():
            return "INSUFFICIENT_CONTEXT"
        return "To return an item, go to Orders, select the item, and submit a Return Request within 14 days."

    else:
        raise LLMBackendError(f"Unsupported LLM backend: '{backend}'")


def _call_gemini(prompt: str) -> str:
    """
    Invokes Google Gemini REST API using the official endpoint.
    Uses 'gemini-2.5-flash' by default, or the model configured in GEMINI_MODEL_NAME.
    Zero-dependency via standard requests library.
    """
    gemini_key = (os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY") or "").strip("'\"")
    if not gemini_key:
        raise LLMBackendError("Cloud LLM configuration error: GEMINI_KEY is not configured.")

    model = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 800
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
            raise LLMBackendError("Gemini returned empty candidate response.")
        else:
            err_detail = f"Gemini provider returned status {response.status_code}"
            try:
                err_json = response.json()
                if "error" in err_json and "message" in err_json["error"]:
                    err_detail += f": {err_json['error']['message']}"
            except Exception:
                pass
            logger.warning(f"Gemini LLM request failed: {err_detail}")
            raise LLMBackendError(f"Gemini LLM unavailable: {err_detail}")
    except requests.exceptions.RequestException as e:
        logger.error("Gemini LLM network request error")
        raise LLMBackendError(f"Gemini LLM connection failed: {type(e).__name__}")


def _call_huggingface(prompt: str) -> str:
    """
    Invokes Hugging Face Inference Providers endpoint.
    """
    hf_token = os.getenv("HF_TOKEN")
    hf_api_url = os.getenv("HF_API_URL", "https://router.huggingface.co/v1/chat/completions")
    hf_model = os.getenv("HF_MODEL_NAME", "deepseek-ai/DeepSeek-R1:fastest")

    if not hf_token:
        raise LLMBackendError("Cloud LLM configuration error: HF_TOKEN is not configured.")

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": hf_model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.0
    }

    try:
        response = requests.post(hf_api_url, headers=headers, json=payload, timeout=25)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            err_detail = f"HuggingFace provider returned status {response.status_code}"
            try:
                err_json = response.json()
                if "error" in err_json:
                    err_detail += f": {err_json['error']}"
            except Exception:
                pass
            logger.warning(f"HuggingFace LLM request failed: {err_detail}")
            raise LLMBackendError(f"Cloud LLM unavailable: {err_detail}")
    except requests.exceptions.RequestException as e:
        logger.error("HuggingFace LLM network request error")
        raise LLMBackendError(f"Cloud LLM connection failed: {type(e).__name__}")


def _call_ollama(prompt: str) -> str:
    """
    Helper function to query a local Ollama instance over HTTP.
    Sends prompt to the local endpoint and extracts the text response.
    """
    ollama_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
    ollama_model = os.getenv("OLLAMA_MODEL_NAME", "llama3:8b")

    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0  # Zero temperature for deterministic responses
        }
    }
    try:
        # POST request to Ollama /api/generate
        res = requests.post(ollama_url, json=payload, timeout=45)
        if res.status_code == 200:
            return res.json().get("response", "").strip()
        else:
            raise LLMBackendError(f"Ollama returned status {res.status_code}")
    except requests.exceptions.RequestException as e:
        raise LLMBackendError(f"Ollama connection error: {type(e).__name__}")


# ---------------------------------------------------------------------------
# 4. LangGraph State Schema & Workflow Nodes
# ---------------------------------------------------------------------------

class SupportState(TypedDict, total=False):
    """
    Schema defining the state passed between LangGraph nodes during execution.
    """
    question: str       # Original customer question
    intent: str         # Detected intent category
    search_query: str   # Query used for vector retrieval
    context: str        # Retrieved FAQ chunks text
    sources: List[str]  # Source file paths for attribution
    draft: str          # Raw LLM draft generation
    retry_count: int    # Number of query rewrite attempts
    final_answer: str   # Reviewed, validated answer returned to user


def classify_intent(state: SupportState):
    """
    Node 1: Fast rule-based intent classification.
    Categorizes the customer question using keyword matching.
    """
    q = state["question"].lower()
    if any(word in q for word in ["refund", "return", "exchange"]):
        intent = "returns_refunds"
    elif any(word in q for word in ["track", "tracking", "shipment", "delivery", "order status"]):
        intent = "order_tracking"
    elif any(word in q for word in ["password", "sign in", "login", "account", "verification"]):
        intent = "account_access"
    elif any(word in q for word in ["charged", "payment", "billing", "invoice", "card"]):
        intent = "billing_payments"
    elif any(word in q for word in ["website", "error", "loading", "browser", "technical"]):
        intent = "technical_support"
    else:
        intent = "general_support"
    return {
        "intent": intent,
        "search_query": state["question"],
        "retry_count": state.get("retry_count", 0)
    }


def retrieve_context(state: SupportState):
    """
    Node 2: Vector retrieval using FAISS.
    Queries the vector store for chunks most similar to search_query.
    """
    global retriever
    if retriever is None:
        init_components()
    docs = retriever.invoke(state["search_query"])
    context = "\n\n".join(d.page_content for d in docs)
    sources = [d.metadata.get("source", "support_faq.txt") for d in docs]
    return {
        "context": context,
        "sources": sources
    }


def draft_answer(state: SupportState):
    """
    Node 3: Prompt construction and LLM generation.
    Formats the system prompt with strict grounding constraints.
    Instructs the LLM to output 'INSUFFICIENT_CONTEXT' if the FAQ lacks the answer.
    """
    prompt = f"""You are a polite customer-support assistant.
Use only the FAQ CONTEXT to answer the CUSTOMER QUESTION.
Do not invent order status, refund approvals, timelines, or company policies.
If the context does not contain the answer, output exactly:
INSUFFICIENT_CONTEXT
Do not ask for passwords, one-time codes, or full payment-card details.
Keep the answer clear and concise.
FAQ CONTEXT:
{state.get("context", "")}
CUSTOMER QUESTION:
{state["question"]}
ANSWER:"""

    # Call the decoupled backend
    draft = call_llm_backend(prompt)
    return {"draft": draft}


def review_answer(state: SupportState):
    """
    Node 4: Review and validate the drafted answer.
    Checks if the model reported INSUFFICIENT_CONTEXT.
    """
    draft = state.get("draft", "").strip()
    if "INSUFFICIENT_CONTEXT" in draft:
        return {"final_answer": ""}  # Leave empty to trigger rewrite/fallback
    return {"final_answer": draft}


def route_after_review(state: SupportState):
    """
    Conditional routing function:
    - If answer is valid -> finish
    - If answer is empty and retries < 1 -> retry (rewrite query)
    - If answer is empty and retries >= 1 -> fallback
    """
    if state.get("final_answer"):
        return "finish"
    if state.get("retry_count", 0) < 1:
        return "retry"
    return "fallback"


def rewrite_query(state: SupportState):
    """
    Node 5: Query expansion / rewriting.
    Expands the question with intent-specific keywords to improve vector search recall.
    """
    category_terms = {
        "returns_refunds": "return eligibility refund process",
        "order_tracking": "order dispatch tracking shipment status",
        "account_access": "account sign in password reset",
        "billing_payments": "billing payment transaction",
        "technical_support": "website technical troubleshooting",
        "general_support": "customer support help contact"
    }
    expanded = state["question"] + " " + category_terms.get(
        state.get("intent", "general_support"), "customer support"
    )
    return {
        "search_query": expanded,
        "retry_count": state.get("retry_count", 0) + 1
    }


def fallback_answer(state: SupportState):
    """
    Node 6: Fallback response.
    Returns a safe, polite message acknowledging lack of information
    without hallucinating policies.
    """
    return {
        "final_answer": (
            "I couldn't find a reliable answer in the available FAQ. "
            "Please contact the support team through the Help section. "
            "Do not share your password, one-time code, or full payment-card details."
        )
    }


# ---------------------------------------------------------------------------
# 5. LangGraph Workflow Graph Assembly
# ---------------------------------------------------------------------------

# Initialize state graph with SupportState schema
graph_builder = StateGraph(SupportState)

# Add processing nodes to the graph
graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("retrieve", retrieve_context)
graph_builder.add_node("draft_answer", draft_answer)
graph_builder.add_node("review_answer", review_answer)
graph_builder.add_node("rewrite_query", rewrite_query)
graph_builder.add_node("fallback", fallback_answer)

# Define sequential execution edges
graph_builder.add_edge(START, "classify_intent")
graph_builder.add_edge("classify_intent", "retrieve")
graph_builder.add_edge("retrieve", "draft_answer")
graph_builder.add_edge("draft_answer", "review_answer")

# Define conditional edge based on review evaluation
graph_builder.add_conditional_edges(
    "review_answer",
    route_after_review,
    {
        "finish": END,
        "retry": "rewrite_query",
        "fallback": "fallback"
    }
)

# Connect rewrite back to retriever for a second search attempt
graph_builder.add_edge("rewrite_query", "retrieve")

# Connect fallback node to graph exit
graph_builder.add_edge("fallback", END)

# Compile graph into a runnable workflow
support_graph = graph_builder.compile()


# ---------------------------------------------------------------------------
# 6. Public Interface Function
# ---------------------------------------------------------------------------

def ask_support(question: str) -> dict:
    """
    Public entrypoint function called by FastAPI app.py.
    Invokes the LangGraph RAG workflow with the given question and returns the result dictionary.
    """
    if vector_store is None:
        init_components()
    result = support_graph.invoke({
        "question": question,
        "retry_count": 0
    })
    return result
