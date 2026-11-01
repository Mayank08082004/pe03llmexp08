"""
test_suite.py - Automated test suite verifying all checklist items and observation table cases for Experiment 7.

Tests Included:
1. Health check endpoint (GET /health) -> Status 200, reports running.
2. Empty/whitespace input validation (POST /ask) -> Status 422.
3. Long input validation >1000 characters (POST /ask) -> Status 422.
4. Backend failure / controlled error handling (POST /ask) -> Status 503, no secret leaks.
5. Supported question (POST /ask) -> Status 200, grounded answer from FAQ, latency recorded.
6. Unsupported question (POST /ask) -> Status 200, fallback response triggered.
"""

import os
import json
from fastapi.testclient import TestClient
from app import app
import support_rag

# Initialize FastAPI test client for in-process HTTP requests
client = TestClient(app)

def run_tests():
    """Runs all 6 test cases sequentially and prints results for the lab observation table."""
    print("=" * 70)
    print("EXPERIMENT 7: TEST SUITE EXECUTION")
    print("=" * 70)

    # Ensure vector store and embeddings are warmed up for test suite
    support_rag.init_components()

    # -----------------------------------------------------------------------
    # Test 1: Health Endpoint Check
    # Verify that GET /health returns HTTP 200 and reports the service is running
    # -----------------------------------------------------------------------
    print("\n--- Test 1: Health Endpoint (GET /health) ---")
    res1 = client.get("/health")
    print(f"Status Code: {res1.status_code}")
    print(f"Response: {res1.json()}")
    assert res1.status_code == 200
    assert res1.json()["status"] == "running"
    print("Result: PASS - API reports running successfully.")

    # -----------------------------------------------------------------------
    # Test 2: Empty Question Input Validation
    # Verify that empty strings and whitespace-only strings are rejected with HTTP 422
    # -----------------------------------------------------------------------
    print("\n--- Test 2: Empty Question Validation (POST /ask) ---")
    res2_empty = client.post("/ask", json={"question": ""})
    res2_space = client.post("/ask", json={"question": "   "})
    print(f"Empty string status: {res2_empty.status_code}, body: {res2_empty.json()}")
    print(f"Whitespace string status: {res2_space.status_code}, body: {res2_space.json()}")
    assert res2_empty.status_code == 422
    assert res2_space.status_code == 422
    print("Result: PASS - Empty question rejected with validation error.")

    # -----------------------------------------------------------------------
    # Test 3: Exceedingly Long Question Input Validation (>1,000 characters)
    # Verify that queries exceeding 1000 chars are rejected with HTTP 422
    # -----------------------------------------------------------------------
    print("\n--- Test 3: Very Long Question (>1000 chars) (POST /ask) ---")
    long_q = "How do I return an item? " * 50  # Generates ~1250 characters
    print(f"Sending question of length: {len(long_q)} characters")
    res3 = client.post("/ask", json={"question": long_q})
    print(f"Status Code: {res3.status_code}")
    print(f"Response: {res3.json()}")
    assert res3.status_code == 422
    print("Result: PASS - Question exceeding 1000 characters rejected with validation error.")

    # -----------------------------------------------------------------------
    # Test 4: Backend Failure / Controlled Error Handling
    # Simulate an unreachable backend (port 59999) and verify controlled HTTP 503
    # Ensure NO private tokens, secrets, or internal stack traces are returned to client
    # -----------------------------------------------------------------------
    print("\n--- Test 4: Backend Failure / Quota Handling (POST /ask) ---")
    orig_backend = os.environ.get("LLM_BACKEND", "ollama")
    orig_ollama_url = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
    
    # Temporarily point to an unreachable address to trigger backend failure
    os.environ["LLM_BACKEND"] = "ollama"
    os.environ["OLLAMA_API_URL"] = "http://127.0.0.1:59999/api/generate"
    
    res4 = client.post("/ask", json={"question": "Where is my order tracking link?"})
    
    # Restore original environment settings
    os.environ["OLLAMA_API_URL"] = orig_ollama_url
    os.environ["LLM_BACKEND"] = orig_backend

    print(f"Status Code: {res4.status_code}")
    print(f"Response: {res4.json()}")
    print(f"X-Response-Time-Ms header: {res4.headers.get('x-response-time-ms')}")
    
    # Verify security constraint: no API keys or Python tracebacks exposed
    res_text = json.dumps(res4.json())
    assert "Bearer" not in res_text
    assert "Traceback" not in res_text
    assert res4.status_code == 503
    assert res4.headers.get("x-response-time-ms") is not None
    print(f"Recorded Latency on Error: {res4.headers.get('x-response-time-ms')} ms")
    print("Result: PASS - Controlled 503 error returned with no secrets/traceback exposed.")

    # -----------------------------------------------------------------------
    # Test 5: Supported Question with Grounded Answer
    # Ask a question covered by the FAQ; verify grounded answer and latency measurement
    # -----------------------------------------------------------------------
    print("\n--- Test 5: Supported Question (Grounded Answer) (POST /ask) ---")
    os.environ["LLM_BACKEND"] = "ollama"
    res5 = client.post("/ask", json={"question": "How do I return an item and get a refund?"})
    print(f"Status Code: {res5.status_code}")
    data5 = res5.json()
    print(f"Answer: {data5.get('answer')}")
    print(f"Sources: {data5.get('sources')}")
    print(f"Intent: {data5.get('intent')}")
    print(f"Latency (ms): {data5.get('latency_ms')} ms")
    print(f"X-Response-Time-Ms header: {res5.headers.get('x-response-time-ms')}")
    assert res5.status_code == 200
    assert len(data5.get("answer", "")) > 10
    assert data5.get("latency_ms") > 0
    print("Result: PASS - Grounded answer and latency returned.")

    # -----------------------------------------------------------------------
    # Test 6: Unsupported Question (Fallback / Insufficient Information)
    # Ask a question not in FAQ (student discount); verify fallback is triggered
    # -----------------------------------------------------------------------
    print("\n--- Test 6: Unsupported Question (Insufficient Information) (POST /ask) ---")
    os.environ["LLM_BACKEND"] = "ollama"
    res6 = client.post("/ask", json={"question": "Can I get a student discount on my order?"})
    print(f"Status Code: {res6.status_code}")
    data6 = res6.json()
    print(f"Answer: {data6.get('answer')}")
    print(f"Intent: {data6.get('intent')}")
    print(f"Latency (ms): {data6.get('latency_ms')} ms")
    assert res6.status_code == 200
    assert "couldn't find a reliable answer" in data6.get("answer", "").lower() or "help section" in data6.get("answer", "").lower()
    print("Result: PASS - Fallback acknowledged insufficient information.")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
