"""
Test suite for FastAPI Multi-Agent Assistant API.

This module tests the core API endpoints including health checks,
root endpoint, and system initialization.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app, multi_agent_app


# ============================================================================
# Test Client Setup
# ============================================================================

@pytest.fixture
def client():
    """Create test client for FastAPI application."""
    return TestClient(app)


# ============================================================================
# Root Endpoint Tests
# ============================================================================

def test_root_endpoint(client):
    """Test that root endpoint returns correct API information."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "message" in data
    assert "version" in data
    assert "docs" in data
    assert "health" in data

    # Check specific values
    assert data["message"] == "Multi-Agent Assistant API"
    assert data["version"] == "1.0.0"
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"


def test_root_endpoint_content_type(client):
    """Test that root endpoint returns JSON content."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"


# ============================================================================
# Health Check Tests
# ============================================================================

def test_health_check_endpoint(client):
    """Test health check endpoint returns proper structure."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "status" in data
    assert "initialized" in data
    assert "agents" in data

    # Check status values
    assert data["status"] in ["healthy", "initializing"]
    assert isinstance(data["initialized"], bool)
    assert isinstance(data["agents"], dict)


def test_health_check_agents_status(client):
    """Test that health check returns all agent statuses."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    # Check all agents are present
    expected_agents = ["crypto", "rag", "stock", "search", "finance_tracker"]
    for agent in expected_agents:
        assert agent in data["agents"]
        assert isinstance(data["agents"][agent], bool)


def test_health_check_response_model(client):
    """Test health check response matches Pydantic model."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    # Validate against expected model structure
    assert set(data.keys()) == {"status", "initialized", "agents"}

    # Check types
    assert isinstance(data["status"], str)
    assert isinstance(data["initialized"], bool)
    assert isinstance(data["agents"], dict)


# ============================================================================
# API Documentation Tests
# ============================================================================

def test_openapi_schema_available(client):
    """Test that OpenAPI schema is accessible."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    assert "openapi" in schema
    assert "info" in schema
    assert schema["info"]["title"] == "Multi-Agent Assistant API"
    assert schema["info"]["version"] == "1.0.0"


def test_docs_endpoint_available(client):
    """Test that Swagger docs are accessible."""
    response = client.get("/docs")

    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "html" in response.headers["content-type"]


def test_redoc_endpoint_available(client):
    """Test that ReDoc documentation is accessible."""
    response = client.get("/redoc")

    assert response.status_code == 200



# ============================================================================
# Error Handling Tests
# ============================================================================

def test_nonexistent_endpoint(client):
    """Test that nonexistent endpoints return 404."""
    response = client.get("/nonexistent")

    assert response.status_code == 404


def test_invalid_method_on_root(client):
    """Test that invalid HTTP methods return 405."""
    response = client.post("/")

    assert response.status_code == 405


# ============================================================================
# Chat Endpoint Tests (Basic)
# ============================================================================

def test_chat_stream_endpoint_requires_post(client):
    """Test that chat stream endpoint only accepts POST."""
    response = client.get("/api/v1/chat/stream")

    assert response.status_code == 405  # Method Not Allowed


def test_chat_stream_endpoint_exists(client):
    """Test that chat stream endpoint exists."""
    # Send minimal valid request
    response = client.post(
        "/api/v1/chat/stream",
        json={"message": "test"}
    )

    # Should return 200 or process the request (not 404)
    assert response.status_code != 404


# ============================================================================
# Upload Endpoint Tests (Basic)
# ============================================================================

def test_upload_endpoint_requires_post(client):
    """Test that upload endpoint only accepts POST."""
    response = client.get("/api/v1/documents/upload")

    assert response.status_code == 405  # Method Not Allowed


def test_upload_endpoint_exists(client):
    """Test that upload endpoint exists."""
    # Try to access without file (should fail validation, not 404)
    response = client.post("/api/v1/documents/upload")

    # Should return 422 (validation error) or other, but not 404
    assert response.status_code != 404


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_system_initialization():
    """Test that multi-agent system can initialize."""
    # Reset initialization state
    multi_agent_app.initialized = False

    # Initialize
    result = await multi_agent_app.initialize()

    # Check initialization
    assert multi_agent_app.initialized is True
    assert multi_agent_app.supervisor is not None
    assert "initialized" in result.lower() or "ready" in result.lower()


def test_agent_status_method():
    """Test get_agent_status returns correct structure."""
    status = multi_agent_app.get_agent_status()

    assert isinstance(status, dict)
    expected_agents = ["crypto", "rag", "stock", "search", "finance_tracker"]

    for agent in expected_agents:
        assert agent in status
        assert isinstance(status[agent], bool)


# ============================================================================
# Performance Tests
# ============================================================================

def test_health_check_response_time(client):
    """Test that health check responds quickly."""
    import time

    start = time.time()
    response = client.get("/health")
    elapsed = time.time() - start

    assert response.status_code == 200
    assert elapsed < 1.0  # Should respond in less than 1 second


def test_root_endpoint_response_time(client):
    """Test that root endpoint responds quickly."""
    import time

    start = time.time()
    response = client.get("/")
    elapsed = time.time() - start

    assert response.status_code == 200
    assert elapsed < 0.5  # Should respond in less than 0.5 seconds


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])