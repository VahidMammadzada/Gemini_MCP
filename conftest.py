"""
Pytest configuration and fixtures for Multi-Agent Assistant API tests.

This module provides shared fixtures and configuration for all tests,
including environment setup and mocking strategies.
"""
import os
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# Environment Configuration
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Setup test environment variables and configuration.

    This fixture runs once per test session and sets up necessary
    environment variables for testing.
    """
    # Set CI flag if not already set
    if "CI" not in os.environ:
        os.environ["CI"] = "false"

    # Ensure required environment variables have defaults for local testing
    env_defaults = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", "test_google_api_key"),
        "CHROMA_API_KEY": os.getenv("CHROMA_API_KEY", "test_chroma_key"),
        "CHROMA_TENANT": os.getenv("CHROMA_TENANT", "test_tenant"),
        "CHROMA_DATABASE": os.getenv("CHROMA_DATABASE", "test_db"),
    }

    # Only set defaults if not in CI and not already set
    is_ci = os.getenv("CI") == "true"
    if not is_ci:
        for key, default_value in env_defaults.items():
            if key not in os.environ:
                os.environ[key] = default_value

    yield

    # Cleanup (optional)
    pass


@pytest.fixture(scope="session")
def has_real_credentials():
    """
    Check if real Google Cloud credentials are available.

    Returns:
        bool: True if credentials are available, False otherwise
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_app_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # Check if we have real credentials (not test/dummy values)
    has_api_key = google_api_key and google_api_key != "test_google_api_key"
    has_app_creds = google_app_creds and os.path.exists(google_app_creds)

    return has_api_key or has_app_creds


# ============================================================================
# Mocking Fixtures
# ============================================================================

@pytest.fixture
def mock_google_genai():
    """
    Mock Google Generative AI to avoid actual API calls during testing.

    This fixture patches the Google GenAI module to return mock responses
    instead of making real API calls.
    """
    with patch('google.generativeai.configure') as mock_configure:
        mock_configure.return_value = None
        yield mock_configure


@pytest.fixture
def mock_chromadb():
    """
    Mock ChromaDB client to avoid database connections during testing.
    """
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    with patch('chromadb.HttpClient', return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_agent_initialization():
    """
    Mock all agent initialization methods.

    This fixture patches the initialize methods of all agents to avoid
    real API calls and external dependencies during testing.
    """
    patches = [
        patch('crypto_agent_mcp.CryptoAgentMCP.initialize', new_callable=AsyncMock),
        patch('rag_agent_mcp.RAGAgentMCP.initialize', new_callable=AsyncMock),
        patch('stock_agent_mcp.StockAgentMCP.initialize', new_callable=AsyncMock),
        patch('search_agent_mcp.SearchAgentMCP.initialize', new_callable=AsyncMock),
        patch('finance_tracker_agent_mcp.FinanceTrackerMCP.initialize', new_callable=AsyncMock),
    ]

    # Start all patches
    mocks = [p.start() for p in patches]

    # Configure mocks to return None (successful initialization)
    for mock in mocks:
        mock.return_value = None

    yield mocks

    # Stop all patches
    for p in patches:
        p.stop()


@pytest.fixture
def mock_all_agents(mock_agent_initialization):
    """
    Convenience fixture that mocks all agent initializations.

    Use this fixture when you want to test API endpoints without
    actually initializing the agents.
    """
    return mock_agent_initialization


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_chat_request():
    """Provide sample chat request data for testing."""
    return {
        "message": "What is the price of Bitcoin?",
        "history": []
    }


@pytest.fixture
def sample_chat_history():
    """Provide sample chat history for testing."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "What is Bitcoin?"}
    ]


# ============================================================================
# Async Test Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop_policy():
    """
    Set event loop policy for async tests.

    This ensures consistent async behavior across different platforms.
    """
    import asyncio
    return asyncio.get_event_loop_policy()


# ============================================================================
# Conditional Test Markers
# ============================================================================

def pytest_configure(config):
    """
    Register custom pytest markers.
    """
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external dependencies"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that may require real API access"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )
    config.addinivalue_line(
        "markers", "requires_credentials: Tests that require real credentials"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to handle conditional skipping.

    This automatically skips integration tests in CI if credentials
    are not available.
    """
    is_ci = os.getenv("CI") == "true"
    has_credentials = bool(
        os.getenv("GOOGLE_API_KEY") and
        os.getenv("GOOGLE_API_KEY") != "test_google_api_key"
    )

    skip_integration = pytest.mark.skip(
        reason="Integration tests skipped in CI without credentials"
    )

    for item in items:
        # Skip integration tests in CI without credentials
        if "integration" in item.keywords and is_ci and not has_credentials:
            item.add_marker(skip_integration)

        # Skip tests that explicitly require credentials
        if "requires_credentials" in item.keywords and not has_credentials:
            item.add_marker(pytest.mark.skip(
                reason="Test requires real credentials"
            ))
