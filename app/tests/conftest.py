"""
Pytest configuration and common fixtures for testing.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    client = Mock()
    client.search = Mock()
    client.set_payload = Mock()
    client.get_collection = Mock()
    return client


@pytest.fixture
def mock_embeddings_model():
    """Mock embeddings model for testing."""
    model = Mock()
    model.encode = Mock(return_value=[[0.1, 0.2, 0.3, 0.4, 0.5]])
    return model


@pytest.fixture
def sample_embedding_vector():
    """Sample embedding vector for testing."""
    return [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def sample_search_payload():
    """Sample search payload for testing."""
    return {
        "text": "This is a sample text for testing purposes.",
        "source_doc_id": "test_doc_123",
        "chunk_index": 0,
        "journal": "Test Journal",
        "publish_year": 2023,
        "link": "https://example.com/test",
        "section_heading": "Introduction",
        "attributes": ["research", "testing", "sample"],
        "usage_count": 0
    }


@pytest.fixture
def sample_qdrant_result():
    """Sample Qdrant search result for testing."""
    result = Mock()
    result.id = "test_chunk_1"
    result.score = 0.95
    result.payload = {
        "text": "This is a sample text for testing purposes.",
        "source_doc_id": "test_doc_123",
        "chunk_index": 0,
        "journal": "Test Journal",
        "publish_year": 2023,
        "link": "https://example.com/test",
        "section_heading": "Introduction",
        "attributes": ["research", "testing", "sample"],
        "usage_count": 0
    }
    return result


@pytest.fixture
def sample_search_results():
    """Sample list of Qdrant search results for testing."""
    results = []
    for i in range(3):
        result = Mock()
        result.id = f"test_chunk_{i+1}"
        result.score = 0.95 - (i * 0.1)
        result.payload = {
            "text": f"This is sample text {i+1} for testing purposes.",
            "source_doc_id": f"test_doc_{i+1}",
            "chunk_index": i,
            "journal": "Test Journal",
            "publish_year": 2023,
            "link": f"https://example.com/test_{i+1}",
            "section_heading": f"Section {i+1}",
            "attributes": ["research", "testing", "sample"],
            "usage_count": i
        }
        results.append(result)
    return results


@pytest.fixture
def mock_async_function():
    """Mock async function for testing."""
    async def mock_func(*args, **kwargs):
        return "mock_result"
    return mock_func


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    ) 