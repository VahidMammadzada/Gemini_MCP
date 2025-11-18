# Testing Guide

## Overview

This project includes a comprehensive test suite for the FastAPI Multi-Agent Assistant API. Tests cover health checks, API endpoints, and system integration.

## Quick Start

### 1. Install Testing Dependencies

```bash
pip install -r requirements-dev.txt
```

### 2. Run Tests

**Run all tests:**
```bash
pytest test_api.py -v
```

**Or use the test runner script:**
```bash
./run_tests.sh
```

## Test Structure

### Test File: `test_api.py`

The test suite is organized into the following sections:

1. **Root Endpoint Tests**
   - Basic API information
   - Response structure validation
   - Content type verification

2. **Health Check Tests**
   - System health status
   - Agent availability
   - Initialization state

3. **API Documentation Tests**
   - OpenAPI schema validation
   - Swagger UI accessibility
   - ReDoc availability

4. **CORS Middleware Tests**
   - Cross-origin request handling

5. **Error Handling Tests**
   - 404 responses for invalid endpoints
   - 405 responses for invalid methods

6. **Chat Endpoint Tests**
   - Streaming endpoint validation
   - WebSocket connectivity

7. **Upload Endpoint Tests**
   - Document upload functionality

8. **Integration Tests**
   - Multi-agent system initialization
   - Agent status reporting

9. **Performance Tests**
   - Response time validation

## Running Specific Tests

### Run specific test functions:
```bash
pytest test_api.py::test_health_check_endpoint -v
```

### Run tests matching a pattern:
```bash
pytest test_api.py -k "health" -v
```

### Run tests by marker:
```bash
# Unit tests only
pytest test_api.py -m unit -v

# Integration tests only
pytest test_api.py -m integration -v

# Async tests only
pytest test_api.py -m asyncio -v
```

## Test Runner Script Options

The `run_tests.sh` script provides convenient testing options:

```bash
# Run all tests
./run_tests.sh

# Quick tests (health checks only)
./run_tests.sh quick

# Unit tests only
./run_tests.sh unit

# Integration tests only
./run_tests.sh integration

# Run with coverage report
./run_tests.sh coverage
```

## Coverage Reports

### Generate coverage report:
```bash
pytest test_api.py --cov=. --cov-report=html --cov-report=term
```

### View coverage report:
```bash
# Open htmlcov/index.html in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Writing New Tests

### Test Function Template

```python
def test_example_endpoint(client):
    """Test description."""
    response = client.get("/endpoint")

    assert response.status_code == 200
    data = response.json()
    assert "expected_key" in data
```

### Async Test Template

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async functionality."""
    result = await some_async_function()
    assert result is not None
```

### Using Fixtures

```python
@pytest.fixture
def sample_data():
    """Provide sample test data."""
    return {"key": "value"}

def test_with_fixture(client, sample_data):
    """Test using fixture data."""
    response = client.post("/endpoint", json=sample_data)
    assert response.status_code == 200
```

## Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_unit_example():
    """Unit test example."""
    pass

@pytest.mark.integration
def test_integration_example():
    """Integration test example."""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Long-running test."""
    pass
```

## Continuous Integration

### GitHub Actions Example

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run tests
      run: pytest test_api.py -v --cov=. --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Troubleshooting

### Common Issues

**1. Import errors:**
```bash
# Ensure you're in the project root
cd /path/to/Gemini_MCP

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**2. Async test failures:**
```bash
# Ensure pytest-asyncio is installed
pip install pytest-asyncio>=0.23.0
```

**3. Tests hanging:**
```bash
# Run with timeout
pytest test_api.py -v --timeout=10
```

**4. Module not found:**
```bash
# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest test_api.py -v
```

## Best Practices

1. **Write descriptive test names** - Use clear, action-based names
2. **One assertion per test** - Keep tests focused
3. **Use fixtures** - Share common setup code
4. **Mock external dependencies** - Isolate unit tests
5. **Test edge cases** - Include error conditions
6. **Keep tests fast** - Mark slow tests with `@pytest.mark.slow`
7. **Document expected behavior** - Use docstrings

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [Pytest-Asyncio](https://pytest-asyncio.readthedocs.io/)

## Support

For issues or questions about testing, please check:
- Project README: `README.md`
- API Documentation: http://localhost:8000/docs (when running)
- GitHub Issues: [Project Issues Page]
