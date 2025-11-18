#!/bin/bash
# Test runner script for Multi-Agent Assistant API

set -e  # Exit on error

echo "🧪 Multi-Agent Assistant API Test Runner"
echo "========================================"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest is not installed"
    echo "📦 Installing development dependencies..."
    pip install -r requirements-dev.txt
    echo "✅ Dependencies installed"
    echo ""
fi

# Parse command line arguments
TEST_TYPE=${1:-"all"}
COVERAGE=${2:-"no"}

case $TEST_TYPE in
    "quick")
        echo "⚡ Running quick tests (health checks only)..."
        pytest test_api.py -k "test_health or test_root" -v
        ;;
    "unit")
        echo "🔬 Running unit tests..."
        pytest test_api.py -m "unit" -v
        ;;
    "integration")
        echo "🔗 Running integration tests..."
        pytest test_api.py -m "integration" -v
        ;;
    "coverage")
        echo "📊 Running tests with coverage report..."
        pytest test_api.py -v --cov=. --cov-report=html --cov-report=term
        echo ""
        echo "📁 Coverage report generated in htmlcov/index.html"
        ;;
    "watch")
        echo "👀 Running tests in watch mode..."
        pytest-watch test_api.py -v
        ;;
    "all"|*)
        echo "🚀 Running all tests..."
        if [ "$COVERAGE" == "coverage" ]; then
            pytest test_api.py -v --cov=. --cov-report=html --cov-report=term
            echo ""
            echo "📁 Coverage report generated in htmlcov/index.html"
        else
            pytest test_api.py -v
        fi
        ;;
esac

echo ""
echo "✅ Tests completed!"
