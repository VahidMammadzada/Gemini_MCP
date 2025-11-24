#!/bin/bash
# Startup script to run both FastAPI backend and Streamlit frontend

set -e

echo "🚀 Starting Multi-Agent Assistant..."

# Function to handle shutdown
shutdown() {
    echo "🛑 Shutting down services..."
    kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null || true
    wait
    exit 0
}

trap shutdown SIGTERM SIGINT

# Start FastAPI backend in background
echo "📡 Starting FastAPI backend on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait for FastAPI to be ready
echo "⏳ Waiting for FastAPI to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ FastAPI is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ FastAPI failed to start"
        kill $FASTAPI_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# Start Streamlit frontend in foreground
echo "🎨 Starting Streamlit UI on port 8501..."
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.enableCORS false \
    --server.enableXsrfProtection true &

STREAMLIT_PID=$!

echo "✅ All services started!"
echo "📡 FastAPI: http://localhost:8000"
echo "🎨 Streamlit: http://localhost:8501"

# Wait for either process to exit
wait -n

# If we get here, one process exited - shut down everything
shutdown