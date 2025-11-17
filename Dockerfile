# Multi-stage Dockerfile for Multi-Agent Assistant
# Supports both Python (uvx) and Node.js (npx) MCP servers

FROM python:3.11-slim as base

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    git \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Verify installations
RUN python --version && node --version && npm --version

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install npx-based MCP servers globally (for CoinGecko)
RUN npm install -g @coingecko/coingecko-mcp

# Copy application code
COPY . .

# Create directory for temporary file uploads
RUN mkdir -p /tmp/uploads && chmod 777 /tmp/uploads

# Expose ports
# 7860 - Gradio UI
# 8000 - FastAPI
EXPOSE 7860 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || curl -f http://localhost:7860 || exit 1

# Default command (can be overridden)
# Use exec form to properly handle signals
CMD ["python", "app.py"]
