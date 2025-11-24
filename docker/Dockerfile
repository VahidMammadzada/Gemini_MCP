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

# Copy application code with new structure
COPY src/ ./src/
COPY ui/ ./ui/
COPY tests/ ./tests/

# Copy startup script
COPY scripts/start.sh ./start.sh

# Create directory for temporary file uploads
RUN mkdir -p /tmp/uploads && chmod 777 /tmp/uploads

# Make startup script executable
RUN chmod +x /app/start.sh

# Copy Streamlit config (includes dark theme)
RUN mkdir -p /root/.streamlit
COPY .streamlit/config.toml /root/.streamlit/config.toml

# Additional Streamlit config for production
RUN echo '\n[browser]\ngatherUsageStats = false\n' >> /root/.streamlit/config.toml


# Expose ports
# 8501 - Streamlit UI
# 8000 - FastAPI Backend
EXPOSE 8501 8000

# Health check - check both FastAPI and Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health && curl -f http://localhost:8501/_stcore/health || exit 1

# Use startup script to run both FastAPI and Streamlit
CMD ["/app/start.sh"]