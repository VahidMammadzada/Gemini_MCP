# Gemini MCP

A multi-agent AI assistant powered by Gemini LLM using the ReAct pattern to orchestrate specialized agents through Model Context Protocol (MCP). Features a LangGraph supervisor with streaming Gradio UI.

## ✨ Features

- **ReAct-based supervisor** – dynamically routes queries to specialized agents
- **5 Specialized Agents** – crypto, stocks, portfolio tracking, RAG document Q&A, web search
- **Real-time streaming** – see AI reasoning steps as they happen
- **Document upload** – chat with PDFs, TXT, and DOCX files
- **Cloud integration** – ChromaDB Cloud for vectors, Google Cloud SQL for portfolio data
- **Docker support** – containerized MCP servers

## 🛠️ Tech Stack

- **AI/LLM**: Google Gemini 2.5 Pro, LangChain, LangGraph
- **UI**: Streamlit (main), Gradio 5.0+ (for demos)
- **Agent Protocol**: FastMCP (Model Context Protocol)
- **Databases**: ChromaDB Cloud (vectors), Google Cloud SQL (PostgreSQL)
- **APIs**: Alpha Vantage, CoinGecko, DuckDuckGo
- **DevOps**: Docker, Docker Compose


## ⚙️ How It Works

The ReAct supervisor uses a reasoning loop (max 5 steps):
1. **Think** – Analyze query and determine needed information
2. **Act** – Call appropriate agent(s)
3. **Observe** – Process agent responses
4. **Repeat** – Continue until sufficient information gathered
5. **Synthesize** – Generate final answer


## 🚀 Quick Start
```bash
# Clone and install
git clone <repository-url>
cd Gemini_MCP
pip install -r requirements.txt
```

Create `.env` file:
```bash
GOOGLE_API_KEY=your_google_ai_api_key
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=your_tenant_id
CHROMA_DATABASE=your_database_name
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
COINGECKO_API_KEY=your_coingecko_key
GCP_PROJECT_ID=your_project_id
CLOUD_SQL_INSTANCE_CONNECTION_NAME=project:region:instance
CLOUD_SQL_DB_NAME=finance_tracker
CLOUD_SQL_DB_USER=your_db_user
CLOUD_SQL_DB_PASS=your_db_password
```

Run database schema:
```bash
psql -h <cloud-sql-host> -U <user> -d finance_tracker -f database_schema.sql
```

Start services:
```bash
# Start MCP Toolbox
docker-compose up -d

# Run app
python app.py
```

Access at: http://localhost:7860

## 💡 Usage Examples

- Crypto: "What's the current Bitcoin price?"
- Stocks: "What's Tesla's stock price?"
- Portfolio: "Add 10 shares of AAPL at $150"
- Documents: Upload PDF → "Summarize this document"
- Search: "Latest AI developments"

## 🔑 API Keys Setup

- **Google AI**: [Get key](https://makersuite.google.com/app/apikey)
- **ChromaDB**: [Sign up](https://www.trychroma.com/)
- **Alpha Vantage**: [Get key](https://www.alphavantage.co/support/#api-key)
- **CoinGecko**: [Get key](https://www.coingecko.com/)
- **Google Cloud SQL**: Create project → Enable Cloud SQL API → Run `gcloud auth application-default login`


## 📁 Project Structure
```
Gemini_MCP/
├── .github/              # CI/CD workflows
├── .streamlit/           # Streamlit configuration
├── docker/               # Docker containerization
├── scripts/              # Startup and setup scripts
├── src/                  # Main application source code
├── tests/                # Unit test suite
├── ui/                   # User interface applications
├── requirements.txt      # Python dependencies (260+ packages)
├── docker-compose.yml    # Multi-container orchestration
├── database_schema.sql   # PostgreSQL schema
└── fastapi_testing.ipynb # Jupyter notebook for API testing
```
---

Built with Google Gemini, LangGraph, Streamlit and MCP