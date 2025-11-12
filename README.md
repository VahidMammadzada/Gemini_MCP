# Multi-Agent Assistant with ReAct Supervisor

A sophisticated multi-agent AI system powered by Google's Gemini LLM, implementing the ReAct (Reasoning + Acting) pattern for intelligent task orchestration. This system coordinates specialized agents through the Model Context Protocol (MCP) to provide comprehensive answers across multiple domains including cryptocurrency, stocks, document analysis, web search, and personal finance tracking.

## Features

- **Intelligent Multi-Agent Orchestration**: ReAct-based supervisor that reasons about queries and dynamically routes to appropriate agents
- **Streaming Real-time Updates**: See the AI's reasoning process with intermediate steps displayed in real-time
- **Multiple Specialized Agents**:
  - Cryptocurrency price tracking and market analysis
  - Stock market data and financial information
  - Personal portfolio tracking and management
  - Document Q&A with RAG (Retrieval-Augmented Generation)
  - Web search capabilities
- **Modern UI**: Interactive Gradio-based chat interface with document upload
- **Cloud Integration**: ChromaDB Cloud for vector storage, Google Cloud SQL for portfolio data
- **Docker Support**: Containerized MCP servers for reliable deployment

## Architecture

### ReAct Pattern Workflow

The system uses a reasoning loop to solve complex queries:

1. **Think**: Supervisor analyzes the query and reasons about what information is needed
2. **Act**: Calls appropriate agent(s) to gather information
3. **Observe**: Reviews and processes agent responses
4. **Repeat**: Continues the loop until sufficient information is gathered (max 5 steps)
5. **Synthesize**: Generates a comprehensive final answer combining all insights

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Gradio UI Layer                       │
│              (User Interface & Interaction)              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              ReAct Supervisor (LangGraph)                │
│        (Reasoning Engine + Task Orchestration)           │
└─┬───────┬─────────┬──────────┬──────────┬──────────────┘
  │       │         │          │          │
  ▼       ▼         ▼          ▼          ▼
┌───┐   ┌───┐    ┌────┐    ┌──────┐   ┌────────┐
│Cry│   │RAG│    │Stok│    │Search│   │Finance │
│pto│   │   │    │    │    │      │   │Tracker │
└─┬─┘   └─┬─┘    └─┬──┘    └──┬───┘   └───┬────┘
  │       │        │           │           │
  ▼       ▼        ▼           ▼           ▼
┌────────────────────────────────────────────────────────┐
│              MCP (Model Context Protocol)              │
│         (Agent Communication & Tool Access)            │
└────────────────────────────────────────────────────────┘
  │       │        │           │           │
  ▼       ▼        ▼           ▼           ▼
CoinGecko ChromaDB Alpha    DuckDuckGo  Cloud SQL
  API     Cloud   Vantage      API      (PostgreSQL)
```

## Requirements

- Python 3.9+
- Docker and Docker Compose (for MCP Toolbox)
- Google Cloud Project (for Cloud SQL integration)
- API Keys:
  - Google AI (Gemini API)
  - ChromaDB Cloud
  - Alpha Vantage (stock data)
  - CoinGecko (optional, for crypto data)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Gemini_MCP
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Google AI
GOOGLE_API_KEY=your_google_ai_api_key

# ChromaDB Cloud
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=your_tenant_id
CHROMA_DATABASE=your_database_name
CHROMA_CLOUD_HOST=api.trychroma.com
DOCUMENTS_COLLECTION=mcp-test

# Stock Market Data
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# CoinGecko (Optional)
COINGECKO_API_KEY=your_coingecko_key

# Google Cloud SQL (for Finance Tracker)
GCP_PROJECT_ID=your_project_id
CLOUD_SQL_REGION=us-central1
CLOUD_SQL_INSTANCE=your_instance_name
CLOUD_SQL_DB_NAME=finance_tracker
CLOUD_SQL_DB_USER=your_db_user
CLOUD_SQL_DB_PASS=your_db_password
CLOUD_SQL_INSTANCE_CONNECTION_NAME=project:region:instance

# MCP Toolbox Server
MCP_TOOLBOX_SERVER_URL=http://localhost:5000
```

### 4. Set Up Cloud SQL Database

Run the database schema to initialize the finance tracker tables:

```bash
# Connect to your Cloud SQL instance and run:
psql -h <cloud-sql-host> -U <user> -d finance_tracker -f database_schema.sql
```

### 5. Start MCP Toolbox Server (Docker)

```bash
docker-compose up -d
```

This starts the MCP Toolbox server on port 5000, which the Finance Tracker agent uses to interact with Cloud SQL.

## Usage

### Starting the Application

```bash
python app.py
```

The Gradio interface will be available at: **http://localhost:7860**

### Example Queries

**Cryptocurrency:**
- "What's the current price of Bitcoin and Ethereum?"
- "Show me the market trends for the top 10 cryptocurrencies"

**Stock Market:**
- "What's Tesla's stock price?"
- "Give me a financial overview of Apple Inc."
- "Show me the latest earnings for NVDA"

**Personal Finance Tracking:**
- "Add 10 shares of AAPL I bought at $150"
- "What's my current portfolio value?"
- "Show me news on my portfolio holdings"
- "What's the performance of my TSLA position?"

**Document Q&A:**
- Upload a PDF/TXT/DOCX file using the upload panel
- "What does my document say about [topic]?"
- "Summarize the key points from my uploaded document"

**Web Search:**
- "Search for the latest AI developments"
- "What did Jerome Powell say in his latest speech?"
- "Find information about quantum computing breakthroughs"

### Uploading Documents

1. Click the file upload area in the right panel
2. Select a PDF, TXT, or DOCX file (max 50MB)
3. Click "Upload to RAG"
4. Wait for confirmation that the document has been processed
5. Ask questions about your document content

## Agent Capabilities

### Crypto Agent (CryptoAgentMCP)
- Real-time cryptocurrency prices
- Market cap and volume data
- Price trends and historical data
- Multi-currency support via CoinGecko API

### Stock Agent (StockAgentMCP)
- Real-time stock quotes
- Company profiles and fundamentals
- Financial statements
- Market analysis via Alpha Vantage API

### Finance Tracker (FinanceTrackerMCP)
- Personal portfolio management
- Buy/sell transaction tracking
- Portfolio performance analytics
- Holdings-specific news and updates
- PostgreSQL database via Cloud SQL

### RAG Agent (RAGAgentMCP)
- Document upload and processing (PDF, TXT, DOCX)
- Vector-based semantic search
- Context-aware question answering
- ChromaDB Cloud integration for scalable storage

### Search Agent (SearchAgentMCP)
- Web search via DuckDuckGo
- Real-time information retrieval
- News and current events
- Docker-based MCP server for reliable search

## Project Structure

```
Gemini_MCP/
├── app.py                          # Main Gradio application
├── langgraph_supervisor.py         # ReAct supervisor implementation
├── config.py                       # Configuration management
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Docker services configuration
├── Dockerfile.toolbox              # MCP Toolbox Docker image
├── database_schema.sql             # Cloud SQL database schema
│
├── crypto_agent_mcp.py             # Cryptocurrency agent
├── stock_agent_mcp.py              # Stock market agent
├── finance_tracker_agent_mcp.py    # Portfolio tracking agent
├── rag_agent_mcp.py                # Document Q&A agent
├── search_agent_mcp.py             # Web search agent
│
├── file_processors.py              # Document processing utilities
├── mcp_client_proper.py            # MCP client implementation
└── decision_maker.py               # Agent routing logic
```

## Configuration

### API Key Setup

#### Google AI (Gemini)
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add to `.env` as `GOOGLE_API_KEY`

#### ChromaDB Cloud
1. Sign up at [ChromaDB Cloud](https://www.trychroma.com/)
2. Create a tenant and database
3. Add credentials to `.env`

#### Alpha Vantage
1. Get free API key at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Add to `.env` as `ALPHA_VANTAGE_API_KEY`

#### CoinGecko (Optional)
1. Sign up at [CoinGecko](https://www.coingecko.com/)
2. Get API key from dashboard
3. Add to `.env` as `COINGECKO_API_KEY`

### Google Cloud Setup

1. Create a Google Cloud Project
2. Enable Cloud SQL Admin API
3. Create a Cloud SQL PostgreSQL instance
4. Set up Application Default Credentials:
   ```bash
   gcloud auth application-default login
   ```
5. Update `.env` with your Cloud SQL connection details

## Technology Stack

- **LLM Framework**: LangChain, LangGraph
- **AI Model**: Google Gemini 2.5 Pro
- **UI**: Gradio 5.0+
- **Agent Protocol**: Model Context Protocol (MCP) via FastMCP
- **Vector Database**: ChromaDB Cloud
- **SQL Database**: Google Cloud SQL (PostgreSQL)
- **Document Processing**: PyPDF2, python-docx
- **APIs**: Alpha Vantage, CoinGecko, DuckDuckGo
- **Containerization**: Docker, Docker Compose

## How It Works

### Streaming Reasoning Process

When you submit a query, you'll see the AI's thought process in real-time:

1. **Reasoning Step**: Shows what the AI is thinking and planning
2. **Action Step**: Indicates which agent is being called
3. **Observation Step**: Displays the results from the agent
4. **Iteration**: May repeat steps 1-3 multiple times
5. **Final Answer**: Synthesized response combining all gathered information

### Example Flow

```
User: "What's Bitcoin's price and should I add it to my portfolio based on my current holdings?"

Step 1 [Thinking]: Need crypto price data → Call crypto agent
Step 2 [Action]: Calling CRYPTO agent ⏳
Step 3 [Observation]: Bitcoin price: $45,231 ✓
Step 4 [Thinking]: Need user's portfolio data → Call finance tracker
Step 5 [Action]: Calling FINANCE_TRACKER agent ⏳
Step 6 [Observation]: Current holdings: AAPL (10 shares), TSLA (5 shares) ✓
Step 7 [Final Answer]: Based on your tech-heavy portfolio and Bitcoin's current
price of $45,231, adding some BTC could provide diversification...
```

## Troubleshooting

### Common Issues

**"System not initialized"**
- Ensure all API keys are properly set in `.env`
- Check that required services (Cloud SQL, ChromaDB) are accessible
- Restart the application

**MCP Toolbox Connection Failed**
- Verify Docker container is running: `docker-compose ps`
- Check logs: `docker-compose logs mcp-toolbox`
- Ensure Google Cloud credentials are mounted correctly

**Document Upload Fails**
- Check file size (max 50MB)
- Verify file type is supported (PDF, TXT, DOCX)
- Ensure ChromaDB Cloud credentials are correct

**Agent Timeout**
- Check API rate limits
- Verify network connectivity
- Review agent-specific error messages in the UI

## Performance Tips

- The supervisor uses a maximum of 5 reasoning steps to prevent infinite loops
- Document uploads are chunked for optimal retrieval
- Agent calls are made asynchronously when possible
- Chat history is limited to the last 20 exchanges

## Security Notes

- Store all API keys in `.env` file (never commit to version control)
- Add `.env` to `.gitignore`
- Use Google Cloud IAM for Cloud SQL access control
- ChromaDB Cloud provides encrypted data storage
- MCP Toolbox runs in isolated Docker container

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request with a clear description

## Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Powered by [Google Gemini](https://ai.google.dev/)
- Uses [Model Context Protocol](https://modelcontextprotocol.io/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

**Built with ❤️ using Google Gemini, LangGraph, and MCP**
