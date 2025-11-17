# 🎨 Streamlit Deployment Architecture

## 📊 Overview

This deployment configuration runs **Streamlit UI + FastAPI Backend** with **all MCP servers** in a single AWS ECS Fargate container.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   AWS ECS Fargate Container              │
│                                                          │
│  ┌────────────────────┐    ┌──────────────────────────┐│
│  │  Streamlit UI      │───▶│  FastAPI Backend         ││
│  │  Port 8501         │    │  Port 8000                ││
│  └────────────────────┘    └───────┬──────────────────┘│
│                                    │                    │
│                           ┌────────▼─────────┐         │
│                           │  Multi-Agent      │         │
│                           │  Supervisor       │         │
│                           └────────┬──────────┘         │
│                                    │                    │
│              ┌─────────────────────┼────────────────┐  │
│              │                     │                │  │
│         ┌────▼────┐   ┌────▼────┐ ┌────▼────┐  ┌──▼───┐
│         │ Crypto  │   │  Stock  │ │   RAG   │  │Search│
│         │ Agent   │   │  Agent  │ │  Agent  │  │Agent │
│         └────┬────┘   └────┬────┘ └────┬────┘  └──┬───┘
│              │             │           │          │    │
│    ┌─────────▼─────────────▼───────────▼──────────▼──┐ │
│    │          MCP Servers (Running in Container)     │ │
│    │  • Chroma MCP (uvx chroma-mcp)                  │ │
│    │  • CoinGecko MCP (npx @coingecko/coingecko-mcp) │ │
│    │  • DuckDuckGo MCP (uvx duckduckgo-mcp-server)   │ │
│    └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│                   External Services                       │
│  • ChromaDB Cloud (Vector Storage)                       │
│  • Google Gemini API (LLM)                               │
│  • CoinGecko API (Crypto Data)                           │
│  • Alpha Vantage API (Stock Data)                        │
│  • GCP Cloud SQL (Finance DB) - via MCP Toolbox          │
└──────────────────────────────────────────────────────────┘
```

## 🔄 How It Works

### 1. **User Interaction Flow**
```
User → Streamlit UI (8501) → FastAPI Backend (8000) →
Multi-Agent System → MCP Servers → External APIs
```

### 2. **Container Startup Process**
The `start.sh` script orchestrates the startup:

```bash
1. FastAPI starts in background (port 8000)
2. Wait for FastAPI health check
3. Streamlit starts in foreground (port 8501)
4. Both services run simultaneously
```

### 3. **MCP Servers**
All MCP servers run **inside the same container**:
- ✅ **Chroma MCP**: Connects to ChromaDB Cloud for document storage
- ✅ **CoinGecko MCP**: Fetches cryptocurrency data
- ✅ **DuckDuckGo MCP**: Provides web search capabilities
- ✅ **Finance Tracker**: Uses MCP Toolbox (separate container) for GCP Cloud SQL

## 🚀 Deployment Options

### **Option 1: Local Testing with Docker Compose** (Recommended First Step)

```bash
# 1. Set up environment
cp .env.example .env
nano .env  # Add your API keys

# 2. Run with Docker Compose
docker-compose up --build

# 3. Access the application
# Streamlit UI: http://localhost:8501
# FastAPI: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### **Option 2: Deploy to AWS ECS** (Production)

```bash
# 1. Configure AWS
aws configure

# 2. Run automated setup
chmod +x scripts/aws-setup.sh
./scripts/aws-setup.sh

# 3. Store secrets
chmod +x scripts/store-secrets.sh
./scripts/store-secrets.sh

# 4. Add GitHub Secrets
# Go to GitHub → Settings → Secrets → Actions
# Add: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

# 5. Deploy via GitHub Actions
git add .
git commit -m "Deploy to AWS"
git push origin main

# 6. Get your app URL
chmod +x scripts/get-app-url.sh
./scripts/get-app-url.sh
```

## 📦 What's Included in the Container

### Python Dependencies (from requirements.txt)
- ✅ Streamlit (UI framework)
- ✅ FastAPI + Uvicorn (Backend)
- ✅ LangChain + LangGraph (Agent framework)
- ✅ Google Generative AI (Gemini)
- ✅ ChromaDB client
- ✅ Document processors (PyPDF2, python-docx)
- ✅ MCP SDK

### Node.js Dependencies (installed globally)
- ✅ @coingecko/coingecko-mcp

### System Tools
- ✅ Python 3.11
- ✅ Node.js 20
- ✅ curl (for health checks)
- ✅ uvx (Python tool runner for MCP servers)
- ✅ npx (Node package runner)

## 🔌 Port Configuration

| Port | Service | Purpose |
|------|---------|---------|
| 8501 | Streamlit | Web UI for users |
| 8000 | FastAPI | Backend API + Agent orchestration |
| 5000 | MCP Toolbox | Cloud SQL connector (separate container) |

## 🌐 Accessing the Deployed Application

After deployment, you'll get:

```bash
🎨 Streamlit UI:  http://YOUR_PUBLIC_IP:8501  ← Main application
🔧 FastAPI:       http://YOUR_PUBLIC_IP:8000  ← Backend API
📚 API Docs:      http://YOUR_PUBLIC_IP:8000/docs  ← Interactive docs
❤️  Health:        http://YOUR_PUBLIC_IP:8000/health  ← Health check
```

## 🔐 Security Configuration

### Secrets (Stored in AWS Secrets Manager)
- `multi-agent/GOOGLE_API_KEY` - Google Gemini API
- `multi-agent/COINGECKO_API_KEY` - CoinGecko API
- `multi-agent/ALPHA_VANTAGE_API_KEY` - Stock data API
- `multi-agent/CHROMA_API_KEY` - ChromaDB Cloud
- `multi-agent/CHROMA_TENANT` - ChromaDB tenant
- `multi-agent/CHROMA_DATABASE` - ChromaDB database

### Network Security
- Security group allows:
  - Port 8501 (Streamlit) from 0.0.0.0/0
  - Port 8000 (FastAPI) from 0.0.0.0/0

**⚠️ Production Recommendation**: Restrict to specific IPs or add authentication

## 💰 Cost Estimation

### AWS ECS Fargate (1 task, 24/7)
- **CPU**: 1 vCPU × $0.04048/hour × 730 hours = $29.55/month
- **Memory**: 2GB × $0.004445/GB/hour × 730 hours = $6.49/month
- **ECR Storage**: ~$0.50/month
- **CloudWatch Logs**: ~$1/month
- **Secrets Manager**: 6 secrets × $0.40 = $2.40/month

**Total**: ~$40/month

### Cost Optimization
```bash
# Stop when not in use
aws ecs update-service \
  --cluster multi-agent-cluster \
  --service multi-agent-service \
  --desired-count 0

# Restart when needed
aws ecs update-service \
  --cluster multi-agent-cluster \
  --service multi-agent-service \
  --desired-count 1
```

## 🐛 Troubleshooting

### Container Won't Start
```bash
# Check logs
aws logs tail /ecs/multi-agent-app --follow

# Common issues:
# 1. Missing secrets → Verify in AWS Secrets Manager
# 2. Health check failing → Check FastAPI is starting
# 3. Out of memory → Increase task memory in task-definition.json
```

### Can't Access Streamlit UI
```bash
# 1. Check security group allows port 8501
aws ec2 describe-security-groups --group-ids YOUR_SG_ID

# 2. Verify task has public IP
./scripts/get-app-url.sh

# 3. Check health status
curl http://YOUR_PUBLIC_IP:8000/health
```

### MCP Servers Not Working
```bash
# Check container logs for MCP server errors
aws logs tail /ecs/multi-agent-app --follow | grep -i mcp

# Common issues:
# 1. Missing API keys
# 2. Network connectivity to external services
# 3. MCP server installation failed during build
```

## 📝 Key Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main Streamlit UI application |
| `api.py` | FastAPI backend with multi-agent system |
| `start.sh` | Container startup script |
| `Dockerfile` | Container build instructions |
| `docker-compose.yml` | Local development setup |
| `.aws/task-definition.json` | ECS task configuration |
| `.github/workflows/deploy-aws.yml` | CI/CD pipeline |

## 🔄 Differences from Gradio Setup

| Aspect | Gradio (Old) | Streamlit (New) |
|--------|--------------|-----------------|
| **UI Port** | 7860 | 8501 |
| **UI Framework** | Gradio | Streamlit |
| **Startup** | Single process | Multi-process (start.sh) |
| **Health Check** | Port 7860 | Port 8000 (FastAPI) |
| **Config** | None | .streamlit/config.toml |

## 🎯 Next Steps

### Immediate
- [x] Deploy locally with Docker Compose
- [ ] Test all agents (crypto, stock, search, RAG)
- [ ] Upload test documents
- [ ] Verify MCP server connectivity

### Production
- [ ] Deploy to AWS ECS
- [ ] Set up custom domain (Route 53)
- [ ] Enable HTTPS (Certificate Manager + ALB)
- [ ] Add authentication (OAuth/Cognito)
- [ ] Configure auto-scaling
- [ ] Set up monitoring alerts

### Advanced
- [ ] Add caching layer (Redis)
- [ ] Implement rate limiting
- [ ] Set up CI/CD staging environment
- [ ] Enable container insights
- [ ] Configure backup/disaster recovery

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [AWS ECS Fargate Guide](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)
- [MCP Documentation](https://modelcontextprotocol.io/introduction)

---

## 🆘 Need Help?

1. **Local issues**: Check Docker logs: `docker-compose logs -f`
2. **AWS issues**: Check CloudWatch: `/ecs/multi-agent-app`
3. **GitHub Actions**: Check Actions tab in repository
4. **Cost concerns**: Use [AWS Cost Calculator](https://calculator.aws/)

---

**Questions or feedback?** Open an issue in the GitHub repository.
