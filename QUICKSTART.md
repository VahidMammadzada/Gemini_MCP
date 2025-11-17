# 🚀 Quick Start Guide

## Option 1: Local Development with Docker (5 minutes)

### 1. Clone and setup
```bash
git clone <your-repo-url>
cd Gemini_MCP
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run with Docker Compose
```bash
docker-compose up --build
```

### 3. Access the application
- Gradio UI: http://localhost:7860
- FastAPI: http://localhost:8000/docs

---

## Option 2: AWS Deployment (30 minutes)

### Prerequisites
- AWS Account
- AWS CLI installed and configured
- Docker installed
- GitHub account

### Quick Deploy Steps

#### 1. AWS Setup (One-time)
```bash
# Set variables
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create ECR repository
aws ecr create-repository --repository-name multi-agent-app --region $AWS_REGION

# Create ECS cluster
aws ecs create-cluster --cluster-name multi-agent-cluster --region $AWS_REGION
```

#### 2. Store Secrets
```bash
# Store your API keys in AWS Secrets Manager
aws secretsmanager create-secret \
    --name multi-agent/GOOGLE_API_KEY \
    --secret-string "your-key-here" \
    --region $AWS_REGION

# Repeat for: COINGECKO_API_KEY, ALPHA_VANTAGE_API_KEY,
# CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE
```

#### 3. Deploy
```bash
# Push to GitHub - GitHub Actions will deploy automatically
git add .
git commit -m "Initial deployment"
git push origin main
```

#### 4. Get Public IP
```bash
# Wait 2-3 minutes for deployment, then:
./scripts/get-app-url.sh
```

---

## Option 3: GitHub Actions Automated Deployment

### Setup (One-time)

1. **Fork/clone this repository**

2. **Add GitHub Secrets:**
   - Go to Settings → Secrets and variables → Actions
   - Add `AWS_ACCESS_KEY_ID`
   - Add `AWS_SECRET_ACCESS_KEY`

3. **Configure AWS resources:**
   ```bash
   # Run the setup script
   chmod +x scripts/aws-setup.sh
   ./scripts/aws-setup.sh
   ```

4. **Push to main branch:**
   ```bash
   git push origin main
   ```

5. **GitHub Actions will automatically:**
   - Build Docker image
   - Push to ECR
   - Deploy to ECS
   - Update running service

---

## 🎯 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    GitHub Actions                        │
│  (Build → Push to ECR → Deploy to ECS)                 │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  AWS ECS Fargate                        │
│  ┌────────────────────────────────────────────────┐    │
│  │  Multi-Agent Container                         │    │
│  │  - Gradio UI (Port 7860)                       │    │
│  │  - FastAPI (Port 8000)                         │    │
│  │  - Python MCP Servers (uvx)                    │    │
│  │  - Node.js MCP Servers (npx)                   │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│             External Services                            │
│  - ChromaDB Cloud (Vector Store)                        │
│  - GCP Cloud SQL (Finance Tracker DB)                   │
│  - Google AI (Gemini LLM)                               │
│  - CoinGecko API (Crypto Data)                          │
│  - Alpha Vantage API (Stock Data)                       │
└─────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing Your Deployment

### 1. Health Check
```bash
curl http://YOUR_PUBLIC_IP:7860
curl http://YOUR_PUBLIC_IP:8000/health
```

### 2. Test Queries
- "What's the current price of Bitcoin?"
- "Show me Tesla's stock price"
- "Search for latest AI developments"

### 3. Upload Document
- Upload a PDF/TXT/DOCX file
- Ask questions about it

---

## 📊 Monitoring

### View Logs
```bash
# AWS CloudWatch
aws logs tail /ecs/multi-agent-app --follow

# Or visit: AWS Console → CloudWatch → Log groups
```

### Service Status
```bash
aws ecs describe-services \
    --cluster multi-agent-cluster \
    --services multi-agent-service
```

---

## 🔧 Common Commands

### Scale Up/Down
```bash
# Scale to 2 instances
aws ecs update-service \
    --cluster multi-agent-cluster \
    --service multi-agent-service \
    --desired-count 2

# Scale to 0 (stop)
aws ecs update-service \
    --cluster multi-agent-cluster \
    --service multi-agent-service \
    --desired-count 0
```

### Force New Deployment
```bash
aws ecs update-service \
    --cluster multi-agent-cluster \
    --service multi-agent-service \
    --force-new-deployment
```

### View Running Tasks
```bash
aws ecs list-tasks --cluster multi-agent-cluster
```

---

## 💰 Cost Management

### Current Configuration Costs
- **1 Task (1 vCPU, 2GB RAM):** ~$36/month
- **With ECR, Logs, Secrets:** ~$44/month total

### Stop When Not Using
```bash
# Stop (no compute costs)
aws ecs update-service \
    --cluster multi-agent-cluster \
    --service multi-agent-service \
    --desired-count 0

# Start again
aws ecs update-service \
    --cluster multi-agent-cluster \
    --service multi-agent-service \
    --desired-count 1
```

---

## 🆘 Troubleshooting

### Task Won't Start
1. Check CloudWatch logs: `/ecs/multi-agent-app`
2. Verify all secrets are set in Secrets Manager
3. Check IAM role permissions

### Can't Access Application
1. Verify security group allows port 7860
2. Ensure task has public IP
3. Wait 2-3 minutes for health checks

### High Costs
1. Check running task count
2. Scale down to 0.5 vCPU / 1GB RAM
3. Stop when not in use

---

## 📚 Next Steps

- [ ] Set up custom domain (Route 53)
- [ ] Enable HTTPS (Certificate Manager)
- [ ] Add authentication (OAuth/JWT)
- [ ] Configure auto-scaling
- [ ] Set up CloudWatch alarms
- [ ] Implement CI/CD for staging/production

---

## 🔗 Useful Links

- [Full Deployment Guide](DEPLOYMENT.md)
- [AWS ECS Console](https://console.aws.amazon.com/ecs)
- [CloudWatch Logs](https://console.aws.amazon.com/cloudwatch)
- [ECR Repositories](https://console.aws.amazon.com/ecr)

---

**Questions?** Check [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.
