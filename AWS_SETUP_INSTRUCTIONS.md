# AWS Fargate Multi-Container Setup - Instructions

## Overview
Your application has been configured to run MCP Toolbox as a sidecar container alongside your main application in AWS Fargate. Both containers will run in the same task and communicate via `localhost`.

## Changes Made

### 1. Task Definition Updates (`task_definition.json`)
- ✅ **Increased resources**: CPU: 1024 → 2048, Memory: 2048MB → 4096MB
- ✅ **Added MCP Toolbox sidecar container** with:
  - Image: `730335436231.dkr.ecr.eu-west-3.amazonaws.com/mcp-toolbox`
  - Port: 5000
  - `essential: false` (main app can continue if toolbox fails)
  - Dependency: main app waits for toolbox to START
- ✅ **Added environment variable**: `MCP_TOOLBOX_SERVER_URL=http://localhost:5000`
- ✅ **Added Cloud SQL configuration** for both containers

### 2. GitHub Actions Workflow Updates (`aws.yaml`)
- ✅ **Builds both images**: multi-agent-app AND mcp-toolbox
- ✅ **Pushes to ECR**: Both images tagged with git SHA
- ✅ **Renders both containers** in task definition before deployment

### 3. Container Communication
```
┌─────────────────────────────────────────────────────────┐
│  ECS Fargate Task (localhost network)                  │
│                                                         │
│  ┌──────────────────┐         ┌──────────────────┐    │
│  │  multi-agent-app │  HTTP   │   mcp-toolbox    │    │
│  │  (ports 8501,    │ ──────> │   (port 5000)    │    │
│  │   8000)          │         │                  │    │
│  └──────────────────┘         └──────────────────┘    │
│                                        │               │
│                                        │ Cloud SQL     │
│                                        │ Connector     │
│                                        ↓               │
└────────────────────────────────────────┼───────────────┘
                                         │
                                         ↓
                              Google Cloud SQL
                              (PostgreSQL)
```

## Required AWS Setup

### Step 1: Add Secrets to AWS Secrets Manager

You need to add the following keys to your existing secret:
**Secret Name**: `API_KEYS-VM91-fuw8iR`
**Region**: `eu-west-3`

**New keys to add** (for Cloud SQL connection):
```json
{
  "GCP_PROJECT_ID": "your-gcp-project-id",
  "CLOUD_SQL_REGION": "us-central1",
  "CLOUD_SQL_INSTANCE": "your-instance-name",
  "CLOUD_SQL_DB_NAME": "finance_tracker",
  "CLOUD_SQL_DB_USER": "your-db-username",
  "CLOUD_SQL_DB_PASS": "your-db-password"
}
```

**AWS CLI command to update**:
```bash
aws secretsmanager update-secret \
  --secret-id "API_KEYS-VM91-fuw8iR" \
  --region eu-west-3 \
  --secret-string '{
    "GOOGLE_API_KEY": "existing-value",
    "ALPHA_VANTAGE_API_KEY": "existing-value",
    "COINGECKO_API_KEY": "existing-value",
    "CHROMA_API_KEY": "existing-value",
    "CHROMA_TENANT": "existing-value",
    "CHROMA_DATABASE": "existing-value",
    "GCP_PROJECT_ID": "your-gcp-project-id",
    "CLOUD_SQL_REGION": "us-central1",
    "CLOUD_SQL_INSTANCE": "your-instance-name",
    "CLOUD_SQL_DB_NAME": "finance_tracker",
    "CLOUD_SQL_DB_USER": "your-db-user",
    "CLOUD_SQL_DB_PASS": "your-db-password"
  }'
```

### Step 2: Create ECR Repository for MCP Toolbox

```bash
aws ecr create-repository \
  --repository-name mcp-toolbox \
  --region eu-west-3
```

### Step 3: Create CloudWatch Log Group

```bash
aws logs create-log-group \
  --log-group-name /ecs/mcp-toolbox \
  --region eu-west-3
```

### Step 4: Update ECS Task IAM Role

Ensure your `ecsTaskRole` (arn:aws:iam::730335436231:role/ecsTaskRole) has permissions to:

1. **Connect to Google Cloud SQL** (if using Cloud SQL Proxy authentication)
2. **Read secrets** from Secrets Manager (already configured)

## Important Notes

### Google Cloud Authentication
⚠️ **IMPORTANT**: The MCP Toolbox container needs to authenticate with Google Cloud to connect to Cloud SQL.

**Current limitation**: The Dockerfile.toolbox expects a credentials file, but Fargate doesn't support file mounts.

**Solutions**:
1. **Option A** (Recommended): Use Cloud SQL built-in authentication via environment variables
   - MCP Toolbox uses connection string format: `project:region:instance`
   - Authentication via database username/password (already configured)

2. **Option B**: Use service account key as environment variable
   - Store entire JSON key as a secret in AWS Secrets Manager
   - Pass as `GOOGLE_APPLICATION_CREDENTIALS_JSON` environment variable
   - Requires modifying Dockerfile.toolbox to write JSON to file at runtime

### Resource Allocation
The task now requires:
- **CPU**: 2048 (2 vCPU) - was 1024
- **Memory**: 4096 MB (4 GB) - was 2048 MB

**Cost Impact**: ~2x increase in ECS Fargate costs

### Container Startup Order
- MCP Toolbox starts first
- Main app waits for toolbox to START (not HEALTHY)
- Both containers share localhost network

## Verification Steps

After deployment:

1. **Check both containers are running**:
```bash
aws ecs describe-tasks \
  --cluster multi-agent-cluster \
  --tasks <task-id> \
  --region eu-west-3
```

2. **View MCP Toolbox logs**:
```bash
aws logs tail /ecs/mcp-toolbox \
  --follow \
  --region eu-west-3
```

3. **View main app logs**:
```bash
aws logs tail /ecs/multi-agent-app \
  --follow \
  --region eu-west-3
```

4. **Test the Finance Tracker**:
   - Access your Streamlit app
   - Ask: "How is my portfolio looking like today?"
   - Should connect to MCP Toolbox at `http://localhost:5000`

## Troubleshooting

### Error: "Cannot connect to localhost:5000"
- Check MCP Toolbox container is running: `docker-compose ps` or ECS console
- Check MCP Toolbox logs for startup errors
- Verify `dependsOn` is configured correctly in task definition

### Error: Cloud SQL connection failed
- Verify Cloud SQL secrets are correctly set in AWS Secrets Manager
- Check MCP Toolbox logs for authentication errors
- Ensure Cloud SQL instance is accessible from the internet (or use Cloud SQL Proxy)
- Verify GCP project ID, region, and instance name are correct

### Container fails to start
- Check CloudWatch logs for both containers
- Verify ECR images exist and are accessible
- Check task execution role has ECR pull permissions

## Next Steps

1. ✅ Add Cloud SQL secrets to AWS Secrets Manager
2. ✅ Create `mcp-toolbox` ECR repository
3. ✅ Create CloudWatch log group for mcp-toolbox
4. ✅ Push code to main branch to trigger CI/CD
5. ✅ Monitor deployment and test Finance Tracker functionality

## Rollback Plan

If the multi-container setup has issues, you can quickly rollback by:

1. Reverting `task_definition.json` to single container (git revert)
2. Deploying previous task definition version via AWS Console
3. The previous deployment remains available in ECS
