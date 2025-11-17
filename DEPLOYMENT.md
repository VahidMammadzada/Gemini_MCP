# 🚀 AWS Deployment Guide for Multi-Agent Assistant

This guide will walk you through deploying your multi-agent application to AWS ECS Fargate with automated GitHub Actions CI/CD.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Testing](#local-testing)
3. [AWS Setup](#aws-setup)
4. [GitHub Secrets Configuration](#github-secrets-configuration)
5. [First Deployment](#first-deployment)
6. [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)
7. [Cost Estimation](#cost-estimation)

---

## 🔧 Prerequisites

### Required Accounts
- ✅ AWS Account ([Sign up here](https://aws.amazon.com/))
- ✅ GitHub Account with this repository
- ✅ API Keys (Google AI, CoinGecko, Alpha Vantage, ChromaDB)

### Install AWS CLI
```bash
# macOS
brew install awscli

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Windows
# Download from: https://aws.amazon.com/cli/
```

Verify installation:
```bash
aws --version
```

### Install Docker
- macOS/Windows: [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Linux:
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose
```

---

## 🧪 Local Testing

Before deploying to AWS, test locally:

### 1. Create `.env` file
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Test with Docker Compose
```bash
# Build and start all services
docker-compose up --build

# Access the app at:
# Gradio UI: http://localhost:7860
# FastAPI: http://localhost:8000
# MCP Toolbox: http://localhost:5000
```

### 3. Verify Everything Works
- Upload a document
- Ask crypto questions
- Test stock queries
- Check finance tracker

### 4. Stop services
```bash
docker-compose down
```

---

## ☁️ AWS Setup

### Step 1: Configure AWS CLI

```bash
aws configure
```

Enter:
- AWS Access Key ID: `[Your access key]`
- AWS Secret Access Key: `[Your secret key]`
- Default region name: `us-east-1` (or your preferred region)
- Default output format: `json`

### Step 2: Create ECR Repository

```bash
# Create repository for Docker images
aws ecr create-repository \
    --repository-name multi-agent-app \
    --region us-east-1

# Save the repositoryUri from output
# Example: 123456789012.dkr.ecr.us-east-1.amazonaws.com/multi-agent-app
```

### Step 3: Create IAM Roles

#### A. ECS Task Execution Role
```bash
# Create trust policy file
cat > ecs-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role
aws iam create-role \
    --role-name ecsTaskExecutionRole \
    --assume-role-policy-document file://ecs-trust-policy.json

# Attach AWS managed policy
aws iam attach-role-policy \
    --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Get role ARN (save this!)
aws iam get-role --role-name ecsTaskExecutionRole --query 'Role.Arn' --output text
```

#### B. ECS Task Role (for application permissions)
```bash
# Create task role
aws iam create-role \
    --role-name ecsTaskRole \
    --assume-role-policy-document file://ecs-trust-policy.json

# Attach Secrets Manager read policy
aws iam attach-role-policy \
    --role-name ecsTaskRole \
    --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite

# Get role ARN (save this!)
aws iam get-role --role-name ecsTaskRole --query 'Role.Arn' --output text
```

### Step 4: Store Secrets in AWS Secrets Manager

```bash
# Get your AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION="us-east-1"

# Store each secret
aws secretsmanager create-secret \
    --name multi-agent/GOOGLE_API_KEY \
    --secret-string "your-google-api-key-here" \
    --region $AWS_REGION

aws secretsmanager create-secret \
    --name multi-agent/COINGECKO_API_KEY \
    --secret-string "your-coingecko-api-key-here" \
    --region $AWS_REGION

aws secretsmanager create-secret \
    --name multi-agent/ALPHA_VANTAGE_API_KEY \
    --secret-string "your-alpha-vantage-api-key-here" \
    --region $AWS_REGION

aws secretsmanager create-secret \
    --name multi-agent/CHROMA_API_KEY \
    --secret-string "your-chroma-api-key-here" \
    --region $AWS_REGION

aws secretsmanager create-secret \
    --name multi-agent/CHROMA_TENANT \
    --secret-string "your-chroma-tenant-here" \
    --region $AWS_REGION

aws secretsmanager create-secret \
    --name multi-agent/CHROMA_DATABASE \
    --secret-string "your-chroma-database-here" \
    --region $AWS_REGION
```

### Step 5: Update Task Definition

Edit `.aws/task-definition.json`:

1. Replace `YOUR_ACCOUNT_ID` with your AWS account ID (from Step 4)
2. Replace `YOUR_ECR_URI` with your ECR repository URI (from Step 2)
3. Replace `REGION` with your region (e.g., `us-east-1`)
4. Update the role ARNs with the ones you got in Step 3

Example:
```json
{
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskRole",
  ...
}
```

### Step 6: Create CloudWatch Log Group

```bash
aws logs create-log-group \
    --log-group-name /ecs/multi-agent-app \
    --region us-east-1
```

### Step 7: Create ECS Cluster

```bash
aws ecs create-cluster \
    --cluster-name multi-agent-cluster \
    --region us-east-1
```

### Step 8: Create VPC and Networking (if you don't have one)

```bash
# Create VPC
VPC_ID=$(aws ec2 create-vpc \
    --cidr-block 10.0.0.0/16 \
    --query 'Vpc.VpcId' \
    --output text)

# Enable DNS
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-support
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-hostnames

# Create subnets (in different AZs for high availability)
SUBNET1=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.1.0/24 \
    --availability-zone us-east-1a \
    --query 'Subnet.SubnetId' \
    --output text)

SUBNET2=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.2.0/24 \
    --availability-zone us-east-1b \
    --query 'Subnet.SubnetId' \
    --output text)

# Create Internet Gateway
IGW_ID=$(aws ec2 create-internet-gateway \
    --query 'InternetGateway.InternetGatewayId' \
    --output text)

aws ec2 attach-internet-gateway \
    --vpc-id $VPC_ID \
    --internet-gateway-id $IGW_ID

# Create route table
ROUTE_TABLE=$(aws ec2 create-route-table \
    --vpc-id $VPC_ID \
    --query 'RouteTable.RouteTableId' \
    --output text)

aws ec2 create-route \
    --route-table-id $ROUTE_TABLE \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id $IGW_ID

# Associate subnets with route table
aws ec2 associate-route-table --subnet-id $SUBNET1 --route-table-id $ROUTE_TABLE
aws ec2 associate-route-table --subnet-id $SUBNET2 --route-table-id $ROUTE_TABLE

# Enable auto-assign public IP
aws ec2 modify-subnet-attribute --subnet-id $SUBNET1 --map-public-ip-on-launch
aws ec2 modify-subnet-attribute --subnet-id $SUBNET2 --map-public-ip-on-launch

# Create security group
SECURITY_GROUP=$(aws ec2 create-security-group \
    --group-name multi-agent-sg \
    --description "Security group for multi-agent app" \
    --vpc-id $VPC_ID \
    --query 'GroupId' \
    --output text)

# Allow inbound traffic on port 7860 (Gradio)
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP \
    --protocol tcp \
    --port 7860 \
    --cidr 0.0.0.0/0

# Allow inbound traffic on port 8000 (FastAPI)
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0

# Save these values!
echo "VPC ID: $VPC_ID"
echo "Subnet 1: $SUBNET1"
echo "Subnet 2: $SUBNET2"
echo "Security Group: $SECURITY_GROUP"
```

### Step 9: Register Task Definition

```bash
aws ecs register-task-definition \
    --cli-input-json file://.aws/task-definition.json \
    --region us-east-1
```

### Step 10: Create ECS Service

```bash
aws ecs create-service \
    --cluster multi-agent-cluster \
    --service-name multi-agent-service \
    --task-definition multi-agent-app \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNET1,$SUBNET2],securityGroups=[$SECURITY_GROUP],assignPublicIp=ENABLED}" \
    --region us-east-1
```

---

## 🔐 GitHub Secrets Configuration

Go to your GitHub repository:
1. Navigate to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**

Add these secrets:

| Secret Name | Value | How to Get |
|------------|-------|------------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key | AWS Console → IAM → Users → Security credentials |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key | Same as above |

### Creating AWS Access Keys for GitHub Actions

```bash
# Create a new IAM user for GitHub Actions
aws iam create-user --user-name github-actions-deployer

# Attach required policies
aws iam attach-user-policy \
    --user-name github-actions-deployer \
    --policy-arn arn:aws:iam::aws:policy/AmazonECS_FullAccess

aws iam attach-user-policy \
    --user-name github-actions-deployer \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser

# Create access key
aws iam create-access-key --user-name github-actions-deployer
```

Copy the `AccessKeyId` and `SecretAccessKey` to GitHub Secrets.

---

## 🎯 First Deployment

### Option 1: Deploy via GitHub Actions (Recommended)

1. **Commit and push your changes:**
```bash
git add .
git commit -m "Add AWS deployment configuration"
git push origin main
```

2. **GitHub Actions will automatically:**
   - Build the Docker image
   - Push to ECR
   - Deploy to ECS

3. **Monitor deployment:**
   - Go to GitHub → Actions tab
   - Watch the deployment progress

### Option 2: Manual Deployment

```bash
# Build Docker image locally
docker build -t multi-agent-app .

# Get ECR login token
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin YOUR_ECR_URI

# Tag and push
docker tag multi-agent-app:latest YOUR_ECR_URI:latest
docker push YOUR_ECR_URI:latest

# Update ECS service
aws ecs update-service \
    --cluster multi-agent-cluster \
    --service multi-agent-service \
    --force-new-deployment \
    --region us-east-1
```

---

## 🔍 Monitoring and Troubleshooting

### Get Service Public IP

```bash
# Get task ARN
TASK_ARN=$(aws ecs list-tasks \
    --cluster multi-agent-cluster \
    --service-name multi-agent-service \
    --query 'taskArns[0]' \
    --output text)

# Get network interface ID
ENI_ID=$(aws ecs describe-tasks \
    --cluster multi-agent-cluster \
    --tasks $TASK_ARN \
    --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
    --output text)

# Get public IP
PUBLIC_IP=$(aws ec2 describe-network-interfaces \
    --network-interface-ids $ENI_ID \
    --query 'NetworkInterfaces[0].Association.PublicIp' \
    --output text)

echo "Access your app at: http://$PUBLIC_IP:7860"
```

### View Logs

```bash
# Via AWS CLI
aws logs tail /ecs/multi-agent-app --follow

# Or use AWS Console
# CloudWatch → Log groups → /ecs/multi-agent-app
```

### Check Service Status

```bash
aws ecs describe-services \
    --cluster multi-agent-cluster \
    --services multi-agent-service \
    --query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount}'
```

### Common Issues

#### 1. Task fails to start
- **Check logs:** CloudWatch logs for error messages
- **Verify secrets:** Ensure all Secrets Manager values are correct
- **Check IAM roles:** Task execution role must have Secrets Manager access

#### 2. Cannot access application
- **Security group:** Verify port 7860 is open
- **Public IP:** Ensure assignPublicIp is ENABLED
- **Health check:** App might still be starting (check logs)

#### 3. High costs
- **Reduce task count:** Scale down to 1 task
- **Use smaller resources:** Change to 0.5 vCPU / 1GB RAM
- **Stop when not in use:** Set desired count to 0

---

## 💰 Cost Estimation

### Monthly Costs (us-east-1)

**ECS Fargate (1 task, 1 vCPU, 2GB RAM):**
- vCPU: $0.04048/hour × 730 hours = ~$29.55/month
- Memory: $0.004445/GB/hour × 2GB × 730 hours = ~$6.49/month
- **Total ECS:** ~$36/month

**ECR Storage:**
- Docker images: ~$0.10/GB/month
- Estimated: ~$0.50/month

**Data Transfer:**
- First 1GB: Free
- Next 10TB: $0.09/GB
- Estimated (light use): ~$5/month

**CloudWatch Logs:**
- First 5GB: Free
- Estimated: ~$0-1/month

**AWS Secrets Manager:**
- $0.40/secret/month × 6 secrets = ~$2.40/month

**💵 Total Estimated Cost: ~$44/month**

### Cost Optimization Tips

1. **Stop when not in use:**
```bash
# Scale down to 0 tasks
aws ecs update-service \
    --cluster multi-agent-cluster \
    --service multi-agent-service \
    --desired-count 0
```

2. **Use smaller instance:**
   - Change to 0.5 vCPU / 1GB RAM
   - Saves ~50% on compute costs

3. **Use spot instances (advanced):**
   - Can save up to 70%
   - Requires capacity provider configuration

---

## 🎉 You're Done!

Your multi-agent assistant is now deployed on AWS with:
- ✅ Automatic deployments via GitHub Actions
- ✅ Secure secret management
- ✅ Auto-scaling capabilities
- ✅ Production-ready infrastructure

### Next Steps

1. **Add custom domain:** Use Route 53 + ALB
2. **Enable HTTPS:** Use AWS Certificate Manager
3. **Add authentication:** Implement OAuth or API keys
4. **Set up alerts:** CloudWatch alarms for errors
5. **Enable auto-scaling:** Based on CPU/memory usage

### Useful Commands

```bash
# View running tasks
aws ecs list-tasks --cluster multi-agent-cluster

# Scale service
aws ecs update-service --cluster multi-agent-cluster --service multi-agent-service --desired-count 2

# Stop service
aws ecs update-service --cluster multi-agent-cluster --service multi-agent-service --desired-count 0

# Delete service (cleanup)
aws ecs delete-service --cluster multi-agent-cluster --service multi-agent-service --force

# Delete cluster
aws ecs delete-cluster --cluster multi-agent-cluster
```

---

## 📚 Additional Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [GitHub Actions for AWS](https://github.com/aws-actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [AWS Cost Calculator](https://calculator.aws/)

---

**Need Help?** Open an issue in the GitHub repository.
