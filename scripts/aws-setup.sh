#!/bin/bash
# Automated AWS ECS Setup Script
# This script sets up all AWS resources needed for deployment

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Multi-Agent Assistant - AWS Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Configuration
REGION=${AWS_REGION:-us-east-1}
CLUSTER_NAME="multi-agent-cluster"
SERVICE_NAME="multi-agent-service"
REPOSITORY_NAME="multi-agent-app"

echo "📍 Region: $REGION"
echo "📦 Repository: $REPOSITORY_NAME"
echo "🎯 Cluster: $CLUSTER_NAME"
echo ""

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed!"
    echo "Install it from: https://aws.amazon.com/cli/"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured!"
    echo "Run: aws configure"
    exit 1
fi

# Get account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "✅ AWS Account: $AWS_ACCOUNT_ID"
echo ""

# Step 1: Create ECR Repository
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Step 1: Creating ECR Repository..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if aws ecr describe-repositories --repository-names $REPOSITORY_NAME --region $REGION &> /dev/null; then
    echo "✅ ECR repository already exists"
else
    aws ecr create-repository \
        --repository-name $REPOSITORY_NAME \
        --region $REGION \
        --image-scanning-configuration scanOnPush=true
    echo "✅ ECR repository created"
fi
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME"
echo "📍 ECR URI: $ECR_URI"
echo ""

# Step 2: Create IAM Roles
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔐 Step 2: Creating IAM Roles..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create trust policy
cat > /tmp/ecs-trust-policy.json << 'EOF'
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

# Create ecsTaskExecutionRole
if aws iam get-role --role-name ecsTaskExecutionRole &> /dev/null; then
    echo "✅ ecsTaskExecutionRole already exists"
else
    aws iam create-role \
        --role-name ecsTaskExecutionRole \
        --assume-role-policy-document file:///tmp/ecs-trust-policy.json

    aws iam attach-role-policy \
        --role-name ecsTaskExecutionRole \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

    echo "✅ ecsTaskExecutionRole created"
fi

# Create ecsTaskRole
if aws iam get-role --role-name ecsTaskRole &> /dev/null; then
    echo "✅ ecsTaskRole already exists"
else
    aws iam create-role \
        --role-name ecsTaskRole \
        --assume-role-policy-document file:///tmp/ecs-trust-policy.json

    aws iam attach-role-policy \
        --role-name ecsTaskRole \
        --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite

    echo "✅ ecsTaskRole created"
fi

EXECUTION_ROLE_ARN=$(aws iam get-role --role-name ecsTaskExecutionRole --query 'Role.Arn' --output text)
TASK_ROLE_ARN=$(aws iam get-role --role-name ecsTaskRole --query 'Role.Arn' --output text)

echo "📍 Execution Role ARN: $EXECUTION_ROLE_ARN"
echo "📍 Task Role ARN: $TASK_ROLE_ARN"
echo ""

# Step 3: Create CloudWatch Log Group
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Step 3: Creating CloudWatch Log Group..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if aws logs describe-log-groups --log-group-name-prefix /ecs/multi-agent-app --region $REGION | grep -q "/ecs/multi-agent-app"; then
    echo "✅ Log group already exists"
else
    aws logs create-log-group \
        --log-group-name /ecs/multi-agent-app \
        --region $REGION
    echo "✅ Log group created"
fi
echo ""

# Step 4: Create VPC and Networking
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Step 4: Setting up VPC and Networking..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check for existing VPC
EXISTING_VPC=$(aws ec2 describe-vpcs \
    --filters "Name=tag:Name,Values=multi-agent-vpc" \
    --query 'Vpcs[0].VpcId' \
    --output text \
    --region $REGION)

if [ "$EXISTING_VPC" != "None" ] && [ -n "$EXISTING_VPC" ]; then
    echo "✅ Using existing VPC: $EXISTING_VPC"
    VPC_ID=$EXISTING_VPC
else
    # Create VPC
    VPC_ID=$(aws ec2 create-vpc \
        --cidr-block 10.0.0.0/16 \
        --region $REGION \
        --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=multi-agent-vpc}]' \
        --query 'Vpc.VpcId' \
        --output text)

    aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-support --region $REGION
    aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-hostnames --region $REGION

    echo "✅ VPC created: $VPC_ID"
fi

# Create subnets
echo "Creating subnets..."
SUBNET1=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.1.0/24 \
    --availability-zone ${REGION}a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=multi-agent-subnet-1}]' \
    --query 'Subnet.SubnetId' \
    --output text \
    --region $REGION 2>/dev/null || \
    aws ec2 describe-subnets \
        --filters "Name=tag:Name,Values=multi-agent-subnet-1" "Name=vpc-id,Values=$VPC_ID" \
        --query 'Subnets[0].SubnetId' \
        --output text \
        --region $REGION)

SUBNET2=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.2.0/24 \
    --availability-zone ${REGION}b \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=multi-agent-subnet-2}]' \
    --query 'Subnet.SubnetId' \
    --output text \
    --region $REGION 2>/dev/null || \
    aws ec2 describe-subnets \
        --filters "Name=tag:Name,Values=multi-agent-subnet-2" "Name=vpc-id,Values=$VPC_ID" \
        --query 'Subnets[0].SubnetId' \
        --output text \
        --region $REGION)

echo "✅ Subnets: $SUBNET1, $SUBNET2"

# Create and attach Internet Gateway
IGW_ID=$(aws ec2 create-internet-gateway \
    --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=multi-agent-igw}]' \
    --query 'InternetGateway.InternetGatewayId' \
    --output text \
    --region $REGION 2>/dev/null || \
    aws ec2 describe-internet-gateways \
        --filters "Name=tag:Name,Values=multi-agent-igw" \
        --query 'InternetGateways[0].InternetGatewayId' \
        --output text \
        --region $REGION)

aws ec2 attach-internet-gateway \
    --vpc-id $VPC_ID \
    --internet-gateway-id $IGW_ID \
    --region $REGION 2>/dev/null || true

echo "✅ Internet Gateway: $IGW_ID"

# Create route table
ROUTE_TABLE=$(aws ec2 create-route-table \
    --vpc-id $VPC_ID \
    --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=multi-agent-rt}]' \
    --query 'RouteTable.RouteTableId' \
    --output text \
    --region $REGION 2>/dev/null || \
    aws ec2 describe-route-tables \
        --filters "Name=tag:Name,Values=multi-agent-rt" "Name=vpc-id,Values=$VPC_ID" \
        --query 'RouteTables[0].RouteTableId' \
        --output text \
        --region $REGION)

aws ec2 create-route \
    --route-table-id $ROUTE_TABLE \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id $IGW_ID \
    --region $REGION 2>/dev/null || true

aws ec2 associate-route-table --subnet-id $SUBNET1 --route-table-id $ROUTE_TABLE --region $REGION 2>/dev/null || true
aws ec2 associate-route-table --subnet-id $SUBNET2 --route-table-id $ROUTE_TABLE --region $REGION 2>/dev/null || true

# Enable auto-assign public IP
aws ec2 modify-subnet-attribute --subnet-id $SUBNET1 --map-public-ip-on-launch --region $REGION
aws ec2 modify-subnet-attribute --subnet-id $SUBNET2 --map-public-ip-on-launch --region $REGION

echo "✅ Route table configured"

# Create security group
SECURITY_GROUP=$(aws ec2 create-security-group \
    --group-name multi-agent-sg \
    --description "Security group for multi-agent app" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text 2>/dev/null || \
    aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=multi-agent-sg" "Name=vpc-id,Values=$VPC_ID" \
        --query 'SecurityGroups[0].GroupId' \
        --output text \
        --region $REGION)

# Allow inbound traffic
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP \
    --protocol tcp \
    --port 7860 \
    --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

echo "✅ Security Group: $SECURITY_GROUP"
echo ""

# Step 5: Create ECS Cluster
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Step 5: Creating ECS Cluster..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if aws ecs describe-clusters --clusters $CLUSTER_NAME --region $REGION | grep -q "ACTIVE"; then
    echo "✅ ECS cluster already exists"
else
    aws ecs create-cluster \
        --cluster-name $CLUSTER_NAME \
        --region $REGION
    echo "✅ ECS cluster created"
fi
echo ""

# Step 6: Update Task Definition
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 Step 6: Updating Task Definition..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Update task definition with correct values
sed -i.bak \
    -e "s|YOUR_ACCOUNT_ID|$AWS_ACCOUNT_ID|g" \
    -e "s|YOUR_ECR_URI|$ECR_URI|g" \
    -e "s|REGION|$REGION|g" \
    -e "s|ACCOUNT_ID|$AWS_ACCOUNT_ID|g" \
    .aws/task-definition.json

# Update execution and task role ARNs
jq --arg exec_role "$EXECUTION_ROLE_ARN" \
   --arg task_role "$TASK_ROLE_ARN" \
   '.executionRoleArn = $exec_role | .taskRoleArn = $task_role' \
   .aws/task-definition.json > .aws/task-definition.json.tmp && \
   mv .aws/task-definition.json.tmp .aws/task-definition.json

echo "✅ Task definition updated"
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ AWS Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Configuration Summary:"
echo "  • Account ID:      $AWS_ACCOUNT_ID"
echo "  • Region:          $REGION"
echo "  • ECR URI:         $ECR_URI"
echo "  • VPC ID:          $VPC_ID"
echo "  • Subnets:         $SUBNET1, $SUBNET2"
echo "  • Security Group:  $SECURITY_GROUP"
echo "  • ECS Cluster:     $CLUSTER_NAME"
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Store your secrets in AWS Secrets Manager:"
echo "   ./scripts/store-secrets.sh"
echo ""
echo "2. Build and push your Docker image:"
echo "   aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI"
echo "   docker build -t $REPOSITORY_NAME ."
echo "   docker tag $REPOSITORY_NAME:latest $ECR_URI:latest"
echo "   docker push $ECR_URI:latest"
echo ""
echo "3. Register task definition:"
echo "   aws ecs register-task-definition --cli-input-json file://.aws/task-definition.json --region $REGION"
echo ""
echo "4. Create ECS service:"
echo "   aws ecs create-service \\"
echo "     --cluster $CLUSTER_NAME \\"
echo "     --service-name $SERVICE_NAME \\"
echo "     --task-definition multi-agent-app \\"
echo "     --desired-count 1 \\"
echo "     --launch-type FARGATE \\"
echo "     --network-configuration \"awsvpcConfiguration={subnets=[$SUBNET1,$SUBNET2],securityGroups=[$SECURITY_GROUP],assignPublicIp=ENABLED}\" \\"
echo "     --region $REGION"
echo ""
echo "5. Or simply push to GitHub (GitHub Actions will deploy automatically)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
