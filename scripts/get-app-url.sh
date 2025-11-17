#!/bin/bash
# Script to get the public URL of your deployed application

set -e

CLUSTER_NAME=${1:-multi-agent-cluster}
SERVICE_NAME=${2:-multi-agent-service}
REGION=${3:-us-east-1}

echo "🔍 Finding your application..."
echo "Cluster: $CLUSTER_NAME"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"
echo ""

# Get task ARN
echo "📋 Getting task ARN..."
TASK_ARN=$(aws ecs list-tasks \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --region $REGION \
    --query 'taskArns[0]' \
    --output text)

if [ "$TASK_ARN" == "None" ] || [ -z "$TASK_ARN" ]; then
    echo "❌ No running tasks found!"
    echo "Check if your service is running:"
    echo "  aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME"
    exit 1
fi

echo "✅ Task ARN: $TASK_ARN"
echo ""

# Get network interface ID
echo "🔌 Getting network interface..."
ENI_ID=$(aws ecs describe-tasks \
    --cluster $CLUSTER_NAME \
    --tasks $TASK_ARN \
    --region $REGION \
    --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
    --output text)

if [ -z "$ENI_ID" ]; then
    echo "❌ Could not get network interface ID!"
    exit 1
fi

echo "✅ Network Interface: $ENI_ID"
echo ""

# Get public IP
echo "🌐 Getting public IP address..."
PUBLIC_IP=$(aws ec2 describe-network-interfaces \
    --network-interface-ids $ENI_ID \
    --region $REGION \
    --query 'NetworkInterfaces[0].Association.PublicIp' \
    --output text)

if [ "$PUBLIC_IP" == "None" ] || [ -z "$PUBLIC_IP" ]; then
    echo "❌ No public IP assigned!"
    echo "Make sure your service has assignPublicIp enabled."
    exit 1
fi

echo "✅ Public IP: $PUBLIC_IP"
echo ""

# Display URLs
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Your application is running!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎨 Streamlit UI:  http://$PUBLIC_IP:8501"
echo "🔧 FastAPI:       http://$PUBLIC_IP:8000"
echo "📚 API Docs:      http://$PUBLIC_IP:8000/docs"
echo "❤️  Health:        http://$PUBLIC_IP:8000/health"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test connectivity
echo "🧪 Testing connectivity..."
if curl -s --max-time 5 "http://$PUBLIC_IP:8000/health" > /dev/null 2>&1; then
    echo "✅ Application is responding!"
else
    echo "⚠️  Application is not responding yet."
    echo "It may still be starting up. Wait 30-60 seconds and try again."
fi
