#!/bin/bash
# Interactive script to store secrets in AWS Secrets Manager

set -e

REGION=${AWS_REGION:-us-east-1}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔐 Store Secrets in AWS Secrets Manager"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Region: $REGION"
echo ""
echo "This script will help you store your API keys securely."
echo ""

# Function to store secret
store_secret() {
    local name=$1
    local description=$2

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📝 $description"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check if secret already exists
    if aws secretsmanager describe-secret --secret-id "$name" --region $REGION &> /dev/null; then
        echo "⚠️  Secret '$name' already exists."
        read -p "Do you want to update it? (y/n): " update
        if [ "$update" != "y" ]; then
            echo "⏭️  Skipping..."
            echo ""
            return
        fi

        read -sp "Enter value: " value
        echo ""

        aws secretsmanager update-secret \
            --secret-id "$name" \
            --secret-string "$value" \
            --region $REGION

        echo "✅ Secret updated"
    else
        read -sp "Enter value: " value
        echo ""

        aws secretsmanager create-secret \
            --name "$name" \
            --description "$description" \
            --secret-string "$value" \
            --region $REGION > /dev/null

        echo "✅ Secret created"
    fi
    echo ""
}

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed!"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured!"
    echo "Run: aws configure"
    exit 1
fi

echo "You can load values from .env file or enter manually."
read -p "Do you have a .env file to load from? (y/n): " has_env

if [ "$has_env" = "y" ]; then
    if [ -f .env ]; then
        echo "✅ Loading from .env file..."
        source .env 2>/dev/null || true
    else
        echo "⚠️  .env file not found, will enter manually"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Required Secrets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Store each secret
store_secret "multi-agent/GOOGLE_API_KEY" "Google AI API Key (for Gemini LLM)"
store_secret "multi-agent/COINGECKO_API_KEY" "CoinGecko API Key (for crypto data)"
store_secret "multi-agent/ALPHA_VANTAGE_API_KEY" "Alpha Vantage API Key (for stock data)"
store_secret "multi-agent/CHROMA_API_KEY" "ChromaDB API Key (for vector storage)"
store_secret "multi-agent/CHROMA_TENANT" "ChromaDB Tenant"
store_secret "multi-agent/CHROMA_DATABASE" "ChromaDB Database Name"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All secrets stored successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Stored Secrets:"
aws secretsmanager list-secrets \
    --filters Key=name,Values=multi-agent/ \
    --query 'SecretList[*].[Name,Description]' \
    --output table \
    --region $REGION

echo ""
echo "💡 Tip: You can view/edit secrets in AWS Console:"
echo "   https://console.aws.amazon.com/secretsmanager/home?region=$REGION"
echo ""
