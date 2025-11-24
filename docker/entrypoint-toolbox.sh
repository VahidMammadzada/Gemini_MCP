#!/bin/bash
set -e

# Write GCP credentials from environment variable to file
if [ -n "$GCP_SERVICE_ACCOUNT_JSON" ]; then
    echo "Configuring GCP credentials from environment variable..."
    mkdir -p /app/credentials
    echo "$GCP_SERVICE_ACCOUNT_JSON" > /app/credentials/credentials.json
    export GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/credentials.json
    echo "GCP credentials configured successfully"
else
    echo "WARNING: GCP_SERVICE_ACCOUNT_JSON not set. Cloud SQL authentication may fail."
fi

# Start MCP Toolbox
echo "Starting MCP Toolbox server on port 5000..."
exec toolbox --prebuilt cloud-sql-postgres --port 5000 --address 0.0.0.0