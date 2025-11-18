# GitHub Repository Secrets Configuration

This document explains the secrets you need to configure in your GitHub repository for the CI/CD pipeline to work properly.

## Required Secrets

Navigate to: **Repository Settings → Secrets and variables → Actions → New repository secret**

### Google Cloud Authentication (Choose One)

#### Option 1: API Key (Recommended for Testing)
- **Name:** `GOOGLE_API_KEY`
- **Value:** Your Google API key from Google Cloud Console
- **How to get it:**
  1. Go to [Google Cloud Console](https://console.cloud.google.com/)
  2. Navigate to **APIs & Services → Credentials**
  3. Click **Create Credentials → API Key**
  4. Copy the generated key

#### Option 2: Service Account JSON (More Secure)
- **Name:** `GCP_SA_KEY`
- **Value:** Base64-encoded service account JSON
- **How to create it:**
  ```bash
  # On your local machine
  cat path/to/service-account.json | base64 -w 0 > encoded.txt
  # Copy content of encoded.txt and paste as secret value
  ```

### ChromaDB Configuration (Optional but Recommended)

#### CHROMA_API_KEY
- **Name:** `CHROMA_API_KEY`
- **Value:** Your ChromaDB Cloud API key
- **Required for:** RAG agent to store/retrieve documents

#### CHROMA_TENANT
- **Name:** `CHROMA_TENANT`
- **Value:** Your ChromaDB tenant ID
- **Required for:** ChromaDB Cloud connection

#### CHROMA_DATABASE
- **Name:** `CHROMA_DATABASE`
- **Value:** Your ChromaDB database name
- **Required for:** ChromaDB Cloud connection

### Optional API Keys

#### COINGECKO_API_KEY
- **Name:** `COINGECKO_API_KEY`
- **Value:** Your CoinGecko API key
- **Required for:** Crypto agent enhanced features

#### ALPHA_VANTAGE_API_KEY
- **Name:** `ALPHA_VANTAGE_API_KEY`
- **Value:** Your Alpha Vantage API key
- **Required for:** Stock market data

## Secrets Summary Table

| Secret Name | Required | Purpose | How to Get |
|------------|----------|---------|------------|
| `GOOGLE_API_KEY` | ✅ Yes* | Google Generative AI | [Google Cloud Console](https://console.cloud.google.com/apis/credentials) |
| `GCP_SA_KEY` | ✅ Yes* | Google Service Account | Base64-encode service account JSON |
| `CHROMA_API_KEY` | ⚠️ Recommended | ChromaDB access | [ChromaDB Cloud](https://www.trychroma.com/) |
| `CHROMA_TENANT` | ⚠️ Recommended | ChromaDB tenant | ChromaDB Cloud dashboard |
| `CHROMA_DATABASE` | ⚠️ Recommended | ChromaDB database | ChromaDB Cloud dashboard |
| `COINGECKO_API_KEY` | ❌ Optional | Crypto data | [CoinGecko](https://www.coingecko.com/api) |
| `ALPHA_VANTAGE_API_KEY` | ❌ Optional | Stock data | [Alpha Vantage](https://www.alphavantage.co/support/#api-key) |

*Choose either `GOOGLE_API_KEY` **OR** `GCP_SA_KEY`, not both.

## How to Add Secrets to GitHub

### Step-by-Step Guide

1. **Navigate to your repository on GitHub**

2. **Go to Settings**
   - Click on the **Settings** tab (you need admin access)

3. **Open Secrets Section**
   - In the left sidebar, click **Secrets and variables**
   - Click **Actions**

4. **Add New Secret**
   - Click **New repository secret**
   - Enter the secret **Name** (e.g., `GOOGLE_API_KEY`)
   - Paste the secret **Value**
   - Click **Add secret**

5. **Repeat for all required secrets**

## Verifying Secrets Configuration

After adding secrets, you can verify they're configured correctly:

1. **Check GitHub Actions**
   - Go to the **Actions** tab in your repository
   - The workflow should now run without credential errors

2. **Test Locally**
   ```bash
   # Set environment variables locally
   export GOOGLE_API_KEY="your-key-here"
   export CHROMA_API_KEY="your-chroma-key"

   # Run tests
   pytest test_api.py -v
   ```

## Security Best Practices

### ✅ DO:
- Use GitHub Secrets for all sensitive information
- Rotate API keys regularly
- Use service accounts with minimal required permissions
- Review secret access logs periodically

### ❌ DON'T:
- Never commit credentials to your repository
- Never log secret values in code
- Never share secrets in public forums
- Never use production credentials for testing

## Troubleshooting

### Tests Still Failing with Credential Errors?

1. **Check secret names match exactly**
   - Secret names are case-sensitive
   - Must match the names used in `.github/workflows/ci.yml`

2. **Verify secret values**
   - No extra spaces or newlines
   - Base64 encoding is correct (for `GCP_SA_KEY`)

3. **Check workflow syntax**
   ```yaml
   env:
     GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}  # ✅ Correct
     GOOGLE_API_KEY: "your-key-here"                # ❌ Wrong - never hardcode
   ```

4. **Re-run the workflow**
   - Go to Actions tab
   - Click on the failed workflow
   - Click **Re-run jobs**

### Local Testing Without Secrets

If you want to run tests locally without real credentials:

```bash
# Tests will use mocks automatically
pytest test_api.py -v

# Run only unit tests (no credentials needed)
pytest test_api.py -m unit -v

# Skip integration tests
pytest test_api.py -m "not integration" -v
```

## Additional Resources

- [GitHub Encrypted Secrets Documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Google Cloud Authentication Guide](https://cloud.google.com/docs/authentication/getting-started)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## Need Help?

If you encounter issues with secrets configuration:

1. Check the workflow logs in the Actions tab
2. Review this documentation
3. Verify all secret names match exactly
4. Ensure secret values are correct and properly encoded
