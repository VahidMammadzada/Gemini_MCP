# Streamlit UI for Multi-Agent Assistant

A modern, production-ready chat interface built with Streamlit that connects to the FastAPI backend.

## Features

✨ **Modern Chat Interface**
- Real-time streaming responses
- Conversation history management
- Message persistence across sessions

🔍 **Intermediate Steps Visualization**
- View AI reasoning process (thinking steps)
- See which agents are called
- Monitor agent responses
- Toggle display on/off

📄 **Document Management**
- Upload PDF, TXT, DOCX files
- Real-time upload status
- Document details display

🔧 **System Monitoring**
- API health check
- Agent status display
- Connection diagnostics

⚙️ **User Controls**
- Clear chat history
- Show/hide reasoning steps
- Quick example queries

---

## Quick Start

### 1. Start the FastAPI Backend

First, ensure the FastAPI server is running:

```bash
# Terminal 1 - FastAPI Backend
python api.py

# OR with uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Run the Streamlit App

```bash
# Terminal 2 - Streamlit UI
streamlit run streamlit_app.py
```

The app will open automatically at: **http://localhost:8501**

---

## Configuration

### API Base URL

By default, the app connects to `http://localhost:8000`.

To change this, edit `.streamlit/secrets.toml`:

```toml
API_BASE_URL = "http://localhost:8000"
```

Or use environment variables:

```bash
export API_BASE_URL="https://your-api-domain.com"
streamlit run streamlit_app.py
```

---

## Usage Guide

### Chat Interface

1. **Type your question** in the chat input at the bottom
2. **Press Enter** to submit
3. **Watch the response stream** in real-time

### Viewing Reasoning Steps

Enable "Show Reasoning Steps" in the sidebar to see:
- 💭 **Thinking**: AI's reasoning process
- 🔧 **Actions**: Which agents are called
- 📊 **Observations**: Agent responses

### Document Upload

1. Click **"Upload Documents"** in the sidebar
2. Select a PDF, TXT, or DOCX file
3. Click **"Upload"**
4. Ask questions about your document!

### Example Queries

Quick-start buttons at the bottom:
- 💰 **Bitcoin price?** - Test crypto agent
- 📈 **AAPL stock info?** - Test stock agent
- 🔍 **AI news?** - Test search agent

---

## Architecture

```
┌─────────────────────────────────────┐
│  Streamlit UI (Port 8501)          │
│  - Chat interface                   │
│  - Document upload                  │
│  - System monitoring                │
└─────────────┬───────────────────────┘
              │ HTTP/SSE
              │
┌─────────────▼───────────────────────┐
│  FastAPI Backend (Port 8000)        │
│  - /api/v1/chat/stream (SSE)       │
│  - /api/v1/documents/upload        │
│  - /health                          │
└─────────────┬───────────────────────┘
              │
              │
┌─────────────▼───────────────────────┐
│  Multi-Agent System                 │
│  - ReAct Supervisor                 │
│  - 5 Specialized Agents             │
└─────────────────────────────────────┘
```

---

## Comparison: Streamlit vs Gradio

| Feature | Streamlit | Gradio |
|---------|-----------|--------|
| **Interface** | Clean, modern | Functional |
| **Customization** | High (CSS, HTML) | Medium |
| **Streaming** | SSE via HTTP | Built-in |
| **State Management** | Session state | Built-in |
| **Production Ready** | ✅ Yes | ✅ Yes |
| **Mobile Friendly** | ✅ Yes | ✅ Yes |
| **Custom Components** | ✅ Extensive | Limited |
| **Learning Curve** | Easy | Very Easy |

---

## Advanced Features

### Custom Styling

The app includes custom CSS for:
- Thinking steps (green border)
- Action steps (yellow border)
- Observation steps (blue border)
- Agent badges

Edit the CSS in `streamlit_app.py` (search for `st.markdown("""<style>`)

### Session State

Chat history is preserved in `st.session_state`:
- `messages`: Full chat display
- `chat_history`: API format (role/content)
- `show_intermediate_steps`: Toggle setting

### Error Handling

The app gracefully handles:
- API connection failures
- Timeout errors
- Upload errors
- Streaming interruptions

---

## Deployment Options

### Option 1: Streamlit Community Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets in the dashboard
5. Deploy!

**Secrets to add:**
```toml
API_BASE_URL = "https://your-fastapi-deployment.com"
```

### Option 2: Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .
COPY .streamlit/ .streamlit/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t streamlit-chat .
docker run -p 8501:8501 streamlit-chat
```

### Option 3: AWS/GCP/Azure

Deploy alongside your FastAPI backend:
- **FastAPI**: Port 8000 (internal)
- **Streamlit**: Port 8501 (public-facing)
- Use reverse proxy (nginx) for routing

---

## Troubleshooting

### "Connection Error" when chatting

**Problem:** Streamlit can't reach FastAPI backend

**Solution:**
1. Verify FastAPI is running: `curl http://localhost:8000/health`
2. Check API_BASE_URL in `.streamlit/secrets.toml`
3. Click "Check Connection" in sidebar

### "API Offline" status

**Problem:** FastAPI server not responding

**Solution:**
1. Start FastAPI: `python api.py`
2. Check for port conflicts
3. Verify agents are initialized

### Streaming not working

**Problem:** Messages not appearing in real-time

**Solution:**
1. Clear browser cache
2. Restart Streamlit: `Ctrl+C` then `streamlit run streamlit_app.py`
3. Check network tab for SSE connection

### Document upload fails

**Problem:** File upload returns error

**Solution:**
1. Check file type (PDF, TXT, DOCX only)
2. Verify file size (<10MB recommended)
3. Check FastAPI logs for details

---

## Environment Variables

```bash
# API Backend URL
export API_BASE_URL="http://localhost:8000"

# Streamlit Port (default: 8501)
export STREAMLIT_SERVER_PORT=8501

# Enable/Disable Auto-reload
export STREAMLIT_SERVER_FILE_WATCHER_TYPE="auto"
```

---

## Development Tips

### Hot Reload

Streamlit auto-reloads when you save files. To manually reload:
- Press `R` in the app
- Or click "Rerun" in the top-right corner

### Debugging

Enable debug mode:
```python
# Add to streamlit_app.py
st.write("Debug:", st.session_state)
```

### Custom Components

Add custom components:
```python
import streamlit.components.v1 as components

components.html("<div>Custom HTML</div>")
```

---

## API Reference

### Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check API status |
| `/api/v1/chat/stream` | POST | Stream chat responses |
| `/api/v1/documents/upload` | POST | Upload documents |

### Request Format

**Chat Request:**
```json
{
  "message": "What is Bitcoin?",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response Events:**
- `type: thinking` - Reasoning step
- `type: action` - Agent call
- `type: observation` - Agent result
- `type: final_token` - Streaming answer
- `type: final_complete` - Done
- `type: error` - Error occurred

---

## Support

For issues or questions:
- Check FastAPI logs: Look for errors in Terminal 1
- Check Streamlit logs: Look for errors in Terminal 2
- Review browser console: F12 → Console tab

---

## Next Steps

- [ ] Add authentication (e.g., OAuth, JWT)
- [ ] Implement conversation export
- [ ] Add voice input/output
- [ ] Create mobile-responsive layout
- [ ] Add chat themes
- [ ] Implement multi-user sessions

---

**Enjoy your Streamlit-powered Multi-Agent Assistant! 🚀**
