"""Streamlit UI for Multi-Agent Assistant connecting to FastAPI backend."""
import streamlit as st
import requests
import json
from typing import Dict, List, Optional
import time
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# FastAPI backend URL
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# API Client Functions
# ============================================================================

def check_api_health() -> Dict:
    """Check if FastAPI backend is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "offline", "error": str(e)}


def stream_chat_response(message: str, history: List[Dict]) -> Dict:
    """
    Stream chat response from FastAPI using Server-Sent Events.

    Yields update dictionaries as they arrive.
    """
    payload = {
        "message": message,
        "history": history
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat/stream",
            json=payload,
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
                    try:
                        event = json.loads(data_str)
                        yield event
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield {"type": "error", "error": str(e)}


def upload_document(file) -> Dict:
    """Upload document to FastAPI backend."""
    try:
        files = {'file': (file.name, file, file.type)}
        response = requests.post(
            f"{API_BASE_URL}/api/v1/documents/upload",
            files=files,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


# ============================================================================
# Session State Initialization
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "api_status" not in st.session_state:
    st.session_state.api_status = None

if "show_intermediate_steps" not in st.session_state:
    st.session_state.show_intermediate_steps = True

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .thinking-step {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #4CAF50;
        margin: 0.5rem 0;
    }

    .action-step {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #ffc107;
        margin: 0.5rem 0;
    }

    .observation-step {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #17a2b8;
        margin: 0.5rem 0;
    }

    .agent-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        background-color: #6c757d;
        color: white;
        font-size: 0.875rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.title("🤖 Multi-Agent Assistant")
    st.markdown("---")

    # API Status
    st.subheader("🔧 System Status")

    if st.button("🔄 Check Connection", use_container_width=True):
        with st.spinner("Checking API status..."):
            st.session_state.api_status = check_api_health()

    if st.session_state.api_status:
        status = st.session_state.api_status
        if status.get("status") == "healthy":
            st.success("✅ API Online")
            if status.get("agents"):
                with st.expander("Agent Status", expanded=False):
                    for agent, is_active in status["agents"].items():
                        emoji = "✅" if is_active else "❌"
                        st.write(f"{emoji} **{agent.title()}**")
        else:
            st.error(f"❌ API Offline: {status.get('error', 'Unknown error')}")
    else:
        st.info("Click 'Check Connection' to verify API status")

    st.markdown("---")

    # Document Upload
    st.subheader("📄 Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        help="Upload documents to the RAG agent for querying"
    )

    if uploaded_file and st.button("📤 Upload", use_container_width=True):
        with st.spinner("Uploading document..."):
            result = upload_document(uploaded_file)
            if result.get("success"):
                st.success(f"✅ {result.get('message', 'Upload successful')}")
                if result.get("details"):
                    with st.expander("Upload Details"):
                        st.json(result["details"])
            else:
                st.error(f"❌ {result.get('message', 'Upload failed')}")

    st.markdown("---")

    # Settings
    st.subheader("⚙️ Settings")
    st.session_state.show_intermediate_steps = st.checkbox(
        "Show Reasoning Steps",
        value=st.session_state.show_intermediate_steps,
        help="Display intermediate thinking, actions, and observations"
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")

    # Information
    st.subheader("ℹ️ Available Agents")
    st.markdown("""
    - 🪙 **Crypto Agent**: Cryptocurrency prices & data
    - 📊 **Stock Agent**: Stock market information
    - 💼 **Finance Tracker**: Portfolio management
    - 📚 **RAG Agent**: Document Q&A
    - 🔍 **Search Agent**: Web search
    """)

    st.markdown("---")
    st.caption("Powered by FastAPI + Gemini 2.5 Pro")

# ============================================================================
# Main Chat Interface
# ============================================================================

st.title("💬 Multi-Agent Chat Assistant")
st.markdown("Ask questions about crypto, stocks, documents, or search the web!")

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        if role == "assistant" and "metadata" in message:
            metadata = message["metadata"]

            # Intermediate steps
            if metadata.get("type") == "thinking":
                if st.session_state.show_intermediate_steps:
                    with st.expander(f"💭 Step {metadata.get('step', '?')}: Reasoning", expanded=False):
                        st.markdown(f"**Thought:** {metadata.get('thought', 'N/A')}")
                        st.markdown(f"**Action:** `{metadata.get('action', 'N/A').upper()}`")
                        st.markdown(f"**Justification:** {metadata.get('justification', 'N/A')}")

            elif metadata.get("type") == "action":
                if st.session_state.show_intermediate_steps:
                    agent = metadata.get('agent', 'unknown')
                    st.info(f"🔧 Calling **{agent.title()}** Agent...")

            elif metadata.get("type") == "observation":
                if st.session_state.show_intermediate_steps:
                    agent = metadata.get('agent', 'unknown')
                    with st.expander(f"📊 {agent.title()} Agent Results", expanded=False):
                        st.write(content)

            else:
                # Regular assistant message
                st.markdown(content)
        else:
            # User or regular message
            st.markdown(content)

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from API
    with st.chat_message("assistant"):
        # Create placeholder for streaming response
        response_placeholder = st.empty()
        thinking_container = st.container()

        final_answer = ""
        current_step = 0

        try:
            for event in stream_chat_response(prompt, st.session_state.chat_history[:-1]):
                event_type = event.get("type", "unknown")

                if event_type == "thinking":
                    current_step = event.get("step", current_step + 1)
                    if st.session_state.show_intermediate_steps:
                        with thinking_container:
                            with st.expander(f"💭 Step {current_step}: Reasoning", expanded=False):
                                st.markdown(f"**Thought:** {event.get('thought', 'N/A')}")
                                st.markdown(f"**Action:** `{event.get('action', 'N/A').upper()}`")
                                st.markdown(f"**Justification:** {event.get('justification', 'N/A')}")

                        # Add to messages for display persistence
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "",
                            "metadata": {
                                "type": "thinking",
                                "step": current_step,
                                "thought": event.get('thought', ''),
                                "action": event.get('action', ''),
                                "justification": event.get('justification', '')
                            }
                        })

                elif event_type == "action":
                    agent = event.get("agent", "unknown")
                    if st.session_state.show_intermediate_steps:
                        with thinking_container:
                            st.info(f"🔧 Calling **{agent.title()}** Agent...")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Calling {agent} agent...",
                            "metadata": {"type": "action", "agent": agent}
                        })

                elif event_type == "observation":
                    agent = event.get("agent", "unknown")
                    summary = event.get("summary", "")
                    if st.session_state.show_intermediate_steps:
                        with thinking_container:
                            with st.expander(f"📊 {agent.title()} Agent Results", expanded=False):
                                st.write(summary)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": summary,
                            "metadata": {"type": "observation", "agent": agent}
                        })

                elif event_type == "final_start":
                    # Clear placeholder for final answer
                    response_placeholder.markdown("Synthesizing final answer...")

                elif event_type == "final_token":
                    final_answer = event.get("accumulated", "")
                    response_placeholder.markdown(final_answer)

                elif event_type == "final_complete":
                    # Display complete final answer
                    if final_answer:
                        response_placeholder.markdown(final_answer)

                elif event_type == "error":
                    error_msg = event.get("error", "Unknown error")
                    response_placeholder.error(f"❌ Error: {error_msg}")
                    final_answer = f"Error: {error_msg}"

        except Exception as e:
            response_placeholder.error(f"❌ Connection Error: {str(e)}")
            final_answer = f"Error: {str(e)}"

    # Add final answer to chat history
    if final_answer:
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": final_answer
        })

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Example Queries:**")
    if st.button("💰 Bitcoin price?"):
        st.session_state.temp_query = "What is the current price of Bitcoin?"
        st.rerun()

with col2:
    if st.button("📈 AAPL stock info?"):
        st.session_state.temp_query = "Tell me about Apple stock (AAPL)"
        st.rerun()

with col3:
    if st.button("🔍 AI news?"):
        st.session_state.temp_query = "Search for latest AI developments"
        st.rerun()

# Handle example query clicks
if hasattr(st.session_state, 'temp_query') and st.session_state.temp_query:
    query = st.session_state.temp_query
    st.session_state.temp_query = None
    st.rerun()
