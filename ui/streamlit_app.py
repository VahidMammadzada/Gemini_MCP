
"""Streamlit UI for Multi-Agent Assistant connecting to FastAPI backend."""
import streamlit as st
import requests
import json
import os
from typing import Dict, List, Optional
import time
from pathlib import Path

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Multi-Agent Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            headers={
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
            },
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        # Use iter_lines with smaller chunk size for faster streaming
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if line:
                if line.startswith('data: '):
                    data_str = line[6:]
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

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_intermediate_steps" not in st.session_state:
    st.session_state.show_intermediate_steps = True

if "processing" not in st.session_state:
    st.session_state.processing = False

st.markdown("""
<style>
    /* Fixed bottom input container */
    .stChatFloatingInputContainer {
        position: sticky;
        bottom: 0;
        background-color: var(--background-color);
        padding: 1rem 0;
        z-index: 100;
    }

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
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #ffc107;
        margin: 0.5rem 0;
    }

    .action-step-done {
        background-color: #d4edda;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #28a745;
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

    /* Search references styling */
    .search-references {
        margin-top: 1rem;
        padding: 0.75rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        border-left: 3px solid #007bff;
    }

    .search-references h4 {
        margin-top: 0;
        font-size: 0.9rem;
        color: #495057;
    }

    .search-references a {
        display: block;
        color: #007bff;
        text-decoration: none;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }

    .search-references a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("🤖 Multi-Agent Assistant")
    st.markdown("---")

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

st.title("💬 Multi-Agent Chat Assistant")
st.markdown("Ask questions about crypto, stocks, documents, or search the web!")

# Container for chat messages
chat_container = st.container()

with chat_container:
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
                        status = metadata.get('status', 'running')
                        if status == 'done':
                            st.success(f"✅ **{agent.title()}** Agent - Done")
                        else:
                            st.info(f"🔧 Calling **{agent.title()}** Agent...")

                elif metadata.get("type") == "observation":
                    if st.session_state.show_intermediate_steps:
                        agent = metadata.get('agent', 'unknown')
                        with st.expander(f"📊 {agent.title()} Agent Results", expanded=False):
                            st.write(content)

                else:
                    st.markdown(content)

                    # Display search references if available
                    if metadata.get("search_references"):
                        refs = metadata["search_references"]
                        st.markdown("---")
                        st.markdown("**🔗 References:**")
                        for ref in refs:
                            st.markdown(f"- [{ref['title']}]({ref['url']})")
            else:
                st.markdown(content)

st.markdown("---")
input_col, upload_col = st.columns([4, 1])

with input_col:
    prompt = st.chat_input("Ask me anything...", key="chat_input")

with upload_col:
    uploaded_file = st.file_uploader(
        "📄",
        type=["pdf", "txt", "docx"],
        help="Upload documents to the RAG agent",
        label_visibility="collapsed",
        key="file_uploader"
    )

# Document upload
if uploaded_file and not st.session_state.processing:
    with st.spinner("Uploading document..."):
        result = upload_document(uploaded_file)
        if result.get("success"):
            st.success(f"✅ {result.get('message', 'Upload successful')}")
        else:
            st.error(f"❌ {result.get('message', 'Upload failed')}")

# Chat input
if prompt and not st.session_state.processing:
    st.session_state.processing = True

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Display user message
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from API
        with st.chat_message("assistant"):
            # Placeholders for streaming (thinking steps first, then final answer)
            thinking_placeholder = st.container()
            response_placeholder = st.empty()

            final_answer = ""
            current_step = 0
            agent_status_placeholders = {}
            search_references = []
            first_event_received = False
            initial_status = None

            # Show initial status while supervisor starts processing
            if st.session_state.show_intermediate_steps:
                with thinking_placeholder:
                    initial_status = st.empty()
                    initial_status.info("🤔 Supervisor is analyzing your query...")

            try:
                for event in stream_chat_response(prompt, st.session_state.chat_history[:-1]):
                    event_type = event.get("type", "unknown")

                    # Clear initial status on first event
                    if not first_event_received and initial_status is not None:
                        initial_status.empty()
                        first_event_received = True

                    if event_type == "thinking":
                        current_step = event.get("step", current_step + 1)
                        if st.session_state.show_intermediate_steps:
                            with thinking_placeholder:
                                with st.expander(f"💭 Step {current_step}: Reasoning", expanded=False):
                                    st.markdown(f"**Thought:** {event.get('thought', 'N/A')}")
                                    st.markdown(f"**Action:** `{event.get('action', 'N/A').upper()}`")
                                    st.markdown(f"**Justification:** {event.get('justification', 'N/A')}")

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
                            with thinking_placeholder:
                                # Placeholder for this agent's status
                                agent_status_placeholders[agent] = st.empty()
                                agent_status_placeholders[agent].info(f"🔧 Calling **{agent.title()}** Agent...")

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Calling {agent} agent...",
                                "metadata": {"type": "action", "agent": agent, "status": "running"}
                            })

                    elif event_type == "observation":
                        agent = event.get("agent", "unknown")
                        summary = event.get("summary", "")

                        # Update the agent status to done
                        if agent in agent_status_placeholders:
                            agent_status_placeholders[agent].success(f"✅ **{agent.title()}** Agent - Done")

                        # Extract search URLs if this is the search agent
                        if agent == "search" and event.get("search_urls"):
                            for url_data in event.get("search_urls", []):
                                if url_data not in search_references:
                                    search_references.append(url_data)

                        if st.session_state.show_intermediate_steps:
                            with thinking_placeholder:
                                with st.expander(f"📊 {agent.title()} Agent Results", expanded=False):
                                    st.write(summary)

                            # Update the message to mark as done
                            for msg in reversed(st.session_state.messages):
                                if (msg.get("role") == "assistant" and
                                    msg.get("metadata", {}).get("type") == "action" and
                                    msg.get("metadata", {}).get("agent") == agent):
                                    msg["metadata"]["status"] = "done"
                                    break

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": summary,
                                "metadata": {"type": "observation", "agent": agent}
                            })

                    elif event_type == "final_start":
                        # Display synthesis status in thinking area, not at top
                        if st.session_state.show_intermediate_steps:
                            with thinking_placeholder:
                                st.info("🔄 Synthesizing final answer...")

                    elif event_type == "final_token":
                        # Stream token by token as plain text to avoid incomplete markdown rendering
                        final_answer = event.get("accumulated", "")
                        response_placeholder.text(final_answer)

                    elif event_type == "final_complete":
                        if final_answer:
                            # Render final complete version with proper markdown formatting
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
        message_data = {
            "role": "assistant",
            "content": final_answer,
            "metadata": {}
        }

        # Add search references if available
        if search_references:
            message_data["metadata"]["search_references"] = search_references

        st.session_state.messages.append(message_data)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": final_answer
        })

    st.session_state.processing = False
    st.rerun()