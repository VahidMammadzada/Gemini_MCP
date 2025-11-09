"""Main application with Gradio Chat UI for multi-agent LLM system."""
import asyncio
import gradio as gr
from gradio import ChatMessage
from crypto_agent_mcp import CryptoAgentMCP
from rag_agent_mcp import RAGAgentMCP
from stock_agent_mcp import StockAgentMCP
from search_agent_mcp import SearchAgentMCP
from finance_tracker_agent_mcp import FinanceTrackerMCP
from typing import Dict, Any, Optional, List, AsyncGenerator
from pathlib import Path
import os
import time
from langgraph_supervisor import ReActSupervisor

class MultiAgentApp:
    """Main application orchestrating LLM supervisor and agents."""

    def __init__(self):
        self.crypto_agent = CryptoAgentMCP()
        self.rag_agent = RAGAgentMCP()
        self.stock_agent = StockAgentMCP()
        self.search_agent = SearchAgentMCP()
        self.finance_tracker = FinanceTrackerMCP()
        self.supervisor = None
        self.chat_history: List[Dict[str, str]] = []
        self.initialized = False

    async def initialize(self):
        """Initialize all agents and supervisor."""
        if not self.initialized:
            print("🚀 Initializing Multi-Agent System...")
            
            # Initialize agents first
            await self.crypto_agent.initialize()
            await self.rag_agent.initialize()
            await self.stock_agent.initialize()
            await self.search_agent.initialize()
            await self.finance_tracker.initialize()

            # Initialize supervisor with agent references
            self.supervisor = ReActSupervisor(
                crypto_agent=self.crypto_agent,
                rag_agent=self.rag_agent,
                stock_agent=self.stock_agent,
                search_agent=self.search_agent,
                finance_tracker=self.finance_tracker
            )
            
            self.initialized = True
            print("✅ System initialized with LangGraph supervisor!")
            return "✅ All agents initialized and ready!"

    async def process_query_streaming(
        self,
        message: str,
        history: List[Dict[str, str]]
    ) -> AsyncGenerator[ChatMessage, None]:
        """
        Process user query with streaming updates showing intermediate steps.

        Args:
            message: User's input message
            history: Chat history in Gradio messages format [{"role": "user/assistant", "content": "..."}]

        Yields:
            ChatMessage objects with metadata for intermediate steps
        """
        if not message.strip():
            yield ChatMessage(role="assistant", content="Please enter a query.")
            return

        try:
            # Check if system is initialized
            if not self.initialized:
                yield ChatMessage(role="assistant", content="❌ System not initialized. Please restart the application.")
                return

            # Convert Gradio messages format to internal format
            internal_history = []
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    internal_history.append({"user": content})
                elif role == "assistant":
                    internal_history.append({"assistant": content})

            # Track message IDs for nested structure
            message_id = 0
            step_messages = {}  # Store step messages to update their status
            final_answer_accumulated = ""  # Accumulate streaming final answer

            async for update in self.supervisor.process_streaming(message, history=internal_history):
                if update.get("type") == "thinking":
                    # Collect thinking step
                    step = update.get("step", 0)
                    thought = update.get("thought", "")
                    action = update.get("action", "")
                    justification = update.get("justification", "")

                    step_content = f"**Thought:** {thought}\n\n"
                    step_content += f"**Action:** {action.upper()}\n\n"
                    step_content += f"**Justification:** {justification}"

                    message_id += 1
                    # Yield intermediate step as ChatMessage with metadata
                    thinking_msg = ChatMessage(
                        role="assistant",
                        content=step_content,
                        metadata={
                            "title": f"💭 Step {step}: Reasoning",
                            "id": message_id,
                            "status": "done"
                        }
                    )
                    step_messages[f"thinking_{step}"] = thinking_msg
                    yield thinking_msg

                elif update.get("type") == "action":
                    # Show agent call as intermediate step with pending status
                    agent = update.get("agent", "unknown")
                    action_content = f"Calling **{agent.upper()}** agent to gather information..."

                    message_id += 1
                    action_msg = ChatMessage(
                        role="assistant",
                        content=action_content,
                        metadata={
                            "title": f"🔧 Calling {agent.title()} Agent",
                            "id": message_id,
                            "status": "pending"
                        }
                    )
                    step_messages[f"action_{agent}"] = action_msg
                    yield action_msg

                elif update.get("type") == "observation":
                    # Show observation as intermediate step - mark the action as done
                    agent = update.get("agent", "unknown")
                    summary = update.get("summary", "")
                    obs_content = f"{summary}"

                    message_id += 1
                    obs_msg = ChatMessage(
                        role="assistant",
                        content=obs_content,
                        metadata={
                            "title": f"📊 {agent.title()} Agent Results",
                            "id": message_id,
                            "status": "done"
                        }
                    )
                    step_messages[f"observation_{agent}"] = obs_msg
                    yield obs_msg

                elif update.get("type") == "final_start":
                    # Start of final answer - reset accumulator
                    final_answer_accumulated = ""

                elif update.get("type") == "final_token":
                    # Stream each token of the final answer
                    final_answer_accumulated = update.get("accumulated", "")
                    # Yield the accumulated answer so far (streaming effect)
                    yield ChatMessage(role="assistant", content=final_answer_accumulated)

                elif update.get("type") == "final_complete":
                    # Final answer is complete - no need to yield again
                    # The last final_token already has the complete content
                    pass

                elif update.get("type") == "error":
                    # Show error
                    error = update.get("error", "Unknown error")
                    yield ChatMessage(
                        role="assistant",
                        content=f"**Error:** {error}"
                    )

            # Update chat history
            self.chat_history.append({"user": message})
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]

        except Exception as e:
            error_msg = f"❌ **Error processing query:** {str(e)}"
            yield ChatMessage(role="assistant", content=error_msg)

    async def upload_document(
        self,
        file_obj,
        progress=gr.Progress()
    ) -> str:
        """
        Handle document upload to ChromaDB Cloud.

        Args:
            file_obj: Gradio file object
            progress: Gradio progress tracker

        Returns:
            Status message
        """
        try:
            if not self.initialized:
                return "❌ System not initialized. Please restart the application."

            if file_obj is None:
                return "❌ No file selected"

            # Get file path from Gradio file object
            file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)

            # Validate file exists
            if not os.path.exists(file_path):
                return f"❌ File not found: {file_path}"

            # Validate file type
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in ['.pdf', '.txt', '.docx']:
                return f"❌ Unsupported file type: {file_extension}. Supported: PDF, TXT, DOCX"

            # Progress callback
            def update_progress(percent, message):
                progress(percent, desc=message)

            # Upload to RAG agent
            result = await self.rag_agent.add_document(
                file_path,
                progress_callback=update_progress
            )

            if result.get("success"):
                return (
                    f"✅ Successfully uploaded {result['filename']}\n"
                    f"📊 Type: {result['file_type']}\n"
                    f"📦 Chunks created: {result['chunks_added']}\n"
                    f"📚 Total documents in collection: {result['total_documents']}"
                )
            else:
                return f"❌ Upload failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            return f"❌ Error uploading document: {str(e)}"

    async def cleanup(self):
        """Cleanup resources."""
        if self.initialized:
            await self.crypto_agent.cleanup()
            await self.rag_agent.cleanup()
            await self.stock_agent.cleanup()
            await self.search_agent.cleanup()
            await self.finance_tracker.cleanup()
            print("🧹 Cleanup complete")
        self.chat_history.clear()


event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(event_loop)

# Global app instance
app = MultiAgentApp()


def create_ui():
    """Create and configure the Gradio Chat UI."""

    with gr.Blocks(title="Multi-Agent Assistant", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🤖 Multi-Agent AI Assistant with Streaming

        This system has five specialized agents working together:
        - **💰 Crypto Agent**: Real-time cryptocurrency data via CoinGecko MCP
        - **📈 Stock Agent**: Stock market data and company information via Alpha Vantage MCP
        - **💼 Finance Tracker**: Personal portfolio tracking with Google Cloud SQL
        - **📚 RAG Agent**: Document Q&A powered by ChromaDB Cloud
        - **🔍 Search Agent**: Web search powered by DuckDuckGo MCP

        Watch the agents think and collaborate in real-time!
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Chat Interface
                chatbot = gr.Chatbot(
                    label="Multi-Agent Assistant",
                    height=400,
                    show_label=True,
                    avatar_images=(None, "🤖"),
                    type='messages',
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Message",
                        placeholder="Ask about crypto, stocks, documents, or search the web...",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", scale=1)
                    retry_btn = gr.Button("Retry Last", scale=1)
                
                gr.Markdown("""
                **Example queries:**
                - What's the current price of Bitcoin and Ethereum?
                - Add 10 shares of AAPL I bought at $150
                - What's my current portfolio value?
                - Show me news on my portfolio holdings
                - What did Jerome Powell say in his latest speech?
                - Show me Tesla's financial overview
                - Search for latest AI developments
                - What does my document say about [topic]?
                """)
            
            with gr.Column(scale=1):
                # Document Upload Section
                gr.Markdown("### 📄 Upload Documents")
                file_upload = gr.File(
                    label="Upload PDF, TXT, or DOCX",
                    file_types=[".pdf", ".txt", ".docx"],
                    type="filepath"
                )
                upload_btn = gr.Button("📤 Upload to RAG", variant="secondary")
                upload_status = gr.Textbox(
                    label="Upload Status",
                    lines=4,
                    interactive=False
                )
                
                # System Status
                gr.Markdown("### 🔧 System Status")
                status_box = gr.Textbox(
                    label="Initialization Status",
                    value="✅ All agents initialized and ready!" if app.initialized else "⏳ Initializing...",
                    lines=2,
                    interactive=False
                )
        
        gr.Markdown("""
        ---
        ### 🏗️ System Architecture

        **ReAct Pattern Supervisor** → Analyzes queries and orchestrates agents through reasoning loops
        
        **Agent Workflow:**
        1. **Think**: Supervisor reasons about what information is needed
        2. **Act**: Calls appropriate agent(s) to gather information
        3. **Observe**: Reviews agent responses
        4. **Repeat**: Continues until sufficient information is gathered
        5. **Synthesize**: Generates comprehensive final answer
        
        Each agent uses Gemini LLM with specialized MCP server tools for their domain.
        """)
        
        # Define async wrappers for Gradio
        async def respond_stream(message, chat_history):
            """Streaming response handler with intermediate steps."""
            # Start with user message
            new_messages = [{"role": "user", "content": message}]
            in_final_answer = False
            final_answer_index = None

            async for chat_msg in app.process_query_streaming(message, chat_history):
                # Check if this is a final answer message (no metadata = final answer)
                is_final_answer = not hasattr(chat_msg, 'metadata') or chat_msg.metadata is None

                if is_final_answer:
                    if not in_final_answer:
                        # First token of final answer - append new message
                        in_final_answer = True
                        final_answer_index = len(new_messages)
                        new_messages.append(chat_msg)
                    else:
                        # Subsequent tokens - update the existing final answer message
                        # Create a new list to ensure Gradio detects the change
                        new_messages = new_messages[:final_answer_index] + [chat_msg]
                else:
                    # Intermediate step - append as new message
                    new_messages.append(chat_msg)

                # Yield accumulated messages (create new list to trigger Gradio update)
                yield chat_history + new_messages
        
        def upload_document_sync(file_obj, progress=gr.Progress()):
            """Synchronous wrapper for async document upload."""
            return event_loop.run_until_complete(app.upload_document(file_obj, progress))
        
        def clear_chat():
            """Clear chat history."""
            app.chat_history.clear()
            return []
        
        def retry_last(chat_history):
            """Retry the last message."""
            if chat_history:
                # Find the last user message
                for i in range(len(chat_history) - 1, -1, -1):
                    if chat_history[i].get("role") == "user":
                        last_user_msg = chat_history[i].get("content", "")
                        # Remove everything from that user message onwards
                        return chat_history[:i], last_user_msg
            return chat_history, ""
        
        # Connect button actions
        msg.submit(
            respond_stream,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "",
            outputs=[msg]
        )
        
        submit_btn.click(
            respond_stream,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "",
            outputs=[msg]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot]
        )
        
        retry_btn.click(
            retry_last,
            inputs=[chatbot],
            outputs=[chatbot, msg]
        )
        
        upload_btn.click(
            upload_document_sync,
            inputs=[file_upload],
            outputs=[upload_status]
        )

    return interface


def main():
    """Main entry point."""
    print("=" * 60)
    print("🚀 Starting Multi-Agent Assistant")
    print("=" * 60)

    # Validate configuration
    try:
        from config import config
        config.validate()
        print("✅ Configuration validated")
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        return

    # Initialize all agents at startup
    print("\n⚡ Initializing all agents at startup...")
    event_loop.run_until_complete(app.initialize())

    # Create and launch UI
    interface = create_ui()

    print("\n📱 Launching Gradio interface...")
    print("🌐 Access the app at: http://localhost:7860")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    try:
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        # Cleanup
        event_loop.run_until_complete(app.cleanup())
        event_loop.close()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()