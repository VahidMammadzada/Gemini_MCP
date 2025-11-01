"""Main application with Gradio Chat UI for multi-agent LLM system."""
import asyncio
import gradio as gr
from crypto_agent_mcp import CryptoAgentMCP
from rag_agent_mcp import RAGAgentMCP
from stock_agent_mcp import StockAgentMCP
from search_agent_mcpp import SearchAgentMCP
from typing import Dict, Any, Optional, List, AsyncGenerator
from pathlib import Path
import os
from langgraph_supervisor import ReActSupervisor

class MultiAgentApp:
    """Main application orchestrating LLM supervisor and agents."""

    def __init__(self):
        self.crypto_agent = CryptoAgentMCP()
        self.rag_agent = RAGAgentMCP()
        self.stock_agent = StockAgentMCP()
        self.search_agent = SearchAgentMCP()
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
            
            # Initialize supervisor with agent references
            self.supervisor = ReActSupervisor(
                crypto_agent=self.crypto_agent,
                rag_agent=self.rag_agent,
                stock_agent=self.stock_agent,
                search_agent=self.search_agent
            )
            
            self.initialized = True
            print("✅ System initialized with LangGraph supervisor!")
            return "✅ All agents initialized and ready!"
    
    async def process_query_streaming(
        self,
        message: str,
        history: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """
        Process user query with streaming updates.
        
        Args:
            message: User's input message
            history: Chat history in Gradio messages format [{"role": "user/assistant", "content": "..."}]
            
        Yields:
            Streaming response updates
        """
        if not message.strip():
            yield "Please enter a query."
            return
        
        try:
            # Check if system is initialized
            if not self.initialized:
                yield "❌ System not initialized. Please restart the application."
                return
            
            # Show initial status
            yield "🎯 **LangGraph Supervisor Processing**\n\n"
            
            # Convert Gradio messages format to internal format
            internal_history = []
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    internal_history.append({"user": content})
                elif role == "assistant":
                    internal_history.append({"assistant": content})
            
            # Process through supervisor with streaming
            current_response = "🎯 **LangGraph Supervisor Processing**\n\n"
            
            async for update in self.supervisor.process_streaming(message, history=internal_history):
                if update.get("type") == "thinking":
                    # Show thinking step
                    step = update.get("step", 0)
                    thought = update.get("thought", "")
                    action = update.get("action", "")
                    justification = update.get("justification", "")
                    
                    thinking_update = f"💭 **Step {step}: Thinking**\n"
                    thinking_update += f"*Thought:* {thought}\n"
                    thinking_update += f"*Action:* {action.upper()}\n"
                    thinking_update += f"*Justification:* {justification}\n\n"
                    
                    current_response += thinking_update
                    yield current_response
                
                elif update.get("type") == "action":
                    # Show agent call
                    agent = update.get("agent", "unknown")
                    action_update = f"🔧 **Calling {agent.upper()} Agent...**\n\n"
                    current_response += action_update
                    yield current_response
                
                elif update.get("type") == "observation":
                    # Show observation
                    agent = update.get("agent", "unknown")
                    summary = update.get("summary", "")
                    obs_update = f"📊 **Observation from {agent.upper()}:**\n{summary}\n\n"
                    current_response += obs_update
                    yield current_response
                
                elif update.get("type") == "final":
                    # Show final answer
                    final_answer = update.get("response", "No response generated")
                    final_update = f"📝 **Final Answer:**\n\n{final_answer}"
                    current_response += final_update
                    yield current_response
                
                elif update.get("type") == "error":
                    # Show error
                    error = update.get("error", "Unknown error")
                    error_update = f"❌ **Error:** {error}"
                    current_response += error_update
                    yield current_response
            
            # Update chat history
            self.chat_history.append({"user": message})
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]
            
        except Exception as e:
            error_msg = f"❌ **Error processing query:** {str(e)}"
            yield error_msg
    
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

        This system has four specialized agents working together:
        - **💰 Crypto Agent**: Real-time cryptocurrency data via CoinGecko MCP
        - **📈 Stock Agent**: Stock market data and company information via Alpha Vantage MCP
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
            """Streaming response handler."""
            async for update in app.process_query_streaming(message, chat_history):
                # Append the new message in messages format
                yield chat_history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": update}
                ]
        
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