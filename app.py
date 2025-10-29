"""Main application with Gradio UI for multi-agent LLM system."""
import asyncio
import gradio as gr
from crypto_agent_mcp import CryptoAgentMCP
from rag_agent_mcp import RAGAgentMCP
from decision_maker import DecisionMaker
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
import os


class MultiAgentApp:
    """Main application orchestrating LLM decision maker and agents."""

    def __init__(self):
        self.decision_maker = DecisionMaker()
        self.crypto_agent = CryptoAgentMCP()
        self.rag_agent = RAGAgentMCP()
        self.chat_history: List[Dict[str, str]] = []
        self.initialized = False

    async def initialize(self):
        """Initialize all agents."""
        if not self.initialized:
            print("🚀 Initializing Multi-Agent System...")
            await self.crypto_agent.initialize()
            await self.rag_agent.initialize()
            self.initialized = True
            print("✅ System initialized and ready!")
    
    async def process_query(self, query: str) -> tuple[str, str]:
        """
        Process user query through the system.
        
        Args:
            query: User's input query
            
        Returns:
            Tuple of (response, debug_info)
        """
        if not query.strip():
            return "Please enter a query.", ""
        
        try:
            # Ensure system is initialized
            if not self.initialized:
                await self.initialize()
            
            debug_info = []
            
            # Step 1: Decision maker decides which agent to use
            debug_info.append("🤔 **Decision Making Phase**")
            decision = await self.decision_maker.decide_agent(query, history=self.chat_history)
            
            agent_name = decision.get('agent', 'none')
            reasoning = decision.get('reasoning', 'No reasoning provided')
            confidence = decision.get('confidence', 0.0)
            
            debug_info.append(f"Selected Agent: **{agent_name}**")
            debug_info.append(f"Reasoning: {reasoning}")
            debug_info.append(f"Confidence: {confidence:.2f}\n")
            
            # Step 2: Route to appropriate agent
            if agent_name == 'crypto':
                debug_info.append("💰 **Executing Crypto Agent**")
                agent_result = await self.crypto_agent.process(query, history=self.chat_history)

                if agent_result.get("success"):
                    final_response = agent_result.get("response", "No response generated")
                else:
                    final_response = f"Error: {agent_result.get('error', 'Unknown error')}"

            elif agent_name == 'rag':
                debug_info.append("📚 **Executing RAG Agent**")
                agent_result = await self.rag_agent.process(query, history=self.chat_history)

                if agent_result.get("success"):
                    final_response = agent_result.get("response", "No response generated")
                else:
                    final_response = f"Error: {agent_result.get('error', 'Unknown error')}"

            elif agent_name == 'none':
                final_response = (
                    "I'm not sure how to handle that query with my current capabilities. "
                    "I'm equipped to help with:\n"
                    "- Cryptocurrency questions (prices, markets, trends)\n"
                    "- Document Q&A (upload documents and ask questions about them)\n\n"
                    "Could you rephrase your question?"
                )
            else:
                final_response = f"Agent '{agent_name}' is not yet implemented."
            
            self.chat_history.append({"user": query, "assistant": final_response})
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]

            debug_output = "\n".join(debug_info)
            return final_response, debug_output
            
        except Exception as e:
            error_msg = f"❌ Error processing query: {str(e)}"
            return error_msg, error_msg
    
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
                await self.initialize()

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
            print("🧹 Cleanup complete")
        self.chat_history.clear()


event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(event_loop)

# Global app instance
app = MultiAgentApp()


def create_ui():
    """Create and configure the Gradio UI."""

    with gr.Blocks(title="Multi-Agent Assistant", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🤖 Multi-Agent AI Assistant

        This system has two specialized agents:
        - **💰 Crypto Agent**: Real-time cryptocurrency data via CoinGecko MCP
        - **📚 RAG Agent**: Document Q&A powered by ChromaDB Cloud

        **Example queries:**
        - What's the current price of Bitcoin?
        - What are the trending coins today?
        - What does the document say about [topic]?
        - Search my documents for [keyword]
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Document Upload Section
                with gr.Accordion("📄 Upload Documents to RAG Agent", open=False):
                    file_upload = gr.File(
                        label="Upload PDF, TXT, or DOCX",
                        file_types=[".pdf", ".txt", ".docx"],
                        type="filepath"
                    )
                    upload_btn = gr.Button("📤 Upload to ChromaDB Cloud", variant="secondary")
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        lines=3,
                        interactive=False
                    )

                gr.Markdown("### 💬 Ask a Question")

                query_input = gr.Textbox(
                    label="Your Query",
                    placeholder="Ask about crypto or your uploaded documents...",
                    lines=3
                )

                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", size="lg")

                response_output = gr.Textbox(
                    label="Response",
                    lines=10,
                    interactive=False
                )

            with gr.Column(scale=1):
                debug_output = gr.Textbox(
                    label="System Debug Info",
                    lines=20,
                    interactive=False
                )
        
        gr.Markdown("""
        ---
        ### System Architecture

        1. **Decision Maker (Gemini)**: Analyzes your query and routes to the appropriate agent
        2. **Crypto Agent**: Connects to CoinGecko MCP Server for real-time crypto data
        3. **RAG Agent**: Uses ChromaDB Cloud for document storage and semantic search
        4. **Response Generation**: Each agent uses Gemini with specialized tools
        """)
        
        # Define the async wrappers for Gradio
        def process_query_sync(query):
            """Synchronous wrapper for async query processing using shared loop."""
            return event_loop.run_until_complete(app.process_query(query))

        def upload_document_sync(file_obj, progress=gr.Progress()):
            """Synchronous wrapper for async document upload using shared loop."""
            return event_loop.run_until_complete(app.upload_document(file_obj, progress))

        def clear_conversation():
            app.chat_history.clear()
            return "", "", ""

        # Connect button actions
        upload_btn.click(
            fn=upload_document_sync,
            inputs=[file_upload],
            outputs=[upload_status]
        )

        submit_btn.click(
            fn=process_query_sync,
            inputs=[query_input],
            outputs=[response_output, debug_output]
        )

        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[query_input, response_output, debug_output]
        )

        # Also submit on Enter
        query_input.submit(
            fn=process_query_sync,
            inputs=[query_input],
            outputs=[response_output, debug_output]
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
