
"""FastAPI application exposing multi-agent system via REST API."""
import asyncio
import json
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import tempfile
import os

from crypto_agent_mcp import CryptoAgentMCP
from rag_agent_mcp import RAGAgentMCP
from stock_agent_mcp import StockAgentMCP
from search_agent_mcp import SearchAgentMCP
from finance_tracker_agent_mcp import FinanceTrackerMCP
from langgraph_supervisor import ReActSupervisor


# ============================================================================
# Pydantic Models
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's query message")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Chat history")


class ChatResponse(BaseModel):
    """Response model for non-streaming chat."""
    response: str = Field(..., description="Assistant's response")
    history: List[ChatMessage] = Field(..., description="Updated chat history")


class StreamEvent(BaseModel):
    """Streaming event model."""
    type: str = Field(..., description="Event type: thinking, action, observation, final_start, final_token, final_complete, error")
    data: Dict = Field(default={}, description="Event data")


class UploadResponse(BaseModel):
    """Response model for document upload."""
    success: bool = Field(..., description="Upload success status")
    message: str = Field(..., description="Status message")
    details: Optional[Dict] = Field(default=None, description="Upload details")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="System status")
    initialized: bool = Field(..., description="Whether system is initialized")
    agents: Dict[str, bool] = Field(..., description="Status of each agent")


# ============================================================================
# Multi-Agent Application
# ============================================================================

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

    async def process_query_streaming(self, message: str, history: List[Dict[str, str]]):
        """
        Process user query with streaming updates.

        Args:
            message: User's input message
            history: Chat history in internal format [{"user": "...", "assistant": "..."}]

        Yields:
            Dictionary updates from supervisor
        """
        if not message.strip():
            yield {"type": "error", "error": "Please enter a query."}
            return

        try:
            # Check if system is initialized
            if not self.initialized:
                yield {"type": "error", "error": "System not initialized."}
                return

            # Stream updates from supervisor
            async for update in self.supervisor.process_streaming(message, history=history):
                yield update

            # Update chat history
            self.chat_history.append({"user": message})
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]

        except Exception as e:
            yield {"type": "error", "error": str(e)}

    async def upload_document(self, file_path: str, filename: str) -> Dict:
        """
        Handle document upload to ChromaDB Cloud.

        Args:
            file_path: Path to the uploaded file
            filename: Original filename

        Returns:
            Upload result dictionary
        """
        try:
            if not self.initialized:
                return {
                    "success": False,
                    "error": "System not initialized"
                }

            # Validate file exists
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }

            # Validate file type
            file_extension = Path(filename).suffix.lower()
            if file_extension not in ['.pdf', '.txt', '.docx']:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_extension}. Supported: PDF, TXT, DOCX"
                }

            # Upload to RAG agent
            result = await self.rag_agent.add_document(file_path)

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

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

    def get_agent_status(self) -> Dict[str, bool]:
        """Get status of all agents."""
        return {
            "crypto": self.crypto_agent is not None,
            "rag": self.rag_agent is not None,
            "stock": self.stock_agent is not None,
            "search": self.search_agent is not None,
            "finance_tracker": self.finance_tracker is not None
        }


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Multi-Agent Assistant API",
    description="FastAPI interface for multi-agent LLM system with ReAct supervisor",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize multi-agent app
multi_agent_app = MultiAgentApp()


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    print("=" * 60)
    print("🚀 Starting FastAPI Multi-Agent Assistant")
    print("=" * 60)

    # Validate configuration
    try:
        from config import config
        config.validate()
        print("✅ Configuration validated")
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        raise

    # Initialize all agents
    await multi_agent_app.initialize()
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\n🛑 Shutting down...")
    await multi_agent_app.cleanup()
    print("👋 Goodbye!")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Multi-Agent Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns system status and agent availability.
    """
    return HealthResponse(
        status="healthy" if multi_agent_app.initialized else "initializing",
        initialized=multi_agent_app.initialized,
        agents=multi_agent_app.get_agent_status()
    )


@app.post("/api/v1/chat/stream", tags=["Chat"])
async def stream_chat(request: ChatRequest):
    """
    Stream chat responses with intermediate reasoning steps.

    Uses Server-Sent Events (SSE) to stream updates in real-time.

    Args:
        request: ChatRequest with message and optional history

    Returns:
        StreamingResponse with SSE events
    """
    # Convert ChatMessage models to internal format
    internal_history = []
    for msg in request.history:
        if msg.role == "user":
            internal_history.append({"user": msg.content})
        elif msg.role == "assistant":
            internal_history.append({"assistant": msg.content})

    async def event_generator():
        """Generate SSE events."""
        try:
            async for update in multi_agent_app.process_query_streaming(
                request.message,
                internal_history
            ):
                # Format as SSE
                event_data = json.dumps(update)
                yield f"data: {event_data}\n\n"
        except Exception as e:
            error_event = json.dumps({"type": "error", "error": str(e)})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.websocket("/ws/v1/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for chat with streaming updates.

    Client sends: {"message": "query", "history": [...]}
    Server streams: {"type": "...", "data": {...}}
    """
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "")
            history = data.get("history", [])

            # Convert to internal format
            internal_history = []
            for msg in history:
                if msg.get("role") == "user":
                    internal_history.append({"user": msg.get("content", "")})
                elif msg.get("role") == "assistant":
                    internal_history.append({"assistant": msg.get("content", "")})

            # Stream responses
            async for update in multi_agent_app.process_query_streaming(message, internal_history):
                await websocket.send_json(update)

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


@app.post("/api/v1/documents/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the RAG agent.

    Supported file types: PDF, TXT, DOCX

    Args:
        file: Uploaded file

    Returns:
        UploadResponse with status and details
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        # Process upload
        result = await multi_agent_app.upload_document(tmp_file_path, file.filename)

        if result.get("success"):
            return UploadResponse(
                success=True,
                message="Document uploaded successfully",
                details={
                    "filename": result.get("filename"),
                    "file_type": result.get("file_type"),
                    "chunks_added": result.get("chunks_added"),
                    "total_documents": result.get("total_documents")
                }
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Upload failed")
            )

    finally:
        # Clean up temp file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
