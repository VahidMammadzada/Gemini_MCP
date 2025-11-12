"""RAG Agent using Chroma MCP Server via LangChain MCP adapters."""
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import config
from file_processors import process_document
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient


class RAGAgentMCP:
    """Agent specialized in document retrieval and Q&A using Chroma MCP Server."""

    @staticmethod
    def _to_text(payload: Any) -> str:
        """Convert tool or model output into a displayable string."""
        if isinstance(payload, str):
            return payload
        try:
            return json.dumps(payload, ensure_ascii=False)
        except TypeError:
            return str(payload)

    def __init__(self):
        self.name = "RAG Agent (MCP)"
        self.description = "Document storage, retrieval, and question answering expert using Chroma MCP Server"
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.model: Optional[ChatGoogleGenerativeAI] = None
        self.tools: List[BaseTool] = []
        self.tool_map: Dict[str, BaseTool] = {}

    async def initialize(self) -> None:
        """Initialize the agent with Chroma MCP server."""
        print(f"🔧 Initializing {self.name}...")

        try:
            # Connect to Chroma MCP Server for tool access
            print(f"  📡 Connecting to Chroma MCP Server...")
            chroma_connection = self._build_chroma_connection()
            self.mcp_client = MultiServerMCPClient({"chroma": chroma_connection})

            # Get tools from MCP server
            self.tools = await self.mcp_client.get_tools(server_name="chroma")
            if not self.tools:
                raise RuntimeError("No tools returned by Chroma MCP server")

            self.tool_map = {tool.name: tool for tool in self.tools}

            print(f"  ✅ Connected to Chroma MCP Server with {len(self.tools)} tools")

            # Initialize Gemini model for answer generation
            self.model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                google_api_key=config.GOOGLE_API_KEY,
            )

            print(f"  ✅ {self.name} ready!")

        except Exception as e:
            import traceback
            print(f"  ❌ Error initializing {self.name}: {e}")
            print(f"  📋 Full error details:")
            traceback.print_exc()
            raise

    async def add_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Add a document to ChromaDB via MCP server."""
        try:
            doc_path = Path(file_path)

            # Validate file type
            if doc_path.suffix.lower() not in config.ALLOWED_FILE_TYPES:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {doc_path.suffix}. Allowed: {config.ALLOWED_FILE_TYPES}"
                }

            print(f"\n📄 Processing {doc_path.suffix.upper()}: {doc_path.name}")

            if progress_callback:
                progress_callback(0.1, "Extracting text from document...")

            # Process document using file_processors
            doc_info = process_document(doc_path, chunk_size=500, overlap=50)

            if progress_callback:
                progress_callback(0.4, f"Extracted {doc_info['num_chunks']} chunks...")

            # Generate document ID
            doc_id = doc_path.stem

            # Prepare metadata
            doc_metadata = {
                "filename": doc_info["filename"],
                "file_type": doc_info["file_type"],
                "file_size": doc_info["file_size"],
                "num_chunks": doc_info["num_chunks"],
                "source": "user_upload",
                **(metadata or {})
            }

            # Upload via Chroma MCP Server
            if progress_callback:
                progress_callback(0.6, "Uploading via Chroma MCP Server...")

            # Prepare data for MCP add_documents tool
            chunks = doc_info["chunks"]
            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {**doc_metadata, "chunk_index": i, "chunk_id": ids[i]}
                for i in range(len(chunks))
            ]

            add_tool = (
                self.tool_map.get("chroma_add_documents")
                or self.tool_map.get("add_documents")
            )
            if not add_tool:
                raise KeyError("add_documents tool not available in Chroma MCP")

            tool_payload = {
                "collection_name": "gemini-embed",  # Use dedicated Gemini collection
                "documents": chunks,
                "ids": ids,
                "metadatas": metadatas,
            }

            result = await add_tool.ainvoke(tool_payload)

            if isinstance(result, dict) and result.get("success"):
                print(f"  ✅ Added via MCP: {len(chunks)} chunks from {doc_path.name}")
            else:
                raise ValueError(result)

            if progress_callback:
                progress_callback(1.0, "Upload complete!")

            return {
                "success": True,
                "document_id": doc_id,
                "filename": doc_path.name,
                "file_type": doc_info["file_type"],
                "chunks_added": len(chunks),
            }

        except Exception as e:
            print(f"  ❌ Error adding document: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    def _build_chroma_connection(self) -> Dict[str, Any]:
        """Create connection configuration for the Chroma MCP server."""
        return {
            "transport": "docker",
            "command": "mcp",
            "args": ["gateway", "run", "--servers=chromadb"],
        }
