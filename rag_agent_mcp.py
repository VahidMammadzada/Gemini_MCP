"""RAG Agent using Chroma MCP Server via LangChain MCP adapters."""
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from config import config
from file_processors import process_document
# from langchain_core.messages import HumanMessage, ToolMessage  # Not needed for direct client approach
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
        # MCP Server client
        self.mcp_client: Optional[MultiServerMCPClient] = None

        # Direct ChromaDB client (for document upload only)
        self.chroma_direct_client = None
        self.collection = None

        # Gemini model
        self.model: Optional[ChatGoogleGenerativeAI] = None
        self.model_with_tools = None
        self.tools: List[BaseTool] = []
        self.tool_map: Dict[str, BaseTool] = {}

    async def initialize(self) -> None:
        """Initialize the agent with direct ChromaDB client."""
        print(f"🔧 Initializing {self.name}...")

        try:
            # MCP Server integration commented out - using direct client instead
            # # 1. Connect to Chroma MCP Server for tool access
            # print(f"  📡 Connecting to Chroma MCP Server...")
            # chroma_connection = self._build_chroma_connection()
            # self.mcp_client = MultiServerMCPClient({"chroma": chroma_connection})
            #
            # # Get tools from MCP server
            # self.tools = await self.mcp_client.get_tools(server_name="chroma")
            # if not self.tools:
            #     raise RuntimeError("No tools returned by Chroma MCP server")
            #
            # self.tool_map = {tool.name: tool for tool in self.tools}
            #
            # print(f"  ✅ Connected to Chroma MCP Server with {len(self.tools)} tools")

            # 1. Connect to ChromaDB Cloud with direct client
            print(f"  📚 Connecting to ChromaDB Cloud...")
            self.chroma_direct_client = chromadb.CloudClient(
                tenant=config.CHROMA_TENANT,
                database=config.CHROMA_DATABASE,
                api_key=config.CHROMA_API_KEY
            )

            # 2. Use Google Gemini embedding function
            print(f"  🔧 Setting up Gemini embeddings...")
            from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
            embedding_function = GoogleGenerativeAiEmbeddingFunction(
                api_key=config.GOOGLE_API_KEY,
                model_name="models/embedding-001"
            )

            # 3. Get or create collection with Gemini embeddings
            collection_name = "gemini-embed"
            self.collection = self.chroma_direct_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Document storage for RAG with Gemini embeddings"},
                embedding_function=embedding_function
            )
            print(f"  📦 Using collection: {collection_name}")
            print(f"  📄 Current documents in collection: {self.collection.count()}")

            # 4. Initialize Gemini model for answer generation
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
        """
        Add a document to ChromaDB via MCP server.

        Note: This uses direct ChromaDB client as MCP add_documents tool
        requires the collection to already exist and documents to be processed.

        Args:
            file_path: Path to the document file
            metadata: Optional metadata for the document
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with status and document info
        """
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

            # MCP Server upload (commented out - using direct client instead)
            # if progress_callback:
            #     progress_callback(0.6, "Uploading via Chroma MCP Server...")
            #
            # # Prepare data for MCP add_documents tool
            # chunks = doc_info["chunks"]
            # ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            # metadatas = [
            #     {**doc_metadata, "chunk_index": i, "chunk_id": ids[i]}
            #     for i in range(len(chunks))
            # ]
            #
            # # Try using MCP server's add_documents tool
            # add_tool = (
            #     self.tool_map.get("chroma_add_documents")
            #     or self.tool_map.get("add_documents")
            # )
            # if not add_tool:
            #     raise KeyError("add_documents tool not available in Chroma MCP")
            #
            # tool_payload = {
            #     "collection_name": "gemini-embed",  # Use dedicated Gemini collection
            #     "documents": chunks,
            #     "ids": ids,
            #     "metadatas": metadatas,
            # }
            #
            # result = await add_tool.ainvoke(tool_payload)
            #
            # if isinstance(result, dict) and result.get("success"):
            #     print(f"  ✅ Added via MCP: {len(chunks)} chunks from {doc_path.name}")
            # else:
            #     raise ValueError(result)

            # Direct ChromaDB client upload
            if progress_callback:
                progress_callback(0.6, "Uploading to ChromaDB with Google embeddings...")

            chunks = doc_info["chunks"]
            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {**doc_metadata, "chunk_index": i, "chunk_id": ids[i]}
                for i in range(len(chunks))
            ]

            self.collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            print(f"  ✅ Added {len(chunks)} chunks from {doc_path.name} to '{self.collection.name}'")

            if progress_callback:
                progress_callback(1.0, "Upload complete!")

            return {
                "success": True,
                "document_id": doc_id,
                "filename": doc_path.name,
                "file_type": doc_info["file_type"],
                "chunks_added": len(chunks),
                "total_documents": self.collection.count()
            }

        except Exception as e:
            print(f"  ❌ Error adding document: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    async def process(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Process a query using direct ChromaDB client and Gemini."""
        try:
            print(f"\n📚 {self.name} processing: '{query}'")

            # MCP Tool-based approach (commented out - using direct client instead)
            # if not self.model_with_tools:
            #     raise RuntimeError("Model not initialized with tools")
            #
            # messages: List[Any] = [HumanMessage(content=query)]
            # final_response = ""
            #
            # while True:
            #     ai_message = await self.model_with_tools.ainvoke(messages)
            #     messages.append(ai_message)
            #
            #     tool_calls = getattr(ai_message, "tool_calls", [])
            #     if not tool_calls:
            #         final_response = self._to_text(ai_message.content)
            #         break
            #
            #     for call in tool_calls:
            #         tool_name = call.get("name")
            #         tool_args = call.get("args", {})
            #         tool_call_id = call.get("id")
            #         print(f"  🔧 MCP Tool call: {tool_name}({tool_args})")
            #
            #         tool = self.tool_map.get(tool_name)
            #         if not tool:
            #             tool_result = {"error": f"Tool '{tool_name}' not found"}
            #         else:
            #             tool_result = await tool.ainvoke(tool_args)
            #
            #         messages.append(
            #             ToolMessage(
            #                 content=self._to_text(tool_result),
            #                 tool_call_id=tool_call_id or "",
            #             )
            #         )

            # Direct ChromaDB query approach
            if not self.model:
                raise RuntimeError("Model not initialized")

            conversation_context = ""
            if history:
                trimmed_history = history[-6:]
                history_lines: List[str] = []
                for turn in trimmed_history:
                    user_text = turn.get("user", "").strip()
                    if user_text:
                        history_lines.append(f"User: {user_text}")
                    assistant_text = turn.get("assistant", "").strip()
                    if assistant_text:
                        history_lines.append(f"Assistant: {assistant_text}")
                if history_lines:
                    conversation_context = "\n".join(history_lines)

            conversational_query = query
            if conversation_context:
                try:
                    rephrase_prompt = """You are a helpful assistant that rewrites follow-up questions.
Use the provided conversation history to rewrite the latest user input so it is a standalone question for document retrieval.

Conversation history:
{conversation_context}

Latest user input: {query}

Respond with only the rewritten standalone question."""
                    rephrase_response = await self.model.ainvoke(rephrase_prompt)
                    candidate = rephrase_response.content if hasattr(rephrase_response, "content") else str(rephrase_response)
                    candidate = (candidate or "").strip()
                    if candidate:
                        conversational_query = candidate
                except Exception as rephrase_error:
                    print(f"  ⚠️ Could not rewrite query from history: {rephrase_error}")

            # Query the collection with Google embeddings
            print(f"  🔍 Querying collection '{self.collection.name}'...")
            results = self.collection.query(
                query_texts=[conversational_query],
                n_results=5  # Get top 5 most relevant chunks
            )

            # Extract documents and metadata
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]

            if not documents:
                print(f"  ⚠️ No relevant documents found")
                return {
                    "success": True,
                    "agent": self.name,
                    "response": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                    "query": query
                }

            print(f"  📄 Found {len(documents)} relevant chunks")

            # Build context from retrieved documents
            context = "\n\n".join([
                f"[Document: {meta.get('filename', 'Unknown')} - Chunk {meta.get('chunk_index', '?')}]\n{doc}"
                for doc, meta in zip(documents, metadatas)
            ])

            # Create prompt with context
            history_section = f"Conversation history:\n{conversation_context}\n\n" if conversation_context else ""
            prompt = f"""Based on the following document excerpts, please answer the question.

{history_section}Context from documents:
{context}

Question: {conversational_query}
Original user input: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so."""

            # Generate answer using Gemini
            print(f"  🤖 Generating answer with Gemini...")
            response = await self.model.ainvoke(prompt)
            final_response = response.content if hasattr(response, 'content') else str(response)

            return {
                "success": True,
                "agent": self.name,
                "response": final_response,
                "query": query,
                "interpreted_query": conversational_query,
                "sources": [meta.get('filename') for meta in metadatas]
            }

        except Exception as e:
            print(f"  ❌ Error in {self.name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "agent": self.name,
                "error": str(e),
                "query": query
            }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        print(f"🧹 {self.name} cleaned up")

    def _build_chroma_connection(self) -> Dict[str, Any]:
        """Create connection configuration for the Chroma MCP server."""
        env = {
            "CHROMA_CLIENT_TYPE": "cloud",
            "CHROMA_TENANT": config.CHROMA_TENANT or "",
            "CHROMA_DATABASE": config.CHROMA_DATABASE or "",
            "CHROMA_API_KEY": config.CHROMA_API_KEY or "",
            "CHROMA_CLOUD_HOST": config.CHROMA_CLOUD_HOST,
        }

        scripts_dir = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
        candidates = [
            str(scripts_dir / "uvx.exe"),
            str(scripts_dir / "uvx.cmd"),
            str(scripts_dir / "uvx"),
            "uvx.exe",
            "uvx.cmd",
            "uvx",
        ]

        uvx_path: Optional[str] = None
        for candidate in candidates:
            if not candidate:
                continue
            resolved = shutil.which(candidate)
            if resolved:
                uvx_path = resolved
                break

        if not uvx_path:
            msg = (
                "Could not locate 'uvx'. Install the 'uv' package inside the virtual "
                "environment or ensure it is on PATH."
            )
            raise FileNotFoundError(msg)

        return {
            "transport": "stdio",
            "command": uvx_path,
            "args": ["--with", "onnxruntime", "chroma-mcp", "--client-type", "cloud"],
            "env": env,
        }
