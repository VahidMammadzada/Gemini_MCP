"""Search Agent using DuckDuckGo MCP Server from Docker."""
import asyncio
from typing import Dict, Any, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from config import config


class SearchAgentMCP:
    """
    Agent for performing web searches using DuckDuckGo MCP Server from Docker.
    Follows the same pattern as StockAgentMCP using langchain-mcp-adapters.
    """

    def __init__(self):
        """Initialize the DuckDuckGo search agent."""
        self.name = "Search Agent (MCP)"
        self.description = "Web search expert using DuckDuckGo MCP Server from Docker"
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.llm_with_tools = None
        self.tools: List[BaseTool] = []
        self.tool_map: Dict[str, BaseTool] = {}
        self.initialized = False

    async def initialize(self):
        """Initialize the agent with MCP client and LLM."""
        if not self.initialized:
            print("🔍 Initializing Search Agent (MCP)...")

            try:
                # Connect to DuckDuckGo MCP Server from Docker
                print("  📡 Connecting to DuckDuckGo MCP Server (Docker)...")

                connection_name = "duckduckgo"
                connections: Dict[str, Dict[str, Any]] = {}

                # DuckDuckGo MCP Server from Docker Hub uses stdio transport
                # Command: docker run -i --rm mcp/duckduckgo
                connections[connection_name] = {
                    "transport": "stdio",
                    "command": "docker",
                    "args": [
                        "run",
                        "-i",           # Interactive (stdin)
                        "--rm",         # Remove container when done
                        "mcp/duckduckgo"  # Official Docker MCP Catalog image
                    ],
                }

                print("    Using Docker MCP Catalog image: mcp/duckduckgo")

                self.mcp_client = MultiServerMCPClient(connections)

                # Load MCP tools as LangChain tools
                print("    Loading tools from DuckDuckGo MCP Server...")
                self.tools = await self.mcp_client.get_tools(server_name=connection_name)

                if not self.tools:
                    raise RuntimeError(
                        "No tools available from DuckDuckGo MCP Server\n"
                        "Make sure Docker is installed and running:\n"
                        "  - Check: docker ps\n"
                        "  - Test: docker run --rm hello-world"
                    )

                self.tool_map = {tool.name: tool for tool in self.tools}

                # Initialize Gemini chat model bound to tools
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.3,
                    google_api_key=config.GOOGLE_API_KEY,
                )
                self.llm_with_tools = self.llm.bind_tools(self.tools)

                print(f"  ✅ Connected to DuckDuckGo MCP Server")
                print(f"  📋 Available tools: {len(self.tools)}")
                for tool in self.tools:
                    print(f"    - {tool.name}")
                print(f"  ✅ Bound {len(self.tools)} tools to LLM")

                self.initialized = True
                print("  ✅ Search Agent (MCP) ready!")

            except Exception as e:
                import traceback
                print(f"  ❌ Error initializing Search Agent: {e}")
                print(f"  📋 Full error details:")
                traceback.print_exc()
                print(f"\n💡 Troubleshooting:")
                print(f"   1. Make sure Docker is running: docker ps")
                print(f"   2. Test Docker: docker run --rm hello-world")
                print(f"   3. Pull the image manually: docker pull mcp/duckduckgo")
                print(f"   4. Check Docker is in PATH: where docker (Windows) or which docker (Linux/Mac)")
                raise

    async def process(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a search query through the DuckDuckGo MCP server.

        Args:
            query: User's search query
            history: Optional conversation history

        Returns:
            Dictionary with agent response
        """
        try:
            if not self.initialized:
                await self.initialize()

            print(f"\n🔍 Search Agent processing: '{query}'")

            # Build system prompt
            system_prompt = """You are a web search assistant with access to DuckDuckGo search.

CRITICAL: You MUST use the available tools to find current information. DO NOT answer from memory.

Your process:
1. Use the search tool with the user's query
2. Read the search results carefully
3. Synthesize a clear, accurate answer with source citations

Always use the search tool first before answering."""

            # Prepare messages
            messages = [SystemMessage(content=system_prompt)]

            # Add conversation history if provided (limit to last 2 turns)
            if history:
                for turn in history[-2:]:
                    if "user" in turn:
                        messages.append(HumanMessage(content=turn["user"]))
                    if "assistant" in turn:
                        messages.append(AIMessage(content=turn["assistant"]))

            # Add current query
            messages.append(HumanMessage(content=query))

            # Tool calling loop
            max_iterations = 3
            for iteration in range(max_iterations):
                # Get LLM response
                response = await self.llm_with_tools.ainvoke(messages)
                messages.append(response)

                # Check for tool calls
                tool_calls = getattr(response, 'tool_calls', None) or []

                if not tool_calls:
                    # No more tool calls, return final response
                    final_content = response.content if hasattr(response, 'content') else str(response)
                    print(f"  ✅ Search complete")
                    return {
                        "success": True,
                        "response": final_content
                    }

                # Execute tool calls
                for tool_call in tool_calls:
                    # Handle both dict and object formats
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('args', {})
                        tool_id = tool_call.get('id', '')
                    else:
                        tool_name = getattr(tool_call, 'name', '')
                        tool_args = getattr(tool_call, 'args', {})
                        tool_id = getattr(tool_call, 'id', '')

                    print(f"  🔧 Executing: {tool_name}({tool_args})")

                    # Get the tool
                    tool = self.tool_map.get(tool_name)
                    if not tool:
                        tool_result = f"Error: Tool '{tool_name}' not found"
                    else:
                        try:
                            # Call the tool with timeout
                            tool_result = await asyncio.wait_for(
                                tool.ainvoke(tool_args),
                                timeout=30.0
                            )
                            print(f"  ✅ Tool executed successfully")

                            # Truncate if too long
                            if isinstance(tool_result, str) and len(tool_result) > 5000:
                                tool_result = tool_result[:5000] + "\n\n[Results truncated]"
                            elif isinstance(tool_result, dict):
                                result_str = str(tool_result)
                                if len(result_str) > 5000:
                                    tool_result = result_str[:5000] + "\n\n[Results truncated]"

                        except asyncio.TimeoutError:
                            tool_result = "Error: Search timed out"
                            print(f"  ⚠️ Tool call timed out")
                        except Exception as e:
                            tool_result = f"Error: {str(e)}"
                            print(f"  ❌ Tool execution failed: {e}")

                    # Add tool result to messages
                    messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_id
                        )
                    )

            # If we hit max iterations, return last response
            print(f"  ⚠️ Max iterations reached")
            return {
                "success": True,
                "response": "Search completed but may be incomplete. Try a more specific query."
            }

        except Exception as e:
            error_msg = f"Error processing search query: {str(e)}"
            print(f"  ❌ {error_msg}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": error_msg,
                "response": f"I encountered an error: {str(e)}"
            }

    async def cleanup(self):
        """Cleanup resources."""
        if self.mcp_client:
            try:
                await self.mcp_client.cleanup()
            except Exception as e:
                print(f"  ⚠️ Warning during cleanup: {e}")
        self.initialized = False
        print("🧹 Search Agent cleanup complete")