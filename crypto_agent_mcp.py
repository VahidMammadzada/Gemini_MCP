"""Crypto Agent using CoinGecko MCP Server via LangChain MCP adapters."""
import json
import shutil
from typing import Any, Dict, List, Optional

from config import config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient


def _resolve_executable(candidates: List[str]) -> str:
    """Return first executable path found in PATH."""
    for name in candidates:
        resolved = shutil.which(name)
        if resolved:
            return resolved
    raise FileNotFoundError(f"Unable to locate any of: {', '.join(candidates)}")


def _to_text(payload: Any) -> str:
    """Convert model or tool output into a printable string."""
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False)
    except TypeError:
        return str(payload)


class CryptoAgentMCP:
    """Agent specialized in cryptocurrency data using CoinGecko MCP Server."""

    def __init__(self, use_public_endpoint: bool = False):
        self.name = "Crypto Agent (MCP)"
        self.description = "Cryptocurrency market data and analysis expert using CoinGecko MCP Server"
        self.use_public_endpoint = use_public_endpoint
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.model: Optional[ChatGoogleGenerativeAI] = None
        self.model_with_tools = None
        self.tools: List[BaseTool] = []
        self.tool_map: Dict[str, BaseTool] = {}

    async def initialize(self) -> None:
        """Initialize the agent with CoinGecko MCP Server."""
        print(f"🔧 Initializing {self.name}...")

        try:
            # Connect to CoinGecko MCP Server
            print(f"  📡 Connecting to CoinGecko MCP Server...")

            connection_name = "coingecko"
            connections: Dict[str, Dict[str, Any]] = {}

            api_key = (config.COINGECKO_API_KEY or "").strip()
            if api_key.lower().startswith("demo"):
                print("    Demo API key detected. Using public endpoint with limited access...")
                self.use_public_endpoint = True

            if self.use_public_endpoint or not api_key:
                print("    Using public SSE endpoint...")
                connections[connection_name] = {
                    "transport": "sse",
                    "url": "https://mcp.api.coingecko.com/sse",
                }
            else:
                print("    Using Pro endpoint with API key...")
                npx_executable = _resolve_executable(["npx.cmd", "npx.exe", "npx"])
                env = {
                    "COINGECKO_PRO_API_KEY": api_key,
                    "COINGECKO_ENVIRONMENT": "pro",
                }
                connections[connection_name] = {
                    "transport": "stdio",
                    "command": npx_executable,
                    "args": ["-y", "@coingecko/coingecko-mcp"],
                    "env": env,
                }

            self.mcp_client = MultiServerMCPClient(connections)

            # Load MCP tools as LangChain tools
            self.tools = await self.mcp_client.get_tools(server_name=connection_name)
            if not self.tools:
                raise RuntimeError("No tools available from CoinGecko MCP Server")

            self.tool_map = {tool.name: tool for tool in self.tools}

            # Initialize Gemini chat model bound to tools
            self.model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0.0,
                google_api_key=config.GOOGLE_API_KEY,
            )
            self.model_with_tools = self.model.bind_tools(self.tools)

            print(f"  ✅ Connected to CoinGecko MCP Server with {len(self.tools)} tools")

            print(f"  ✅ {self.name} ready!")

        except Exception as e:
            import traceback
            print(f"  ❌ Error initializing {self.name}: {e}")
            print(f"  📋 Full error details:")
            traceback.print_exc()
            raise

    async def process(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Process a query using CoinGecko MCP Server tools."""
        try:
            print(f"\n💰 {self.name} processing: '{query}'")
            messages: List[Any] = []
            if history:
                trimmed_history = history[-10:]
                for turn in trimmed_history:
                    user_text = turn.get("user")
                    if user_text:
                        messages.append(HumanMessage(content=user_text))
                    assistant_text = turn.get("assistant")
                    if assistant_text:
                        messages.append(AIMessage(content=assistant_text))
            messages.append(HumanMessage(content=query))
            final_response = ""

            while True:
                if not self.model_with_tools:
                    raise RuntimeError("Model not initialized with tools")

                ai_message = await self.model_with_tools.ainvoke(messages)
                messages.append(ai_message)

                tool_calls = getattr(ai_message, "tool_calls", [])
                if not tool_calls:
                    final_response = _to_text(ai_message.content)
                    break

                for call in tool_calls:
                    tool_name = call.get("name")
                    tool_args = call.get("args", {})
                    tool_call_id = call.get("id")
                    print(f"  🔧 MCP Tool call: {tool_name}({tool_args})")

                    tool = self.tool_map.get(tool_name)
                    if not tool:
                        tool_result = {"error": f"Tool '{tool_name}' not found"}
                    else:
                        tool_result = await tool.ainvoke(tool_args)

                    messages.append(
                        ToolMessage(
                            content=_to_text(tool_result),
                            tool_call_id=tool_call_id or "",
                        )
                    )

            return {
                "success": True,
                "agent": self.name,
                "response": final_response,
                "query": query
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
