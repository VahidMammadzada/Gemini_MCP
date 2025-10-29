"""Proper MCP client using official Python SDK for MCP servers."""
import asyncio
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from config import config


class ProperMCPClient:
    """Base class for connecting to MCP servers using official SDK."""

    def __init__(self, name: str, server_command: str, server_args: List[str], env: Optional[Dict[str, str]] = None):
        """
        Initialize MCP client.

        Args:
            name: Name of the MCP server
            server_command: Command to start the MCP server (e.g., 'npx', 'uvx')
            server_args: Arguments for the server command
            env: Optional environment variables
        """
        self.name = name
        self.server_command = server_command
        self.server_args = server_args
        self.env = env or {}
        self.session: Optional[ClientSession] = None
        self.exit_stack = None
        self.available_tools: List[Dict[str, Any]] = []

    async def connect(self) -> bool:
        """
        Connect to the MCP server using stdio transport.

        Returns:
            True if connection successful
        """
        try:
            # Prepare environment for subprocess (ensure venv scripts on PATH)
            env_vars = os.environ.copy()
            if sys.prefix:
                scripts_dir = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
                if scripts_dir.exists():
                    current_path = env_vars.get("PATH", "")
                    env_vars["PATH"] = f"{scripts_dir}{os.pathsep}{current_path}" if current_path else str(scripts_dir)

            if self.env:
                env_vars.update(self.env)

            # Create server parameters
            server_params = StdioServerParameters(
                command=self.server_command,
                args=self.server_args,
                env=env_vars
            )

            # Create stdio client and session
            from contextlib import AsyncExitStack
            self.exit_stack = AsyncExitStack()

            # Connect to server via stdio
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )

            read, write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )

            # Initialize the session
            await self.session.initialize()

            # List available tools
            tools_result = await self.session.list_tools()
            self.available_tools = tools_result.tools if hasattr(tools_result, 'tools') else []

            print(f"  ✅ Connected to {self.name} MCP Server")
            print(f"  📋 Available tools: {len(self.available_tools)}")

            # Print tool names for debugging
            for tool in self.available_tools:
                tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                print(f"    - {tool_name}")

            return True

        except Exception as e:
            print(f"  ❌ Failed to connect to {self.name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool

        Returns:
            Tool execution result
        """
        if not self.session:
            raise RuntimeError(f"MCP client not connected. Call connect() first.")

        try:
            # Call the tool using the session
            result = await self.session.call_tool(tool_name, arguments)

            # Parse result
            if hasattr(result, 'content'):
                # Extract content from MCP response
                content = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        content.append(item.text)
                    elif hasattr(item, 'data'):
                        content.append(item.data)

                return {
                    "success": True,
                    "result": content[0] if len(content) == 1 else content
                }

            return {
                "success": True,
                "result": str(result)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }

    def get_tools_for_gemini(self) -> List[Dict[str, Any]]:
        """
        Convert MCP tools to Gemini function declarations format.

        Returns:
            List of function declarations for Gemini
        """
        def sanitize_schema(node: Any) -> None:
            if isinstance(node, dict):
                node.pop("title", None)
                any_of = node.pop("anyOf", None)
                if any_of:
                    replacement = any_of[0] if isinstance(any_of, list) and any_of else None
                    if isinstance(replacement, dict):
                        # merge first option into current node
                        for key, value in replacement.items():
                            node.setdefault(key, deepcopy(value))
                    sanitize_schema(replacement)

                for key, value in list(node.items()):
                    sanitize_schema(value)
            elif isinstance(node, list):
                for item in node:
                    sanitize_schema(item)

        function_declarations = []

        for tool in self.available_tools:
            # Extract tool information
            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
            tool_description = tool.description if hasattr(tool, 'description') else f"MCP tool: {tool_name}"

            # Extract input schema
            tool_schema = tool.inputSchema if hasattr(tool, 'inputSchema') else {}

            if tool_schema:
                if hasattr(tool_schema, "to_dict"):
                    parameters = tool_schema.to_dict()
                elif isinstance(tool_schema, dict):
                    parameters = deepcopy(tool_schema)
                else:
                    try:
                        parameters = deepcopy(tool_schema)
                    except Exception:
                        parameters = tool_schema
            else:
                parameters = {
                    "type": "object",
                    "properties": {}
                }

            sanitize_schema(parameters)

            # Convert to Gemini format
            function_decl = {
                "name": tool_name,
                "description": tool_description,
                "parameters": parameters
            }

            function_declarations.append(function_decl)

        return [{"function_declarations": function_declarations}]

    async def cleanup(self) -> None:
        """Close the MCP session."""
        if self.exit_stack:
            await self.exit_stack.aclose()
            print(f"  🧹 {self.name} MCP client closed")


class ChromaMCPClient(ProperMCPClient):
    """MCP client for Chroma MCP Server."""

    def __init__(self):
        """Initialize Chroma MCP client."""
        # Prepare environment variables for Chroma Cloud
        env = {
            "CHROMA_CLIENT_TYPE": "cloud",
            "CHROMA_TENANT": config.CHROMA_TENANT,
            "CHROMA_DATABASE": config.CHROMA_DATABASE,
            "CHROMA_API_KEY": config.CHROMA_API_KEY,
        }

        # Initialize with uvx command to run chroma-mcp
        super().__init__(
            name="Chroma",
            server_command="uvx",
            server_args=[
                "chroma-mcp",
                "--client-type", "cloud"
            ],
            env=env
        )


class CoinGeckoMCPClient(ProperMCPClient):
    """MCP client for CoinGecko MCP Server."""

    def __init__(self):
        """Initialize CoinGecko MCP client."""
        # Prepare environment variables
        env = {}
        if config.COINGECKO_API_KEY:
            env["COINGECKO_PRO_API_KEY"] = config.COINGECKO_API_KEY
            env["COINGECKO_ENVIRONMENT"] = "pro"

        # Initialize with npx command to run CoinGecko MCP
        super().__init__(
            name="CoinGecko",
            server_command="npx",
            server_args=[
                "-y",
                "@coingecko/coingecko-mcp"
            ],
            env=env if env else None
        )


# Alternative: Use public CoinGecko MCP endpoint
class CoinGeckoPublicMCPClient(ProperMCPClient):
    """MCP client for CoinGecko Public MCP Server."""

    def __init__(self):
        """Initialize CoinGecko Public MCP client."""
        # Use the public endpoint via mcp-remote
        super().__init__(
            name="CoinGecko Public",
            server_command="npx",
            server_args=[
                "mcp-remote",
                "https://mcp.api.coingecko.com/sse"
            ],
            env=None
        )
