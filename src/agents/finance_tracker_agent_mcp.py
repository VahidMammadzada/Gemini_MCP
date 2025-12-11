"""Finance Tracker Agent using MCP Toolbox Docker server via HTTP."""
import json
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

from src.core.config import config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from toolbox_langchain import ToolboxClient


def _to_text(payload: Any) -> str:
    """Convert model or tool output into a printable string."""
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False)
    except TypeError:
        return str(payload)


# Pydantic schemas for structured output
class PortfolioHolding(BaseModel):
    """A single stock holding in the portfolio."""
    symbol: str = Field(description="Stock ticker symbol (e.g., AAPL, MSFT)")
    quantity: float = Field(description="Number of shares held")
    avg_cost_basis: float = Field(description="Average cost per share")
    total_invested: float = Field(description="Total amount invested (quantity * avg_cost_basis)")
    realized_gains: Optional[float] = Field(default=None, description="Realized gains/losses from sales")


class TransactionRecord(BaseModel):
    """A stock transaction record."""
    symbol: str = Field(description="Stock ticker symbol")
    transaction_type: Literal["BUY", "SELL"] = Field(description="Type of transaction")
    quantity: float = Field(description="Number of shares")
    price: float = Field(description="Price per share")
    transaction_date: str = Field(description="Date of transaction")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class PortfolioResponse(BaseModel):
    """Structured response for portfolio queries."""
    response_type: Literal["portfolio_view", "transaction_added", "transaction_history", "general_response"] = Field(
        description="Type of response being provided"
    )
    summary: str = Field(description="Brief summary of the response")
    holdings: Optional[List[PortfolioHolding]] = Field(
        default=None,
        description="Current portfolio holdings (for portfolio_view)"
    )
    transaction: Optional[TransactionRecord] = Field(
        default=None,
        description="Transaction details (for transaction_added)"
    )
    transactions: Optional[List[TransactionRecord]] = Field(
        default=None,
        description="List of transactions (for transaction_history)"
    )
    total_portfolio_value: Optional[float] = Field(
        default=None,
        description="Total portfolio value"
    )
    insights: Optional[List[str]] = Field(
        default=None,
        description="Key insights or recommendations"
    )


def format_portfolio_response(response: PortfolioResponse) -> str:
    """Format structured portfolio response into readable text."""
    output = []

    output.append(response.summary)
    output.append("")

    if response.holdings:
        output.append("## Current Portfolio Holdings")
        output.append("")
        # Create table header
        output.append("| Symbol | Shares | Avg Cost | Total Invested | Realized Gains |")
        output.append("|--------|--------|----------|----------------|----------------|")
        # Add each holding as a table row
        for holding in response.holdings:
            realized_gains = f"${holding.realized_gains:.2f}" if holding.realized_gains is not None else "$0.00"
            output.append(
                f"| {holding.symbol} | {holding.quantity} | "
                f"${holding.avg_cost_basis:.2f} | ${holding.total_invested:.2f} | {realized_gains} |"
            )
        output.append("")

    # Format single transaction
    if response.transaction:
        output.append("## Transaction Details")
        output.append("")
        t = response.transaction
        output.append(f"- **Type**: {t.transaction_type}")
        output.append(f"- **Symbol**: {t.symbol}")
        output.append(f"- **Quantity**: {t.quantity} shares")
        output.append(f"- **Price**: ${t.price:.2f} per share")
        output.append(f"- **Date**: {t.transaction_date}")
        output.append(f"- **Total Amount**: ${t.quantity * t.price:.2f}")
        if t.notes:
            output.append(f"- **Notes**: {t.notes}")
        output.append("")

    # Format transaction history
    if response.transactions:
        output.append("## Transaction History")
        output.append("")
        for t in response.transactions:
            output.append(f"**{t.transaction_date}** - {t.transaction_type} {t.quantity} shares of {t.symbol} @ ${t.price:.2f}")
            if t.notes:
                output.append(f"  Notes: {t.notes}")
        output.append("")

    # Add total portfolio value
    if response.total_portfolio_value is not None:
        output.append(f"**Total Portfolio Value**: ${response.total_portfolio_value:.2f}")
        output.append("")

    # Add insights
    if response.insights:
        output.append("## Key Insights")
        output.append("")
        for insight in response.insights:
            output.append(f"- {insight}")
        output.append("")

    return "\n".join(output).strip()


class FinanceTrackerMCP:
    """
    Agent for managing personal investment portfolio using MCP Toolbox HTTP server.

    Features:
    - Portfolio Management: Track stock purchases/sales via Cloud SQL
    - Position Analysis: View holdings, cost basis, realized gains
    - Performance Metrics: Calculate gains/losses with current prices
    - Database Operations: Full CRUD via MCP Toolbox Docker container
    - Secure Access: Uses Google Cloud SQL with MCP Toolbox connector
    """

    def __init__(self):
        self.name = "Finance Tracker (MCP)"
        self.description = "Personal portfolio tracking and analysis using Cloud SQL via MCP Toolbox"
        self.toolbox_client: Optional[ToolboxClient] = None
        self.model: Optional[ChatGoogleGenerativeAI] = None
        self.model_with_tools = None
        self.model_structured = None
        self.tools: List[Any] = []
        self.tool_map: Dict[str, Any] = {}
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize the agent with MCP Toolbox HTTP server."""
        print(f"💼 Initializing {self.name}...")

        try:
            # Connect to MCP Toolbox HTTP server (Docker container)
            print(f"  📡 Connecting to MCP Toolbox HTTP server...")
            print(f"    Server URL: {config.MCP_TOOLBOX_SERVER_URL}")

            # Create toolbox client and enter async context manager
            self.toolbox_client = ToolboxClient(config.MCP_TOOLBOX_SERVER_URL)
            # Manually enter the async context manager to keep connection alive
            await self.toolbox_client.__aenter__()

            # Load database tools from toolbox
            print("    Loading tools from MCP Toolbox...")
            self.tools = self.toolbox_client.load_toolset()

            if not self.tools:
                raise RuntimeError(
                    f"No tools available from MCP Toolbox server at {config.MCP_TOOLBOX_SERVER_URL}\n"
                    f"Make sure the Docker container is running: docker-compose up -d"
                )

            self.tool_map = {tool.name: tool for tool in self.tools}

            # Initialize Gemini chat model bound to tools
            self.model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                google_api_key=config.GOOGLE_API_KEY,
            )
            self.model_with_tools = self.model.bind_tools(self.tools)

            # Initialize model with structured output for final responses
            self.model_structured = self.model.with_structured_output(PortfolioResponse)

            print(f"  ✅ Connected to MCP Toolbox with {len(self.tools)} tools")
            print(f"  📋 Available MCP Toolbox capabilities:")
            print(f"    - Database queries (SELECT, INSERT, UPDATE, DELETE)")
            print(f"    - Schema introspection (tables, columns, relationships)")
            print(f"    - Transaction management")
            print(f"    - Natural language to SQL conversion")
            print(f"    Total: {len(self.tools)} tools available")

            self.initialized = True
            print(f"  ✅ {self.name} ready!")

        except Exception as e:
            import traceback
            print(f"  ❌ Error initializing {self.name}: {e}")
            print(f"  📋 Full error details:")
            traceback.print_exc()
            print(f"\n💡 Troubleshooting:")
            print(f"   1. Make sure Docker is running")
            print(f"   2. Start MCP Toolbox container: docker-compose up -d")
            print(f"   3. Check container status: docker-compose ps")
            print(f"   4. View container logs: docker-compose logs mcp-toolbox")
            print(f"   5. Verify server is accessible: curl {config.MCP_TOOLBOX_SERVER_URL}/health")
            raise

    async def process(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Process a query using MCP Toolbox for Cloud SQL operations."""
        try:
            if not self.initialized:
                await self.initialize()

            print(f"\n💼 {self.name} processing: '{query}'")

            # Build system message with portfolio context
            system_message = """You are a Finance Tracker Agent specialized in portfolio management.

You have access to a Cloud SQL PostgreSQL database through MCP Toolbox with the following schema:

Tables:
- stock_transactions: Stores all BUY/SELL transactions (symbol, transaction_type, quantity, price, transaction_date, notes)
- portfolio_positions: Current aggregated positions (symbol, total_quantity, avg_cost_basis, total_invested, realized_gains)
- portfolio_snapshots: Historical portfolio value tracking
- stock_metadata: Cached stock information (company_name, sector, industry, market_cap)

When users want to:
- ADD a transaction: INSERT into stock_transactions (triggers will update portfolio_positions automatically)
- VIEW portfolio: SELECT from portfolio_positions
- CHECK transaction history: SELECT from stock_transactions
- ANALYZE performance: Query portfolio_positions with calculations

Always use the MCP Toolbox database tools to:
1. Query the database for current data
2. Insert/update records as needed
3. Calculate metrics (gains, losses, allocations)
4. Provide clear, actionable insights

Be helpful, accurate, and provide investment insights based on their data."""

            messages: List[Any] = [HumanMessage(content=system_message)]

            # Add conversation history
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

            # Tool calling loop - gather data from database
            tool_calls_info = []
            while True:
                if not self.model_with_tools:
                    raise RuntimeError("Model not initialized with tools")

                ai_message = await self.model_with_tools.ainvoke(messages)
                messages.append(ai_message)

                tool_calls = getattr(ai_message, "tool_calls", [])
                if not tool_calls:
                    break

                for call in tool_calls:
                    tool_name = call.get("name")
                    tool_args = call.get("args", {})
                    tool_call_id = call.get("id")
                    print(f"  🔧 MCP Toolbox call: {tool_name}({json.dumps(tool_args, indent=2)})")
                    tool_calls_info.append(f"🔧 MCP Toolbox call: {tool_name}({json.dumps(tool_args)})")

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

            # Generate structured response based on all gathered information
            print("  📊 Generating structured response...")
            structured_prompt = """Based on the database query results, provide a structured response.

Guidelines:
- For portfolio viewing: Set response_type to "portfolio_view" and populate holdings list
- For adding transactions: Set response_type to "transaction_added" and populate transaction field
- For transaction history: Set response_type to "transaction_history" and populate transactions list
- For other queries: Set response_type to "general_response"
- Always provide a clear summary
- Include relevant insights when possible"""

            messages.append(HumanMessage(content=structured_prompt))

            if not self.model_structured:
                raise RuntimeError("Structured output model not initialized")

            structured_response = await self.model_structured.ainvoke(messages)

            # Format the structured response into readable text
            final_response = format_portfolio_response(structured_response)

            return {
                "success": True,
                "agent": self.name,
                "response": final_response,
                "query": query,
                "structured_data": structured_response.model_dump(),
                "tool_calls": tool_calls_info
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
        if self.toolbox_client:
            try:
                await self.toolbox_client.__aexit__(None, None, None)
            except Exception as e:
                print(f"  ⚠️  Warning during cleanup: {e}")
        print(f"🧹 {self.name} cleaned up")