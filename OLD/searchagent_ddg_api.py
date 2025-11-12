"""DuckDuckGo Search Agent for web search capabilities."""
import asyncio
from typing import Dict, Any, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from ddgs import DDGS
from config import config


class SearchAgentMCP:
    """
    Agent for performing web searches using DuckDuckGo.
    Follows the same pattern as other MCP agents in the system.
    """

    def __init__(self):
        """Initialize the DuckDuckGo search agent."""
        self.llm = None
        self.ddgs = DDGS()
        self.initialized = False

    async def initialize(self):
        """Initialize the agent with LLM."""
        if not self.initialized:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key=config.GOOGLE_API_KEY,
            )
            self.initialized = True
            print("✅ Search Agent initialized")

    async def search_web(
        self,
        query: str,
        max_results: int = 4,
        region: str = "wt-wt"
    ) -> Dict[str, Any]:
        """
        Perform a web search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 5)
            region: Region for search (default: wt-wt for worldwide)

        Returns:
            Dictionary with search results
        """
        try:
            # Perform search
            results = []
            # Pass query as positional argument (API changed in newer versions)
            search_results = list(self.ddgs.text(
                query,
                region=region,
                safesearch='moderate',
                max_results=max_results
            ))

            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "link": result.get("href", ""),
                    "snippet": result.get("body", "")
                })

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }

    async def search_news(
        self,
        query: str,
        max_results: int = 4
    ) -> Dict[str, Any]:
        """
        Search for news articles using DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            Dictionary with news results
        """
        try:
            results = []
            # Pass query as positional argument (API changed in newer versions)
            news_results = list(self.ddgs.news(
                query,
                safesearch='moderate',
                max_results=max_results
            ))

            for result in news_results:
                results.append({
                    "title": result.get("title", ""),
                    "link": result.get("url", ""),
                    "snippet": result.get("body", ""),
                    "source": result.get("source", ""),
                    "date": result.get("date", "")
                })

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }

    def _format_search_results(self, search_data: Dict[str, Any]) -> str:
        """
        Format search results into a readable string.

        Args:
            search_data: Raw search results

        Returns:
            Formatted string of results
        """
        if not search_data.get("success"):
            return f"Search failed: {search_data.get('error', 'Unknown error')}"

        results = search_data.get("results", [])
        if not results:
            return f"No results found for query: {search_data.get('query')}"

        formatted = f"Found {len(results)} results for '{search_data.get('query')}':\n\n"

        for idx, result in enumerate(results, 1):
            formatted += f"{idx}. {result.get('title', 'No title')}\n"
            formatted += f"   URL: {result.get('link', 'N/A')}\n"
            formatted += f"   {result.get('snippet', 'No description')}\n"
            
            # Add source and date for news results
            if result.get('source'):
                formatted += f"   Source: {result['source']}"
                if result.get('date'):
                    formatted += f" | Date: {result['date']}"
                formatted += "\n"
            
            formatted += "\n"

        return formatted

    async def process(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a search query with LLM interpretation.

        Args:
            query: User's search query
            history: Optional conversation history

        Returns:
            Dictionary with agent response
        """
        try:
            if not self.initialized:
                await self.initialize()

            # Determine search type and execute
            search_type = await self._determine_search_type(query)
            
            if search_type == "news":
                search_results = await self.search_news(query, max_results=5)
            else:
                search_results = await self.search_web(query, max_results=5)

            # Format results
            formatted_results = self._format_search_results(search_results)

            # Generate LLM response with context
            system_prompt = """You are a web search assistant. Your role is to:
1. Interpret web search results
2. Synthesize information from multiple sources
3. Provide clear, accurate answers with source attribution
4. Highlight the most relevant findings
5. Note any limitations or conflicting information

Always cite sources by referring to their result numbers (e.g., "According to result #1...")."""

            user_prompt = f"""User Query: {query}

Search Results:
{formatted_results}

Based on these search results, provide a comprehensive answer to the user's query. 
Include relevant details and cite your sources."""

            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            return {
                "success": True,
                "response": response.content,
                "raw_results": search_results,
                "search_type": search_type,
                "metadata": {
                    "results_count": search_results.get("count", 0),
                    "query": query
                }
            }

        except Exception as e:
            error_msg = f"Error processing search query: {str(e)}"
            return {
                "success": False,
                "error": error_msg,
                "response": f"I encountered an error while searching: {str(e)}"
            }

    async def _determine_search_type(self, query: str) -> str:
        """
        Determine if query is asking for news or general web search.

        Args:
            query: User query

        Returns:
            "news" or "web"
        """
        query_lower = query.lower()
        news_keywords = ["news", "latest", "recent", "today", "breaking", "update"]
        
        if any(keyword in query_lower for keyword in news_keywords):
            return "news"
        
        return "web"

    async def cleanup(self):
        """Cleanup resources."""
        if self.initialized:
            # DuckDuckGo search doesn't need explicit cleanup
            self.initialized = False
            print("🧹 Search Agent cleanup complete")