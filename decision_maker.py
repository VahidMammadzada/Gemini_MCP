"""Decision Maker to route queries to appropriate agents."""
from typing import Any, Dict, List, Optional

from config import config
from langchain_google_genai import ChatGoogleGenerativeAI


class DecisionMaker:
    """LLM-based decision maker to route queries to appropriate agents."""
    
    def __init__(self):
        self.name = "Decision Maker"
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.0,
            google_api_key=config.GOOGLE_API_KEY,
        )
        
        self.agent_descriptions = {
            "crypto": """
            Crypto Agent: Specializes in cryptocurrency market data, prices, trends, 
            market analysis, and blockchain information. Use this for queries about:
            - Cryptocurrency prices and market data
            - Trading volumes and market caps
            - Trending coins and top gainers/losers
            - Blockchain statistics
            - Crypto market analysis
            """,
            "rag": """
            RAG Agent: Specializes in document retrieval and question answering from
            uploaded documents (PDF, TXT, DOCX) stored in ChromaDB Cloud. Use this for queries about:
            - Questions about uploaded documents
            - Information retrieval from your document library
            - Document search and semantic analysis
            - Knowledge base queries
            - Content from stored files
            - Finding specific information in documents
            """
        }
    
    async def decide_agent(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Decide which agent should handle the query."""
        try:
            history_block = ""
            if history:
                recent_history = history[-6:]
                history_lines: List[str] = []
                for turn in recent_history:
                    user_text = turn.get("user", "").strip()
                    if user_text:
                        history_lines.append(f"User: {user_text}")
                    assistant_text = turn.get("assistant", "").strip()
                    if assistant_text:
                        history_lines.append(f"Assistant: {assistant_text}")
                if history_lines:
                    history_block = "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"

            prompt = f"""You are a routing system for a multi-agent AI application.

{history_block}Available agents:
{self.agent_descriptions['crypto']}

{self.agent_descriptions['rag']}

User query: "{query}"

Analyze the query and determine which agent should handle it.
Respond with ONLY a JSON object in this exact format:
{{
    "agent": "crypto" or "rag",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Choose carefully based on the query content."""

            response = await self.model.ainvoke(prompt)
            result_text = response.content.strip() if isinstance(response.content, str) else str(response.content)

            # Extract JSON from response
            import json
            # Remove markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            decision = json.loads(result_text)

            print(f"\n🎯 {self.name} Decision:")
            print(f"  Agent: {decision['agent']}")
            print(f"  Confidence: {decision['confidence']}")
            print(f"  Reasoning: {decision['reasoning']}")

            return decision

        except Exception as e:
            print(f"❌ Error in decision maker: {e}")
            # Default to crypto agent if decision fails
            return {
                "agent": "crypto",
                "confidence": 0.5,
                "reasoning": f"Error in decision maker, defaulting to crypto agent: {str(e)}"
            }