from typing import Any, Dict, List, Optional, TypedDict, Annotated, Sequence, AsyncGenerator
import operator

import asyncio

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


from config import config


class AgentState(TypedDict):
    """State for the ReAct agent pattern."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    agent_outputs: Dict[str, Any]
    reasoning_steps: List[str]
    final_answer: Optional[str]
    current_step: int
    max_steps: int


class ReActSupervisor:
    """Supervisor using ReAct pattern for multi-agent orchestration with streaming."""
    
    def __init__(
        self,
        crypto_agent=None,
        rag_agent=None,
        stock_agent=None,
        search_agent=None,
        finance_tracker=None,
        max_steps: int = 5
    ):
        """
        Initialize the ReAct supervisor.

        Args:
            crypto_agent: Crypto agent instance
            rag_agent: RAG agent instance
            stock_agent: Stock agent instance
            search_agent: Web search agent instance (DuckDuckGo)
            finance_tracker: Finance tracker agent instance
            max_steps: Maximum reasoning steps before forcing completion
        """
        self.crypto_agent = crypto_agent
        self.rag_agent = rag_agent
        self.stock_agent = stock_agent
        self.search_agent = search_agent
        self.finance_tracker = finance_tracker
        self.max_steps = max_steps
        self.streaming_callback = None  # For streaming updates
        
        # Initialize supervisor LLM with structured output
        self.supervisor_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.1,
            google_api_key=config.GOOGLE_API_KEY,
        )
        
        # Build the ReAct workflow
        self.graph = self._build_react_graph()
        self.compiled_graph = self.graph.compile()
    
    def _build_react_graph(self) -> StateGraph:
        """Build the ReAct pattern graph."""
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("think", self.think_node)
        workflow.add_node("act_crypto", self.act_crypto_node)
        workflow.add_node("act_rag", self.act_rag_node)
        workflow.add_node("act_stock", self.act_stock_node)
        workflow.add_node("act_search", self.act_search_node)
        workflow.add_node("act_finance_tracker", self.act_finance_tracker_node)
        workflow.add_node("observe", self.observe_node)
        workflow.add_node("finish", self.finish_node)
        
        # Set entry point
        workflow.set_entry_point("think")
        
        # Add routing from think node
        workflow.add_conditional_edges(
            "think",
            self.route_from_thinking,
            {
                "crypto": "act_crypto",
                "rag": "act_rag",
                "stock": "act_stock",
                "search": "act_search",
                "finance_tracker": "act_finance_tracker",
                "finish": "finish",
            }
        )

        # Actions lead to observe
        workflow.add_edge("act_crypto", "observe")
        workflow.add_edge("act_rag", "observe")
        workflow.add_edge("act_stock", "observe")
        workflow.add_edge("act_search", "observe")
        workflow.add_edge("act_finance_tracker", "observe")
        
        # Observe leads back to think (or finish if max steps)
        workflow.add_conditional_edges(
            "observe",
            self.should_continue,
            {
                "continue": "think",
                "finish": "finish"
            }
        )
        
        # Finish ends the graph
        workflow.add_edge("finish", END)
        
        return workflow
    
    async def _emit_update(self, update: Dict[str, Any]):
        """Emit streaming update if callback is set."""
        if self.streaming_callback:
            await self.streaming_callback(update)
    
    async def think_node(self, state: AgentState) -> Dict[str, Any]:
        """Reasoning step: Analyze current state and decide next action."""
        current_step = state.get("current_step", 0) + 1
        
        # Build context from previous outputs
        context = self._build_context(state)
        
        # Create reasoning prompt
        think_prompt = f"""You are a ReAct supervisor orchestrating multiple agents to answer user queries.

Current Query: {state['query']}

Available Actions:
- CALL_CRYPTO: Get cryptocurrency market data, prices, trends
- CALL_STOCK: Get stock market data, company information, financial data
- CALL_FINANCE_TRACKER: Manage personal stock portfolio (add transactions, view positions, analyze performance, get portfolio news)
- CALL_RAG: Search and retrieve information from uploaded documents
- CALL_SEARCH: Search the web for current information, news, or general knowledge
- FINISH: Provide final answer (use when you have sufficient information)

Current Step: {current_step}/{self.max_steps}

Information Gathered So Far:
{context}

IMPORTANT INSTRUCTIONS:
1. Check what information you ALREADY HAVE in the context above
2. Do NOT call the same agent twice unless you need different information
3. If you already have an answer from any agent, move to FINISH
4. Only call another agent if you need ADDITIONAL different information
5. Use CALL_SEARCH for general knowledge, current events, and news
6. FINISH when you have enough information to answer the user's query

Based on what you know so far, reason about what to do next.
Format your response as:

THOUGHT: [Analyze what you have and what you still need]
ACTION: [CALL_CRYPTO | CALL_STOCK | CALL_RAG | CALL_SEARCH | FINISH]
JUSTIFICATION: [Why this action will help]"""

        response = await self.supervisor_llm.ainvoke([
            SystemMessage(content="You are a ReAct supervisor. Think step by step and avoid redundant actions."),
            HumanMessage(content=think_prompt)
        ])
        
        # Parse the response
        content = response.content
        thought = ""
        action = "finish"
        justification = ""
        
        if "THOUGHT:" in content:
            thought = content.split("THOUGHT:")[1].split("ACTION:")[0].strip()
        
        if "ACTION:" in content:
            action_text = content.split("ACTION:")[1].split("\n")[0].strip().upper()
            if "CRYPTO" in action_text:
                action = "crypto"
            elif "FINANCE_TRACKER" in action_text or "FINANCE" in action_text:
                action = "finance_tracker"
            elif "STOCK" in action_text:
                action = "stock"
            elif "RAG" in action_text:
                action = "rag"
            elif "SEARCH" in action_text:
                action = "search"
            else:
                action = "finish"
        
        if "JUSTIFICATION:" in content:
            justification = content.split("JUSTIFICATION:")[1].strip()
        
        # Add reasoning step
        reasoning_steps = state.get("reasoning_steps", [])
        reasoning_steps.append(f"Step {current_step}: {thought} -> Action: {action}")
        
        print(f"\nThinking (Step {current_step}):")
        print(f"   Thought: {thought}")
        print(f"   Action: {action}")
        print(f"   Justification: {justification}")
        
        # Emit streaming update
        await self._emit_update({
            "type": "thinking",
            "step": current_step,
            "thought": thought,
            "action": action,
            "justification": justification
        })
        
        return {
            "current_step": current_step,
            "reasoning_steps": reasoning_steps,
            "messages": [AIMessage(content=f"Thought: {thought}\nAction: {action}")],
            "next_action": action
        }
    
    async def act_crypto_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute crypto agent and return raw output."""
        if not self.crypto_agent:
            return {"agent_outputs": {"crypto_error": "Crypto agent not available"}}
        
        print("   Calling Crypto Agent...")
        await self._emit_update({"type": "action", "agent": "crypto"})
        
        result = await self.crypto_agent.process(
            state["query"],
            history=self._extract_history(state["messages"])
        )
        
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["crypto"] = result
        
        return {"agent_outputs": agent_outputs}
    
    async def act_rag_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute RAG agent and return raw output."""
        if not self.rag_agent:
            return {"agent_outputs": {"rag_error": "RAG agent not available"}}
        
        print("   Calling RAG Agent...")
        await self._emit_update({"type": "action", "agent": "rag"})
        
        result = await self.rag_agent.process(
            state["query"],
            history=self._extract_history(state["messages"])
        )
        
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["rag"] = result
        
        return {"agent_outputs": agent_outputs}
    
    async def act_stock_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute stock agent and return raw output."""
        if not self.stock_agent:
            return {"agent_outputs": {"stock_error": "Stock agent not available"}}
        
        print("   Calling Stock Agent...")
        await self._emit_update({"type": "action", "agent": "stock"})
        
        result = await self.stock_agent.process(
            state["query"],
            history=self._extract_history(state["messages"])
        )
        
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["stock"] = result
        
        return {"agent_outputs": agent_outputs}
    
    async def act_search_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute web search agent and return raw output."""
        if not self.search_agent:
            return {"agent_outputs": {"search_error": "Search agent not available"}}

        print("   Calling Web Search Agent...")
        await self._emit_update({"type": "action", "agent": "search"})

        result = await self.search_agent.process(
            state["query"],
            history=self._extract_history(state["messages"])
        )

        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["search"] = result

        return {"agent_outputs": agent_outputs}

    async def act_finance_tracker_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute finance tracker agent and return raw output."""
        if not self.finance_tracker:
            return {"agent_outputs": {"finance_tracker_error": "Finance Tracker agent not available"}}

        print("   Calling Finance Tracker Agent...")
        await self._emit_update({"type": "action", "agent": "finance_tracker"})

        result = await self.finance_tracker.process(
            state["query"],
            history=self._extract_history(state["messages"])
        )

        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["finance_tracker"] = result

        return {"agent_outputs": agent_outputs}

    async def observe_node(self, state: AgentState) -> Dict[str, Any]:
        """Process and observe the latest agent output."""
        agent_outputs = state.get("agent_outputs", {})
        
        latest_output = "No new observations"
        latest_agent = "unknown"
        
        if agent_outputs:
            for agent_name, output in list(agent_outputs.items())[-1:]:
                if isinstance(output, dict) and output.get("success"):
                    response = output.get('response', 'No response')
                    latest_output = response
                    latest_agent = agent_name
                    break
        
        # Log summary
        summary = latest_output[:250] + "..." if len(latest_output) > 250 else latest_output
        print(f"   Observation from {latest_agent}: {summary}")
        
        # Emit streaming update
        await self._emit_update({
            "type": "observation",
            "agent": latest_agent,
            "summary": summary
        })
        
        return {
            "messages": [AIMessage(content=f"Observation from {latest_agent}:\n{latest_output}")]
        }
    
    async def finish_node(self, state: AgentState) -> Dict[str, Any]:
        """Synthesize all agent outputs and generate final answer with streaming."""
        agent_outputs = state.get("agent_outputs", {})
        reasoning_steps = state.get("reasoning_steps", [])

        print("\nSupervisor Synthesizing Final Answer...")

        # Build synthesis prompt
        synthesis_prompt = f"""You are synthesizing information to answer this query: {state['query']}

Your reasoning process:
{chr(10).join(reasoning_steps)}

Information gathered from agents:"""

        for agent_name, output in agent_outputs.items():
            if isinstance(output, dict) and output.get("success"):
                synthesis_prompt += f"\n\n{agent_name.upper()} Agent Response:\n{output.get('response', 'No response')}"

        synthesis_prompt += """

Now provide a comprehensive, well-structured answer that:
1. Directly addresses the user's query
2. Integrates insights from all relevant agent outputs
3. Is clear and actionable
4. Highlights any important findings or recommendations
5. Cites sources when appropriate

Final Answer:"""

        # Emit start of final answer
        await self._emit_update({
            "type": "final_start"
        })

        # Stream the final answer token by token
        final_answer = ""
        async for chunk in self.supervisor_llm.astream([
            SystemMessage(content="You are providing the final, synthesized answer."),
            HumanMessage(content=synthesis_prompt)
        ]):
            if hasattr(chunk, 'content') and chunk.content:
                final_answer += chunk.content
                # Emit each token/chunk as it arrives
                await self._emit_update({
                    "type": "final_token",
                    "token": chunk.content,
                    "accumulated": final_answer
                })

        # Emit completion of final answer
        await self._emit_update({
            "type": "final_complete",
            "response": final_answer
        })

        return {
            "final_answer": final_answer,
            "messages": [AIMessage(content=final_answer)]
        }
    
    def route_from_thinking(self, state: AgentState) -> str:
        """Route based on thinking decision."""
        last_message = state["messages"][-1] if state["messages"] else None
        
        if last_message and "Action:" in last_message.content:
            try:
                action_line = last_message.content.split("Action:")[1].split("\n")[0].strip().upper()
                
                if "CRYPTO" in action_line or "CALL_CRYPTO" in action_line:
                    return "crypto"
                elif "FINANCE_TRACKER" in action_line or "CALL_FINANCE_TRACKER" in action_line or "FINANCE" in action_line:
                    return "finance_tracker"
                elif "STOCK" in action_line or "CALL_STOCK" in action_line:
                    return "stock"
                elif "RAG" in action_line or "CALL_RAG" in action_line:
                    return "rag"
                elif "SEARCH" in action_line or "CALL_SEARCH" in action_line:
                    return "search"
                elif "FINISH" in action_line:
                    return "finish"
            except (IndexError, AttributeError):
                pass
        
        return "finish"
    
    def should_continue(self, state: AgentState) -> str:
        """Decide whether to continue reasoning or finish."""
        current_step = state.get("current_step", 0)
        
        if current_step >= self.max_steps:
            print(f"   Max steps ({self.max_steps}) reached, finishing...")
            return "finish"
        
        return "continue"
    
    def _build_context(self, state: AgentState) -> str:
        """Build context string from current state."""
        agent_outputs = state.get("agent_outputs", {})
        
        if not agent_outputs:
            return "No information gathered yet."
        
        context_parts = []
        for agent_name, output in agent_outputs.items():
            if isinstance(output, dict) and output.get("success"):
                response = output.get("response", "No response")
                if len(response) > 1000:
                    response = response[:1000] + f"... [Response continues for {len(response)} total chars]"
                context_parts.append(f"=== {agent_name.upper()} Agent ===\n{response}")
        
        return "\n\n".join(context_parts) if context_parts else "No successful agent outputs yet."
    
    def _extract_history(self, messages: Sequence[BaseMessage]) -> List[Dict[str, str]]:
        """Extract chat history from messages."""
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"user": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"assistant": msg.content})
        return history[-10:]
    
    async def process(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Process query through ReAct supervisor pattern."""
        initial_state: AgentState = {
            "messages": [],
            "query": query,
            "agent_outputs": {},
            "reasoning_steps": [],
            "final_answer": None,
            "current_step": 0,
            "max_steps": self.max_steps
        }
        
        if history:
            for turn in history[-3:]:
                if "user" in turn:
                    initial_state["messages"].append(HumanMessage(content=turn["user"]))
                if "assistant" in turn:
                    initial_state["messages"].append(AIMessage(content=turn["assistant"]))
        
        try:
            print(f"\nReAct Supervisor starting for query: '{query}'")
            print(f"   Max steps: {self.max_steps}")
            
            final_state = await self.compiled_graph.ainvoke(initial_state)
            
            return {
                "success": True,
                "response": final_state.get("final_answer", "No answer generated"),
                "reasoning_steps": final_state.get("reasoning_steps", []),
                "agent_outputs": final_state.get("agent_outputs", {}),
                "steps_taken": final_state.get("current_step", 0)
            }
            
        except Exception as e:
            print(f"   ReAct Supervisor error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": f"Supervisor error: {str(e)}"
            }
    
    async def process_streaming(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process query with streaming updates.
        
        Yields update dictionaries with types: thinking, action, observation, final, error
        """
        updates_queue = []
        
        async def callback(update: Dict[str, Any]):
            """Callback to capture streaming updates."""
            updates_queue.append(update)
        
        # Set streaming callback
        self.streaming_callback = callback
        
        # Start processing in background
        result_task = asyncio.create_task(self.process(query, history))
        
        # Yield updates as they come in
        while not result_task.done():
            if updates_queue:
                update = updates_queue.pop(0)
                yield update
            await asyncio.sleep(0.1)
        
        # Yield any remaining updates
        while updates_queue:
            update = updates_queue.pop(0)
            yield update
        
        # Get final result
        try:
            result = await result_task
            if result.get("success"):
                # Final update already emitted by finish_node
                pass
            else:
                yield {
                    "type": "error",
                    "error": result.get("error", "Unknown error")
                }
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e)
            }
        finally:
            self.streaming_callback = None