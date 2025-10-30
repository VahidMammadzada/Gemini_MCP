from typing import Any, Dict, List, Optional, TypedDict, Annotated, Sequence, Literal
from enum import Enum
import operator
import json

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import config


class AgentState(TypedDict):
    """State for the ReAct agent pattern."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    agent_outputs: Dict[str, Any]  # Stores raw outputs from each agent
    reasoning_steps: List[str]  # Supervisor's reasoning trace
    final_answer: Optional[str]
    current_step: int
    max_steps: int


class ReActSupervisor:
    """Supervisor using ReAct pattern for multi-agent orchestration."""
    
    def __init__(self, crypto_agent=None, rag_agent=None, stock_agent=None, max_steps: int = 3):
        """
        Initialize the ReAct supervisor.
        
        Args:
            crypto_agent: Crypto agent instance
            rag_agent: RAG agent instance
            stock_agent: Stock agent instance
            max_steps: Maximum reasoning steps before forcing completion
        """
        self.crypto_agent = crypto_agent
        self.rag_agent = rag_agent
        self.stock_agent = stock_agent
        self.max_steps = max_steps
        
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
        workflow.add_node("think", self.think_node)  # Reasoning step
        workflow.add_node("act_crypto", self.act_crypto_node)  # Action: call crypto agent
        workflow.add_node("act_rag", self.act_rag_node)  # Action: call RAG agent
        workflow.add_node("act_stock", self.act_stock_node)  # Action: call stock agent
        workflow.add_node("observe", self.observe_node)  # Process agent outputs
        workflow.add_node("finish", self.finish_node)  # Generate final answer
        
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
                "finish": "finish",
            }
        )
        
        # Actions lead to observe
        workflow.add_edge("act_crypto", "observe")
        workflow.add_edge("act_rag", "observe")
        workflow.add_edge("act_stock", "observe")
        
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
    
    async def think_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Reasoning step: Analyze current state and decide next action.
        """
        current_step = state.get("current_step", 0) + 1
        
        # Build context from previous outputs
        context = self._build_context(state)
        
        # Create reasoning prompt with better agent awareness
        think_prompt = f"""You are a ReAct supervisor orchestrating multiple agents to answer user queries.

Current Query: {state['query']}

Available Actions:
- CALL_CRYPTO: Get cryptocurrency market data, prices, trends
- CALL_STOCK: Get stock market data, company information, financial data
- CALL_RAG: Search and retrieve information from uploaded documents
- FINISH: Provide final answer (use when you have sufficient information)

Current Step: {current_step}/{self.max_steps}

Information Gathered So Far:
{context}

IMPORTANT INSTRUCTIONS:
1. Check what information you ALREADY HAVE in the context above
2. Do NOT call the same agent twice unless you need different information
3. If you already have an answer from RAG, Crypto, or Stock, move to FINISH
4. Only call another agent if you need ADDITIONAL different information
5. FINISH when you have enough information to answer the user's query

Based on what you know so far, reason about what to do next.
Format your response as:

THOUGHT: [Analyze what you have and what you still need. Be specific about whether you already called RAG, Crypto, or Stock]
ACTION: [CALL_CRYPTO | CALL_STOCK | CALL_RAG | FINISH]
JUSTIFICATION: [Why this action will help. If FINISH, explain why current info is sufficient]"""

        response = await self.supervisor_llm.ainvoke([
            SystemMessage(content="You are a ReAct supervisor. Think step by step and avoid redundant actions."),
            HumanMessage(content=think_prompt)
        ])
        
        # Parse the response
        content = response.content
        thought = ""
        action = "finish"  # default
        justification = ""
        
        if "THOUGHT:" in content:
            thought = content.split("THOUGHT:")[1].split("ACTION:")[0].strip()
        
        if "ACTION:" in content:
            action_text = content.split("ACTION:")[1].split("\n")[0].strip().upper()
            if "CRYPTO" in action_text:
                action = "crypto"
            elif "STOCK" in action_text:
                action = "stock"
            elif "RAG" in action_text:
                action = "rag"
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
        
        # Get agent output
        result = await self.crypto_agent.process(
            state["query"],
            history=self._extract_history(state["messages"])
        )
        
        # Store raw output
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["crypto"] = result
        
        return {"agent_outputs": agent_outputs}
    
    async def act_rag_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute RAG agent and return raw output."""
        if not self.rag_agent:
            return {"agent_outputs": {"rag_error": "RAG agent not available"}}
        
        print("   Calling RAG Agent...")
        
        # Get agent output
        result = await self.rag_agent.process(
            state["query"],
            history=self._extract_history(state["messages"])
        )
        
        # Store raw output
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["rag"] = result
        
        return {"agent_outputs": agent_outputs}
    
    async def act_stock_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute stock agent and return raw output."""
        if not self.stock_agent:
            return {"agent_outputs": {"stock_error": "Stock agent not available"}}
        
        print("   Calling Stock Agent...")
        
        # Get agent output
        result = await self.stock_agent.process(
            state["query"],
            history=self._extract_history(state["messages"])
        )
        
        # Store raw output
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["stock"] = result
        
        return {"agent_outputs": agent_outputs}
    
    async def observe_node(self, state: AgentState) -> Dict[str, Any]:
        """Process and observe the latest agent output - FULL RESPONSE, NO TRUNCATION."""
        agent_outputs = state.get("agent_outputs", {})
        
        # Get the most recent agent output
        latest_output = "No new observations"
        latest_agent = "unknown"
        
        if agent_outputs:
            # Get last added output
            for agent_name, output in list(agent_outputs.items())[-1:]:
                if isinstance(output, dict) and output.get("success"):
                    # Get FULL response without truncation
                    response = output.get('response', 'No response')
                    latest_output = response
                    latest_agent = agent_name
                    break
        
        # Log a summary for display
        summary = latest_output[:250] + "..." if len(latest_output) > 250 else latest_output
        print(f"   Observation from {latest_agent}: {summary}")
        
        # Store FULL observation in messages
        return {
            "messages": [AIMessage(content=f"Observation from {latest_agent}:\n{latest_output}")]
        }
    
    async def finish_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesize all agent outputs and generate final answer.
        This is where the supervisor summarizes everything.
        """
        agent_outputs = state.get("agent_outputs", {})
        reasoning_steps = state.get("reasoning_steps", [])
        
        print("\nSupervisor Synthesizing Final Answer...")
        
        # Build synthesis prompt
        synthesis_prompt = f"""You are synthesizing information to answer this query: {state['query']}

Your reasoning process:
{chr(10).join(reasoning_steps)}

Information gathered from agents:"""
        
        # Add all agent outputs
        for agent_name, output in agent_outputs.items():
            if isinstance(output, dict) and output.get("success"):
                synthesis_prompt += f"\n\n{agent_name.upper()} Agent Response:\n{output.get('response', 'No response')}"
        
        synthesis_prompt += """

Now provide a comprehensive, well-structured answer that:
1. Directly addresses the user's query
2. Integrates insights from all relevant agent outputs
3. Is clear and actionable
4. Highlights any important findings or recommendations

Final Answer:"""
        
        # Generate final synthesized answer
        response = await self.supervisor_llm.ainvoke([
            SystemMessage(content="You are providing the final, synthesized answer."),
            HumanMessage(content=synthesis_prompt)
        ])
        
        final_answer = response.content
        
        return {
            "final_answer": final_answer,
            "messages": [AIMessage(content=final_answer)]
        }
    
    def route_from_thinking(self, state: AgentState) -> str:
        """Route based on thinking decision."""
        # Get the last message which contains the action decision
        last_message = state["messages"][-1] if state["messages"] else None
        
        if last_message and "Action:" in last_message.content:
            # Extract ONLY the action line to avoid matching words in thought
            try:
                action_line = last_message.content.split("Action:")[1].split("\n")[0].strip().upper()
                
                if "CRYPTO" in action_line or "CALL_CRYPTO" in action_line:
                    return "crypto"
                elif "STOCK" in action_line or "CALL_STOCK" in action_line:
                    return "stock"
                elif "RAG" in action_line or "CALL_RAG" in action_line:
                    return "rag"
                elif "FINISH" in action_line:
                    return "finish"
            except (IndexError, AttributeError):
                pass
        
        return "finish"
    
    def should_continue(self, state: AgentState) -> str:
        """Decide whether to continue reasoning or finish."""
        current_step = state.get("current_step", 0)
        
        # Force finish if max steps reached
        if current_step >= self.max_steps:
            print(f"   Max steps ({self.max_steps}) reached, finishing...")
            return "finish"
        
        return "continue"
    
    def _build_context(self, state: AgentState) -> str:
        """Build context string from current state with FULL responses."""
        agent_outputs = state.get("agent_outputs", {})
        
        if not agent_outputs:
            return "No information gathered yet."
        
        context_parts = []
        for agent_name, output in agent_outputs.items():
            if isinstance(output, dict) and output.get("success"):
                response = output.get("response", "No response")
                # Include MORE context (up to 1000 chars) so supervisor knows what it has
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
        return history[-10:]  # Keep last 10 turns
    
    async def process(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process query through ReAct supervisor pattern.
        
        The supervisor will:
        1. Think about what information is needed
        2. Act by calling appropriate agents
        3. Observe the results
        4. Repeat until sufficient information is gathered
        5. Synthesize and return final answer
        """
        # Prepare initial state
        initial_state: AgentState = {
            "messages": [],
            "query": query,
            "agent_outputs": {},
            "reasoning_steps": [],
            "final_answer": None,
            "current_step": 0,
            "max_steps": self.max_steps
        }
        
        # Add history if provided
        if history:
            for turn in history[-3:]:  # Keep last 3 turns for context
                if "user" in turn:
                    initial_state["messages"].append(HumanMessage(content=turn["user"]))
                if "assistant" in turn:
                    initial_state["messages"].append(AIMessage(content=turn["assistant"]))
        
        # Run the ReAct graph
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