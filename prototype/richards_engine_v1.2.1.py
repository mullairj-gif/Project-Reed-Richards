import json
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults

# --- 1. STATE DEFINITION ---
class KnowledgeNode(TypedDict):
    id: str
    thought: str
    speculation_level: str
    parent_id: str

class RichardsState(TypedDict):
    problem: str
    domain: str
    knowledge_graph: Dict[str, KnowledgeNode]
    current_focus: List[str]
    final_synthesis: str
    iteration: int

# --- 2. REASONING BRANCH NODE ---
def reasoning_branch_node(state: RichardsState):
    """The Core Thinking Engine: Generates 3 research-backed paths."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    search_tool = TavilySearchResults(k=3)
    tools = [search_tool]
    
    # Create the agent for real-world grounding
    agent = create_tool_calling_agent(llm, tools, 
        "You are the Reed Richards core. Generate 3 distinct paths grounded in 2025 data.")
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    
    prompt = f"Analyze: {state['problem']} in {state['domain']}. Provide 3 distinct solution branches."
    response = agent_executor.invoke({"input": prompt})
    raw_thought = response["output"]
    
    # Store results in the Graph structure
    new_nodes = {}
    focus_ids = []
    for i in range(3):
        node_id = f"iter_{state['iteration']}_path_{i}"
        new_nodes[node_id] = {
            "id": node_id,
            "thought": f"PATH {i+1}: " + raw_thought, # Simplified for prototype
            "speculation_level": "Research-Grounded",
            "parent_id": "root"
        }
        focus_ids.append(node_id)
        
    return {
        "knowledge_graph": {**state["knowledge_graph"], **new_nodes},
        "current_focus": focus_ids,
        "iteration": state["iteration"] + 1
    }

# --- 3. SYNTHESIZER NODE ---
def synthesizer_node(state: RichardsState):
    """The Evaluator: Collapses branches into a ranked plan."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    thoughts = [state['knowledge_graph'][nid]['thought'] for nid in state['current_focus']]
    
    synthesis_prompt = f"Synthesize these paths into a ranked, actionable 2025 plan: {thoughts}"
    response = llm.invoke(synthesis_prompt)
    return {"final_synthesis": response.content}

# --- 4. GRAPH CONSTRUCTION ---
builder = StateGraph(RichardsState)
builder.add_node("reasoner", reasoning_branch_node)
builder.add_node("synthesizer", synthesizer_node)

builder.set_entry_point("reasoner")
builder.add_edge("reasoner", "synthesizer")
builder.add_edge("synthesizer", END)

# Compile the application
richards_engine = builder.compile()
