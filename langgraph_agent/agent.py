import json
import requests
from typing import Literal
from langgraph.graph import StateGraph, END
from state import AgentState
from tools import AVAILABLE_TOOLS, RAG_TOOL_DESCRIPTION_V1, IMAGE_TOOL_DESCRIPTION_V1

# Local Llama 3 API running on Bare Metal / GPU (port 8080)
LLM_API_URL = "http://127.0.0.1:11434/api/generate"

def call_local_llm(prompt: str) -> str:
    """Helper function to send prompts to Ollama."""
    try:
        res = requests.post(LLM_API_URL, json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }, timeout=30)
        
        if res.status_code == 200:
            return res.json().get("response", "").strip()
        return f"Error: LLM returned status {res.status_code}"
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

# ==========================================
# Node 1: Planner Node
# ==========================================
def planner_node(state: AgentState) -> AgentState:
    query = state["query"]
    
    prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a smart planner for a real estate AI agent. 
Analyze the user's query and decide which tools are needed to answer it.
Available tools:
1. "rag_search": {RAG_TOOL_DESCRIPTION_V1}
2. "image_analyser": {IMAGE_TOOL_DESCRIPTION_V1}

Respond ONLY with a valid JSON array of tool names. Example: ["rag_search", "image_analyser"].
If no tools are needed, return an empty array: [].
<|eot_id|><|start_header_id|>user<|end_header_id|>
Query: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    llm_response = call_local_llm(prompt)
    
    # Extract the JSON array from the LLM's response
    try:
        selected_tools = json.loads(llm_response)
        if not isinstance(selected_tools, list):
            selected_tools = []
    except:
        selected_tools = []

    state["selected_tools"] = selected_tools
    state["reasoning_steps"].append(f"Planner decided to use tools: {selected_tools}")
    return state

# ==========================================
# Node 2: Tool Execution Node
# ==========================================
def tool_execution_node(state: AgentState) -> AgentState:
    selected_tools = state["selected_tools"]
    tool_outputs = state.get("tool_outputs", {})
    query = state["query"]
    
    for tool_name in selected_tools:
        if tool_name in AVAILABLE_TOOLS:
            # For this iteration, we pass the full query. We will refine parameter extraction later.
            if tool_name == "rag_search":
                result = AVAILABLE_TOOLS[tool_name](query)
            elif tool_name == "image_analyser":
                # Naive URL extraction for testing purposes
                words = query.split()
                urls = [w for w in words if w.startswith("http")]
                url = urls[0] if urls else "http://example.com/placeholder.jpg"
                result = AVAILABLE_TOOLS[tool_name](url)
                
            tool_outputs[tool_name] = result
            state["reasoning_steps"].append(f"Executed tool: {tool_name}")

    state["tool_outputs"] = tool_outputs
    return state

# ==========================================
# Node 3: Synthesiser Node
# ==========================================
def synthesiser_node(state: AgentState) -> AgentState:
    query = state["query"]
    tool_outputs = state.get("tool_outputs", {})
    
    context_summary = ""
    if "rag_search" in tool_outputs:
        rag_data = tool_outputs["rag_search"]
        if isinstance(rag_data, dict):
            insight = rag_data.get("insight", "")
            listings = rag_data.get("similar_listings", [])[:2] # נגביל ל-2 דירות בלבד
            context_summary += f"RAG Insight: {insight}\nListings: {listings}\n"
        else:
            context_summary += f"RAG Output: {str(rag_data)[:500]}\n"

    if "image_analyser" in tool_outputs:
        context_summary += f"Image Output: {tool_outputs['image_analyser']}\n"
    
    if not context_summary:
        context_summary = "No tools were executed."

    prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful real estate AI assistant. Answer the user query using the context below.
Context:
{context_summary}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Query: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    final_answer = call_local_llm(prompt)
    
    state["final_answer"] = final_answer
    state["reasoning_steps"].append("Synthesiser compiled the final answer.")
    return state

# ==========================================
# Conditional Edge Logic
# ==========================================
def should_run_tools(state: AgentState) -> Literal["execute_tools", "synthesise"]:
    # If the planner selected tools, execute them. Otherwise, skip to synthesis.
    if len(state.get("selected_tools", [])) > 0:
        return "execute_tools"
    return "synthesise"

# ==========================================
# Graph Compilation
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("execute_tools", tool_execution_node)
workflow.add_node("synthesise", synthesiser_node)

workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", should_run_tools)
workflow.add_edge("execute_tools", "synthesise")
workflow.add_edge("synthesise", END)

agent_graph = workflow.compile()