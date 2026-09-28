from fastapi import FastAPI
from pydantic import BaseModel
from agent import agent_graph

# Initialize the application
app = FastAPI(title="LangGraph Agent Service")

# Define the expected incoming JSON schema based on the assignment guidelines
class AgentQueryInput(BaseModel):
    query: str

# Create the POST endpoint
@app.post("/agent/run")
async def run_agent(payload: AgentQueryInput):
    # Initialize the AgentState with the user's query
    initial_state = {
        "query": payload.query,
        "messages": [],
        "selected_tools": [],
        "tool_outputs": {},
        "reasoning_steps": [],
        "final_answer": None
    }
    
    # Run the LangGraph state machine
    final_state = agent_graph.invoke(initial_state)
    
    # Return the exact structure required by the project guidelines
    return {
        "answer": final_state.get("final_answer", "Error generating answer"),
        "tools_used": final_state.get("selected_tools", []),
        "reasoning_steps": final_state.get("reasoning_steps", [])
    }