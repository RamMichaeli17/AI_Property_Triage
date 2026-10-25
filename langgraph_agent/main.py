from fastapi import FastAPI
from pydantic import BaseModel
from agent import agent_graph

app = FastAPI(
    title="LangGraph Agent Service",
    description="Stateful multi-step reasoning agent for property triage.",
    version="1.0.0",
)


class AgentQueryInput(BaseModel):
    query: str


@app.post("/agent/run")
async def run_agent(payload: AgentQueryInput):
    initial_state = {
        "query": payload.query,
        "messages": [],
        "selected_tools": [],
        "tool_outputs": {},
        "reasoning_steps": [],
        "final_answer": None,
    }

    final_state = agent_graph.invoke(initial_state)

    return {
        "answer": final_state.get("final_answer", "Error generating answer"),
        "tools_used": final_state.get("selected_tools", []),
        "reasoning_steps": final_state.get("reasoning_steps", []),
    }