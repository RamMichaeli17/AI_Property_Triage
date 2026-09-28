from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    query: str                       # User query
    messages: List[Dict[str, str]]   # Chat history
    selected_tools: List[str]        # Tools selected by Planner
    tool_outputs: Dict[str, Any]     # Raw JSON tool outputs
    reasoning_steps: List[str]       # Execution log
    final_answer: Optional[str]      # Final answer