from fastapi import FastAPI
from pydantic import BaseModel

# Initialize the application
app = FastAPI(title="Mock Guardrails Service")

# Define the expected incoming JSON schema shared by both endpoints
class TextPayload(BaseModel):
    text: str

# Define the shared mock response
mock_response = {
    "pass": True,
    "reason": "none",
    "safe_text": "mocked safe text"
}

# Create the POST endpoint for input validation
@app.post("/check/input")
async def check_input(payload: TextPayload):
    # In production, this would scan the user's prompt for prompt injection or toxicity.
    return mock_response

# Create the POST endpoint for output validation
@app.post("/check/output")
async def check_output(payload: TextPayload):
    # In production, this would scan the LLM's response for hallucinations or unsafe content.
    return mock_response