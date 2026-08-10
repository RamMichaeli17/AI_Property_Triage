from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

print("Loading LLM model to GPU...")
llm = Llama(
    model_path="./Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    verbose=False
)
print("LLM loaded successfully to GPU!")

class PromptReq(BaseModel):
    prompt: str

@app.post("/generate")
def generate(req: PromptReq):
    res = llm(req.prompt, max_tokens=200, temperature=0.3)
    return {"text": res["choices"][0]["text"]}