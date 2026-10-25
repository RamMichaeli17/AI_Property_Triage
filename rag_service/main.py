import json
import os
import requests
from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions

# LangChain Imports
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM

# Dynamic host configuration: defaults to localhost for local execution,
# or overrides via environment variable (e.g. host.docker.internal in Docker)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Initialize the FastAPI application
app = FastAPI(
    title="RAG Property Service",
    description="Vector retrieval and LLM market insight microservice.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    description: str


chroma_client = None
collection = None


# ==========================================
# 1. LangChain Custom LLM Wrapper
# ==========================================
class LocalLlamaMicroservice(LLM):
    """Custom LangChain LLM wrapper connecting to the Ollama inference server."""

    api_url: str = f"{OLLAMA_HOST}/api/generate"
    model_name: str = OLLAMA_MODEL

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=30,
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return f"Ollama Error: {response.text}"
        except Exception as e:
            return f"Could not connect to Ollama service: {e}"

    @property
    def _llm_type(self) -> str:
        return "local_llama_microservice"


# Instantiate the custom LangChain LLM
llm = LocalLlamaMicroservice()


# ==========================================
# 2. Vector Database Initialization
# ==========================================
@app.on_event("startup")
def startup_event():
    global chroma_client, collection

    db_path = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    collection = chroma_client.get_or_create_collection(
        name="properties_collection",
        embedding_function=embedding_fn,
    )

    if collection.count() == 0:
        json_file_path = "listings.json"
        if os.path.exists(json_file_path):
            with open(json_file_path, "r", encoding="utf-8") as f:
                listings = json.load(f)

            documents, metadatas, ids = [], [], []

            for item in listings:
                rich_content = (
                    f"ID: {item['id']} | Property in {item['city']}, "
                    f"Neighborhood: {item['neighborhood']}, Street: {item['street']}. "
                    f"It has {item['rooms']} rooms and costs {item['price']} NIS. "
                    f"Description: {item['description']}"
                )
                documents.append(rich_content)
                metadatas.append({
                    "city": item["city"],
                    "neighborhood": item["neighborhood"],
                    "street": item["street"],
                    "rooms": item["rooms"],
                    "price": item["price"],
                })
                ids.append(item["id"])

            collection.add(documents=documents, metadatas=metadatas, ids=ids)


# ==========================================
# 3. RAG Query Endpoint
# ==========================================
@app.post("/query")
@app.post("/query/")
def query_rag(payload: QueryRequest):
    global collection
    if not collection:
        raise HTTPException(status_code=500, detail="Database not initialized")

    # PHASE 1: Retrieval (ChromaDB)
    results = collection.query(
        query_texts=[payload.description],
        n_results=3,
    )

    similar_listings, context_list = [], []

    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        for doc in docs:
            context_list.append(doc)
            similar_listings.append(doc)

    context_str = "\n".join(context_list)

    # PHASE 2: Prompt Construction (LangChain)
    with open("prompt_template.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
    formatted_prompt = prompt.format(context=context_str, query=payload.description)

    # PHASE 3: LLM Inference via LangChain Wrapper
    insight = llm.invoke(formatted_prompt)

    return {
        "similar_listings": similar_listings,
        "insight": insight,
    }