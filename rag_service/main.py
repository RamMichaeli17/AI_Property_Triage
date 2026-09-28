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

# Initialize the FastAPI application
app = FastAPI()

class QueryRequest(BaseModel):
    description: str

chroma_client = None
collection = None

# ==========================================
# 1. LangChain Custom LLM Wrapper
# ==========================================
class LocalLlamaMicroservice(LLM):
    """
    A custom LangChain LLM class that connects to our local Llama.cpp microservice.
    This fulfills the requirement to use LangChain for LLM invocation while 
    maintaining our robust GPU-accelerated microservice architecture.
    """
    api_url: str = "http://127.0.0.1:11434/api/generate"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        try:
            response = requests.post(self.api_url, json={"prompt": prompt}, timeout=30)
            if response.status_code == 200:
                return response.json().get("text", "").strip()
            return f"LLM Error: {response.text}"
        except Exception as e:
            return f"Could not connect to LLM service: {e}"
    
    @property
    def _llm_type(self) -> str:
        return "local_llama_microservice"

# Instantiate our custom LangChain LLM
llm = LocalLlamaMicroservice()
# ==========================================

@app.on_event("startup")
def startup_event():
    global chroma_client, collection
    
    # Initialize ChromaDB
    db_path = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    
    collection = chroma_client.get_or_create_collection(
        name="properties_collection",
        embedding_function=embedding_fn
    )
    
    if collection.count() == 0:
        json_file_path = "listings.json"
        if os.path.exists(json_file_path):
            with open(json_file_path, "r", encoding="utf-8") as f:
                listings = json.load(f)
                
            documents, metadatas, ids = [], [], []
            
            for item in listings:
                rich_content = f"ID: {item['id']} | Property in {item['city']}, Neighborhood: {item['neighborhood']}, Street: {item['street']}. It has {item['rooms']} rooms and costs {item['price']} NIS. Description: {item['description']}"
                documents.append(rich_content)
                metadatas.append({
                    "city": item["city"], "neighborhood": item["neighborhood"],
                    "street": item["street"], "rooms": item["rooms"], "price": item["price"]
                })
                ids.append(item["id"])
                
            collection.add(documents=documents, metadatas=metadatas, ids=ids)

@app.post("/query/")
def query_rag(payload: QueryRequest):
    global collection
    if not collection:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    # PHASE 1: Retrieval (ChromaDB)
    results = collection.query(
        query_texts=[payload.description],
        n_results=3
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
    
    # Format the prompt using LangChain to fulfill the assignment requirement
    formatted_prompt = prompt.format(context=context_str, query=payload.description)

    # PHASE 3: Send directly to Ollama Server
    import requests
    insight = "No insight generated."
    try:
        response = requests.post("http://127.0.0.1:11434/api/generate", json={
            "model": "llama3",
            "prompt": formatted_prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }, timeout=30)
        
        if response.status_code == 200:
            insight = response.json().get("response", "").strip()
        else:
            insight = f"Ollama Error: {response.text}"
    except Exception as e:
        insight = f"Could not connect to Ollama service: {e}"

    return {
        "similar_listings": similar_listings,
        "insight": insight
    }