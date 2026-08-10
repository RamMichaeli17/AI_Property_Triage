import json
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.prompts import PromptTemplate

# Initialize the FastAPI application
app = FastAPI()

# Define the expected JSON payload structure for the incoming request
class QueryRequest(BaseModel):
    description: str

# Global variables to maintain database connection state across requests
chroma_client = None
collection = None

# The URL of the local Llama.cpp microservice running on the GPU
LLM_API_URL = "http://127.0.0.1:8080/generate"

@app.on_event("startup")
def startup_event():
    """
    This function runs once when the FastAPI server starts.
    It handles database initialization and data seeding (Cold Start).
    """
    global chroma_client, collection
    
    # Initialize ChromaDB with persistent storage to save vectors to disk
    db_path = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Use HuggingFace's all-MiniLM-L6-v2 model for text embeddings (ChromaDB default)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    
    # Connect to or create the vector collection for our properties
    collection = chroma_client.get_or_create_collection(
        name="properties_collection",
        embedding_function=embedding_fn
    )
    
    # Pre-populate the database if it's currently empty
    if collection.count() == 0:
        json_file_path = "listings.json"
        if os.path.exists(json_file_path):
            with open(json_file_path, "r", encoding="utf-8") as f:
                listings = json.load(f)
                
            documents, metadatas, ids = [], [], []
            
            # Parse the JSON and prepare documents for vectorization
            for item in listings:
                # Create a rich text representation for better semantic search matching
                rich_content = f"ID: {item['id']} | Property in {item['city']}, Neighborhood: {item['neighborhood']}, Street: {item['street']}. It has {item['rooms']} rooms and costs {item['price']} NIS. Description: {item['description']}"
                
                documents.append(rich_content)
                # Store structured metadata for potential future filtering capabilities
                metadatas.append({
                    "city": item["city"], "neighborhood": item["neighborhood"],
                    "street": item["street"], "rooms": item["rooms"], "price": item["price"]
                })
                ids.append(item["id"])
                
            # Add the data to ChromaDB (this triggers the automatic embedding process)
            collection.add(documents=documents, metadatas=metadatas, ids=ids)

@app.post("/query/")
def query_rag(payload: QueryRequest):
    """
    The main RAG endpoint. It retrieves context from ChromaDB,
    builds a LangChain prompt, and queries the local LLM microservice.
    """
    global collection
    if not collection:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    # PHASE 1: Retrieval (Vector Similarity Search)
    # Fetch the top 3 most relevant property listings based on semantic similarity
    results = collection.query(
        query_texts=[payload.description],
        n_results=3
    )
    
    similar_listings, context_list = [], []
    
    # Extract the retrieved documents from the ChromaDB results
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        for doc in docs:
            context_list.append(doc)
            similar_listings.append(doc)
            
    # Combine the retrieved listings into a single context string for the prompt
    context_str = "\n".join(context_list)

    # PHASE 2: Prompt Construction
    # Read the strict system instructions from the external text file
    with open("prompt_template.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    # Use LangChain to dynamically inject the context and user query into the template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
    formatted_prompt = prompt.format(context=context_str, query=payload.description)
    
    # PHASE 3: LLM Generation (Calling the Microservice)
    insight = "No insight generated."
    try:
        # Send the formatted prompt to our local Llama 3 API running on the GPU
        response = requests.post(LLM_API_URL, json={"prompt": formatted_prompt}, timeout=30)
        
        if response.status_code == 200:
            # Extract and clean the generated insight from the LLM
            insight = response.json().get("text", "").strip()
        else:
            insight = f"LLM Error: {response.text}"
    except Exception as e:
        insight = f"Could not connect to LLM service: {e}"

    # Return the final required output schema (similar properties + AI insight)
    return {
        "similar_listings": similar_listings,
        "insight": insight
    }