import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions

app = FastAPI()

# Define the incoming data structure based on the API contract
class QueryRequest(BaseModel):
    description: str

# Global variables for the database and collection
chroma_client = None
collection = None

@app.on_event("startup")
def startup_event():
    global chroma_client, collection
    
    # Initialize ChromaDB with persistent storage on disk
    db_path = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Use the default lightweight embedding model from Hugging Face
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    
    # Create or retrieve the collection in the database
    collection = chroma_client.get_or_create_collection(
        name="properties_collection",
        embedding_function=embedding_fn
    )
    
    # Load data from the JSON file and index it if the collection is empty
    if collection.count() == 0:
        json_file_path = "listings.json"
        if os.path.exists(json_file_path):
            with open(json_file_path, "r", encoding="utf-8") as f:
                listings = json.load(f)
                
            documents = []
            metadatas = []
            ids = []
            
            for item in listings:
                # Use the description as the string for vector search
                documents.append(item["description"])
                # Save the other fields as metadata so we can retrieve them later
                metadatas.append({
                    "city": item["city"],
                    "neighborhood": item["neighborhood"],
                    "street": item["street"],
                    "rooms": item["rooms"],
                    "price": item["price"]
                })
                ids.append(item["id"])
                
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Loaded {len(ids)} listings into ChromaDB successfully!")

@app.post("/query/")
def query_rag(payload: QueryRequest):
    global collection
    if not collection:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    # Perform a vector search in ChromaDB based on the requested description
    results = collection.query(
        query_texts=[payload.description],
        n_results=2
    )
    
    # Extract the results and build the output according to the API contract
    similar_listings = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        for doc, meta in zip(docs, metas):
            similar_listings.append(f"{meta['city']}, {meta['neighborhood']} - {doc} (Price: {meta['price']} NIS)")

    # Generate a simple insight based on the retrieved results
    insight = f"Found {len(similar_listings)} similar properties matching your request using vector search."

    return {
        "similar_listings": similar_listings,
        "insight": insight
    }