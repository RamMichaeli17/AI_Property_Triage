import requests
from typing import Dict, Any

RAG_TOOL_DESCRIPTION_V1 = """
Useful for searching real estate property listings, prices, locations, amenities, and general property specifications in the database.
"""

IMAGE_TOOL_DESCRIPTION_V1 = """
Useful for analyzing property images, detecting room types (e.g., kitchen, bathroom), and calculating condition scores (1-5).
"""

RAG_SERVICE_URL = "http://127.0.0.1:8001/query/"
IMAGE_SERVICE_URL = "http://127.0.0.1:8002/analyse"

def call_rag_service(query: str) -> Dict[str, Any]:
    try:
        res = requests.post(RAG_SERVICE_URL, json={"description": query}, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"error": f"Failed to call RAG Service: {str(e)}"}

def call_image_analyser_service(image_url: str) -> Dict[str, Any]:
    try:
        res = requests.post(IMAGE_SERVICE_URL, json={"image_url": image_url}, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"error": f"Failed to call Image Analyser Service: {str(e)}"}

AVAILABLE_TOOLS = {
    "rag_search": call_rag_service,
    "image_analyser": call_image_analyser_service
}