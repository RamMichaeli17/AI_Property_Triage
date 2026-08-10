from fastapi import FastAPI
from pydantic import BaseModel

# Initialize the application
app = FastAPI(title="Mock Image Analyser Service")

# Define the expected incoming JSON schema
class AnalyseInput(BaseModel):
    image_url: str

# Create the POST endpoint
@app.post("/analyse")
async def analyse_image(payload: AnalyseInput):
    # In production, this would download the image from the URL 
    # and pass it through a computer vision model.
    return {
        "room_type": "kitchen",
        "condition_score": 4,
        "confidence": 0.91
    }