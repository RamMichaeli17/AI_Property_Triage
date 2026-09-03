import os
import torch
import requests
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torchvision.transforms as transforms
import torch.nn.functional as F
from model import RealEstateAnalyzer

# Initialize FastAPI
app = FastAPI(title="Image Analyser Service")

# Define the Input Schema
class ImageRequest(BaseModel):
    image_url: str

# Configuration & Setup
DEVICE = torch.device("cpu")
ROOM_CLASSES = ['bathroom', 'bedroom', 'exterior', 'kitchen', 'living room', 'other']
CONFIDENCE_THRESHOLD = 0.60  # If confidence is below 60%, return 'uncertain'

# Load the trained model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pth")

model = RealEstateAnalyzer(num_room_classes=len(ROOM_CLASSES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval() # Set to evaluation mode

# Image Preprocessing (Must perfectly match the test_transform from dataset_loader)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/analyse")
async def analyse_image(request: ImageRequest):
    # 1. Download the image
    try:
        response = requests.get(request.image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from URL: {str(e)}")

    # 2. Preprocess the image and add batch dimension
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # 3. Inference (Pass through the neural network)
    with torch.no_grad():
        room_logits, condition_score = model(input_tensor)

    # 4. Calculate Room Confidence (Softmax converts logits to percentages 0.0 - 1.0)
    probabilities = F.softmax(room_logits, dim=1)[0]
    max_prob, predicted_idx = torch.max(probabilities, 0)
    confidence = max_prob.item()

    # 5. Apply Confidence Threshold logic
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_room = "uncertain"
    else:
        predicted_room = ROOM_CLASSES[predicted_idx.item()]

    # 6. Format Condition Score (Clamp between 1 and 5, round to nearest integer)
    raw_score = condition_score.item()
    clamped_score = max(1.0, min(5.0, raw_score))
    final_score = int(round(clamped_score))

    # 7. Return exact JSON Response format required by assignment
    return {
        "room_type": predicted_room,
        "condition_score": final_score,
        "confidence": round(confidence, 2)
    }