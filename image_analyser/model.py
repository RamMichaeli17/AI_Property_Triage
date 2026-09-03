import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class RealEstateAnalyzer(nn.Module):
    def __init__(self, num_room_classes=6):
        super(RealEstateAnalyzer, self).__init__()
        
        # 1. Load the pre-trained ResNet50 backbone
        # We use ResNet50_Weights.DEFAULT to get the most accurate, up-to-date ImageNet weights
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # 2. Freeze the backbone to prevent the weights from being updated during training
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Extract the input size of the original final layer (2048 for ResNet50)
        num_ftrs = self.backbone.fc.in_features
        
        # 4. Remove the original final layer by replacing it with an Identity function
        self.backbone.fc = nn.Identity()
        
        # 5. Build our dual-head architecture
        # Head A: Classification (6 output nodes for the room types)
        self.room_classifier = nn.Sequential(
            nn.Dropout(p=0.3), # Dropout prevents overfitting on small datasets
            nn.Linear(num_ftrs, num_room_classes)
        )
        
        # Head B: Regression (1 output node for the 1.0 - 5.0 condition score)
        self.condition_scorer = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 1)
        )

    def forward(self, x):
        # Pass the image through the frozen backbone to extract features
        features = self.backbone(x)
        
        # Route the extracted features through both heads simultaneously
        room_logits = self.room_classifier(features)
        condition_score = self.condition_scorer(features)
        
        return room_logits, condition_score

if __name__ == "__main__":
    # Test the model structure to ensure the tensor math aligns
    print("Initializing model and downloading weights (if first run)...")
    model = RealEstateAnalyzer()
    
    # Simulate a batch of 4 images
    dummy_input = torch.randn(4, 3, 224, 224) 
    rooms, conditions = model(dummy_input)
    
    print("\nSuccess! Model built correctly.")
    print(f"Room Logits Shape: {rooms.shape} (Expected: [4, 6])")
    print(f"Condition Scores Shape: {conditions.shape} (Expected: [4, 1])")