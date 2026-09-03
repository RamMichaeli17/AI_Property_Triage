import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import get_data_loaders
from model import RealEstateAnalyzer

def train_model(epochs=15, batch_size=16, learning_rate=0.001):
    # 1. Device configuration (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load data
    train_loader, test_loader, class_to_idx = get_data_loaders(batch_size=batch_size)
    
    # 3. Initialize model
    model = RealEstateAnalyzer(num_room_classes=len(class_to_idx)).to(device)

    # 4. Define Loss Functions
    # Classification head loss (Room Type)
    criterion_room = nn.CrossEntropyLoss()
    # Regression head loss (Condition Score)
    criterion_condition = nn.MSELoss()

    # 5. Define Optimizer (Only optimize trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)

    print(f"Starting training for {epochs} epochs...\n")

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_rooms = 0
        total_samples = 0

        for images, room_targets, condition_targets in train_loader:
            images = images.to(device)
            room_targets = room_targets.to(device)
            # Reshape condition targets to match model output shape [batch_size, 1]
            condition_targets = condition_targets.unsqueeze(1).to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            room_preds, condition_preds = model(images)

            # Compute individual losses
            loss_room = criterion_room(room_preds, room_targets)
            loss_condition = criterion_condition(condition_preds, condition_targets)

            # Total loss combination
            total_loss = loss_room + loss_condition

            # Backward pass & optimize
            total_loss.backward()
            optimizer.step()

            # Metrics tracking
            running_loss += total_loss.item() * images.size(0)
            _, predicted_rooms = torch.max(room_preds, 1)
            correct_rooms += (predicted_rooms == room_targets).sum().item()
            total_samples += images.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = (correct_rooms / total_samples) * 100

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} | Room Acc: {epoch_acc:.2f}%")

    print("\nTraining Complete!")

    # --- Evaluation Loop ---
    evaluate_model(model, test_loader, device, criterion_room, criterion_condition)

    # --- Save Model Weights ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(SCRIPT_DIR, "model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved successfully to: {save_path}")

def evaluate_model(model, test_loader, device, criterion_room, criterion_condition):
    model.eval()
    test_loss = 0.0
    correct_rooms = 0
    total_samples = 0
    total_condition_error = 0.0

    with torch.no_grad():
        for images, room_targets, condition_targets in test_loader:
            images = images.to(device)
            room_targets = room_targets.to(device)
            condition_targets = condition_targets.unsqueeze(1).to(device)

            room_preds, condition_preds = model(images)

            loss_room = criterion_room(room_preds, room_targets)
            loss_condition = criterion_condition(condition_preds, condition_targets)
            total_loss = loss_room + loss_condition

            test_loss += total_loss.item() * images.size(0)
            _, predicted_rooms = torch.max(room_preds, 1)
            correct_rooms += (predicted_rooms == room_targets).sum().item()
            total_samples += images.size(0)

            # Average absolute difference in condition rating
            total_condition_error += torch.sum(torch.abs(condition_preds - condition_targets)).item()

    avg_loss = test_loss / total_samples
    test_acc = (correct_rooms / total_samples) * 100
    avg_mae_condition = total_condition_error / total_samples

    print("\n" + "="*40)
    print("           TEST SET RESULTS           ")
    print("="*40)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Room Classification Accuracy: {test_acc:.2f}%")
    print(f"Condition Score Mean Absolute Error (MAE): {avg_mae_condition:.2f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    train_model()