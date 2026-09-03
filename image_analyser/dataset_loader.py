import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Define the exact classes based on the assignment requirements
ROOM_CLASSES = ['bathroom', 'bedroom', 'exterior', 'kitchen', 'living room', 'other']
# Create a dictionary to map strings to integers (e.g., 'bathroom' -> 0)
CLASS_TO_IDX = {room_name: idx for idx, room_name in enumerate(ROOM_CLASSES)}

class PropertyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the labels_scored.csv file.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Load Image
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        # Convert to RGB to ensure no grayscale or RGBA images crash the model
        image = Image.open(img_name).convert('RGB') 

        # 2. Extract Room Type (Classification - expects Integer)
        room_str = str(self.data_frame.iloc[idx, 1]).strip().lower()
        # Fallback to 'other' if a label is somehow misspelled
        room_idx = CLASS_TO_IDX.get(room_str, CLASS_TO_IDX['other']) 
        
        # 3. Extract Condition Score (Regression - expects Float)
        condition_score = float(self.data_frame.iloc[idx, 2])

        # 4. Apply Transformations
        if self.transform:
            image = self.transform(image)

        # Return the image tensor and a tuple of the two labels
        return image, torch.tensor(room_idx, dtype=torch.long), torch.tensor(condition_score, dtype=torch.float32)

def get_data_loaders(batch_size=32):
    """
    Sets up the transformations and returns training and testing DataLoaders.
    """
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TARGET_BASE_DIR = os.path.join(SCRIPT_DIR, "dataset")
    IMG_DIR = os.path.join(TARGET_BASE_DIR, "images")
    CSV_PATH = os.path.join(TARGET_BASE_DIR, "labels_scored.csv")

    # --- Data Augmentation Strategy ---
    # We use data augmentation to artificially expand our small dataset and prevent overfitting.
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224), # Randomly crop 224x224 to simulate different camera angles
        transforms.RandomHorizontalFlip(p=0.5), # Mirror the image 50% of the time
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Simulate different lighting conditions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
    ])

    # For testing, we only resize, crop to center, and normalize. No random augmentations!
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the full dataset
    full_dataset = PropertyDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=train_transform)

    # Split the dataset: 80% for training, 20% for testing
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Important: Overwrite the transform for the test set so it doesn't get random augmentations
    test_dataset.dataset.transform = test_transform 

    # Create the DataLoaders (these batch the images and feed them to the GPU)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, CLASS_TO_IDX

if __name__ == "__main__":
    # Test the loader to make sure it works
    print("Testing Dataset Loader...")
    train_loader, test_loader, classes = get_data_loaders(batch_size=4)
    
    # Grab one batch of images and labels
    images, room_labels, condition_labels = next(iter(train_loader))
    
    print(f"Loaded {len(train_loader.dataset)} training images and {len(test_loader.dataset)} testing images.")
    print(f"Classes mapping: {classes}")
    print(f"Batch Image Tensor Shape: {images.shape}")
    print(f"Batch Room Labels: {room_labels}")
    print(f"Batch Condition Scores: {condition_labels}")
    print("Success! Data pipeline is ready.")