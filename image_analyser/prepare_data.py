import os
import shutil
import random
import csv

# --- Configuration ---
# UPDATE THIS PATH to where your downloaded dataset folders are located
SOURCE_DATASET_PATH = r"E:\archive\REI-Dataset\REI-Dataset" 

# Get the exact directory where this script is located (image_analyser)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Anchor the target directory to the script's location
TARGET_BASE_DIR = os.path.join(SCRIPT_DIR, "dataset")
TARGET_IMG_DIR = os.path.join(TARGET_BASE_DIR, "images")
CSV_PATH = os.path.join(TARGET_BASE_DIR, "labels.csv")

# Define how many images to pull from each source folder and their new target label
SAMPLING_CONFIG = {
    "kitchen": {"label": "kitchen", "count": 40},
    "bathroom": {"label": "bathroom", "count": 40},
    "bedroom": {"label": "bedroom", "count": 40},
    "livingRoom": {"label": "living room", "count": 40},
    "frontyard": {"label": "exterior", "count": 20}, # Combined into exterior
    "backyard": {"label": "exterior", "count": 20}   # Combined into exterior
}

def setup_directories():
    """Creates the target directories if they don't exist."""
    if not os.path.exists(TARGET_IMG_DIR):
        os.makedirs(TARGET_IMG_DIR)
        print(f"Created target directory: {TARGET_IMG_DIR}")

def prepare_dataset():
    setup_directories()
    
    csv_data = []
    
    for source_folder, config in SAMPLING_CONFIG.items():
        folder_path = os.path.join(SOURCE_DATASET_PATH, source_folder)
        target_label = config["label"]
        sample_size = config["count"]
        
        if not os.path.exists(folder_path):
            print(f"WARNING: Source folder not found -> {folder_path}")
            continue
            
        # Get all valid image files in the directory
        all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(all_files) < sample_size:
            print(f"WARNING: Not enough images in {source_folder}. Requested {sample_size}, found {len(all_files)}.")
            sample_size = len(all_files)
            
        # Randomly sample the files
        sampled_files = random.sample(all_files, sample_size)
        
        # Copy files and record their metadata
        for idx, filename in enumerate(sampled_files):
            # Prefix the filename with the folder name to guarantee unique file names
            new_filename = f"{source_folder}_{idx+1:03d}.jpg"
            
            source_file_path = os.path.join(folder_path, filename)
            target_file_path = os.path.join(TARGET_IMG_DIR, new_filename)
            
            # Copy the image to the new flat folder
            shutil.copy2(source_file_path, target_file_path)
            
            # Append to our CSV data list (leaving condition_score blank for manual entry)
            csv_data.append({
                "filename": new_filename,
                "room_type": target_label,
                "condition_score": "" 
            })
            
        print(f"Successfully processed {sample_size} images from '{source_folder}' -> mapped to '{target_label}'")

    # Write the data to labels.csv
    with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['filename', 'room_type', 'condition_score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
        
    print(f"\nDone! Extracted {len(csv_data)} images.")
    print(f"Images saved to: {TARGET_IMG_DIR}")
    print(f"CSV saved to: {CSV_PATH}")

if __name__ == "__main__":
    prepare_dataset()