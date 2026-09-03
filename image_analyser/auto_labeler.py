import os
import csv
import base64
import requests
import time

# --- Configuration ---
# Your local Ollama server endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_BASE_DIR = os.path.join(SCRIPT_DIR, "dataset")
TARGET_IMG_DIR = os.path.join(TARGET_BASE_DIR, "images")
INPUT_CSV = os.path.join(TARGET_BASE_DIR, "labels.csv")
OUTPUT_CSV = os.path.join(TARGET_BASE_DIR, "labels_scored.csv")

def encode_image(image_path):
    """Converts an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_condition_score(image_path):
    """Sends the image to local LLaVA via Ollama and requests a 1-5 score."""
    base64_image = encode_image(image_path)
    
    prompt = (
        "You are an expert real estate appraiser. Rate the physical condition "
        "and maintenance of this property on a strict integer scale of 1 to 5.\n"
        "1 = Dilapidated, ruined.\n"
        "2 = Below average, dated, dirty.\n"
        "3 = Average, standard, livable.\n"
        "4 = Good, clean, modern.\n"
        "5 = Excellent, luxury, pristine.\n\n"
        "Output ONLY a single digit number from 1 to 5. Say absolutely nothing else."
    )

    payload = {
        "model": "llava",
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 5 # Force it to keep the answer extremely short
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if response.status_code == 200:
            # Extract just the number from the response
            score = response.json().get("response", "").strip()
            # Clean up any stray punctuation LLaVA might add (like "4.")
            score = ''.join(filter(str.isdigit, score)) 
            return score
        else:
            print(f"Error from Ollama: {response.text}")
            return ""
    except Exception as e:
        print(f"Connection Error on {image_path}: {e}")
        return ""

def process_dataset():
    rows = []
    with open(INPUT_CSV, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            rows.append(row)
            
    print(f"Starting automated labeling for {len(rows)} images using local LLaVA...")

    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['filename', 'room_type', 'condition_score']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows):
            filename = row['filename']
            image_path = os.path.join(TARGET_IMG_DIR, filename)
            
            if not os.path.exists(image_path):
                print(f"[{idx+1}/{len(rows)}] Image missing: {filename} - Skipping")
                writer.writerow(row)
                continue

            print(f"[{idx+1}/{len(rows)}] Analyzing {filename}...")
            
            if not row['condition_score']:
                score = get_condition_score(image_path)
                
                if score in ['1', '2', '3', '4', '5']:
                    row['condition_score'] = score
                    print(f"    --> Score: {score}")
                else:
                    print(f"    --> Invalid score received: '{score}'")

            writer.writerow(row)
            outfile.flush() 

    print(f"\nFinished! Scored dataset saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    # Required library: pip install requests
    process_dataset()