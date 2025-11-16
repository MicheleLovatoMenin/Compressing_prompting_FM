import os
import json
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
# 1. MODIFICATION: Define the output folder
OUTPUT_DIR = "datasets"
OUTPUT_FILENAME = "gsm8k_test_set.json"
NUM_SAMPLES = 50 # Number of samples to extract

# --- MAIN FUNCTION ---

def extract_and_save_dataset(num_samples: int = NUM_SAMPLES, output_dir: str = OUTPUT_DIR, output_file: str = OUTPUT_FILENAME):
    """
    Loads the GSM8K dataset (test split) and saves the first N samples as a JSON file
    in the specified folder.
    """
    print(f"Loading dataset {DATASET_NAME} ({DATASET_SPLIT} split)...")
    try:
        # Load the specified dataset
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    except Exception as e:
        print(f"Error loading the dataset: {e}")
        return

    # Limit to the specified number of samples or the total available
    if num_samples > len(dataset):
        num_samples = len(dataset)
        print(f"Warning: Using all {num_samples} available samples in the split.")
    
    # Convert the first N samples into a list of Python dictionaries
    data_list = []
    print(f"Extracting and converting the first {num_samples} samples...")
    for i in tqdm(range(num_samples)):
        data_list.append(dataset[i])
        
    # 2. MODIFICATION: Create the output folder
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the full file path
    full_output_path = os.path.join(output_dir, output_file)
        
    # --- DATA SAVING ---
    try:
        # Open the file and write the JSON data
        with open(full_output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4)
            
        print(f"\n--- SAVING COMPLETE ---")
        print(f"Data saved to: {os.path.abspath(full_output_path)}")
        print(f"File contains {len(data_list)} GSM8K samples.")
    except Exception as e:
        print(f"Error while saving the file: {e}")


if __name__ == "__main__":
    extract_and_save_dataset()