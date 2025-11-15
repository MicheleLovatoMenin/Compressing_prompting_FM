import os
import json
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURAZIONE ---
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
# 1. MODIFICA: Definiamo la cartella di output
OUTPUT_DIR = "datasets"
OUTPUT_FILENAME = "gsm8k_test_set.json"
NUM_SAMPLES = 50 # Numero di campioni da estrarre

# --- FUNZIONE PRINCIPALE ---

def extract_and_save_dataset(num_samples: int = NUM_SAMPLES, output_dir: str = OUTPUT_DIR, output_file: str = OUTPUT_FILENAME):
    """
    Carica il dataset GSM8K (split test) e salva i primi N campioni come file JSON
    nella cartella specificata.
    """
    print(f"Caricamento dataset {DATASET_NAME} ({DATASET_SPLIT} split)...")
    try:
        # Carica il dataset specificato
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")
        return

    # Limita al numero di campioni specificato o al totale disponibile
    if num_samples > len(dataset):
        num_samples = len(dataset)
        print(f"Attenzione: Usando tutti i {num_samples} esempi disponibili nello split.")
    
    # Converte i primi N campioni in una lista di dizionari Python
    data_list = []
    print(f"Estrazione e conversione dei primi {num_samples} campioni...")
    for i in tqdm(range(num_samples)):
        data_list.append(dataset[i])
        
    # 2. MODIFICA: Creazione della cartella di output
    os.makedirs(output_dir, exist_ok=True)
    
    # Crea il percorso completo del file
    full_output_path = os.path.join(output_dir, output_file)
        
    # --- SALVATAGGIO DEI DATI ---
    try:
        with open(full_output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4)
            
        print(f"\n--- SALVATAGGIO COMPLETATO ---")
        print(f"Dati salvati in: {os.path.abspath(full_output_path)}")
        print(f"File contiene {len(data_list)} campioni di GSM8K.")
    except Exception as e:
        print(f"Errore durante il salvataggio del file: {e}")


if __name__ == "__main__":
    extract_and_save_dataset()