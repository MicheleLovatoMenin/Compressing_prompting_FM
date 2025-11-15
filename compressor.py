import json
import os
from tqdm import tqdm
import re 

# --- CONFIGURAZIONE ---
INPUT_FILE = os.path.join("datasets", "gsm8k_test_set.json")
OUTPUT_FILE = os.path.join("datasets", "experiment_data_compressed.json")

# --- REGOLE DI COMPRESSIONE RULE-BASED ---

def apply_minimalist_rules(text: str) -> str:
    """
    Applica le regole di compressione basate su principi linguistici.
    Queste regole sono applicate solo al prompt_compressed.
    """
    
    # 1. Rimozione di frasi/parole di cortesia o introduzione superflue
    text = re.sub(r'(?i)\b(please|kindly|could you|i would like to know)\b', '', text)
    text = re.sub(r'(?i)\b(In order to|In addition,)\b', '', text)
    
    # Rimozione degli Articoli (Determinativi e Indeterminativi)
    text = re.sub(r'\b(the|a|an)\b', '', text, flags=re.IGNORECASE)

    # 2. Semplificazione di connettivi e avverbi ridondanti
    text = re.sub(r'(?i)\b(and then|subsequently|furthermore|moreover)\b', ', ', text)
    
    # 3. Semplificazione di frasi matematiche verbose
    text = text.replace(" is equal to ", " = ").replace(" the total of ", " sum of ")
    
    # 4. Rimozione di Aggettivi/Avverbi di Intensità (e ridondanti)
    text = re.sub(r'(?i)\b(very|mini|estremamente|actually|basically|just|detailed)\b', '', text)

    # 5. Pulizia e normalizzazione
    text = re.sub(r'[;:"\'()]', '', text) 
    
    # Rimuove gli spazi multipli
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- FUNZIONE PRINCIPALE DI COMPRESSIONE ---

def compress_prompts_and_save():
    if not os.path.exists(INPUT_FILE):
        print(f"Errore: File di input non trovato: {INPUT_FILE}")
        print("Assicurati di aver eseguito 'datasets.py' prima.")
        return

    print(f"Caricamento dati da {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data_for_experiment = []

    print("Applicazione delle regole di compressione e generazione dei prompt...")
    for i, sample in enumerate(tqdm(data)):
        # Il testo del problema originale
        question = sample['question']
        
        # 1. PROMPT ORIGINALE (Baseline)
        # Deve essere identico alla domanda originale (question)
        prompt_original = question 
        
        # 2. PROMPT COMPRESSO (Target)
        # La domanda con applicate le regole di compressione
        prompt_compressed = apply_minimalist_rules(question)
        
        data_for_experiment.append({
            'id': i,
            'ground_truth': sample['answer'], 
            # In questo schema, question_text e prompt_original sono equivalenti
            'prompt_original': prompt_original, 
            'prompt_compressed': prompt_compressed, 
            'metrics': {} 
        })

    # --- SALVATAGGIO DEI DATI ---
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_for_experiment, f, indent=4)
        
    print(f"\n--- COMPRESSIONE COMPLETATA (Domanda Pura vs. Domanda Compressa) ---")
    print(f"File sperimentale salvato in: {os.path.abspath(OUTPUT_FILE)}")
    print(f"Contiene {len(data_for_experiment)} coppie Originale/Compressa pronte per l'inferenza.")

if __name__ == "__main__":
    compress_prompts_and_save()