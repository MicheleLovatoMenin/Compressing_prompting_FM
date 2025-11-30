import json
import random

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
INPUT_FILE = "datasets/gsm8k_test_set_300.json"
OUTPUT_FILE = "datasets/dataset_gsm8k_formatted_8shot.json"

# Numero di esempi (Shots) da inserire nel contesto.
# Per simulare un prompt "Full-Shot" pesante (come nel paper), 
# idealmente dovresti metterne tra 4 e 8, dipendente dalla lunghezza media.
NUM_SHOTS = 8 

random.seed(42) # Per riproducibilità

# ==========================================
# 2. CARICAMENTO DATI
# ==========================================
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
except FileNotFoundError:
    print(f"Errore: File {INPUT_FILE} non trovato. Creo dati dummy per test.")
    input_data = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)]

# ==========================================
# 3. GENERAZIONE PROMPT
# ==========================================
processed_data = []

print(f"Elaborazione di {len(input_data)} elementi con {NUM_SHOTS}-shot CoT...")

for i, target_item in enumerate(input_data):
    
    # --- A. Selezione Esempi (Context) ---
    # Escludiamo l'elemento corrente per evitare data leakage
    candidates = input_data[:i] + input_data[i+1:]
    
    # Se non ci sono abbastanza candidati, ne prendiamo il massimo possibile
    k = min(NUM_SHOTS, len(candidates))
    examples = random.sample(candidates, k)
    
    # --- B. Costruzione della parte CONTESTO (da comprimere) ---
    # Questa è la parte "grassa" che LLMLingua dovrà aggredire.
    # Include l'istruzione generale e gli esempi risolti.
    context_str = "Instruction: Answer the following math problems reasoning step by step.\n\n"
    
    for ex in examples:
        context_str += f"Question: {ex['question']}\n"
        context_str += f"Answer: {ex['answer']}\n"
        context_str += "###\n\n" # Separatore chiaro
        
    # --- C. Costruzione della parte TARGET (da preservare) ---
    # Questa è la domanda attuale. È fondamentale che il modello veda i numeri
    # di QUESTA domanda, quindi idealmente questa parte non va compressa (o pochissimo).
    # Aggiungiamo il trigger CoT "Let's think step by step".
    target_str = f"Question: {target_item['question']}\n"
    target_str += "Answer: Let's think step by step."

    # --- D. Unione (Full Prompt) ---
    full_prompt = context_str + target_str

    # --- E. Salvataggio ---
    entry = {
        # I campi richiesti tassativamente da te:
        "question": full_prompt,      # Il prompt intero (Context + Target)
        "answer": target_item['answer'], # La risposta corretta (Ground Truth)
        
        # Campi EXTRA (Utili per LLMLingua per separare la compressione):
        "context_only": context_str,
        "target_only": target_str
    }
    processed_data.append(entry)

# ==========================================
# 4. SALVATAGGIO OUTPUT
# ==========================================
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)

print(f"Salvato in: {OUTPUT_FILE}")
print("-" * 30)
print(f"Esempio struttura finale (primo elemento):")
print(f"LUNGHEZZA TOTALE: ~{len(processed_data[0]['question'].split())} parole")
print(f"KEYS DISPONIBILI: {list(processed_data[0].keys())}")