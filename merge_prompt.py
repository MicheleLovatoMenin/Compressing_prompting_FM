import json
import random

# ==========================================
# 1. CARICAMENTO DATI (Sostituisci questo blocco)
# ==========================================
# Qui devi caricare la tua lista di dizionari. 
# Deve essere una lista di oggetti con chiavi "question" e "answer".
INPUT_FILE = "datasets/gsm8k_test_set_300.json"

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    input_data = json.load(f)

# ==========================================
# 2. CONFIGURAZIONE
# ==========================================
NUM_SHOTS = 5  # Numero di esempi da includere prima della domanda target
OUTPUT_FILE = "datasets/dataset_formatted_5shot.json"
random.seed(42) # Per riproducibilità

# ==========================================
# 3. GENERAZIONE PROMPT (Few-Shot CoT)
# ==========================================
processed_data = []

print(f"Elaborazione di {len(input_data)} elementi...")

for i, target_item in enumerate(input_data):
    
    # Crea una lista di candidati escludendo l'elemento corrente
    # (Non vogliamo che la risposta sia già negli esempi!)
    candidates = input_data[:i] + input_data[i+1:]
    
    # Se il dataset è troppo piccolo, prendiamo quello che c'è
    current_shots = NUM_SHOTS if len(candidates) >= NUM_SHOTS else len(candidates)
    examples = random.sample(candidates, current_shots)
    
    # Costruzione del Prompt
    # Header opzionale
    full_prompt = "Instruction: Answer the following math problems reasoning step by step.\n\n"
    
    # Aggiunta degli esempi (Il contesto "comprimibile")
    for ex in examples:
        full_prompt += f"Question: {ex['question']}\n"
        full_prompt += f"Answer: {ex['answer']}\n"
        full_prompt += "###\n\n" # Separatore tra esempi
        
    # Aggiunta della domanda Target (Quella da risolvere)
    full_prompt += f"Question: {target_item['question']}\n"
    full_prompt += "Answer: Let's think step by step." 
    
    # Salviamo il risultato
    processed_data.append({
        "answer": target_item['answer'],
        "question": full_prompt # <--- Questo è quello da dare a LLMLingua
    })

# ==========================================
# 4. SALVATAGGIO OUTPUT
# ==========================================
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)

print(f"Fatto! Salvati {len(processed_data)} prompt formattati in '{OUTPUT_FILE}'")
print(f"Esempio di lunghezza prompt generato: {len(processed_data[0]['question'].split())} parole (circa).")