import json
import re
import time
import os
import pandas as pd
from groq import Groq, RateLimitError

# ==========================================
# CONFIGURAZIONE
# ==========================================
# Assicurati che questo sia il file generato dallo script di compressione
INPUT_FILE = "datasets/gsm8k_compressed.json" 
OUTPUT_FILE = "results_evaluation_api.json"

# Chiave API (Meglio usare le variabili d'ambiente se possibile)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

# Modello
MODEL_ID = "llama-3.1-8b-instant" 

# ==========================================
# UTILS
# ==========================================
def extract_answer_gsm8k(text):
    """
    Estrae il numero finale dopo ####.
    Gestisce casi come '#### $1,230.50' o '#### 5.'
    """
    if not text: return "N/A"
    
    # 1. Cerca il marker standard di GSM8K
    parts = text.split("####")
    
    # Se trova il marker, prende l'ultima parte, altrimenti analizza tutto il testo
    # (fallback nel caso il modello dimentichi '####')
    candidate = parts[-1] if len(parts) > 1 else text
    
    # 2. Pulisci: Rimuovi tutto ciò che non è numero, punto o meno
    # Sostituiamo le virgole delle migliaia (1,000 -> 1000) PRIMA di cercare i numeri
    candidate_clean = candidate.replace(',', '')
    
    # 3. Trova tutti i numeri (anche decimali e negativi)
    nums = re.findall(r'-?\d+\.?\d*', candidate_clean)
    
    # Restituisce l'ultimo numero trovato (di solito la risposta finale)
    return nums[-1] if nums else "N/A"

def check_correctness(pred, gold):
    try:
        # Normalizzazione brutale per il confronto
        pred_float = float(str(pred).strip())
        gold_float = float(str(gold).replace(',', '').strip())
        # Confronto con tolleranza minima per errori di virgola mobile
        return abs(pred_float - gold_float) < 1e-4
    except:
        return False

# ==========================================
# MAIN
# ==========================================
def main():
    print(f"--- Loading Dataset: {INPUT_FILE} ---")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERRORE: Non trovo {INPUT_FILE}. Hai eseguito lo script di compressione?")
        return

    # Per test veloce, decommenta:
    # data = data[:10]

    print("--- Connecting to Groq API ---")
    if not GROQ_API_KEY:
        print("ERRORE: Manca la GROQ_API_KEY!")
        return
        
    client = Groq(api_key=GROQ_API_KEY)

    # Mappa delle chiavi nel JSON -> Nome metodo per il report
    # Verifica che queste chiavi esistano nel tuo JSON prodotto prima
    methods_map = [
        ('Original', 'question_original'),  # O 'question' se non hai rinominato
        ('RuleBased', 'question_rulebased'),
        ('LLMLingua2', 'question_llmlingua2')
    ]

    final_results = []
    stats = []

    print("\n--- Starting API Inference ---")
    
    for i, entry in enumerate(data):
        print(f"\nProcessing Question {i+1}/{len(data)}")
        
        # Gold Answer dal dataset
        gold_val = extract_answer_gsm8k(entry.get('answer', ''))
        
        result_entry = {
            'id': i, 
            'gold': gold_val, 
            'evaluations': {}
        }

        for method_name, json_key in methods_map:
            prompt_text = entry.get(json_key, "")
            
            # Se la compressione ha fallito o la chiave manca, saltiamo
            if not prompt_text: 
                print(f"  [{method_name}] Skipped (Empty prompt)")
                continue

            # Preparazione messaggi
            messages = [
                {
                    "role": "system", 
                    "content": "You are a math expert. Solve the problem step by step. IMPORTANT: At the end, output the final answer after '####'."
                },
                {
                    "role": "user", 
                    "content": prompt_text
                }
            ]

            # RETRY LOGIC PER RATE LIMIT
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    start_t = time.time()
                    
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=MODEL_ID,
                        temperature=0.0,
                        max_tokens=512,
                        stop=None
                    )
                    end_t = time.time()
                    
                    # Estrazione Dati
                    response_text = chat_completion.choices[0].message.content
                    input_tokens = chat_completion.usage.prompt_tokens
                    
                    pred_val = extract_answer_gsm8k(response_text)
                    is_correct = check_correctness(pred_val, gold_val)
                    latency = end_t - start_t

                    # Salvataggio Risultati Parziali
                    result_entry['evaluations'][method_name] = {
                        # Salviamo solo i primi 50 chars del prompt per non intasare il log
                        'prompt_snippet': prompt_text[:50] + "...", 
                        'response': response_text,
                        'prediction': pred_val,
                        'correct': is_correct,
                        'tokens': input_tokens,
                        'latency': latency
                    }
                    
                    stats.append({
                        'Method': method_name, 
                        'Correct': 1 if is_correct else 0, 
                        'Tokens': input_tokens, 
                        'Latency': latency
                    })
                    
                    print(f"  [{method_name:<11}] Tok: {input_tokens:<4} | Lat: {latency:.2f}s | OK: {str(is_correct):<5} | Pred: {pred_val}")
                    
                    # Successo, usciamo dal loop dei retry
                    break 

                except RateLimitError:
                    wait_time = (attempt + 1) * 5 # Backoff esponenziale: 5s, 10s, 15s
                    print(f"  [{method_name}] Rate Limit hit! Waiting {wait_time}s...")
                    time.sleep(wait_time)
                except Exception as e:
                    print(f"  [{method_name}] Generic Error: {e}")
                    break # Errori non di rete (es. bad request) non si ritentano

        final_results.append(result_entry)

        # Salvataggio incrementale (utile se crasha a metà)
        if i % 5 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=4)

    # Salvataggio finale
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)

    # REPORT FINALE PANDAS
    if stats:
        df = pd.DataFrame(stats)
        summary = df.groupby('Method').agg({
            'Correct': 'mean', 
            'Tokens': 'mean', 
            'Latency': 'mean'
        }).reset_index()
        
        # Formattazione carina
        summary['Accuracy'] = (summary['Correct'] * 100).map('{:.2f}%'.format)
        summary['Tokens'] = summary['Tokens'].map('{:.1f}'.format)
        summary['Latency'] = summary['Latency'].map('{:.2f}s'.format)
        
        print("\n" + "="*40)
        print("          FINAL RESULTS          ")
        print("="*40)
        print(summary[['Method', 'Accuracy', 'Tokens', 'Latency']].to_string(index=False))
        print("="*40)
    else:
        print("Nessuna statistica raccolta. Qualcosa è andato storto.")

if __name__ == "__main__":
    main()