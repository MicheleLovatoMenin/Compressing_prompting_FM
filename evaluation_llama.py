import json
import re
import time
import os
import pandas as pd
from groq import Groq

# ==========================================
# CONFIGURAZIONE
# ==========================================
INPUT_FILE = "datasets/gsm8k_compressed_0.8.json"
OUTPUT_FILE = "results_evaluation_api.json"

# Inserisci qui la tua chiave API di Groq
# (O meglio, impostala come variabile d'ambiente, ma per ora scrivila qui se è un test locale)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  


# Modello da usare su Groq (Llama 3 8B è perfetto per il confronto)
MODEL_ID = "llama-3.1-8b-instant"

# ==========================================
# UTILS
# ==========================================
def extract_answer_gsm8k(text):
    match = re.search(r'####\s*(-?\d+\.?\d*)', text)
    if match: return match.group(1).strip()
    cleaned = text.split("####")[-1] if "####" in text else text
    nums = re.findall(r'-?\d+\.?\d*', cleaned)
    return nums[-1] if nums else "N/A"

def check_correctness(pred, gold):
    try:
        pred = str(pred).replace(',', '').strip()
        gold = str(gold).replace(',', '').strip()
        return float(pred) == float(gold)
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
        print("Dataset non trovato.")
        return

    print("--- Connecting to Groq API ---")
    client = Groq(api_key=GROQ_API_KEY)

    methods_map = [
        ('Original', 'question_original'),
        ('RuleBased', 'question_rulebased'),
        ('LLMLingua2', 'question_llmlingua2')
    ]

    final_results = []
    stats = []

    print("\n--- Starting API Inference ---")
    for i, entry in enumerate(data):
        print(f"\nProcessing Question {i+1}/{len(data)}")
        gold_val = extract_answer_gsm8k(entry.get('answer', ''))
        
        result_entry = {'id': i, 'gold': gold_val, 'evaluations': {}}

        for method, key in methods_map:
            prompt_text = entry.get(key, "")
            if not prompt_text: continue

            # Chiamata API
            messages = [
                {"role": "system", "content": "You are a math expert. Solve step by step. End answer with '####' and the number."},
                {"role": "user", "content": prompt_text}
            ]

            try:
                start_t = time.time()
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=MODEL_ID,
                    temperature=0.0, # Deterministico
                    max_tokens=512
                )
                end_t = time.time()
                
                response_text = chat_completion.choices[0].message.content
                
                # Calcolo Token (Stimato o fornito dall'API)
                # Groq ci dice quanti token ha usato nel 'usage'
                input_tokens = chat_completion.usage.prompt_tokens
                
                pred_val = extract_answer_gsm8k(response_text)
                is_correct = check_correctness(pred_val, gold_val)
                latency = end_t - start_t

                # Salvataggio
                result_entry['evaluations'][method] = {
                    'prompt': prompt_text,
                    'response': response_text,
                    'prediction': pred_val,
                    'correct': is_correct,
                    'tokens': input_tokens,
                    'latency': latency
                }
                
                stats.append({'Method': method, 'Correct': 1 if is_correct else 0, 'Tokens': input_tokens, 'Latency': latency})
                print(f"  [{method:<10}] Tok: {input_tokens:<3} | Lat: {latency:.2f}s | OK: {is_correct} | Pred: {pred_val}")

            except Exception as e:
                print(f"  [{method}] API Error: {e}")
                time.sleep(2) # Aspetta un attimo se c'è errore

        final_results.append(result_entry)

    # Salvataggio finale
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)

    # Tabella riassuntiva
    if stats:
        df = pd.DataFrame(stats)
        summary = df.groupby('Method').agg({'Correct': 'mean', 'Tokens': 'mean', 'Latency': 'mean'}).reset_index()
        summary['Accuracy'] = (summary['Correct'] * 100).map('{:.2f}%'.format)
        print("\n=== FINAL RESULTS ===")
        print(summary[['Method', 'Accuracy', 'Tokens', 'Latency']])

if __name__ == "__main__":
    main()