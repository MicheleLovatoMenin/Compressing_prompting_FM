import json
import re
import time
import os
import pandas as pd
import google.generativeai as genai
from google.api_core import retry

# ==========================================
# CONFIGURAZIONE
# ==========================================
INPUT_FILE = "datasets/gsm8k_compressed.json"
OUTPUT_FILE = "results_evaluation_gemini.json"

# INCOLLA QUI LA TUA CHIAVE GOOGLE (quella presa da AI Studio)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")   

MODEL_NAME = "models/gemini-2.0-flash"

# ==========================================
# SETUP
# ==========================================
genai.configure(api_key=GOOGLE_API_KEY)

# Configurazione del modello
generation_config = {
    "temperature": 0.0, # Deterministico
    "max_output_tokens": 512,
}
model = genai.GenerativeModel(model_name=MODEL_NAME, generation_config=generation_config)

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

    print(f"--- Starting Inference with {MODEL_NAME} ---")
    
    methods_map = [
        ('Original', 'question_original'),
        ('RuleBased', 'question_rulebased'),
        ('LLMLingua2', 'question_llmlingua2')
    ]

    final_results = []
    stats = []

    for i, entry in enumerate(data[:20]):
        print(f"\nProcessing Question {i+1}/{len(data)}")
        gold_val = extract_answer_gsm8k(entry.get('answer', ''))
        
        result_entry = {'id': i, 'gold': gold_val, 'evaluations': {}}

        for method, key in methods_map:
            prompt_text = entry.get(key, "")
            if not prompt_text: continue

            # Gemini preferisce prompt diretti
            full_prompt = (
                "You are a math expert. Solve the following problem step by step. "
                "End your answer strictly with '####' followed by the number.\n\n"
                f"Problem: {prompt_text}"
            )

            try:
                start_t = time.time()
                # Chiamata API
                response = model.generate_content(full_prompt)
                end_t = time.time()
                
                try:
                    response_text = response.text
                except ValueError:
                    # A volte Gemini blocca la risposta per sicurezza (raro in math)
                    response_text = "BLOCKED_BY_SAFETY"

                # Stimiamo i token (Gemini ha un metodo count_tokens ma rallenta, usiamo stima char)
                # Oppure chiamiamo model.count_tokens(prompt_text) se vuoi precisione
                input_tokens = model.count_tokens(full_prompt).total_tokens
                
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
                
                # Rispetta i rate limits del piano free (fondamentale!)
                time.sleep(7) 

            except Exception as e:
                print(f"  [{method}] API Error: {e}")
                time.sleep(5)

        final_results.append(result_entry)

    # Salvataggio Finale
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)

    # Report
    if stats:
        df = pd.DataFrame(stats)
        summary = df.groupby('Method').agg({'Correct': 'mean', 'Tokens': 'mean', 'Latency': 'mean'}).reset_index()
        summary['Accuracy'] = (summary['Correct'] * 100).map('{:.2f}%'.format)
        print("\n=== FINAL GEMINI RESULTS ===")
        print(summary[['Method', 'Accuracy', 'Tokens', 'Latency']])

if __name__ == "__main__":
    main()