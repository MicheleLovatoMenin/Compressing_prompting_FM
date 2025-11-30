import json
import re
import time
import torch
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ==========================================
# CONFIGURAZIONE
# ==========================================
INPUT_FILE = "datasets/gsm8k_compressed.json" 
OUTPUT_FILE = "results_evaluation_qwen.json"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Usa True se hai poca VRAM (sotto i 16GB) per caricare il modello a 4-bit
USE_4BIT = False

# ==========================================
# UTILS (Estrazione e Check - Invariati)
# ==========================================
def extract_answer_gsm8k(text):
    if not text: return "N/A"
    parts = text.split("####")
    candidate = parts[-1] if len(parts) > 1 else text
    candidate_clean = candidate.replace(',', '')
    nums = re.findall(r'-?\d+\.?\d*', candidate_clean)
    return nums[-1] if nums else "N/A"

def check_correctness(pred, gold):
    try:
        pred_float = float(str(pred).strip())
        gold_float = float(str(gold).replace(',', '').strip())
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
        print("Dataset non trovato.")
        return
    
    # data = data[:5] # Decommenta per test rapido

    print(f"--- Loading Model: {MODEL_ID} ---")
    print(f"--- Quantization 4-bit: {USE_4BIT} ---")

    # Configurazione Quantizzazione (per risparmiare memoria)
    bnb_config = None
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Caricamento Modello VL (Vision-Language)
    # Nota: Anche se usiamo solo testo, usiamo la classe specifica per Qwen-VL
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Caricamento Processor (gestisce tokenizzazione e immagini)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    methods_map = [
        #('Original', 'question_original'),
        #('RuleBased', 'question_rulebased'),
        ('LLMLingua2', 'question_llmlingua2')
    ]

    final_results = []
    stats = []

    print("\n--- Starting Local Inference ---")
    
    for i, entry in enumerate(data):
        print(f"\nProcessing Question {i+1}/{len(data)}")
        gold_val = extract_answer_gsm8k(entry.get('answer', ''))
        
        result_entry = {'id': i, 'gold': gold_val, 'evaluations': {}}

        for method_name, json_key in methods_map:
            prompt_text = entry.get(json_key, "")
            if not prompt_text: continue

            # Preparazione Messaggio (Formato Chat Qwen)
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful math assistant. Solve step by step. End with '####' and the number."
                },
                {
                    "role": "user", 
                    "content": prompt_text
                }
            ]

            # Preparazione Input
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Nota: 'images=None' perché è una task solo testo
            inputs = processor(
                text=[text_input],
                images=None, 
                videos=None,
                padding=True,
                return_tensors="pt"
            )
            
            # Spostiamo gli input sulla GPU
            inputs = inputs.to(model.device)

            # Generazione
            start_t = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    temperature=0.01, # Quasi deterministico (0.0 a volte dà errori su hf)
                    do_sample=False
                )
            end_t = time.time()

            # Decodifica (rimuoviamo i token di input dall'output)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Valutazione
            input_tokens_count = inputs.input_ids.shape[1] # Conteggio esatto token input
            pred_val = extract_answer_gsm8k(response_text)
            is_correct = check_correctness(pred_val, gold_val)
            latency = end_t - start_t

            # Log e Salvataggio
            print(f"  [{method_name:<11}] Tok: {input_tokens_count:<4} | Lat: {latency:.2f}s | OK: {str(is_correct):<5} | Pred: {pred_val}")
            
            result_entry['evaluations'][method_name] = {
                'response': response_text,
                'prediction': pred_val,
                'correct': is_correct,
                'tokens': int(input_tokens_count),
                'latency': latency
            }
            stats.append({'Method': method_name, 'Correct': 1 if is_correct else 0, 'Tokens': input_tokens_count, 'Latency': latency})

        final_results.append(result_entry)
        
        # Salvataggio incrementale
        if i % 5 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=4)

    # Salvataggio finale
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)

    # Report
    if stats:
        df = pd.DataFrame(stats)
        summary = df.groupby('Method').agg({'Correct': 'mean', 'Tokens': 'mean', 'Latency': 'mean'}).reset_index()
        summary['Accuracy'] = (summary['Correct'] * 100).map('{:.2f}%'.format)
        print("\n=== FINAL LOCAL RESULTS ===")
        print(summary)

if __name__ == "__main__":
    main()