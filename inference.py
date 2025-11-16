import json
import time
import os
from datetime import datetime
from groq import Groq

# Configurazione
INPUT_FILE = "datasets/output_compressed_rulebased_aggressive.json"
OUTPUT_FILE = "datasets/results_rb_agg.json"
API_KEY = os.getenv("GROQ_API_KEY")  # Sostituisci con la tua API key
MODEL = "llama-3.3-70b-versatile"   # Puoi cambiare con llama-3.1-8b-instant per più velocità
PROMPT_TEMPLATE = "Solve this math problem"

def process_instance(client, instance, model):
    """Processa una singola istanza con entrambi i prompt"""
    
    results = {
        "original_prompt": {},
        "compressed_prompt": {}
    }
    
    # Processa prompt originale
    try:
        start_time = time.time()
        
        response_original = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"{PROMPT_TEMPLATE}: {instance['question']}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        end_time = time.time()
        
        results["original_prompt"] = {
            "prompt_used": "question",
            "response": response_original.choices[0].message.content,
            "tokens_used": response_original.usage.total_tokens,
            "generation_time_ms": int((end_time - start_time) * 1000),
            "model": model
        }
        
    except Exception as e:
        results["original_prompt"] = {
            "error": str(e),
            "prompt_used": "question"
        }
    
    # Piccola pausa per evitare rate limits
    time.sleep(0.1)
    
    # Processa prompt compresso
    try:
        start_time = time.time()
        
        response_compressed = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"{PROMPT_TEMPLATE}: {instance['question_cut']}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        end_time = time.time()
        
        results["compressed_prompt"] = {
            "prompt_used": "question_cut",
            "response": response_compressed.choices[0].message.content,
            "tokens_used": response_compressed.usage.total_tokens,
            "generation_time_ms": int((end_time - start_time) * 1000),
            "model": model
        }
        
    except Exception as e:
        results["compressed_prompt"] = {
            "error": str(e),
            "prompt_used": "question_cut"
        }
    
    return results

def main():
    # Inizializza client Groq
    client = Groq(api_key=API_KEY)
    
    # Carica dati input
    print(f"Caricamento dati da {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Trovate {len(data)} istanze da processare")
    
    # Processa ogni istanza
    processed_data = []
    
    for i, instance in enumerate(data, 1):
        print(f"Processando istanza {i}/{len(data)}...")
        
        # Crea nuova istanza con risultati
        new_instance = instance.copy()
        
        # Processa con LLM
        llm_responses = process_instance(client, instance, MODEL)
        
        # Aggiungi risultati
        new_instance["llm_responses"] = llm_responses
        new_instance["metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "api_used": "groq"
        }
        
        processed_data.append(new_instance)
        
        # Salva progressivamente ogni 10 istanze
        if i % 10 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Salvate {i} istanze")
    
    # Salvataggio finale
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Completato! Risultati salvati in {OUTPUT_FILE}")
    print(f"Processate {len(processed_data)} istanze")

if __name__ == "__main__":
    main()