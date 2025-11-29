import json
import re
import sys
import torch
import gc
from llmlingua import PromptCompressor

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "datasets/dataset_formatted_5shot.json"
OUTPUT_FILE = "datasets/gsm8k_compressed_5_shot.json"
TARGET_RATE = 0.5

# ==========================================
# RULE-BASED LOGIC
# ==========================================
ARTICLES = {'a', 'an', 'the'}
CONJUNCTIONS = {'and', 'but', 'or', 'so', 'yet', 'for', 'nor'}
PREPOSITIONS = {
    'in', 'on', 'at', 'to', 'from', 'with', 'by', 'about', 'as', 'into',
    'like', 'through', 'after', 'over', 'between', 'out', 'against',
    'during', 'without', 'before', 'under', 'around', 'among', 'of',
    'per', 'within', 'upon', 'beneath', 'beside', 'beyond', 'off',
    'above', 'below', 'near', 'behind', 'across', 'along', 'toward',
    'towards', 'throughout', 'until', 'since'
}
ADVERBS = {
    'very', 'really', 'quite', 'just', 'only', 'also', 'too', 'much',
    'most', 'more', 'well', 'even', 'however', 'then', 'now', 'every',
    'daily', 'always', 'never', 'often', 'sometimes', 'usually', 'rarely',
    'frequently', 'seldom', 'hardly', 'barely', 'nearly', 'almost',
    'extremely', 'completely', 'totally', 'absolutely', 'quite', 'rather',
    'fairly', 'pretty', 'enough', 'still', 'yet', 'already'
}
PRONOUNS = {
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'mine', 'yours', 'hers', 'ours', 'theirs', 'myself', 'yourself',
    'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves'
}
REMOVE_WORDS = ARTICLES | CONJUNCTIONS | PREPOSITIONS | ADVERBS | PRONOUNS

def rule_based_compress(text):
    tokens = text.split()
    compressed_tokens = [t for t in tokens if re.sub(r'[^\w]', '', t.lower()) not in REMOVE_WORDS]
    compressed = ' '.join(compressed_tokens)
    return re.sub(r'\s+', ' ', compressed).strip()

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==========================================
# MAIN PROCESSING
# ==========================================

def main():
    print(f"--- Reading {INPUT_FILE} ---")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Test su 10 righe
    data = data[:300]
    print(f"Processing {len(data)} rows...")

    # [1/2] RULE BASED
    print("\n[1/2] Running Rule-Based Compression...")
    for entry in data:
        if 'question' in entry:
            entry['question_original'] = entry.pop('question')
        if 'question_original' in entry:
            entry['question_rulebased'] = rule_based_compress(entry['question_original'])
            entry['question_llmlingua1'] = ""

    # [2/2] LLMLINGUA-1
    print("\n[2/2] Initializing LLMLingua-1...")
    clean_memory()
    
    # Usa un modello più piccolo e stabile
    model_id = "gpt2"  # Alternativa: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    try:
        print(f"      Loading model: {model_id}...")
        
        # FORZA CPU - questo è il fix critico
        device = "cpu"
        print(f"      Using device: {device}")
        
        compressor_v1 = PromptCompressor(
            model_name=model_id,
            device_map=device,  # FORZA CPU
            model_config={
                "trust_remote_code": True,
                "torch_dtype": torch.float32  # CPU richiede float32
            },
            open_api_config={}
        )
        
        # Verifica che il tokenizer sia caricato correttamente
        print(f"      Tokenizer loaded: {type(compressor_v1.tokenizer)}")
        
        # FIX: Assicurati che pad_token sia settato
        if compressor_v1.tokenizer.pad_token is None:
            compressor_v1.tokenizer.pad_token = compressor_v1.tokenizer.eos_token
            print(f"      Set pad_token to eos_token: {compressor_v1.tokenizer.pad_token}")
        
        # IMPORTANTE: Forza il device anche internamente
        compressor_v1.device = device
            
    except Exception as e:
        print(f"!!! CRASH DURING LOADING !!!: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n      Compressing with LLMLingua-1 (Target Rate: {TARGET_RATE})...")
    
    for i, entry in enumerate(data):
        if 'question_original' not in entry:
            continue
            
        original_txt = entry['question_original']
        
        try:
            print(f"\n   Processing row {i}...")
            print(f"   Original: {original_txt[:100]}...")
            
            # Calcola il target_token basato sul rate desiderato
            tokens_orig = compressor_v1.tokenizer.encode(original_txt)
            num_tokens_orig = len(tokens_orig)
            target_tokens = int(num_tokens_orig * TARGET_RATE)
            
            print(f"   Original tokens: {num_tokens_orig} -> Target: {target_tokens}")
            
            # USA target_token invece di rate - molto più affidabile!
            result = compressor_v1.compress_prompt(
                original_txt,
                target_token=target_tokens,  # Specifica numero esatto di token
                # Disabilita filtri che potrebbero causare problemi
                use_sentence_level_filter=False,
                use_context_level_filter=False,
                use_token_level_filter=True
            )
            
            compressed_txt = result.get('compressed_prompt', original_txt)
            
            # Calcola statistiche
            tokens_comp = compressor_v1.tokenizer.encode(compressed_txt)
            num_tokens_comp = len(tokens_comp)
            actual_rate = num_tokens_comp / num_tokens_orig if num_tokens_orig > 0 else 1.0
            
            print(f"   Compressed tokens: {num_tokens_comp}")
            print(f"   Actual rate: {actual_rate:.2%}")
            print(f"   Compressed: {compressed_txt[:100]}...")
            
            # Verifica se la compressione ha funzionato
            if original_txt == compressed_txt:
                print("   ⚠️ WARNING: No compression occurred!")
            elif actual_rate > 0.9:
                print("   ⚠️ WARNING: Very low compression rate!")
            
            entry['question_llmlingua1'] = compressed_txt
            
        except Exception as e:
            print(f"   ❌ Error on row {i}: {e}")
            import traceback
            traceback.print_exc()
            entry['question_llmlingua1'] = original_txt

    print("\nCleaning up model...")
    del compressor_v1
    clean_memory()

    print(f"\nSaving results to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("✓ Done!")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()