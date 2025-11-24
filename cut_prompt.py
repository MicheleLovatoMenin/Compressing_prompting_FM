import json
import re
import sys
import torch
import gc
from llmlingua import PromptCompressor

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "datasets/gsm8k_trial_set.json"
OUTPUT_FILE = "datasets/gsm8k_compressed_0.8.json"

# How much do we want to compress with LLMLingua?
# rate=0.5 means "keep 50% of the tokens" (halve the length).
# You can lower it to 0.3 to be more aggressive (cut 70%).
TARGET_RATE = 0.8

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
    # Keep token if it's NOT in the remove list (normalized) or if it's empty
    compressed_tokens = [t for t in tokens if re.sub(r'[^\w]', '', t.lower()) not in REMOVE_WORDS or t == '']
    compressed = ' '.join(compressed_tokens)
    return re.sub(r'\s+', ' ', compressed).strip()

def clean_memory():
    gc.collect()
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

    # Limit to 50 for testing purposes (remove this line to process all)
    data = data[:50]
    print(f"Processing {len(data)} rows...")

    # 1. EXECUTE RULE-BASED COMPRESSION
    print("\n[1/2] Running Rule-Based Compression...")
    for entry in data:
        # If the key 'question' exists, rename it to 'question_original'
        # .pop() retrieves the value, removes the old key, and we assign it to the new key
        if 'question' in entry:
            entry['question_original'] = entry.pop('question')
        
        # Now use 'question_original' as the base
        if 'question_original' in entry:
            entry['question_rulebased'] = rule_based_compress(entry['question_original'])
            # Initialize the field for the next step
            entry['question_llmlingua2'] = ""
        else:
            print("Warning: Found entry without a question!")

    # 2. EXECUTE LLMLINGUA-2 (BERT-based, fast)
    print("\n[2/2] Loading LLMLingua-2 (Microsoft BERT)...")
    clean_memory()
    
    compressor_v2 = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("      Compressing with LLMLingua-2...")
    for i, entry in enumerate(data):
        if 'question_original' in entry:
            result = compressor_v2.compress_prompt(
                entry['question_original'], 
                rate=TARGET_RATE, 
                force_tokens=['?', '.', '=']
            )
            # Write directly into the original list (in-place modification)
            entry['question_llmlingua2'] = result['compressed_prompt']
        
        if i % 10 == 0: print(f"      Done {i}/{len(data)}")

    del compressor_v2
    clean_memory()

    print(f"\nSaving clean dataset to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("Done! You can now run the evaluation script.")

if __name__ == "__main__":
    main()