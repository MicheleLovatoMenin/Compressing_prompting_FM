import json
import re
import sys

# Define removal lists (all lowercase for comparison)
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

# Combine all removal words
REMOVE_WORDS = ARTICLES | CONJUNCTIONS | PREPOSITIONS | ADVERBS | PRONOUNS


def compress_text(text):
    """
    Compress text by removing articles, conjunctions, prepositions, adverbs, and pronouns.
    
    Args:
        text (str): Original text to compress
        
    Returns:
        str: Compressed text
    """
    # Split text into tokens (words and punctuation)
    # This regex keeps punctuation attached to words when appropriate
    tokens = text.split()
    
    compressed_tokens = []
    
    for token in tokens:
        # Extract the word part (remove punctuation for checking)
        # But keep the original token to preserve punctuation
        word_lower = re.sub(r'[^\w]', '', token.lower())
        
        # Keep token if the word is not in removal list
        if word_lower not in REMOVE_WORDS or word_lower == '':
            compressed_tokens.append(token)
    
    # Join tokens and clean up extra spaces
    compressed = ' '.join(compressed_tokens)
    
    # Clean up multiple spaces
    compressed = re.sub(r'\s+', ' ', compressed).strip()
    
    return compressed


def process_json_file(input_file, output_file):
    """
    Process JSON file containing questions and answers.
    Compress questions using rule-based approach.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
    """
    # Read input JSON
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{input_file}' is not valid JSON.")
        sys.exit(1)
    
    # Process each entry
    processed_data = []
    for entry in data:
        if 'question' not in entry:
            print(f"Warning: Entry missing 'question' field, skipping: {entry}")
            continue
            
        question_cut = compress_text(entry['question'])

        compressed_entry = {
            'question': entry['question'],
            'question_cut': question_cut,
            'original_tokens' : len(entry['question'].split()),
            'compressed_tokens' : len(question_cut.split()),
            'answer': entry.get('answer', '')  # Keep answer as is, default to empty string if missing
        }
        processed_data.append(compressed_entry)
    
    # Write output JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    
    print(f"Processing complete!")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Processed {len(processed_data)} entries")
    
    # Print sample compression
    if processed_data:
        print("\n--- Sample Compression ---")
        sample = processed_data[0]
        print(f"Original: {sample['question']}")
        print(f"Compressed: {sample['question_cut']}")
        
        # Calculate compression ratio
        original_tokens = len(sample['question'].split())
        compressed_tokens = len(sample['question_cut'].split())
        compression_ratio = (1 - compressed_tokens / original_tokens) * 100
        print(f"Compression: {original_tokens} tokens → {compressed_tokens} tokens ({compression_ratio:.1f}% reduction)")


if __name__ == "__main__":
    # Default file names
    input_file = "datasets/gsm8k_trial_set.json"
    output_file = "datasets/output_compressed_baseline.json"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    process_json_file(input_file, output_file)