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

# Common non-essential adjectives
ADJECTIVES_COMMON = {
    'good', 'bad', 'big', 'small', 'large', 'little', 'new', 'old',
    'great', 'high', 'different', 'important', 'public', 'poor', 'major',
    'available', 'popular', 'likely', 'natural', 'similar', 'common',
    'recent', 'certain', 'full', 'simple', 'sure', 'clear', 'whole',
    'better', 'best', 'worse', 'worst', 'nice', 'beautiful', 'ugly',
    'happy', 'sad', 'fresh', 'lovely', 'wonderful', 'terrible'
}


def count_tokens(text):
    """Count tokens (words) in text."""
    return len(text.split())


def compress_text(text, level='light'):
    """
    Compress text by removing linguistic elements based on compression level.
    
    Args:
        text (str): Original text to compress
        level (str): Compression level - 'light', 'medium', or 'aggressive'
        
    Returns:
        str: Compressed text
    """
    tokens = text.split()
    compressed_tokens = []
    
    # Define removal sets based on compression level
    if level == 'light':
        # Level 1: Remove only articles and some common adverbs
        remove_set = ARTICLES | {'very', 'really', 'quite', 'just', 'also', 'too'}
    elif level == 'medium':
        # Level 2: Articles + adverbs + conjunctions + some pronouns
        remove_set = ARTICLES | ADVERBS | CONJUNCTIONS | {'he', 'she', 'it', 'they', 'them', 'him', 'her'}
    else:  # aggressive
        # Level 3: Everything (articles, adverbs, conjunctions, prepositions, pronouns, common adjectives)
        remove_set = ARTICLES | ADVERBS | CONJUNCTIONS | PREPOSITIONS | PRONOUNS | ADJECTIVES_COMMON
    
    for token in tokens:
        # Extract the word part (remove punctuation for checking)
        word_lower = re.sub(r'[^\w]', '', token.lower())
        
        # Keep token if the word is not in removal list
        if word_lower not in remove_set or word_lower == '':
            compressed_tokens.append(token)
    
    # Join tokens and clean up extra spaces
    compressed = ' '.join(compressed_tokens)
    compressed = re.sub(r'\s+', ' ', compressed).strip()
    
    return compressed


def process_json_file(input_file, output_file_prefix, level='light'):
    """
    Process JSON file containing questions and answers.
    Compress questions using rule-based approach with specified compression level.
    
    Args:
        input_file (str): Path to input JSON file
        output_file_prefix (str): Prefix for output JSON file
        level (str): Compression level - 'light', 'medium', or 'aggressive'
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
    total_original_tokens = 0
    total_compressed_tokens = 0
    
    for entry in data:
        if 'question' not in entry:
            print(f"Warning: Entry missing 'question' field, skipping: {entry}")
            continue
        
        original_question = entry['question']
        compressed_question = compress_text(original_question, level)
        original_tokens = count_tokens(original_question)
        compressed_tokens = count_tokens(compressed_question)
        
        total_original_tokens += original_tokens
        total_compressed_tokens += compressed_tokens
        
        compressed_entry = {
            'question': original_question,
            'question_cut': compressed_question,
            'original_tokens': original_tokens,
            'compressed_tokens': compressed_tokens,
            'answer': entry.get('answer', '')
        }
        processed_data.append(compressed_entry)
    
    # Create output filename with level
    output_file = f"datasets/{output_file_prefix}_{level}.json"
    
    # Write output JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    
    # Calculate overall statistics
    if total_original_tokens > 0:
        overall_compression = (1 - total_compressed_tokens / total_original_tokens) * 100
    else:
        overall_compression = 0
    
    print(f"\n{'='*60}")
    print(f"Compression Level: {level.upper()}")
    print(f"{'='*60}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Processed entries: {len(processed_data)}")
    print(f"Total original tokens: {total_original_tokens}")
    print(f"Total compressed tokens: {total_compressed_tokens}")
    print(f"Overall compression: {overall_compression:.1f}% reduction")
    
    # Print sample compression
    if processed_data:
        print(f"\n--- Sample Compression ({level}) ---")
        sample = processed_data[0]
        print(f"Original ({sample['original_tokens']} tokens):")
        print(f"  {sample['question']}")
        print(f"Compressed ({sample['compressed_tokens']} tokens):")
        print(f"  {sample['question_cut']}")
        sample_compression = (1 - sample['compressed_tokens'] / sample['original_tokens']) * 100
        print(f"Sample compression: {sample_compression:.1f}% reduction")


if __name__ == "__main__":
    input_file = "datasets/gsm8k_trial_set.json"
    output_prefix = "output_compressed_rulebased"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_prefix = sys.argv[2]
    
    # Process all three levels
    print("\nProcessing all compression levels...\n")
    
    for level in ['light', 'medium', 'aggressive']:
        process_json_file(input_file, output_prefix, level)
    
    print(f"\n{'='*60}")
    print("All compression levels completed!")
    print(f"{'='*60}")