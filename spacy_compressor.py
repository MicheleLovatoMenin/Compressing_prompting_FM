import json
import sys
import spacy

"""
SpaCy-based Prompt Compressor

Usage:
    python script.py input.json output_prefix [level]
    
    level: 'light', 'medium', or 'aggressive' (default: processes all three)
"""

LIGHT_REMOVE_POS = {
    'DET',  # Determiners (a, an, the, this, that)
}

MEDIUM_REMOVE_POS = {
    'DET',   # Determiners
    'ADP',   # Adpositions (prepositions and postpositions)
    'CCONJ', # Coordinating conjunctions (and, or, but)
    'ADV',   # Adverbs
}

AGGRESSIVE_REMOVE_POS = {
    'DET',   # Determiners
    'ADP',   # Adpositions
    'CCONJ', # Coordinating conjunctions
    'SCONJ', # Subordinating conjunctions (if, while, that)
    'ADV',   # Adverbs
    'PRON',  # Pronouns
    'AUX',   # Auxiliary verbs (is, are, was, were, have, has)
    'PART',  # Particles (possessive marker 's, not)
}

# Additional refinement: remove non-essential adjectives in aggressive mode
ADJECTIVE_POS = 'ADJ'


def count_tokens(text):
    """Count tokens (words) in text."""
    return len(text.split())


def compress_with_spacy(text, nlp, level='light'):
    """
    Compress text using SpaCy POS tagging.
    
    Args:
        text (str): Original text to compress
        nlp: SpaCy language model
        level (str): Compression level - 'light', 'medium', or 'aggressive'
        
    Returns:
        str: Compressed text
    """
    # Choose removal set based on level
    if level == 'light':
        remove_pos = LIGHT_REMOVE_POS
        remove_adjectives = False
    elif level == 'medium':
        remove_pos = MEDIUM_REMOVE_POS
        remove_adjectives = False
    else:  # aggressive
        remove_pos = AGGRESSIVE_REMOVE_POS
        remove_adjectives = True  # Remove adjectives in aggressive mode
    
    # Process text with SpaCy
    doc = nlp(text)
    
    # Keep tokens that are not in the removal set
    kept_tokens = []
    for token in doc:
        # Check if token should be removed
        should_remove = token.pos_ in remove_pos
        
        # In aggressive mode, also remove adjectives
        if remove_adjectives and token.pos_ == ADJECTIVE_POS:
            should_remove = True
        
        # Keep token if it should not be removed
        if not should_remove:
            # Preserve whitespace for proper spacing
            if token.whitespace_:
                kept_tokens.append(token.text + ' ')
            else:
                kept_tokens.append(token.text)
    
    # Join and clean up
    compressed = ''.join(kept_tokens).strip()
    
    return compressed


def process_json_file(input_file, output_file_prefix, nlp, level='light'):
    """
    Process JSON file containing questions and answers.
    Compress questions using SpaCy-based approach with specified compression level.
    
    Args:
        input_file (str): Path to input JSON file
        output_file_prefix (str): Prefix for output JSON file
        nlp: SpaCy language model
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
        compressed_question = compress_with_spacy(original_question, nlp, level)
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
    # Load SpaCy model
    print("Loading SpaCy model (en_core_web_sm)...")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Model loaded successfully!\n")
    except OSError:
        print("Error: SpaCy model 'en_core_web_sm' not found.")
        print("Please install it using: python -m spacy download en_core_web_sm")
        sys.exit(1)
    
    # Parse command line arguments
    input_file = "datasets/gsm8k_trial_set.json"
    output_prefix = "output_compressed_spacy"
    specific_level = None
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_prefix = sys.argv[2]
    if len(sys.argv) > 3:
        specific_level = sys.argv[3]
        if specific_level not in ['light', 'medium', 'aggressive']:
            print(f"Error: Invalid level '{specific_level}'. Must be 'light', 'medium', or 'aggressive'")
            sys.exit(1)
    
    # Process specified level(s)
    if specific_level:
        process_json_file(input_file, output_prefix, nlp, specific_level)
    else:
        # Process all three levels
        print("Processing all compression levels...\n")
        for level in ['light', 'medium', 'aggressive']:
            process_json_file(input_file, output_prefix, nlp, level)
    
    print(f"\n{'='*60}")
    print("Compression completed!")
    print(f"{'='*60}")