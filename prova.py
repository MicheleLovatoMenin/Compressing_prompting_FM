import json
import sys

input = "datasets/output_compressed_spacy_aggressive.json"

lista = []

with open(input, 'r', encoding='utf-8') as f:
    data = json.load(f)
    for entry in data:
        a = entry['compressed_tokens'] / (entry['original_tokens']) 
        lista.append(a)
media = sum(lista) / len(lista)
print(f"Average compression rate: {media:.2%}")