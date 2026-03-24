#!/usr/bin/env python3
"""Quick script to show validation attempt details."""
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python show_attempt.py <jsonl_file> [attempt_number]")
    sys.exit(1)

jsonl_file = sys.argv[1]
attempt_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

with open(jsonl_file, encoding='utf-8') as f:
    lines = f.readlines()
    record = json.loads(lines[-1])  # Last line (skip metadata)

attempts = record['validation_attempts']
if attempt_num > len(attempts):
    print(f"Only {len(attempts)} attempts available")
    sys.exit(1)

attempt = attempts[attempt_num - 1]

print(f"=== POGING {attempt_num} ===")
print(f"Status: {'[OK] VALID' if attempt['is_valid'] else '[FAIL] INVALID'}")

# Number validation
num_val = attempt['number_validation']
print(f"\nNummer validatie: {'[OK]' if num_val['is_valid'] else '[FAIL]'}")
if not num_val['is_valid']:
    print("Errors:")
    for err in num_val['errors']:
        print(f"  - {err}")

# B1 validation
b1_val = attempt['b1_validation']
print(f"\nB1 validatie: {'[OK]' if b1_val['is_valid'] else '[FAIL]'}")
if not b1_val['is_valid']:
    print("Warnings:")
    for warn in b1_val['warnings'][:5]:
        print(f"  - {warn}")

print("\nB1 Details:")
details = b1_val['details']
print(f"  Gemiddelde zinlengte: {details['avg_sentence_length']} woorden")
print(f"  Langste zin: {details['max_sentence_length']} woorden")
print(f"  Lange zinnen (>25): {details['long_sentences_count']}")
print(f"  Complexe woorden (>5 lettergrepen): {len(details['complex_words'])}")
print(f"  Jargon woorden: {len(details['jargon_found'])}")
print(f"  Totaal zinnen: {details['total_sentences']}")
print(f"  Totaal woorden: {details['total_words']}")

# Tokens
print(f"\nTokens gebruikt:")
print(f"  Input: {attempt['tokens_used']['input']}")
print(f"  Output: {attempt['tokens_used']['output']}")

# Show full explanation
print(f"\n=== VOLLEDIGE UITLEG ===")
explanation = attempt['explanation']
print(explanation)
