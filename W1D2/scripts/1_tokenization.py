"""
Concept 1: Tokenization
Exercises 1-5: Load tokenizer, examine tokenization, verify round-trip, identify special tokens, tokenize JSONL file
"""
import json
from transformers import AutoTokenizer
from pathlib import Path

# Sample dataset path
SAMPLE_JSONL = "/mnt/weka/aisg/data/raw_knowledge/EN/EN_Wikipedia/raw/text/enwiki_dedup.jsonl"

if __name__ == "__main__":
    print("=" * 60)
    print("Concept 1: Tokenization")
    print("=" * 60)
    
    # Exercise 1: Load tokenizer
    print("\n[Exercise 1] Loading Qwen3-4B tokenizer...")
    print("-" * 60)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    print(f"Tokenizer loaded successfully!")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"\nSpecial tokens:")
    print(f"  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # Exercise 2: Tokenize and examine
    print("\n[Exercise 2] Tokenizing sample text and examining token IDs vs subword tokens...")
    print("-" * 60)
    sample_text = "Hello! How are you today? This is a test of tokenization."
    print(f"Original text: {sample_text}\n")
    
    token_ids = tokenizer.encode(sample_text, add_special_tokens=True)
    tokens = tokenizer.tokenize(sample_text, add_special_tokens=True)
    
    print(f"Token IDs (integers): {token_ids}")
    print(f"Number of tokens: {len(token_ids)}\n")
    print(f"Subword tokens (strings): {tokens}")
    print(f"Number of subword tokens: {len(tokens)}\n")
    
    print("Mapping between token IDs and subword tokens:")
    for i, (token_id, token_str) in enumerate(zip(token_ids, tokens)):
        print(f"  Position {i}: ID {token_id:6d} -> '{token_str}'")
    
    converted_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print(f"\nUsing tokenizer.convert_ids_to_tokens(): {converted_tokens}")
    
    # Exercise 3: Verify round-trip
    print("\n[Exercise 3] Verifying round-trip consistency...")
    print("-" * 60)
    test_texts = [
        "Hello world!",
        "This is a longer sentence with punctuation, numbers 123, and special characters: @#$",
        "Tokenization is important for NLP models."
    ]
    
    for text in test_texts:
        print(f"\nOriginal text: '{text}'")
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        decoded_with = tokenizer.decode(token_ids)
        decoded_without = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        print(f"Decoded (with special tokens): '{decoded_with}'")
        print(f"Decoded (skip special tokens): '{decoded_without}'")
        print(f"Match: {text == decoded_without} (may differ due to normalization)")
    
    # Exercise 4: Identify special tokens
    print("\n[Exercise 4] Identifying special tokens...")
    print("-" * 60)
    special_token_attrs = {
        'bos_token': 'BOS (Beginning of Sequence)',
        'eos_token': 'EOS (End of Sequence)',
        'pad_token': 'PAD (Padding)',
        'unk_token': 'UNK (Unknown)',
    }
    
    print("Special tokens found:")
    for attr, description in special_token_attrs.items():
        token_str = getattr(tokenizer, attr, None)
        token_id = getattr(tokenizer, f"{attr}_id", None)
        if token_str is not None:
            print(f"  {description}: '{token_str}' (ID: {token_id})")
    
    vocab_size = tokenizer.vocab_size
    print(f"\nVerifying token IDs are in vocabulary (size: {vocab_size}):")
    if tokenizer.bos_token_id is not None:
        in_vocab = 0 <= tokenizer.bos_token_id < vocab_size
        print(f"  BOS (ID {tokenizer.bos_token_id}): {'✓ In vocabulary' if in_vocab else '✗ Out of range'}")
    if tokenizer.eos_token_id is not None:
        in_vocab = 0 <= tokenizer.eos_token_id < vocab_size
        print(f"  EOS (ID {tokenizer.eos_token_id}): {'✓ In vocabulary' if in_vocab else '✗ Out of range'}")
    
    print("\nExample: How special tokens appear in tokenization:")
    sample_text = "Hello world"
    token_ids = tokenizer.encode(sample_text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print(f"  Text: '{sample_text}'")
    print(f"  Token IDs: {token_ids}")
    print(f"  Tokens: {tokens}")
    if tokens:
        print(f"  First token (BOS?): {tokens[0]}")
        print(f"  Last token (EOS?): {tokens[-1]}")
    
    # Exercise 5: Tokenize JSONL file
    print("\n[Exercise 5] Tokenizing JSONL file...")
    print("-" * 60)
    jsonl_path = Path(SAMPLE_JSONL)
    
    if not jsonl_path.exists():
        print(f"Warning: JSONL file not found: {jsonl_path}")
        print("Note: This exercise requires the sample dataset. Skipping...")
    else:
        print(f"Reading JSONL file: {jsonl_path}")
        print(f"Processing up to 5 samples...\n")
        
        tokenized_samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num > 5:
                    break
                
                try:
                    data = json.loads(line.strip())
                    text = None
                    for key in ['text', 'content', 'raw_content', 'raw_contents', 'contents']:
                        if key in data:
                            text = data[key]
                            break
                    
                    if text is None:
                        print(f"  Sample {line_num}: No text field found, skipping")
                        continue
                    
                    token_ids = tokenizer.encode(text, add_special_tokens=True)
                    tokenized_samples.append({
                        'line_num': line_num,
                        'text_length': len(text),
                        'num_tokens': len(token_ids),
                    })
                    
                    print(f"  Sample {line_num}:")
                    print(f"    Text length: {len(text)} characters")
                    print(f"    Number of tokens: {len(token_ids)}")
                    print(f"    Text preview: {text[:100] + '...' if len(text) > 100 else text}")
                    print()
                    
                except json.JSONDecodeError as e:
                    print(f"  Sample {line_num}: JSON decode error: {e}")
                    continue
                except Exception as e:
                    print(f"  Sample {line_num}: Error: {e}")
                    continue
        
        if tokenized_samples:
            print(f"Processed {len(tokenized_samples)} samples successfully")
            avg_tokens = sum(s['num_tokens'] for s in tokenized_samples) / len(tokenized_samples)
            print(f"Average tokens per sample: {avg_tokens:.1f}")
    
    print("\n" + "=" * 60)
    print("All exercises completed!")

