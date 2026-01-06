"""
Concept 3: Sequence Length Management in Pretraining Preprocessing
Exercises 1-4: Examine document storage, check sequence lengths, understand constraints, review data loading
"""
import numpy as np
import json
import sys
from pathlib import Path

# Helper function to create test dataset if it doesn't exist
def create_test_dataset_if_needed(dataset_path, jsonl_path, num_samples=10):
    """Create a small test dataset for exercises if it doesn't exist."""
    bin_file = Path(f"{dataset_path}.bin")
    idx_file = Path(f"{dataset_path}.idx")
    
    if bin_file.exists() and idx_file.exists():
        return True  # Dataset already exists
    
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"Note: Could not import transformers: {e}")
        return False
    
    try:
        sys.path.insert(0, '/mnt/weka/aisg/source_files/megatron_yuli')
        from megatron.core.datasets import indexed_dataset
    except ImportError as e:
        print(f"Note: Could not import Megatron-LM indexed_dataset: {e}")
        return False
    
    try:
        print(f"Creating test dataset from {num_samples} samples...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        
        # Use optimal_dtype method if available, otherwise default to uint32
        try:
            dtype = indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size)
        except (AttributeError, TypeError):
            # Fallback: determine dtype based on vocab size
            vocab_size = tokenizer.vocab_size
            if vocab_size < 65536:
                dtype = indexed_dataset.DType.uint16
            else:
                dtype = indexed_dataset.DType.uint32
        
        builder = indexed_dataset.IndexedDatasetBuilder(
            str(bin_file),
            dtype=dtype
        )
        
        count = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= num_samples:
                    break
                try:
                    data = json.loads(line.strip())
                    text = None
                    for key in ['text', 'content', 'raw_content', 'raw_contents', 'contents']:
                        if key in data:
                            text = data[key]
                            break
                    if text and len(text.strip()) > 0:
                        token_ids = tokenizer.encode(text, add_special_tokens=True)
                        if len(token_ids) > 0:
                            builder.add_document(token_ids, [len(token_ids)])
                            count += 1
                except Exception:
                    continue
        
        builder.finalize(str(idx_file))
        print(f"Created test dataset: {count} samples\n")
        return True
    except Exception as e:
        print(f"Note: Could not create test dataset: {e}")
        return False

SAMPLE_JSONL = "/mnt/weka/aisg/data/raw_knowledge/EN/EN_Wikipedia/raw/text/enwiki_dedup.jsonl"
EXAMPLE_DATASET = "test_dataset"  # Test dataset path

if __name__ == "__main__":
    print("=" * 60)
    print("Concept 3: Sequence Length Management in Pretraining Preprocessing")
    print("=" * 60)
    
    # Exercise 1: Examine document storage
    print("\n[Exercise 1] Examining how documents are stored in indexed dataset...")
    print("-" * 60)
    
    # Create test dataset if it doesn't exist
    jsonl_path = Path(SAMPLE_JSONL)
    if jsonl_path.exists():
        create_test_dataset_if_needed(EXAMPLE_DATASET, SAMPLE_JSONL, num_samples=10)
    
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDataset
        
        # Check if files exist first
        bin_file = Path(f"{EXAMPLE_DATASET}.bin")
        idx_file = Path(f"{EXAMPLE_DATASET}.idx")
        
        if not bin_file.exists() or not idx_file.exists():
            print(f"Note: Dataset files not found ({bin_file}, {idx_file})")
            print("Concept: Documents are stored as complete sequences, regardless of length.")
            print("  No truncation or splitting happens during preprocessing.")
        else:
            dataset = IndexedDataset(EXAMPLE_DATASET)
            print(f"Total documents: {len(dataset)}")
            print(f"Sequence lengths shape: {dataset.sequence_lengths.shape}\n")
            
            print("Examining sample documents:\n")
            for i in range(min(3, len(dataset))):
                seq = dataset[i]
                seq_len = len(seq)
                print(f"Document {i}:")
                print(f"  Length: {seq_len} tokens")
                print(f"  Stored as: Complete sequence (not truncated)")
                print(f"  First 5 tokens: {seq[:5].tolist()}")
                print(f"  Last 5 tokens: {seq[-5:].tolist()}")
                print()
            
            seq_lengths = dataset.sequence_lengths
            print("Document length statistics:")
            print(f"  Min length: {seq_lengths.min()} tokens")
            print(f"  Max length: {seq_lengths.max()} tokens")
            print(f"  Mean length: {seq_lengths.mean():.1f} tokens")
            print(f"  Median length: {np.median(seq_lengths):.1f} tokens")
            print()
            print("Key observation:")
            print("  Documents are stored as complete sequences, regardless of length.")
            print("  No truncation or splitting happens during preprocessing.")
            # Explicitly close to avoid __del__ issues
            del dataset
        
    except ImportError:
        print("Note: Requires Megatron-LM IndexedDataset")
        print("Concept: Documents are stored as-is, no truncation during preprocessing")
    except Exception as e:
        print(f"Note: {e}")
        print("Concept: Each document stored as complete sequence")
    
    # Exercise 2: Check sequence lengths
    print("\n[Exercise 2] Checking sequence lengths of stored documents...")
    print("-" * 60)
    
    # Ensure test dataset exists
    jsonl_path = Path(SAMPLE_JSONL)
    if jsonl_path.exists():
        create_test_dataset_if_needed(EXAMPLE_DATASET, SAMPLE_JSONL, num_samples=10)
    
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDataset
        
        # Check if files exist first
        bin_file = Path(f"{EXAMPLE_DATASET}.bin")
        idx_file = Path(f"{EXAMPLE_DATASET}.idx")
        
        if not bin_file.exists() or not idx_file.exists():
            print(f"Note: Dataset files not found ({bin_file}, {idx_file})")
            print("Concept: Use dataset.sequence_lengths to get all document lengths")
            print("  - Returns numpy array with length of each document")
            print("  - Can compute statistics: min, max, mean, percentiles")
        else:
            dataset = IndexedDataset(EXAMPLE_DATASET)
            seq_lengths = dataset.sequence_lengths
            
            print(f"Total documents: {len(seq_lengths)}")
            print(f"Sequence lengths array shape: {seq_lengths.shape}")
            print(f"Data type: {seq_lengths.dtype}\n")
            
            print("Sequence length statistics:")
            print(f"  Min: {seq_lengths.min()} tokens")
            print(f"  Max: {seq_lengths.max()} tokens")
            print(f"  Mean: {seq_lengths.mean():.1f} tokens")
            print(f"  Median: {np.median(seq_lengths):.1f} tokens")
            print(f"  Std: {seq_lengths.std():.1f} tokens")
            print()
            
            print("Length distribution (percentiles):")
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                val = np.percentile(seq_lengths, p)
                print(f"  {p}th percentile: {val:.1f} tokens")
            print()
            
            print("Sample sequence lengths (first 10):")
            for i in range(min(10, len(seq_lengths))):
                print(f"  Document {i}: {seq_lengths[i]} tokens")
            # Explicitly close to avoid __del__ issues
            del dataset
        
    except ImportError:
        print("Note: Requires Megatron-LM IndexedDataset")
        print("Concept: Use dataset.sequence_lengths to get all document lengths")
    except Exception as e:
        print(f"Note: {e}")
        print("Concept: Sequence lengths show wide variation in document sizes")
    
    # Exercise 3: Understand when constraints apply
    print("\n[Exercise 3] Understanding when sequence length constraints are applied...")
    print("-" * 60)
    print("Key Concept: When are sequence length constraints applied?\n")
    
    print("1. During Preprocessing:")
    print("   - Documents are tokenized and stored as-is")
    print("   - No truncation or splitting")
    print("   - Documents can be any length")
    print("   - Example: A 10,000 token document is stored as 10,000 tokens\n")
    
    print("2. During Training Data Loading:")
    print("   - Model has fixed context window (e.g., 2048, 4096 tokens)")
    print("   - Data loader handles sequence length constraints")
    print("   - Strategies:")
    print("     a) Concatenate multiple short documents up to context window")
    print("     b) Split long documents across multiple training examples")
    print("     c) Truncate if necessary (less common)")
    print("   - Example: 10,000 token document → 5 training examples of 2048 tokens each\n")
    
    print("3. Why this design?")
    print("   - Preprocessing: Fast, simple, preserves all data")
    print("   - Training: Flexible, can adjust to different context windows")
    print("   - Efficiency: Random access during preprocessing, batching during training\n")
    
    print("Example workflow:")
    print("  Preprocessing:")
    print("    Document 1: 500 tokens  → stored as 500 tokens")
    print("    Document 2: 3000 tokens → stored as 3000 tokens")
    print("    Document 3: 100 tokens  → stored as 100 tokens\n")
    
    print("  Training (context window = 2048):")
    print("    Batch 1: [Doc1 (500) + Doc3 (100) + Doc2 (1448)] = 2048 tokens")
    print("    Batch 2: [Doc2 remaining (1552)] = 1552 tokens")
    
    # Exercise 4: Review data loading strategy
    print("\n[Exercise 4] Reviewing Megatron-Bridge data loading strategy...")
    print("-" * 60)
    print("Conceptual Overview:\n")
    
    print("1. Data Loading Process:")
    print("   - Load documents from IndexedDataset")
    print("   - Each document has variable length")
    print("   - Model requires fixed-length sequences (context window)\n")
    
    print("2. Concatenation Strategy:")
    print("   - Short documents: Concatenate multiple documents")
    print("   - Fill up to context window (e.g., 2048 tokens)")
    print("   - Add separator tokens between documents if needed\n")
    
    print("3. Splitting Strategy:")
    print("   - Long documents: Split into multiple sequences")
    print("   - Each sequence fits within context window")
    print("   - May use overlap to preserve context\n")
    
    print("4. Example:")
    print("   Context window: 2048 tokens")
    print("   Document 1: 500 tokens")
    print("   Document 2: 3000 tokens")
    print("   Document 3: 100 tokens")
    print()
    print("   Result:")
    print("     Sequence 1: [Doc1 (500) + Doc3 (100) + padding or next doc]")
    print("     Sequence 2: [Doc2 first 2048 tokens]")
    print("     Sequence 3: [Doc2 remaining 952 tokens + padding or next doc]\n")
    
    print("To find actual implementation:")
    print("  - Look in Megatron-Bridge repository")
    print("  - Check data loading modules")
    print("  - Search for 'concatenate', 'sequence_length', 'context_window'")
    
    print("\n" + "=" * 60)
    print("All exercises completed!")

