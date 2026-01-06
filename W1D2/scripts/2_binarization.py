"""
Concept 2: Binarization (Indexed Dataset Format)
Exercises 1-5: Examine bin/idx structure, read IndexedDataset, compare memory, inspect builder, verify integrity
"""
import json
import os
from pathlib import Path

# Helper function for memory comparison (reusable)
def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return None

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
        import sys
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
        print(f"Created test dataset: {count} samples in {bin_file.name} and {idx_file.name}\n")
        return True
    except Exception as e:
        print(f"Note: Could not create test dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

# Sample paths
SAMPLE_JSONL = "/mnt/weka/aisg/data/raw_knowledge/EN/EN_Wikipedia/raw/text/enwiki_dedup.jsonl"
EXAMPLE_DATASET = "test_dataset"  # Test dataset path

if __name__ == "__main__":
    print("=" * 60)
    print("Concept 2: Binarization (Indexed Dataset Format)")
    print("=" * 60)
    
    # Exercise 1: Examine bin/idx structure
    print("\n[Exercise 1] Examining .bin and .idx file structure...")
    print("-" * 60)
    
    # Create test dataset if it doesn't exist
    jsonl_path = Path(SAMPLE_JSONL)
    if jsonl_path.exists():
        create_test_dataset_if_needed(EXAMPLE_DATASET, SAMPLE_JSONL, num_samples=10)
    
    example_bin = f"{EXAMPLE_DATASET}.bin"
    example_idx = f"{EXAMPLE_DATASET}.idx"
    
    if Path(example_bin).exists() and Path(example_idx).exists():
        bin_size = Path(example_bin).stat().st_size
        idx_size = Path(example_idx).stat().st_size
        print(f".bin file: {example_bin}")
        print(f"  Size: {bin_size:,} bytes ({bin_size / 1024 / 1024:.2f} MB)")
        print(f"  Contains: Binary token ID data")
        print(f"\n.idx file: {example_idx}")
        print(f"  Size: {idx_size:,} bytes ({idx_size / 1024:.2f} KB)")
        print(f"  Contains: Index metadata (offsets and lengths)")
    else:
        print(f"Note: Example files not found ({example_bin}, {example_idx})")
        print("This exercise examines the structure. You need processed data to run it.")
        print("\n.idx file structure (typical):")
        print("  - Header: Number of sequences, dtype info")
        print("  - Sequence offsets: Where each sequence starts in .bin file")
        print("  - Sequence lengths: Length of each sequence in tokens")
        print("\n.bin file structure (typical):")
        print("  - Raw binary data: Token IDs stored as integers")
        print("  - Dtype: Usually uint16 or uint32 depending on vocab size")
        print("  - Layout: Sequences concatenated sequentially")
    
    # Exercise 2: Read from IndexedDataset
    print("\n[Exercise 2] Reading from IndexedDataset...")
    print("-" * 60)
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDataset
        
        # Check if files exist first
        bin_file = Path(f"{EXAMPLE_DATASET}.bin")
        idx_file = Path(f"{EXAMPLE_DATASET}.idx")
        
        if not bin_file.exists() or not idx_file.exists():
            print(f"Note: Dataset files not found ({bin_file}, {idx_file})")
            print("This exercise requires processed data. Showing conceptual usage:\n")
            print("Usage:")
            print("  from megatron.core.datasets.indexed_dataset import IndexedDataset")
            print("  dataset = IndexedDataset('path/to/dataset')  # without .bin/.idx extension")
            print("  sequence = dataset[0]  # Get first sequence")
            print("  lengths = dataset.sequence_lengths  # Get all sequence lengths")
        else:
            dataset = IndexedDataset(EXAMPLE_DATASET)
            print(f"Dataset loaded successfully!")
            print(f"Total sequences: {len(dataset)}")
            print(f"Sequence lengths shape: {dataset.sequence_lengths.shape}")
            print(f"Total tokens: {dataset.sequence_lengths.sum()}\n")
            
            num_to_read = min(3, len(dataset))
            print(f"Reading {num_to_read} samples:\n")
            for i in range(num_to_read):
                sequence = dataset[i]
                seq_len = len(sequence)
                print(f"Sample {i}:")
                print(f"  Length: {seq_len} tokens")
                print(f"  First 10 token IDs: {sequence[:10].tolist()}")
                print(f"  Last 10 token IDs: {sequence[-10:].tolist()}")
                print()
            # Explicitly close to avoid __del__ issues
            del dataset
    except ImportError:
        print("Error: Could not import IndexedDataset from Megatron-LM")
        print("Note: This requires Megatron-LM to be installed and in PYTHONPATH.")
        print("  Set PYTHONPATH to include Megatron-LM directory, e.g.:")
        print("  export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH")
    except Exception as e:
        print(f"Note: {e}")
        print("Concept: IndexedDataset provides O(1) random access to sequences")
        print("  - dataset[i] returns sequence i")
        print("  - dataset.sequence_lengths returns array of all lengths")
    
    # Exercise 3: Compare memory usage
    print("\n[Exercise 3] Comparing memory usage: JSONL vs memory-mapped binary...")
    print("-" * 60)
    jsonl_path = Path(SAMPLE_JSONL)
    
    # Ensure test dataset exists for comparison
    if jsonl_path.exists():
        create_test_dataset_if_needed(EXAMPLE_DATASET, SAMPLE_JSONL, num_samples=100)
    
    if jsonl_path.exists():
        mem_before = get_memory_usage()
        if mem_before is not None:
            print(f"Memory before loading: {mem_before:.2f} MB")
            
            # Load JSONL
            documents = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 1000:  # Limit for demo
                        break
                    data = json.loads(line.strip())
                    text = None
                    for key in ['text', 'content', 'raw_content', 'raw_contents', 'contents']:
                        if key in data:
                            text = data[key]
                            break
                    if text:
                        documents.append(text)
            
            mem_after = get_memory_usage()
            if mem_after is not None:
                mem_used = mem_after - mem_before
                print(f"Loaded {len(documents)} documents")
                print(f"Memory after loading: {mem_after:.2f} MB")
                print(f"Memory used: {mem_used:.2f} MB")
                print(f"File size: {jsonl_path.stat().st_size / 1024 / 1024:.2f} MB")
                
                # Compare with binary (conceptual)
                print("\nMemory-mapped binary (conceptual):")
                print("  - Uses minimal memory (only accessed portions loaded)")
                print("  - For same data: ~few MB vs ~hundreds MB for JSONL")
                print("  - Enables processing datasets larger than RAM")
        else:
            print("Note: psutil not available for memory measurement")
            print("Concept: Memory-mapped binary uses much less memory than loading entire JSONL")
    else:
        print(f"Note: JSONL file not found: {jsonl_path}")
        print("Concept: Memory-mapped binary is more memory-efficient than loading entire files")
    
    # Exercise 4: Inspect IndexedDatasetBuilder
    print("\n[Exercise 4] Inspecting IndexedDatasetBuilder usage...")
    print("-" * 60)
    preprocess_script = Path("../../ARF-Training/data_prep/megatron/src/preprocess_data_spark.py")
    if not preprocess_script.exists():
        preprocess_script = Path("/mnt/weka/aisg/users/yuli/ARF-Training/repos/PAIP-Pretrain/ARF-Training/data_prep/megatron/src/preprocess_data_spark.py")
    
    if preprocess_script.exists():
        print(f"Found preprocessing script: {preprocess_script}")
        with open(preprocess_script, 'r') as f:
            content = f.read()
        
        if "IndexedDatasetBuilder" in content:
            print("\nIndexedDatasetBuilder usage found:")
            print("  1. Import: from megatron.core.datasets import indexed_dataset")
            print("  2. Create builder:")
            print("     builder = indexed_dataset.IndexedDatasetBuilder(")
            print("         bin_file_path,")
            print("         dtype=indexed_dataset.DType.optimal_dtype(vocab_size)")
            print("     )")
            print("  3. Add documents:")
            print("     builder.add_document(token_ids, sentence_lengths)")
            print("  4. Finalize (creates .idx file):")
            print("     builder.finalize(idx_file_path)")
    else:
        print("Preprocessing script not found. Showing conceptual usage:")
        print("\nIndexedDatasetBuilder workflow:")
        print("  1. Create builder with output .bin file path")
        print("  2. Add documents one at a time with add_document()")
        print("  3. Finalize to write .idx file with offsets and lengths")
        print("  4. Result: .bin file (data) + .idx file (index)")
    
    # Exercise 5: Verify data integrity
    print("\n[Exercise 5] Verifying data integrity...")
    print("-" * 60)
    try:
        from transformers import AutoTokenizer
        from megatron.core.datasets.indexed_dataset import IndexedDataset
        
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        jsonl_path = Path(SAMPLE_JSONL)
        
        # Check if dataset files exist
        bin_file = Path(f"{EXAMPLE_DATASET}.bin")
        idx_file = Path(f"{EXAMPLE_DATASET}.idx")
        
        if jsonl_path.exists() and bin_file.exists() and idx_file.exists():
            dataset = IndexedDataset(EXAMPLE_DATASET)
            print("Verifying sample documents...\n")
            
            # Load JSONL texts (only first few to match test dataset)
            jsonl_texts = []
            max_samples = min(10, len(dataset))
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(jsonl_texts) >= max_samples:
                        break
                    try:
                        data = json.loads(line.strip())
                        text = None
                        for key in ['text', 'content', 'raw_content', 'raw_contents', 'contents']:
                            if key in data:
                                text = data[key]
                                break
                        if text and len(text.strip()) > 0:
                            jsonl_texts.append(text)
                    except Exception:
                        continue
            
            # Verify samples
            num_to_check = min(3, len(dataset), len(jsonl_texts))
            matches = 0
            
            for i in range(num_to_check):
                print(f"Sample {i}:")
                original_text = jsonl_texts[i]
                token_ids = dataset[i].tolist()
                decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                
                # Compare first 200 chars
                compare_len = min(200, len(original_text), len(decoded_text))
                if original_text[:compare_len] == decoded_text[:compare_len]:
                    print(f"  ✓ MATCH")
                    matches += 1
                else:
                    print(f"  ✗ MISMATCH (may be due to tokenizer normalization)")
                    print(f"    Original preview: {original_text[:100]}...")
                    print(f"    Decoded preview: {decoded_text[:100]}...")
                print()
            
            print(f"Verification: {matches}/{num_to_check} samples matched")
            # Explicitly close to avoid __del__ issues
            del dataset
        else:
            print("Note: Requires both processed dataset and original JSONL file")
            print("Concept: Decode token IDs from .bin and compare with original JSONL text")
            
    except ImportError as e:
        print(f"Note: {e}")
        print("Concept: Verify by decoding token IDs and comparing with original text")
    except Exception as e:
        print(f"Note: {e}")
        print("Concept: Round-trip verification ensures data integrity")
    
    print("\n" + "=" * 60)
    print("All exercises completed!")
