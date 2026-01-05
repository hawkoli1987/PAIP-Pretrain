# Week 1, Day 2b: Data Preprocessing for Pretraining Framework

## Key Knowledge-Points

### 1. Tokenization
- **Intuition**: Raw text must be converted to numerical token IDs that the model can process. Tokenization breaks text into subword units (BPE/WordPiece/SentencePiece) that balance vocabulary size with coverage. This is required because neural networks operate on numerical tensors, not raw strings. In practice, tokenizers are trained on large corpora to learn optimal subword splits that minimize out-of-vocabulary tokens while keeping vocabulary manageable.
- **Exercise Steps**:
  1. Load a tokenizer (e.g., Qwen3-4B) using HuggingFace's tokenizer
  2. Tokenize a sample text and examine token IDs vs. subword tokens
  3. Compare tokenization of the same text across different tokenizers
  4. Measure tokenization speed and vocabulary size trade-offs
  5. Practice handling special tokens (BOS, EOS, padding, etc.)

### 2. Binarization (Indexed Dataset Format)
- **Intuition**: Training on large datasets requires efficient random access and memory mapping. Binarization converts tokenized sequences into a binary format (`.bin` + `.idx`) that allows fast seeking without loading entire datasets into memory. This is critical for multi-epoch training where we need to randomly sample batches from terabytes of data. The indexed format stores offsets and lengths, enabling O(1) access to any sequence.
- **Exercise Steps**:
  1. Examine the structure of `.bin` (data) and `.idx` (index) files
  2. Write a script to read a few samples from a binarized dataset
  3. Compare memory usage: loading JSONL vs. memory-mapped binary
  4. Practice creating indexed datasets from tokenized JSONL files
  5. Verify data integrity by comparing original vs. reconstructed samples

### 3. Sequence Length Management
- **Intuition**: Models have fixed context windows (e.g., 2048, 4096 tokens). Sequences must be truncated or split to fit, and padding may be needed for batching. Long sequences are often split into multiple training examples to maximize data utilization. This is required because transformer architectures have quadratic attention complexity, making very long sequences computationally prohibitive.
- **Exercise Steps**:
  1. Implement sequence truncation and padding logic
  2. Practice splitting long documents into multiple sequences with overlap
  3. Calculate effective token utilization (non-padding tokens / total tokens)
  4. Compare different splitting strategies (sentence-aware vs. fixed-length chunks)
  5. Visualize sequence length distributions before and after processing

### 4. Parallel Processing
- **Intuition**: Processing terabytes of data requires parallelization across multiple workers/nodes. Sharding splits data into chunks that can be processed independently, enabling horizontal scaling. This is required because single-machine processing would take weeks/months. In practice, data is split by file or by document count, with each worker processing its shard independently.
- **Exercise Steps**:
  1. Implement a simple sharding script that splits JSONL files by line count
  2. Process multiple shards in parallel using multiprocessing
  3. Merge processed shards back into a single dataset
  4. Practice handling edge cases (uneven shards, empty files, etc.)
  5. Measure speedup from parallelization (1 vs. 4 vs. 8 workers)

### 5. Data Verification and Quality Checks
- **Intuition**: Preprocessing bugs can silently corrupt training data. Verification ensures token counts match, sequences are valid, and no data is lost during conversion. This is required because training on corrupted data wastes weeks of compute. In practice, we verify checksums, sample random sequences, and compare statistics (token counts, sequence lengths) between raw and processed data.
- **Exercise Steps**:
  1. Write a verification script that checks dataset integrity
  2. Compare token counts: original text → tokenized → binarized
  3. Sample and display random sequences from processed data
  4. Check for common issues: empty sequences, invalid token IDs, misaligned indices
  5. Generate a verification report with statistics and sample outputs

