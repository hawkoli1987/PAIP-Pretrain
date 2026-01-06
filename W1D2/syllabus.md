# Week 1, Day 2b: Data Preprocessing for Pretraining Framework

## Concept 1: Tokenization

### Socratic Questions

**Overall Intuition Questions:**
- Why can't neural networks directly process raw text strings? What form of data do they require?
- What problem does tokenization solve when dealing with natural language? Why not just use individual words or characters?
- How do subword tokenization methods (BPE/WordPiece/SentencePiece) balance between vocabulary size and coverage? What happens if the vocabulary is too small or too large?
- Why are tokenizers trained on large corpora? What would happen if we used a tokenizer trained on a different domain?

**Exercise Step Questions:**
1. How do we load a pre-trained tokenizer in Python? What information does a tokenizer contain?
2. When we tokenize text, what are the two representations we get? How are token IDs different from subword tokens?
3. If we tokenize text and then decode the token IDs, should we get back the exact same text? Why or why not?
4. What are special tokens and why do we need them? How can we identify them in a tokenizer's vocabulary?
5. How do we apply tokenization to a real dataset file? What considerations are there when processing JSONL files?

### Intuition
Raw text must be converted to numerical token IDs that the model can process. Tokenization breaks text into subword units (BPE/WordPiece/SentencePiece) that balance vocabulary size with coverage. This is required because neural networks operate on numerical tensors, not raw strings. In practice, tokenizers are trained on large corpora to learn optimal subword splits that minimize out-of-vocabulary tokens while keeping vocabulary manageable.

### Exercise Steps
1. Load the Qwen3-4B tokenizer using `AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")` from transformers
2. Tokenize a sample text string and examine the resulting token IDs (integers) vs. subword tokens (strings)
3. Decode token IDs back to text and verify round-trip consistency
4. Identify special tokens (BOS, EOS) in the tokenizer vocabulary and understand their token IDs
5. Practice tokenizing a JSONL file using the tokenizer

**Script**: `scripts/1_tokenization.py`

---

## Concept 2: Binarization (Indexed Dataset Format)

### Socratic Questions

**Overall Intuition Questions:**
- Why is loading entire datasets into memory problematic for large-scale training? What happens when we have terabytes of data?
- How does memory mapping work? Why is it more efficient than loading entire files?
- What information do we need to store to enable O(1) random access to any sequence in a dataset?
- Why do we need both `.bin` (data) and `.idx` (index) files? What would happen if we only had one?

**Exercise Step Questions:**
1. What is the structure of `.bin` and `.idx` files? How are they organized?
2. How do we read data from an indexed dataset? What methods does `IndexedDataset` provide?
3. How much memory does loading a JSONL file use compared to memory-mapping a binary file? Why is there a difference?
4. How are `.bin` and `.idx` files created? What does `IndexedDatasetBuilder` do?
5. How can we verify that the data in a binary file matches the original JSONL? What could go wrong?

### Intuition
Training on large datasets requires efficient random access and memory mapping. Binarization converts tokenized sequences into a binary format (`.bin` + `.idx`) that allows fast seeking without loading entire datasets into memory. This is critical for multi-epoch training where we need to randomly sample batches from terabytes of data. The indexed format stores offsets and lengths, enabling O(1) access to any sequence.

### Exercise Steps
1. Examine the structure of `.bin` (data) and `.idx` (index) files created by Megatron-Bridge preprocessing
2. Use `IndexedDataset` from `megatron.core.datasets.indexed_dataset` to read a few samples from a binarized dataset
3. Compare memory usage: loading entire JSONL file vs. memory-mapped binary (use `psutil` to measure)
4. Inspect the `IndexedDatasetBuilder` usage in `ARF-Training/data_prep/megatron/src/preprocess_data_spark.py` to understand how `.bin` and `.idx` files are created
5. Verify data integrity by decoding a sample from `.bin` file and comparing with original JSONL text

**Script**: `scripts/2_binarization.py`

---

## Concept 3: Sequence Length Management in Pretraining Preprocessing

### Socratic Questions

**Overall Intuition Questions:**
- When should sequence length management happen - during preprocessing or during training? What are the trade-offs?
- Why do we store documents as-is during preprocessing instead of truncating them immediately?
- How does the data loader handle documents that are longer or shorter than the model's context window?
- What happens to a very long document? Is it discarded or split into multiple training examples?

**Exercise Step Questions:**
1. How are documents stored in the indexed dataset? Are they truncated or stored completely?
2. How can we check the length of each document in a dataset? What does `sequence_lengths` tell us?
3. When are sequence length constraints applied? Why not during preprocessing?
4. How does Megatron-Bridge concatenate or split sequences during training? What's the strategy?

### Intuition
During pretraining data preprocessing, documents are tokenized and stored as-is in the indexed dataset format. Sequence length management (truncation, splitting, concatenation) happens **during training**, not during preprocessing. In Megatron-Bridge, the data loader handles sequence length by concatenating multiple documents up to the model's context window (e.g., 2048, 4096 tokens), or splitting long documents across multiple training examples. This approach maximizes data utilization while maintaining efficient random access during preprocessing.

### Exercise Steps
1. Examine how documents are stored in the indexed dataset: each document is stored as a complete sequence regardless of length
2. Check the sequence lengths of stored documents using `dataset.sequence_lengths` from an `IndexedDataset` object
3. Understand that sequence length constraints are applied during training data loading, not during preprocessing
4. Review Megatron-Bridge data loading code to see how sequences are concatenated/split to fit context windows during training

**Script**: `scripts/3_sequence_length.py`

---

## Concept 4: Parallel Processing with PySpark

### Socratic Questions

**Overall Intuition Questions:**
- Why is sequential processing insufficient for large datasets? How long would it take to process terabytes of data?
- What are the different approaches to parallelization? What are their limitations?
- How does PySpark enable distributed processing? What makes it different from simple multiprocessing?
- When should we use PySpark vs multiprocessing vs single-processing? What are the trade-offs?

**Working Logic Questions:**
- **Single-processing**: What happens when we process documents one by one? Why is this simple but slow?
- **Multi-processing**: How does Python's multiprocessing work? What limits its scalability?
- **PySpark**: How does PySpark partition data? How does it manage tasks across workers?

**Exercise Step Questions:**
1. How is a SparkSession created? What configuration options control the number of workers?
2. How are input files loaded into Spark DataFrames? What happens to the data structure?
3. How does `rdd.mapPartitionsWithIndex` enable parallel processing? What does each partition process?
4. How are partition outputs merged? Why do we need to merge them?
5. What are the key differences between PySpark and multiprocessing? When would you choose each?

### Intuition
Processing terabytes of data requires parallelization. PySpark enables distributed processing across multiple cores/nodes by partitioning data and processing each partition independently. This is required because single-machine processing would take weeks/months. In practice, we use PySpark to partition JSONL/parquet files and process each partition in parallel, then merge the results.

**Working Logic Comparison:**
- **Single-processing**: Sequential processing, one document at a time. Simple but slow for large datasets.
- **Multi-processing**: Uses Python's `multiprocessing` to spawn worker processes on a single machine. Limited by CPU cores and memory on one machine.
- **PySpark**: Distributed processing framework that partitions data across workers (can span multiple machines). Handles data partitioning, task scheduling, and fault tolerance automatically. Better for very large datasets and cluster environments.

### Exercise Steps
1. Walk through `ARF-Training/data_prep/megatron/src/preprocess_data_spark.py` to understand the PySpark workflow:
   - How SparkSession is created with worker configuration
   - How input files (JSONL/parquet) are loaded into Spark DataFrames
   - How `rdd.mapPartitionsWithIndex` processes each partition in parallel
   - How partition outputs are merged into final `.bin`/`.idx` files
2. Compare the PySpark approach with the single-processing version in `preprocess_data.py` (same directory)
3. Identify key differences: PySpark handles partitioning automatically, manages memory across partitions, and can scale to cluster environments
4. Run a small preprocessing job using PySpark script and observe the parallel execution
5. Understand the trade-offs: PySpark has higher overhead but better scalability; multiprocessing is simpler but limited to single machine

**Script**: `scripts/4_parallel_processing.py`

---

## Concept 5: Data Verification and Quality Checks

### Socratic Questions

**Overall Intuition Questions:**
- Why is data verification critical? What happens if corrupted data is used for training?
- What types of errors can occur during preprocessing? How can we detect them?
- Why do we need multiple verification checks? Can one check catch all problems?

**Issue-Specific Questions:**
- **Data Loss**: How can documents be lost during conversion? What causes encoding errors?
- **Tokenization Mismatches**: Why might decoded text not match original text? Is this always a problem?
- **Zero-size Files**: What causes empty output files? How can processing fail silently?
- **Missing Files**: Why might expected output files be missing? What could cause incomplete processing?
- **Invalid Token IDs**: How can token IDs be out of range? What would happen during training?

**Exercise Step Questions:**
1. How do we run verification checks? What information do they provide?
2. What does each verification check validate? What would a failure indicate?
3. How do we interpret verification results? What actions should we take for different failures?
4. How do we fix common issues? What are typical solutions?
5. How do we generate a comprehensive verification report?

### Intuition
Preprocessing bugs can silently corrupt training data. Verification ensures token counts match, sequences are valid, and no data is lost during conversion. This is required because training on corrupted data wastes weeks of compute. In practice, we verify checksums, sample random sequences, and compare statistics (token counts, sequence lengths) between raw and processed data.

**Potential Issues and Check Steps:**
- **Issue: Data loss during conversion** - Some documents may be skipped due to encoding errors or empty content
  - **Check**: Compare total line counts between raw JSONL files and processed dataset using `check_total_line_counts_per_subset()` in `verify_processed_data.py`
- **Issue: Tokenization mismatches** - Decoded text may not match original due to tokenizer normalization or encoding errors
  - **Check**: Sample random documents and verify round-trip consistency using `check_bin_idx_vs_raw_text()` in `verify_processed_data.py`
- **Issue: Zero-size or corrupted output files** - Processing may fail silently, creating empty files
  - **Check**: Verify all output `.bin` and `.idx` files are non-zero size using `check_files_created_by_preprocess_are_not_zero_in_size_per_subset()`
- **Issue: Missing or incomplete output files** - Some input file groups may not produce expected output files
  - **Check**: Verify expected number of output files matches input file groups using `check_number_of_files_created_by_preprocess_per_subset()`
- **Issue: Invalid token IDs** - Token IDs may be out of vocabulary range
  - **Check**: Decode samples and verify token IDs are within valid vocabulary range

### Exercise Steps
1. Run `ARF-Training/data_prep/megatron/src/verify_processed_data.py` on a processed dataset
2. Examine each verification check: file count, file sizes, line counts, and sample verification
3. Interpret verification results: understand what each check validates and what failures indicate
4. Practice fixing common issues: handle encoding errors, empty documents, and file path mismatches
5. Generate a verification report summarizing all checks and their results

**Script**: `scripts/5_data_verification.py`
