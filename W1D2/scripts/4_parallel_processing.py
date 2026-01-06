"""
Concept 4: Parallel Processing with PySpark
Exercises 1-5: Walk through PySpark workflow, compare approaches, identify differences, run preprocessing, understand trade-offs
"""
from pathlib import Path

if __name__ == "__main__":
    print("=" * 60)
    print("Concept 4: Parallel Processing with PySpark")
    print("=" * 60)
    
    # Exercise 1: Walk through PySpark workflow
    print("\n[Exercise 1] Walking through PySpark workflow...")
    print("-" * 60)
    preprocess_script = Path("../../ARF-Training/data_prep/megatron/src/preprocess_data_spark.py")
    if not preprocess_script.exists():
        preprocess_script = Path("/mnt/weka/aisg/users/yuli/ARF-Training/repos/PAIP-Pretrain/ARF-Training/data_prep/megatron/src/preprocess_data_spark.py")
    
    if preprocess_script.exists():
        print(f"Analyzing: {preprocess_script}\n")
        with open(preprocess_script, 'r') as f:
            content = f.read()
        
        print("PySpark Workflow Steps:\n")
        print("1. SparkSession Creation:")
        if "SparkSession.builder" in content:
            print("   - Found: SparkSession.builder")
            print("   - Configures: master, driver memory, executor memory")
            print("   - Sets partition size based on available memory")
        print()
        
        print("2. Loading Input Files:")
        if "spark.read" in content:
            print("   - Found: spark.read.json() or spark.read.parquet()")
            print("   - Converts to Spark DataFrame")
            print("   - Preserves ordering with metadata columns")
        print()
        
        print("3. Partition Processing:")
        if "mapPartitionsWithIndex" in content:
            print("   - Found: rdd.mapPartitionsWithIndex()")
            print("   - Each partition processed independently")
            print("   - Tokenization happens per partition")
            print("   - Creates IndexedDatasetBuilder per partition")
        print()
        
        print("4. Writing Partition Outputs:")
        if "add_document" in content:
            print("   - Found: builder.add_document() calls")
            print("   - Finalizes partition files")
        print()
        
        print("5. Merging Partitions:")
        if "add_index" in content:
            print("   - Found: builder.add_index() for merging")
            print("   - Creates final .bin and .idx files")
            print("   - Cleans up intermediate partition files")
    else:
        print("Preprocessing script not found. Showing conceptual workflow:\n")
        print("1. Initialize SparkSession:")
        print("   spark = SparkSession.builder \\")
        print("       .master('local[N]') \\")
        print("       .config('spark.driver.memory', '32g') \\")
        print("       .getOrCreate()\n")
        print("2. Load Data:")
        print("   df = spark.read.json('input.jsonl')")
        print("   rdd = df.rdd\n")
        print("3. Process Partitions:")
        print("   rdd.mapPartitionsWithIndex(process_partition).count()\n")
        print("4. Merge Results:")
        print("   # Merge all partition .bin/.idx files into final output")
    
    # Exercise 2: Compare with single-processing
    print("\n[Exercise 2] Comparing PySpark with single-processing...")
    print("-" * 60)
    print("Comparison: PySpark vs Single-processing\n")
    
    print("1. Data Loading:")
    print("   PySpark:")
    print("     - spark.read.json() → DataFrame → RDD")
    print("     - Automatic partitioning")
    print("     - Handles large files efficiently")
    print()
    print("   Single-processing:")
    print("     - open(file) → read line by line")
    print("     - Sequential file reading")
    print("     - Simple but limited by memory")
    print()
    
    print("2. Parallelization:")
    print("   PySpark:")
    print("     - Automatic partition distribution")
    print("     - Can span multiple machines")
    print("     - Task scheduling and fault tolerance")
    print()
    print("   Single-processing:")
    print("     - Sequential processing")
    print("     - One document at a time")
    print("     - No parallelization")
    print()
    
    print("3. Memory Management:")
    print("   PySpark:")
    print("     - Partitions processed independently")
    print("     - Automatic memory management")
    print("     - Can handle datasets larger than RAM")
    print()
    print("   Single-processing:")
    print("     - Loads entire file into memory (if small)")
    print("     - Or processes line by line")
    print("     - Limited by available RAM")
    
    # Exercise 3: Identify key differences
    print("\n[Exercise 3] Identifying key differences: PySpark vs multiprocessing...")
    print("-" * 60)
    print("Key Differences:\n")
    
    print("1. Architecture:")
    print("   PySpark: Distributed, can span multiple machines")
    print("   Multiprocessing: Single-machine, multiple processes")
    print()
    
    print("2. Data Partitioning:")
    print("   PySpark: Automatic partitioning, distributed across workers")
    print("   Multiprocessing: Manual splitting required")
    print()
    
    print("3. Fault Tolerance:")
    print("   PySpark: Automatic task retry, handles worker failures")
    print("   Multiprocessing: No built-in fault tolerance")
    print()
    
    print("4. Scalability:")
    print("   PySpark: Scales to hundreds of nodes, petabytes of data")
    print("   Multiprocessing: Limited by CPU cores on one machine")
    print()
    
    print("5. Overhead:")
    print("   PySpark: Higher startup overhead, network communication")
    print("   Multiprocessing: Lower overhead, faster startup")
    
    # Exercise 4: Run small preprocessing job
    print("\n[Exercise 4] Running small preprocessing job (demo)...")
    print("-" * 60)
    print("To run preprocessing with PySpark:\n")
    
    print("1. Prepare input data:")
    print("   - JSONL file with text data")
    print("   - Ensure tokenizer is available")
    print()
    
    print("2. Run preprocessing script:")
    print("   python preprocess_data_spark.py \\")
    print("       --input /path/to/input.jsonl \\")
    print("       --output-prefix /path/to/output \\")
    print("       --tokenizer-type HuggingFaceTokenizer \\")
    print("       --tokenizer-model Qwen/Qwen3-4B \\")
    print("       --json-keys text \\")
    print("       --append-eod \\")
    print("       --workers 4")
    print()
    
    print("3. Observe execution:")
    print("   - SparkSession initialization")
    print("   - Data loading and partitioning")
    print("   - Parallel processing across partitions")
    print("   - Progress logs from each partition")
    print("   - Merging of partition outputs")
    print()
    
    print("Expected observations:")
    print("   - Multiple partitions processed simultaneously")
    print("   - CPU cores utilized across workers")
    print("   - Total time < sum of individual times")
    
    # Exercise 5: Understand trade-offs
    print("\n[Exercise 5] Understanding trade-offs...")
    print("-" * 60)
    print("Decision Matrix:\n")
    
    print("Dataset Size | Best Choice | Reason")
    print("-" * 60)
    print("Small (< 1GB)  | Single-processing | Low overhead, simple")
    print("Medium (1-100GB) | Multiprocessing | Good balance, local speed")
    print("Large (> 100GB) | PySpark | Scalability, fault tolerance")
    print()
    
    print("Trade-off Summary:")
    print("  Single-processing: Simplest, fast for small data, no parallelization")
    print("  Multiprocessing: Good speedup, single machine, manual splitting")
    print("  PySpark: Best scalability, fault tolerance, higher overhead")
    print()
    
    print("When to Choose Each:")
    print("  Single-processing: Small datasets, quick prototyping")
    print("  Multiprocessing: Medium datasets, single machine available")
    print("  PySpark: Large datasets, cluster access, need fault tolerance")
    
    print("\n" + "=" * 60)
    print("All exercises completed!")

