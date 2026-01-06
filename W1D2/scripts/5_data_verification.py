"""
Concept 5: Data Verification and Quality Checks
Exercises 1-5: Run verification, examine checks, interpret results, fix issues, generate report
"""
import json
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    print("=" * 60)
    print("Concept 5: Data Verification and Quality Checks")
    print("=" * 60)
    
    # Exercise 1: Run verification script
    print("\n[Exercise 1] Running verification script (demo)...")
    print("-" * 60)
    print("To run verification on processed data:\n")
    
    print("1. Basic usage:")
    print("   python verify_processed_data.py \\")
    print("       --output_root_dir /path/to/processed/data \\")
    print("       --dataset_name enwiki \\")
    print("       --input_dir /path/to/raw/data \\")
    print("       --megatron_prefix EN/EN_Wikipedia \\")
    print("       --text_key text \\")
    print("       --input_file_groups '[[\"file1.jsonl\"], [\"file2.jsonl\"]]'")
    print()
    
    print("2. What it checks:")
    print("   - Number of output files matches expected")
    print("   - Output files are non-zero size")
    print("   - Total line counts match between raw and processed")
    print("   - Sample documents match (round-trip verification)")
    print()
    
    print("Expected Output:")
    print("  ==== LOOK AT ONE SAMPLE ====")
    print("  Raw file: /path/to/raw/file.jsonl")
    print("  Processed file: /path/to/processed/file")
    print("  PASS: raw text == decoded text")
    print("  Number of passes: 20/20")
    
    # Exercise 2: Examine verification checks
    print("\n[Exercise 2] Examining verification checks...")
    print("-" * 60)
    print("Verification Check Details:\n")
    
    print("1. check_number_of_files_created_by_preprocess_per_subset()")
    print("   Purpose: Verify expected number of output files")
    print("   Checks: Counts .bin and .idx files, compares with input file groups")
    print("   Failure indicates: Incomplete processing, missing files")
    print()
    
    print("2. check_files_created_by_preprocess_are_not_zero_in_size_per_subset()")
    print("   Purpose: Detect empty or corrupted files")
    print("   Checks: File size of each .bin and .idx file")
    print("   Failure indicates: Processing failed silently, empty input data")
    print()
    
    print("3. check_total_line_counts_per_subset()")
    print("   Purpose: Verify no data loss during conversion")
    print("   Checks: Counts lines in raw JSONL vs sequences in processed dataset")
    print("   Failure indicates: Documents skipped, encoding errors, empty documents")
    print()
    
    print("4. check_bin_idx_vs_raw_text()")
    print("   Purpose: Verify data integrity (round-trip)")
    print("   Checks: Randomly samples documents, decodes and compares with original")
    print("   Failure indicates: Tokenization errors, data corruption, wrong tokenizer")
    print()
    
    print("5. Token ID Range Check (conceptual)")
    print("   Purpose: Verify token IDs are valid")
    print("   Checks: All token IDs within vocabulary range")
    print("   Failure indicates: Incorrect tokenizer, data corruption")
    
    # Exercise 3: Interpret results
    print("\n[Exercise 3] Interpreting verification results...")
    print("-" * 60)
    print("Understanding Verification Results:\n")
    
    print("1. File Count Mismatch:")
    print("   Result: Expected 10 files, found 8")
    print("   Interpretation: Some input file groups didn't produce output")
    print("   Action: Check preprocessing logs, verify input files, re-run preprocessing")
    print()
    
    print("2. Zero-size Files:")
    print("   Result: File 'output.bin' has zero size")
    print("   Interpretation: Processing failed for this file")
    print("   Action: Check input data, review preprocessing errors, re-process")
    print()
    
    print("3. Line Count Mismatch:")
    print("   Result: Raw lines: 1000, Processed: 950")
    print("   Interpretation: 50 documents were lost during processing")
    print("   Action: Check preprocessing logs, verify encoding, review filtering logic")
    print()
    
    print("4. Round-trip Mismatch:")
    print("   Result: Sample 5/20 failed round-trip check")
    print("   Interpretation: May be acceptable (normalization) or indicate corruption")
    print("   Action: Check if mismatch is due to normalization, verify tokenizer")
    print("           If < 10% failures, usually acceptable")
    print()
    
    print("Decision Tree:")
    print("  All pass → Use data for training")
    print("  File count mismatch → Re-run preprocessing")
    print("  Zero-size files → Fix and re-process")
    print("  Line count mismatch → Investigate data loss")
    print("  Round-trip failures → Check if acceptable (< 10% usually OK)")
    
    # Exercise 4: Fix common issues
    print("\n[Exercise 4] Fixing common issues...")
    print("-" * 60)
    print("Common Issues and Fixes:\n")
    
    print("Issue 1: Encoding Errors")
    print("  Symptom: Documents skipped, line count mismatch")
    print("  Fix:")
    print("    1. Check file encoding: file -i input.jsonl")
    print("    2. Convert to UTF-8: iconv -f ISO-8859-1 -t UTF-8 input.jsonl > output.jsonl")
    print("    3. Re-run preprocessing")
    print()
    
    print("Issue 2: Empty Documents")
    print("  Symptom: Zero-size files, line count mismatch")
    print("  Fix:")
    print("    1. Filter empty documents before preprocessing")
    print("    2. Or filter input JSONL to remove empty lines")
    print("    3. Re-run preprocessing")
    print()
    
    print("Issue 3: File Path Mismatches")
    print("  Symptom: File count mismatch, missing files")
    print("  Fix:")
    print("    1. Verify input file paths exist")
    print("    2. Check output directory permissions")
    print("    3. Update paths in config and re-run")
    print()
    
    print("Issue 4: Wrong Tokenizer")
    print("  Symptom: Round-trip failures, gibberish text")
    print("  Fix:")
    print("    1. Verify tokenizer model matches preprocessing")
    print("    2. Check tokenizer configuration")
    print("    3. Re-run preprocessing with correct tokenizer")
    print()
    
    print("Issue 5: Corrupted Files")
    print("  Symptom: Zero-size files, read errors")
    print("  Fix:")
    print("    1. Check disk space and health")
    print("    2. Remove corrupted files")
    print("    3. Re-run preprocessing")
    
    # Exercise 5: Generate report
    print("\n[Exercise 5] Generating verification report...")
    print("-" * 60)
    
    # Example report generation
    example_results = {
        "dataset_name": "enwiki",
        "file_count_status": "pass",
        "file_count_expected": 2,
        "file_count_actual": 2,
        "file_sizes_status": "pass",
        "files_checked": 2,
        "zero_size_files": [],
        "line_counts_status": "pass",
        "raw_lines": 1000,
        "processed_sequences": 1000,
        "round_trip_status": "pass",
        "samples_checked": 20,
        "samples_passed": 20,
        "samples_failed": 0,
    }
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": example_results["dataset_name"],
        "checks": {
            "file_count": {
                "status": example_results["file_count_status"],
                "expected": example_results["file_count_expected"],
                "actual": example_results["file_count_actual"],
            },
            "file_sizes": {
                "status": example_results["file_sizes_status"],
                "files_checked": example_results["files_checked"],
                "zero_size_files": example_results["zero_size_files"],
            },
            "line_counts": {
                "status": example_results["line_counts_status"],
                "raw_lines": example_results["raw_lines"],
                "processed_sequences": example_results["processed_sequences"],
                "difference": example_results["raw_lines"] - example_results["processed_sequences"],
            },
            "round_trip": {
                "status": example_results["round_trip_status"],
                "samples_checked": example_results["samples_checked"],
                "samples_passed": example_results["samples_passed"],
                "samples_failed": example_results["samples_failed"],
                "pass_rate": example_results["samples_passed"] / example_results["samples_checked"],
            }
        },
        "summary": {
            "all_passed": True,
            "total_checks": 4,
            "passed_checks": 4,
            "failed_checks": 0
        }
    }
    
    print("Example Verification Report:\n")
    print(f"Dataset: {report['dataset']}")
    print(f"Timestamp: {report['timestamp']}\n")
    
    print("Check Results:")
    for check_name, check_data in report["checks"].items():
        status_symbol = "✓" if check_data["status"] == "pass" else "✗"
        print(f"  {status_symbol} {check_name}: {check_data['status']}")
    
    print(f"\nOverall: {'PASS' if report['summary']['all_passed'] else 'FAIL'}")
    print(f"  Passed: {report['summary']['passed_checks']}/{report['summary']['total_checks']}")
    print(f"  Failed: {report['summary']['failed_checks']}/{report['summary']['total_checks']}")
    print()
    
    print("Report Structure:")
    print("  - timestamp: When verification was run")
    print("  - dataset: Dataset name")
    print("  - checks: Results for each check")
    print("  - summary: Overall status and statistics")
    print()
    print("To save report:")
    print("  import json")
    print("  with open('verification_report.json', 'w') as f:")
    print("      json.dump(report, f, indent=2)")
    
    print("\n" + "=" * 60)
    print("All exercises completed!")

