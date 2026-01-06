# Syllabus Outline

## Week 1: Data for PreTraining
day 1: data pipeline in general, 
- dedup
- filtering
- data generation
- quality classification
- toxicity removal
- PII protection

day 2a: rule-based and model-based quality classification

day 2b: data preprocessing for Pretraining Framework, ref: ARF-Training/data_prep/megatron

day 3: data-mixing, stratification and sampling, ref: ARF-Training/train/configs/data_config.yaml, ARF-Training/train/scripts/common/datamix

day 4-5 (optional): readup on SOTA model training stuff

## Week 2: Model Training, ref: ARF-Training/train/scripts/smc_megatron_bridge

day 1: logs, checkpointing, resuming

day 2: scheduler, optimizer, learning rate, GBS/MBS, etc.

day 3: parallelism, memory management, throughput

day 4-5 (optional): readup on SOTA model training stuff
