# Exercise: Rebuilding the Data Pipeline for Multi-Stage Training

## Background

In continual pre-training (CPT), we often need to change the dataset mix partway through training. For example:

- **Stage 1**: 90% general web data + 10% domain-specific data (warm up on the domain)
- **Stage 2**: 10% general web data + 90% domain-specific data (focus on the domain)

MegatronBridge's multi-stage training implements this by **rebuilding the data pipeline at phase boundaries** while the model, optimizer, and scheduler continue uninterrupted. Your task is to implement the function that builds a fresh data pipeline for a new phase.

## The Megatron Data Pipeline

When MegatronBridge trains, data flows through a 5-layer pipeline:

```
blend config                    "10% dataset_A, 90% dataset_B"
    |                           (weights + paths)
    v
BlendedDataset                  Interleaves samples from multiple
    |                           Megatron .bin/.idx datasets according
    v                           to the blend weights
DataLoader + Sampler            Batches samples, distributes across
    |                           data-parallel ranks, skips consumed samples
    v
cyclic_iter()                   Makes the finite dataloader infinite
    |                           (wraps around when exhausted)
    v
RerunDataIterator               Wraps iterator for fault-tolerance
                                (supports replay on failure)
```

When we change the blend mid-training, we need to **rebuild this entire chain** with the new blend weights, while keeping everything else (model weights, optimizer state, LR schedule) intact.

## Your Task

Open `loaders.py` in this directory. Find `rebuild_train_data_iterator()` (around line 108). The function signature and docstring are complete, but the body is replaced with TODO comments. Implement the 5 steps described in the comments.

**Time budget:** ~45 minutes

## Step-by-Step Guide

### Step 1: Update the blend config

The `blend` argument arrives as a flat list: `["0.1", "/path/a", "0.9", "/path/b"]`.

MegatronBridge's `GPTDatasetConfig.blend` expects a parsed tuple: `(["/path/a", "/path/b"], [0.1, 0.9])`.

Use `get_blend_from_list()` (already imported at the top of the file, line 19) to convert:

```python
cfg.dataset.blend = get_blend_from_list(blend)
cfg.dataset.blend_per_split = None   # clear per-split config
```

### Step 2: Build the dataset

Get a dataset provider — this is a callable that creates `BlendedDataset` objects:

```python
from megatron.bridge.data.utils import get_dataset_provider
provider = get_dataset_provider(cfg.dataset)
```

Call it with a 3-tuple of sample counts `(train, valid, test)`. We only need the train dataset:

```python
train_ds, _, _ = provider((phase_train_samples, 0, 0), cfg.dataset)
```

**Why `phase_train_samples` instead of the total?** Because we're building a dataset sized for this phase only. The blend weights have changed — we want fresh samples from the new mix, not a massive dataset where most entries use the old blend.

### Step 3: Get data-parallel rank and world size

The dataloader needs to know which slice of data belongs to this GPU:

```python
dp_rank = torch.distributed.get_rank(group=dp_group)
dp_size = torch.distributed.get_world_size(group=dp_group)
```

### Step 4: Build the DataLoader

Look at `build_train_valid_test_data_loaders()` around line 272 of this file for the reference pattern. Your call is similar but simpler (no validation/test, no signal handler):

```python
train_dataloader = build_pretraining_data_loader(
    train_ds,                              # the dataset from step 2
    consumed_in_phase,                     # sampler offset (0 for fresh start)
    cfg.dataset.dataloader_type,           # "single" or "cyclic"
    cfg.train.micro_batch_size,            # samples per GPU per step
    cfg.dataset.num_workers,               # dataloader workers
    cfg.dataset.data_sharding,             # data sharding flag
    pin_memory=cfg.dataset.pin_memory,
    persistent_workers=cfg.dataset.persistent_workers,
    data_parallel_rank=dp_rank,
    data_parallel_size=dp_size,
    global_batch_size=cfg.train.global_batch_size,
)
```

**Why `consumed_in_phase` instead of `train_state.consumed_train_samples`?** The sampler offset tells it how many samples to skip. For a new phase, we start fresh (offset = 0). The global `consumed_train_samples` counts ALL samples across ALL phases — using it would skip into the wrong part of the new dataset.

### Step 5: Wrap in cyclic iterator + RerunDataIterator

The DataLoader is finite (it has `phase_train_samples` entries). Training needs an infinite stream, so we wrap it:

```python
train_iter = iter(cyclic_iter(train_dataloader))    # infinite cycling
train_data_iterator = RerunDataIterator(train_iter)  # fault-tolerance wrapper

return train_data_iterator, None, None
```

We return `None` for valid and test because those datasets don't change between phases.

## Verification

### Step 1: Copy your completed file into the MegatronBridge source

```bash
# Back up the original first
cp /mnt/weka/aisg/source_files/megatron-bridge_yuli/src/megatron/bridge/data/loaders.py \
   /mnt/weka/aisg/source_files/megatron-bridge_yuli/src/megatron/bridge/data/loaders.py.bak

# Copy your implementation
cp W1D3/tutorial/loaders.py \
   /mnt/weka/aisg/source_files/megatron-bridge_yuli/src/megatron/bridge/data/loaders.py
```

### Step 2: Run unit tests (no GPU needed, runs inside the container)

```bash
cd /mnt/weka/aisg/source_files/megatron-bridge_yuli
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/unit_tests/training/test_multi_stage.py -v
```

Expected: **15 passed**. These tests validate the multi-stage config logic. They don't test `rebuild_train_data_iterator()` directly, but they confirm your file doesn't have syntax errors.

### Step 3: Run the smoke test (requires 8 GPUs)

```bash
cd /mnt/weka/aisg/model_training_team/code_forge/yuli/megatron_bridge/scripts/multi_stage

# Set environment
export CKPT_DIR=/mnt/weka/aisg/ckpt/mb/multi_stage/exercise_test
export LOG_DIR=/mnt/weka/aisg/log/mb/multi_stage/exercise_test/$(date +%s)
export DATA_ROOT=/mnt/weka/aisg/data/megatron/qwen3
export WANDB_MODE=disabled
mkdir -p $LOG_DIR $CKPT_DIR

# Launch 2-stage training (~4 min total)
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --rdzv_endpoint=localhost:29500 \
    multi_stage_resolve_config.py --config multi_stage_test.yaml \
    2>&1 | tee $LOG_DIR/train.log
```

**Expected output** (key lines to look for):

```
Multi-stage training: phase transitions at steps [39]
Starting training loop at iteration 0
iteration   10/  78 | ... lm loss: 2.2xxx ...
iteration   20/  78 | ... lm loss: 2.1xxx ...
iteration   30/  78 | ... lm loss: 2.0xxx ...
Phase transition at step 39: entering stage 2/2 (stage2_mathcode_heavy)
saving checkpoint at iteration 39 ...
> rebuilding train dataset for phase: 312 samples    <-- YOUR FUNCTION WAS CALLED
Data iterators rebuilt for stage 2 (stage2_mathcode_heavy)
iteration   40/  78 | ... lm loss: 2.1xxx ...
iteration   50/  78 | ... lm loss: 1.4xxx ...       <-- loss changes with new blend
...
iteration   78/  78 | ...
saving checkpoint at iteration 78 ...
```

If you see "Phase transition at step 39" followed by "rebuilding train dataset" and training continuing to iteration 78 — your implementation works.

### Step 4: Restore the original (if needed)

```bash
cp /mnt/weka/aisg/source_files/megatron-bridge_yuli/src/megatron/bridge/data/loaders.py.bak \
   /mnt/weka/aisg/source_files/megatron-bridge_yuli/src/megatron/bridge/data/loaders.py
```

## Discussion Questions

1. **Why `(phase_train_samples, 0, 0)` instead of the full training sample count?**

   The dataset is sized for the current phase only. If total training is 10B tokens across 2 phases, each phase needs a 5B-token dataset with its own blend. Building a 10B dataset with the new blend would waste memory and include samples that were "meant for" the first phase's blend.

2. **Why `consumed_in_phase=0` instead of `train_state.consumed_train_samples`?**

   `consumed_train_samples` is the cumulative count across all phases. If phase 1 consumed 312 samples, using that as the sampler offset in the new dataset would skip the first 312 entries of the new blend — entries that were never seen. Starting at 0 means "begin this new blend from the beginning."

3. **What would happen if we didn't wrap in `cyclic_iter()`?**

   The dataloader would be finite. When it runs out of samples (after `phase_train_samples / (micro_batch_size * dp_size)` batches), `next()` would raise `StopIteration` and training would crash. `cyclic_iter()` makes it infinite by looping back to the start.

4. **How does the training loop know WHEN to call this function?**

   Look at `train.py` in the MegatronBridge source. Before the main `while` loop, it pre-computes a set of transition step numbers from the `MultiStageConfig`. Each iteration, it checks `if step in _phase_transition_steps` — if true, it saves a checkpoint and calls `rebuild_data_fn()` (which calls your function).
