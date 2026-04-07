# Week 2, Day 1: Logs, Checkpointing, Resuming

## MegatronBridge Reference: Checkpoint Workflows

The following summarizes what documentation exists in the MegatronBridge repo
(`/mnt/weka/aisg/users/yuli/ARF-Training/repos/megatron-bridge_yuli/`) for each
checkpoint workflow, and what is missing.

Copied reference docs live in `W2D1/ref/`:
- `ref/megatronbridge-checkpointing.md` — save/load/resume during training
- `ref/megatronbridge-bridge-guide.md` — HF ↔ Megatron conversion

---

### Workflow 1: Import checkpoint from HuggingFace → Megatron

**Status: DOCUMENTED**

Source: `docs/bridge-guide.md` (copied to `ref/megatronbridge-bridge-guide.md`)

The one-call convenience:
```bash
python -c "from megatron.bridge import AutoBridge; AutoBridge.import_ckpt('meta-llama/Llama-3.2-1B', './checkpoints/llama32_1b')"
```

Or via the CLI script:
```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model meta-llama/Llama-3.2-1B \
  --megatron-path ./checkpoints/llama32_1b \
  --torch-dtype bfloat16
```

Python API:
```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
AutoBridge.save_megatron_model(model, "./checkpoints/llama32_1b")
```

Result is a Megatron-format checkpoint (`iter_0000000/`, `latest_train_state.pt`).

---

### Workflow 2: Shard checkpoint for a given distribution strategy

**Status: DOCUMENTED (automatic — no separate sharding step needed)**

Source: `docs/training/checkpointing.md` (copied to `ref/megatronbridge-checkpointing.md`)

MegatronBridge uses Megatron Core's **distributed checkpointing** (`torch_dist` format).
Shards (`.distcp` files) are written directly from each rank, so no explicit resharding
step is needed when changing TP/PP/DP/CP/EP between save and load.

Enabling full reshardability of the optimizer state (across TP/PP/EP, not just DP):
```yaml
checkpoint:
  dist_ckpt_optim_fully_reshardable: true
```

Loading a checkpoint saved with different parallelism:
```python
checkpoint = CheckpointConfig(
    load="/path/to/checkpoint_dir",
    dist_ckpt_strictness="assume_ok_unexpected",  # default; tolerates key mismatches
)
```

**What's missing**: There is no standalone script to reshard an existing checkpoint from
one TP/PP configuration to another without re-running training. The resharding happens
automatically on the next training run when you change parallelism sizes.

---

### Workflow 3: Checkpoint saving during training

**Status: DOCUMENTED**

Source: `docs/training/checkpointing.md` (copied to `ref/megatronbridge-checkpointing.md`)

Configured via `CheckpointConfig`:
```yaml
checkpoint:
  save: /path/to/checkpoint_dir
  save_interval: 1000          # save every 1000 steps
  save_optim: true             # include optimizer state
  save_rng: true               # include RNG state
  save_tokenizer_assets: true  # save tokenizer files into checkpoint
  async_save: false            # true = background save (non-blocking)
```

Checkpoint directory structure:
```
checkpoint_dir/
├── latest_train_state.pt          # pointer to latest iteration
└── iter_N/
    ├── __0_0.distcp               # distributed weight/optimizer shards
    ├── __0_1.distcp
    ├── ...
    ├── .metadata                  # PyTorch DCP metadata
    ├── common.pt                  # rank-0 misc states
    ├── metadata.json              # MCore dist ckpt metadata
    ├── run_config.yaml            # full ConfigContainer snapshot
    ├── train_state.pt             # step count, consumed samples, LR scheduler state
    ├── tokenizer/                 # tokenizer files (portable)
    └── dataloader_state/          # per-DP-rank data iterator positions
        ├── train_dataloader_dprank000.pt
        └── ...
```

---

### Workflow 4: Checkpoint loading on resume

**Status: DOCUMENTED**

Source: `docs/training/checkpointing.md` (copied to `ref/megatronbridge-checkpointing.md`)

By default, loads the **latest** checkpoint from `latest_train_state.pt`:
```yaml
checkpoint:
  load: /path/to/checkpoint_dir
  load_optim: true   # restore optimizer state
  load_rng: true     # restore RNG state
```

To load a specific iteration:
```python
checkpoint = CheckpointConfig(
    load="/path/to/checkpoint_dir",
    ckpt_step=5000,  # loads iter_0005000; fails with FileNotFoundError if missing
)
```

For fine-tuning from a pretrained base (frozen weights):
```yaml
checkpoint:
  pretrained_checkpoint: /path/to/pretrained_megatron_ckpt
```

Both `load` (adapter/resume) and `pretrained_checkpoint` (base weights) can be set
simultaneously for PEFT fine-tuning.

---

### Workflow 5: Export checkpoint from Megatron → HuggingFace (including merge of shards)

**Status: DOCUMENTED**

Source: `docs/bridge-guide.md` (copied to `ref/megatronbridge-bridge-guide.md`)

The `.distcp` shards are merged automatically during export — no manual merge step.

One-call convenience:
```bash
python -c "
from megatron.bridge import AutoBridge
b = AutoBridge.from_hf_pretrained('meta-llama/Llama-3.2-1B')
b.export_ckpt('./checkpoints/llama32_1b', './exports/llama32_1b_hf')
"
```

CLI script:
```bash
python examples/conversion/convert_checkpoints.py export \
  --hf-model meta-llama/Llama-3.2-1B \
  --megatron-path ./checkpoints/llama32_1b \
  --hf-path ./exports/llama32_1b_hf
```

Three export options depending on need:
```python
# Full model (config + tokenizer + weights) — use for deployment
bridge.save_hf_pretrained(model, "./exports/llama32_1b_hf")

# Weights only in safetensors — faster, smaller
bridge.save_hf_weights(model, "./exports/weights_only")

# Stream weights without writing to disk — use in RL or eval pipelines
for name, weight in bridge.export_hf_weights(model, cpu=True):
    process(name, weight)
```

**Important**: Always use `AutoBridge.from_hf_pretrained()` (not `from_hf_config()`)
for export, as `from_hf_config()` lacks the tokenizer artifacts needed to produce a
complete HF checkpoint.

---

## Coverage Summary

| Workflow | MegatronBridge docs | Notes |
|---|---|---|
| 1. Import HF → Megatron | ✅ `bridge-guide.md` | CLI + Python API + one-liner |
| 2. Shard for parallelism | ✅ `checkpointing.md` + `parallelisms.md` | Automatic via `torch_dist`; no standalone reshard tool |
| 3. Save during training | ✅ `checkpointing.md` | Full config table + dir structure |
| 4. Load on resume | ✅ `checkpointing.md` | `load`, `ckpt_step`, `pretrained_checkpoint` |
| 5. Export Megatron → HF | ✅ `bridge-guide.md` | Shards merged automatically; 3 export methods |
