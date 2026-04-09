# Week 2, Day 1: Logs, Checkpointing, Resuming

## Key Knowledge-Points

### 1. Log Files and Log Directory Organization
- **Questions**: When a multi-node training job crashes at 2am, how do you figure out which node failed? If you need to reproduce a run from 3 weeks ago, how do you recover the exact config and code that was used? Why does rank-0 show output in the terminal but other ranks don't?
- **Intuition**: Every training submission creates a dedicated `LOG_DIR` at `{shared_fs}/log/mb/{sweep_name}/{job_name}/{job_id}/`. The `{job_id}` makes each submission's logs unique — you always get a fresh directory even when resuming. The `CKPT_DIR` (at `{shared_fs}/ckpt/mb/{sweep_name}/{job_name}/`) deliberately omits `{job_id}` so resumed jobs continue writing to the same checkpoint.

  **Full directory layout:**
  ```
  {shared_fs}/log/mb/{sweep_name}/{job_name}/{job_id}/
  ├── slurm.log  OR  pbs.log                   # scheduler stdout/stderr
  ├── launch.slurm  OR  launch.pbs             # copy of the generated job script
  ├── launcher.py                               # copy of launcher at submission time
  ├── resolve_config.py                         # copy of training entry point
  ├── utils.py                                  # copy of utility module
  ├── config.yaml                               # copy of the config used
  ├── data_config.yaml                          # copy of data config (if exists)
  ├── calc_proportion.py                        # copy of proportion calculator (if exists)
  ├── <rank>_python_master.log                  # master node torchrun output (tee'd to terminal)
  ├── <rank>_python.log                         # non-master node torchrun output (per node)
  ├── <rank>_<hostname>_sh.log                  # python/framework version info (first node only)
  ├── recipe/
  │   ├── original.yaml                         # recipe before overrides
  │   └── override.yaml                         # recipe after all overrides applied
  ├── nccl/
  │   └── <hostname>.log                        # per-node NCCL debug log
  ├── env/
  │   ├── EnvVar_hostOS.log                     # host OS environment variables
  │   └── EnvVar_<hostname>.log                 # per-node container environment variables
  └── tb_logs/
      └── events.out.tfevents.*                 # TensorBoard event files
  ```

  **Execution context — where each artifact is created:**

  Understanding _where_ each file is created helps you reason about what's available when a job fails at different stages:

  | File / Directory | Created by | Runs on |
  |---|---|---|
  | `slurm.log` / `pbs.log` | Scheduler stdout redirect (`#SBATCH --output`) | Head node (job shell) |
  | `launch.slurm` / `launch.pbs` | `cp "$0"` in outer bash script | Head node (job shell) |
  | `launcher.py`, `resolve_config.py`, `utils.py`, `config.yaml`, `data_config.yaml`, `calc_proportion.py` | `cp` commands in outer bash script | Head node (job shell) |
  | `EnvVar_hostOS.log` | `printenv` in outer bash script | Head node (job shell) |
  | `<rank>_python_master.log` | torchrun output via `tee` on master node | Master node (SLURM: last node; PBS: node 0) |
  | `<rank>_python.log` | torchrun output redirect on non-master nodes | Each non-master node |
  | `<rank>_<hostname>_sh.log` | Shell block gated on `node_rank == 0` | First node only |
  | `EnvVar_<hostname>.log` | `env` inside container | Each node |
  | `nccl/<hostname>.log` | `NCCL_DEBUG_FILE` env var | Each node |
  | `recipe/original.yaml`, `recipe/override.yaml` | `resolve_config.py` saves these during config resolution | DP_rank 0 (inside training process) |
  | `tb_logs/events.*` | TensorBoard writer | Last rank (logger rank) |

  The key distinction: the **outer bash script** (the generated SLURM/PBS job script) runs once on the head node and handles directory creation, artifact copying, and environment variable setup. Then it launches `torchrun` (via `srun`/`mpirun`) which spawns processes on **each node**. Within each node, torchrun creates `gpus_per_node` worker processes, but log files are per-node (not per-GPU) — only the logger rank writes TensorBoard and WandB metrics.

  **Why we structure logs this way:**

  1. **Group by nature and source**: Scheduler logs, training output, NCCL communication logs, and environment dumps are separated into different files and directories. NCCL debug output is especially verbose — isolating it in `nccl/` prevents it from drowning out training output. Environment variables go in `env/` so you can quickly check container configuration without scrolling through training logs.
  2. **Identify hardware failures from individual nodes/GPUs**: Per-node `<rank>_python.log` files and per-node `nccl/<hostname>.log` files let you pinpoint exactly which machine failed. In a 16-node job, if one node has a bad GPU, its `nccl/<hostname>.log` will show the NCCL timeout while all other nodes show normal communication. Without per-node logs, you'd only see "NCCL timeout" in the master log with no indication of the faulty node.

  **Why we save artifact copies (scripts and configs):**

  1. **Immediate script iteration**: After `sbatch`/`qsub`, the generated job script copies all source files (`launcher.py`, `resolve_config.py`, `config.yaml`, etc.) into `LOG_DIR`. The running job reads from these copies, not from your working directory. This means you can immediately edit scripts in your workdir and submit a new experiment — the already-running job is not affected, and there is no risk of contaminating its configuration mid-training.
  2. **Full traceability**: Weeks later, when you need to understand why a run behaved differently, the `LOG_DIR` contains the exact `config.yaml`, `resolve_config.py`, and `launcher.py` that were used. Combined with `recipe/original.yaml` (before overrides) and `recipe/override.yaml` (after all overrides), you have complete reproducibility without relying on git history or memory of what you changed.

  **Where local log folders live for each logging backend:**

  | Backend | Directory | Set by | Lifetime |
  |---|---|---|---|
  | TensorBoard | `{LOG_DIR}/tb_logs/` | `cfg.logger.tensorboard_dir` in `resolve_config.py` | Per-submission (fresh each job ID) |
  | WandB | `{CKPT_DIR}/wandb/` | `WANDB_DIR={CKPT_DIR}` env var in launcher.py | Persistent across resumes |
  | Checkpoints (NeMo) | `{CKPT_DIR}/` | `cfg.checkpoint.save` and `cfg.checkpoint.load` | Persistent across resumes |

  TensorBoard lives in `LOG_DIR` because each submission gets its own TB event files (matching the job's log lifetime). WandB lives in `CKPT_DIR` because WandB auto-resumes the run using its local state in `wandb/` — if the job is resubmitted, WandB picks up where it left off rather than creating a new run.

- **Exercise Steps**:
  1. Navigate to a real `LOG_DIR` and list its contents; match each file to its purpose in the directory layout above.
  2. Open `<rank>_python_master.log` and a non-master `<rank>_python.log` side-by-side — what's different? Why does only the master node get tee'd?
  3. Examine `env/EnvVar_hostOS.log` — which secrets are filtered out and how? Find the grep pattern in `launcher.py` that does the filtering.
  4. Compare `recipe/original.yaml` vs `recipe/override.yaml` — which fields changed? What does this tell you about how feature flags modify the base recipe?
  5. Explain why `LOG_DIR` includes `{job_id}` but `CKPT_DIR` does not — what would break if both used job_id, or if both omitted it?
  6. Using the execution-context table, identify which files would be missing if a job was killed before torchrun started on any node. Which files would still exist?

### 2. W&B Integration and Metrics Dashboard
- **Questions**: If `launcher.py` contains no `wandb.init()` calls, how does W&B know about the run? What does `WANDB_RUN_GROUP` control and why does it matter when you have dozens of experiments? If throughput suddenly drops by 30%, which W&B metric would you look at first?
- **Intuition**: W&B is wired entirely through environment variables set in the generated bash script — `WANDB_PROJECT`, `WANDB_RUN_GROUP` (= `SWEEP_NAME`), `WANDB_EXP_NAME` (= `{JOB_NAME}-{JOB_ID}`), `WANDB_MODE=online`, and `WANDB_DIR` (= `CKPT_DIR`). The actual `wandb.init()` and `wandb.log()` calls live inside MegatronBridge's training loop. The `resolve_config.py` `apply_logging_config()` function configures which metric groups to enable — all of the following are enabled by default in our setup:

  ```python
  cfg.logger.log_throughput = True
  cfg.logger.log_progress = True
  cfg.logger.log_l2_norm_grad_to_tensorboard = True
  cfg.logger.log_memory_to_tensorboard = True
  cfg.logger.log_params_norm = True
  cfg.logger.log_runtime_to_tensorboard = True
  cfg.logger.log_throughput_to_tensorboard = True
  cfg.logger.throughput_window_size = 20
  cfg.logger.runtime_time_unit = "hours"
  ```

  Below is a comprehensive walkthrough of all major metrics logged during MegatronBridge training, grouped by their original grouping in the codebase.

  ---

  **Core Training Metrics** (logged every `tensorboard_log_interval` steps by the training loop)

  | Metric | Unit | Description | How obtained |
  |---|---|---|---|
  | `lm loss` | nats | Language model cross-entropy loss. Primary training signal. | Computed in `forward_step()` loss function, averaged over tokens, reduced across DP ranks and microbatches. |
  | `mtp_1 loss`, `mtp_2 loss`, ... | nats | Per-head auxiliary loss for Multi-Token Prediction (only when MTP enabled). | Each MTP prediction head returns a separate loss via `MTPLossLoggingHelper`. |
  | `learning-rate` | float | Current learning rate from the LR scheduler. | Read from the optimizer's param scheduler via `get_canonical_lr_for_logging()`. |
  | `grad-norm` | float | Global L2 norm of all gradients after clipping. | Computed during the optimizer step as part of gradient clipping. |
  | `params-norm` | float | L2 norm of all model parameters. | Computed via `calc_params_l2_norm()` over all model parameters with `requires_grad`. |
  | `loss-scale` | float | Dynamic loss scaler value for mixed-precision training. | Read from the optimizer's internal AMP loss scaler via `optimizer.get_loss_scale()`. |
  | `batch-size` | count | Global batch size for the step (= `micro_batch_size * data_parallel_size * num_microbatches`). | Computed from config. |
  | `iteration-time` | seconds | Wall-clock time for one training step. | Derived: `elapsed_time / total_iterations` over the logging interval. |
  | `samples vs steps` | count | Cumulative consumed training samples, indexed by global step. | Tracked by the training state counter. |
  | `skipped-train-samples` | count | Samples skipped due to loss-scale underflow or NaN detection. | Tracked by the training state counter. |

  Interpretation: If `loss-scale` is monotonically decreasing and hitting the minimum floor, gradients contain too many NaN/Inf values — a sign of training instability. If `grad-norm` spikes without a corresponding loss spike, a single layer may be exploding (check per-layer norms below).

  ---

  **Throughput Metrics** (rolling window average, from `report_throughput()`)

  These are rolling averages over a window (default: 20 steps, set by `cfg.logger.throughput_window_size`).

  | Metric | Unit | Scope | How obtained |
  |---|---|---|---|
  | `throughput/batches_per_sec` | batches/sec | Global (all GPUs) | Derived: `elapsed_batches / elapsed_wall_clock_time` over the rolling window. |
  | `throughput/samples_per_sec` | samples/sec | Global | Derived: `elapsed_samples / elapsed_wall_clock_time`. |
  | `throughput/tokens_per_sec` | tokens/sec | Global | Derived: `elapsed_tokens / elapsed_wall_clock_time`. |
  | `throughput/device/batches_per_sec` | batches/sec | Per GPU | Derived: global value / `world_size`. |
  | `throughput/device/samples_per_sec` | samples/sec | Per GPU | Derived: global value / `world_size`. |
  | `throughput/device/tokens_per_sec` | tokens/sec | Per GPU | Derived: global value / `world_size`. |
  | `throughput/tflops/device` | TFLOP/s | Per GPU | Derived from model FLOP count (see formula below). |
  | `throughput/tflops` | TFLOP/s | Global (all GPUs) | `throughput/tflops/device * world_size`. |

  `throughput/tflops/device` is the key efficiency indicator — it tells you how well you're utilizing the hardware. The formula:

  ```
  throughput/tflops/device = num_floating_point_operations(config, batch_size)
                             / elapsed_time_per_iteration
                             / world_size
                             / 1e12
  ```

  Where `num_floating_point_operations` counts the FLOPs for one training step (forward + backward). For a standard Transformer model, the FLOP count is:

  ```
  flops = batch_size * seq_length * (
      # Attention (per layer): Q/K/V projections + attention scores + output projection
      12 * num_layers * hidden_size^2 * (1 + GQA_ratio + seq_length / (2 * hidden_size))
      # MLP (per layer): up-projection + down-projection (3/2x for SwiGLU gating)
    + 12 * num_layers * hidden_size * ffn_hidden_size * gated_multiplier
      # Logits: embedding-to-vocabulary projection (+ MTP heads if enabled)
    + 6  * hidden_size * padded_vocab_size * (1 + mtp_num_layers)
  )
  ```

  The 12x factor comes from: 3x (forward + backward wgrad + backward dgrad) * 2x (two stacked GEMMs per block) * 2x (multiply-accumulate = 2 FLOPs per element). See [Narayanan et al., 2021, Appendix](https://arxiv.org/abs/2104.04473) for the derivation.

  Interpretation: For H100 GPUs with BF16, well-optimized configs achieve 150-180 MODEL_TFLOP/s per GPU. A sudden 30% drop usually indicates a straggler node (check per-node NCCL logs) or a data loading bottleneck.

  ---

  **Memory Metrics** (from `report_memory()`, converted to GB)

  All memory metrics are read from `torch.cuda.memory_stats()` and converted from bytes to gigabytes (`GB = bytes / 1e9`).

  | Metric | Description | Use case |
  |---|---|---|
  | `memory/current_allocated_gigabytes` | Currently allocated GPU memory. | Baseline memory footprint per step. |
  | `memory/current_active_gigabytes` | Currently active (in-use) memory. | Memory actually being used by tensors right now. |
  | `memory/current_inactive_gigabytes` | Inactive but non-releasable memory. | Fragmentation indicator — high values mean the allocator is holding blocks it can't free. |
  | `memory/current_reserved_gigabytes` | Reserved by PyTorch's CUDA caching allocator. | Total memory claimed from the GPU driver. |
  | `memory/peak_allocated_gigabytes` | High-water mark for allocated memory. | Maximum memory needed during training (usually during backward pass). |
  | `memory/peak_active_gigabytes` | High-water mark for active memory. | Maximum simultaneously active tensors. |
  | `memory/peak_inactive_gigabytes` | High-water mark for inactive memory. | Worst-case fragmentation. |
  | `memory/peak_reserved_gigabytes` | High-water mark for reserved memory. | How close you are to OOM — compare with GPU total memory. |
  | `memory/alloc_retries` | Count of failed `cudaMalloc` calls that triggered a cache flush and retry. | `alloc_retries > 0` means the allocator had to evict cached blocks to satisfy a request — a warning sign you are near OOM. |

  ---

  **Runtime Metrics** (from `report_runtime()`, time unit = hours)

  | Metric | Unit | Description | How obtained |
  |---|---|---|---|
  | `time/remaining_estimate` | hours | Estimated time to `train_iters` completion. | Derived: `(elapsed_time / elapsed_fraction) * remaining_fraction`, where `elapsed_fraction = current_step / train_iters`. |
  | `time/tokens` | count | Total consumed tokens so far. | Derived: `consumed_train_samples * seq_length`. |
  | `time/samples` | count | Total consumed samples so far. | Directly from training state counter. |
  | `time/batches` | count | Total consumed batches (= global step). | Directly from training state counter. |
  | `time/total` | hours | Total elapsed wall-clock training time. | Derived: `(time.time() - start_time) / 3600`. |

  ---

  **Per-Layer Gradient Norms** (from `report_l2_norm_grad()`)

  | Metric | Description |
  |---|---|
  | `l2_norm/grad/global` | L2 norm across all model parameters. Equivalent to `grad-norm` but computed after gradient unscaling. |
  | `l2_norm/grad/<layer_name>` | L2 norm for each individual named parameter that has a gradient (e.g., `l2_norm/grad/decoder.layers.0.self_attention.linear_qkv.weight`). |

  These norms are computed by iterating over `model.named_parameters()` and calling `torch.linalg.vector_norm(p.main_grad)` for each parameter. The global norm is `sqrt(sum(per_layer_norm^2))`.

  Interpretation: This is the most granular diagnostic for gradient health. If `grad-norm` spikes, use per-layer norms to identify the responsible layer. Common culprits: embedding layers, the final LM head, or the first attention layer. Watch for layers where the norm is orders of magnitude larger than peers — this indicates localized instability.

  ---

  **Energy Metrics** (hardware-dependent, requires NVML support)

  | Metric | Unit | Description | How obtained |
  |---|---|---|---|
  | `iter-energy/gpu` | Joules/iter/GPU | Energy consumed per training iteration per GPU. | Derived: `energy_monitor.lap() / total_iterations / world_size`. The energy monitor reads cumulative GPU energy via NVML. |
  | `power/gpu` | Watts/GPU | Instantaneous power draw per GPU. | Derived: `energy / elapsed_time_per_iteration`. |

  These metrics require compatible GPU hardware and drivers (NVML). On systems without NVML support, they are simply not logged.

  ---

  **Validation Metrics** (logged every `eval_interval` steps)

  | Metric | When logged |
  |---|---|
  | `lm loss validation` | Every `eval_interval` steps (single validation set). |
  | `lm loss validation <dataset_name>` | Per-dataset validation loss (when `multiple_validation_sets: true`). The `<dataset_name>` is the basename of the validation data path. |
  | `lm loss validation (aggregated)` | Mean across all validation datasets. |

  See Section 3 below for the rationale behind multiple validation datasets and how to detect domain collapse.

- **Exercise Steps**:
  1. Open a real W&B run for one of our training jobs. Find the W&B run using `Group = {SWEEP_NAME}` and `Name = {JOB_NAME}-{JOB_ID}`. Identify which metric groups are present.
  2. Locate where `WANDB_EXP_NAME` is set in `launcher.py` and trace how it flows into `cfg.logger.wandb_exp_name` in `resolve_config.py`.
  3. Look at `throughput/tflops/device` over the first 100 steps of a run — does it stabilize? What causes the initial ramp-up?
  4. Compare `grad-norm` across two training runs with different learning rates — what do high vs. healthy grad-norm curves look like?
  5. If `loss-scale` is monotonically decreasing and hitting the minimum floor, what does that indicate about training stability?
  6. Find `report_throughput()` and `report_memory()` in the MegatronBridge `train_utils.py` source. What is the `throughput_window_size`? How would changing it affect the smoothness of the throughput curve?
  7. Calculate the expected `throughput/tflops/device` for a known model config (e.g., Qwen3-4B with `seq_length=8192`, `GBS=1024`) using the TFLOP formula above. Compare your calculation with the actual W&B value — what accounts for the difference?

### 3. Multiple Validation Datasets
- **Questions**: If you're training on a mix of English Wikipedia, Malay news, and math code, and validation loss improves overall — how do you know the model isn't getting worse on math? Why might a single aggregated validation loss hide important regression? What's the minimum number of validation sets you'd want for a multilingual continual pretraining run?
- **Intuition**: LLM pretraining uses mixed data from many domains. A single validation set gives one number that hides per-domain behavior. We enable multiple validation datasets via `multiple_validation_sets: true` in the config plus a list of separate `valid_data` paths. During evaluation (every `eval_interval` steps), the trainer iterates over each validation dataset independently and logs per-dataset losses. W&B metrics follow the pattern `lm loss validation <dataset_name>` (where `<dataset_name>` is the path basename), plus `lm loss validation (aggregated)` (mean across all datasets). This is critical for catching **domain collapse** — a failure mode where training on a new domain pushes the model to forget previously learned domains, which the aggregated loss may not reveal until it's severe.
- **Exercise Steps**:
  1. Find `multiple_validation_sets` in `resolve_config.py` and `launcher.py` — trace the full path from the `--multival` CLI flag through the config preset to the final `cfg.dataset.multiple_validation_sets = True`.
  2. On a W&B run with multi-validation enabled, plot `lm loss validation EN_Wikipedia` and `lm loss validation MY_Fineweb2` on the same chart alongside `lm loss validation (aggregated)` — do they diverge?
  3. Design a scenario where `lm loss validation (aggregated)` improves while one domain's loss gets worse. What type of training data mix would cause this?
  4. What is `eval_iters` and how does it affect the reliability of per-dataset validation loss estimates? What's the tradeoff with evaluation frequency?
