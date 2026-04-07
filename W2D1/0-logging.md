# Week 2, Day 1: Logs, Checkpointing, Resuming

## Key Knowledge-Points

### 1. Log Files and Log Directory Organization
- **Socratic questions**: When a multi-node training job crashes at 2am, how do you figure out which node failed? If you need to reproduce a run from 3 weeks ago, how do you recover the exact config and code that was used? Why does rank-0 show output in the terminal but other ranks don't?
- **Intuition**: Every training submission creates a dedicated `LOG_DIR` at `{shared_fs}/log/mb/{sweep_name}/{job_name}/{job_id}/`. The `{job_id}` makes each submission's logs unique — you always get a fresh directory even when resuming. The `CKPT_DIR` (at `{shared_fs}/ckpt/mb/{sweep_name}/{job_name}/`) deliberately omits `{job_id}` so resumed jobs continue writing to the same checkpoint. Within `LOG_DIR`, different files serve different debugging purposes:
  - **`slurm.log` / `pbs.log`**: scheduler-level stdout/stderr from the job script itself. First place to check for job launch failures or environment errors.
  - **`<rank>_python_master.log`**: torchrun output from rank-0, tee'd to both this file and the terminal. This is the main training log you watch live.
  - **`<rank>_python.log`** (non-rank-0 nodes): each worker's torchrun output, written to file only — not echoed to terminal. Essential for diagnosing per-node hangs or errors.
  - **`nccl/<hostname>.log`**: per-node NCCL debug log (one per host, via `NCCL_DEBUG=INFO`). Needed when diagnosing inter-node communication failures or NCCL hangs.
  - **`env/EnvVar_hostOS.log`** and **`env/EnvVar_<hostname>.log`**: full environment variable dump from host OS and from inside each container. Secrets (`WANDB_API_KEY`, `HF_TOKEN`) are filtered out. Useful for reproducing environment issues.
  - **`<rank>_<hostname>_sh.log`**: rank-0 only — records python path, megatron-core/bridge/transformer-engine version and install location. Critical for identifying framework version mismatches after upgrades.
  - **`recipe/original.yaml`** and **`recipe/override.yaml`**: snapshots of the training recipe before and after all overrides are applied. Lets you see exactly what was trained even if the source YAML changed later.
  - **`tb_logs/`**: TensorBoard event files, generated in parallel with W&B.
  - **Artifact copies**: `launch.slurm`, `launcher.py`, `resolve_config.py`, `config.yaml`, `data_config.yaml` — exact versions of all scripts and configs at submission time, making the run self-contained and reproducible.
- **Exercise Steps**:
  1. Navigate to a real `LOG_DIR` and list its contents; match each file to its purpose in the table above.
  2. Open `<rank>_python_master.log` and a non-master `<rank>_python.log` side-by-side — what's different? Why does only rank-0 get tee'd?
  3. Examine `env/EnvVar_hostOS.log` — which secrets are filtered out and how? Find the grep pattern in `launcher.py` that does the filtering.
  4. Compare `recipe/original.yaml` vs `recipe/override.yaml` — which fields changed? What does this tell you about how feature flags modify the base recipe?
  5. Explain why `LOG_DIR` includes `{job_id}` but `CKPT_DIR` does not — what would break if both used job_id, or if both omitted it?

### 2. W&B Integration and Metrics Dashboard
- **Socratic questions**: If `launcher.py` contains no `wandb.init()` calls, how does W&B know about the run? What does `WANDB_RUN_GROUP` control and why does it matter when you have dozens of experiments? If throughput suddenly drops by 30%, which W&B metric would you look at first?
- **Intuition**: W&B is wired entirely through environment variables set in the generated bash script — `WANDB_PROJECT`, `WANDB_RUN_GROUP` (= `SWEEP_NAME`), `WANDB_EXP_NAME` (= `{JOB_NAME}-{JOB_ID}`), `WANDB_MODE=online`, and `WANDB_DIR` (= `CKPT_DIR`). The actual `wandb.init()` and `wandb.log()` calls live inside `megatron.bridge.training.pretrain`. The launcher's `resolve_config.py` configures which metric groups to enable via flags like `log_throughput_to_tensorboard=True`, `log_memory_to_tensorboard=True`, etc. Metrics logged to W&B during training:
  - **Loss**: `lm loss` (language model cross-entropy, in nats). With MTP enabled: `mtp_1 loss`, `mtp_2 loss`, … (per prediction-head auxiliary loss).
  - **Optimization health**: `learning-rate` (float), `grad-norm` (global L2 norm of all gradients after clipping), `params-norm` (L2 norm of all parameters), `loss-scale` (FP16 dynamic loss scaler value).
  - **Throughput** (rolling 20-step window): `throughput/tokens_per_sec` (tokens/sec across all GPUs), `throughput/device/tokens_per_sec` (per-GPU), `throughput/tflops/device` (model TFLOP/s per GPU — key efficiency indicator).
  - **Memory** (GB): `memory/current_allocated_gigabytes`, `memory/peak_allocated_gigabytes`, `memory/peak_reserved_gigabytes`, and others. `memory/alloc_retries` counts failed cudaMalloc calls.
  - **Runtime**: `time/remaining_estimate` (hours), `time/total` (hours elapsed), `time/tokens` and `time/samples` (consumed so far).
  - **Per-layer gradient norms**: `l2_norm/grad/global` (whole model), `l2_norm/grad/<layer_name>` (per named parameter with grad). Useful for detecting gradient vanishing/explosion in specific layers.
  - **Energy** (hardware-dependent): `iter-energy/gpu` (Joules/iter/GPU), `power/gpu` (Watts/GPU).
- **Exercise Steps**:
  1. Open a real W&B run for one of our training jobs. Find the W&B run using `Group = {SWEEP_NAME}` and `Name = {JOB_NAME}-{JOB_ID}`. Identify which metric groups are present.
  2. Locate where `WANDB_EXP_NAME` is set in `launcher.py` and trace how it flows into `cfg.logger.wandb_exp_name` in `resolve_config.py`.
  3. Look at `throughput/tflops/device` over the first 100 steps of a run — does it stabilize? What causes the initial ramp-up?
  4. Compare `grad-norm` across two training runs with different learning rates — what do high vs. healthy grad-norm curves look like?
  5. If `loss-scale` is monotonically decreasing and hitting the minimum floor, what does that indicate about training stability?

### 3. Multiple Validation Datasets
- **Socratic questions**: If you're training on a mix of English Wikipedia, Malay news, and math code, and validation loss improves overall — how do you know the model isn't getting worse on math? Why might a single aggregated validation loss hide important regression? What's the minimum number of validation sets you'd want for a multilingual continual pretraining run?
- **Intuition**: LLM pretraining uses mixed data from many domains. A single validation set gives one number that hides per-domain behavior. We enable multiple validation datasets via `multiple_validation_sets: true` in the config plus a list of separate `valid_data` paths. During evaluation (every `eval_interval` steps), the trainer iterates over each validation dataset independently and logs per-dataset losses. W&B metrics follow the pattern `lm loss validation <dataset_name>` (where `<dataset_name>` is the path basename), plus `lm loss validation (aggregated)` (mean across all datasets). This is critical for catching **domain collapse** — a failure mode where training on a new domain pushes the model to forget previously learned domains, which the aggregated loss may not reveal until it's severe.
- **Exercise Steps**:
  1. Find `multiple_validation_sets` in `resolve_config.py` and `launcher.py` — trace the full path from the `--multival` CLI flag through the config preset to the final `cfg.dataset.multiple_validation_sets = True`.
  2. On a W&B run with multi-validation enabled, plot `lm loss validation EN_Wikipedia` and `lm loss validation MY_Fineweb2` on the same chart alongside `lm loss validation (aggregated)` — do they diverge?
  3. Design a scenario where `lm loss validation (aggregated)` improves while one domain's loss gets worse. What type of training data mix would cause this?
  4. What is `eval_iters` and how does it affect the reliability of per-dataset validation loss estimates? What's the tradeoff with evaluation frequency?
