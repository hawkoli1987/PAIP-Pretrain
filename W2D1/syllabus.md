# Week 2, Day 1: Logs, Checkpointing, Resuming

## Key Knowledge-Points

### 1. Training Logging Infrastructure and Monitoring
- **Intuition**: Training runs for days/weeks and generates massive amounts of information (losses, metrics, system stats). Logging captures this information for monitoring, debugging, and analysis. Long training runs need monitoring to detect issues early (loss spikes, NaN values, OOM errors, hardware failures). Alerting notifies engineers when issues occur. This is required because we need to detect training issues (divergence, hardware failures) and track experiment progress, and undetected issues waste days of compute. In practice, logs are written to files (stdout/stderr) and/or monitoring systems (W&B, TensorBoard), with different verbosity levels for different audiences. We monitor loss trends, gradient norms, learning rate, throughput, and system metrics (GPU utilization, memory), with automated alerts for anomalies.
- **Exercise Steps**:
  1. Examine log files from a training run (stdout, stderr, W&B logs)
  2. Identify key information in logs: loss curves, learning rate, throughput, memory usage
  3. Write a script to parse and extract metrics from log files
  4. Practice setting up logging in a training script (Python logging module)
  5. Compare different log formats and choose appropriate verbosity levels
  6. Set up basic monitoring: log loss, learning rate, throughput to file
  7. Implement loss spike detection: alert if loss increases significantly
  8. Practice using W&B or TensorBoard to visualize training metrics
  9. Write a script to detect common training issues from logs (NaN, divergence, slowdown)
  10. Compare monitoring approaches: file-based vs. cloud-based (W&B) monitoring

### 2. Checkpoint Structure and Contents
- **Intuition**: Checkpoints save model state (weights, optimizer state, training step) to disk, enabling training to resume after interruptions. Checkpoints are large (GBs for large models) and must be saved efficiently. This is required because training can fail due to hardware issues, job time limits, or manual stops. In practice, checkpoints include model weights, optimizer states (momentum, Adam statistics), RNG states, and metadata (step, epoch, config).
- **Exercise Steps**:
  1. Examine a checkpoint directory structure (model weights, optimizer states, metadata)
  2. Load a checkpoint and inspect its contents (keys, shapes, data types)
  3. Compare checkpoint sizes: model-only vs. full checkpoint (with optimizer)
  4. Practice saving and loading checkpoints in a simple training loop
  5. Understand distributed checkpointing: how checkpoints are sharded across ranks

### 3. Checkpoint Saving Strategies
- **Intuition**: Checkpoints are expensive to save (I/O time, storage cost), so we use strategies: periodic saves (every N steps), keep only latest K checkpoints, save best checkpoints based on validation metrics. This is required because saving every step is infeasible (too slow, too much storage). In practice, we save every 1000-10000 steps, keep 2-3 recent checkpoints, and optionally save best validation checkpoints.
- **Exercise Steps**:
  1. Implement periodic checkpoint saving (save every N steps)
  2. Implement checkpoint rotation: keep only latest K checkpoints, delete older ones
  3. Implement best-checkpoint saving: save when validation metric improves
  4. Compare storage usage: saving all vs. rotating checkpoints
  5. Practice handling checkpoint save failures (retry logic, partial saves)

### 4. Resuming Training from Checkpoints
- **Intuition**: Resuming requires restoring exact training state: model weights, optimizer state, learning rate schedule position, data loader position, RNG state. This ensures training continues exactly as if it never stopped. This is required because training must be reproducible and continuous across job restarts. In practice, we load checkpoint, restore optimizer state, set training step, and resume data iteration from the correct position.
- **Exercise Steps**:
  1. Implement checkpoint loading in a training script
  2. Verify resumption correctness: compare loss curves from continuous vs. resumed training
  3. Practice handling missing checkpoints or corrupted checkpoints gracefully
  4. Implement data loader state restoration (resume from correct batch/step)
  5. Test resumption across different scenarios: step boundary, epoch boundary, mid-epoch

