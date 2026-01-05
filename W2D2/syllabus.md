# Week 2, Day 2: Scheduler, Optimizer, Learning Rate, GBS/MBS

## Key Knowledge-Points

### 1. Learning Rate Schedules
- **Intuition**: Learning rate must change during training: warmup (start small to stabilize), decay (reduce to fine-tune). Different schedules (linear, cosine, inverse-square-root) have different convergence properties. This is required because fixed learning rates either converge slowly (too small) or diverge (too large). In practice, we use warmup (1000-10000 steps) followed by decay, with schedules chosen based on dataset size and training length.
- **Exercise Steps**:
  1. Implement different LR schedules: constant, linear, cosine, inverse-square-root
  2. Plot LR curves over training steps for each schedule
  3. Compare schedules on a small training task (convergence speed, final loss)
  4. Practice implementing custom schedules (e.g., step-wise, polynomial)
  5. Understand WSD (weighted step decay) scheduler and its parameters

### 2. Optimizer Selection and Configuration
- **Intuition**: Optimizers (Adam, AdamW, SGD) have different convergence properties and memory requirements. Adam/AdamW are standard for LLMs due to adaptive learning rates. Weight decay regularizes to prevent overfitting. This is required because naive SGD converges slowly for large models. In practice, we use AdamW with β₁=0.9, β₂=0.95-0.999, weight decay=0.1-0.01, with optional gradient clipping.
- **Exercise Steps**:
  1. Compare optimizers: SGD, Adam, AdamW on a small task
  2. Experiment with optimizer hyperparameters: β₁, β₂, weight decay
  3. Implement gradient clipping and measure its effect on training stability
  4. Compare memory usage: Adam (stores momentum) vs. SGD
  5. Practice configuring optimizer in Megatron-Bridge (via config/recipe)

### 3. Global Batch Size (GBS) vs. Micro Batch Size (MBS)
- **Intuition**: GBS is the effective batch size across all devices (GBS = MBS × data parallelism × gradient accumulation). MBS is the batch size per device per forward pass. Gradient accumulation allows large GBS with limited memory. This is required because large GBS improves training stability and final performance, but is limited by GPU memory. In practice, we choose MBS to fit in GPU memory, then set GBS via data parallelism and gradient accumulation to achieve desired effective batch size (e.g., 512-2048 for large models).
- **Exercise Steps**:
  1. Understand the relationship: GBS = MBS × DP × GA (data parallelism × gradient accumulation)
  2. Calculate GBS for given MBS, number of GPUs, and gradient accumulation steps
  3. Experiment with different MBS values: measure memory usage and throughput
  4. Compare training dynamics: small GBS (64) vs. large GBS (1024) on a toy task
  5. Practice configuring GBS/MBS in training scripts and understand trade-offs

### 4. Learning Rate Scaling with Batch Size
- **Intuition**: Larger batch sizes require larger learning rates to maintain training dynamics (linear scaling rule: LR ∝ GBS). This is required because larger batches provide more stable gradients, allowing higher learning rates. In practice, we scale base LR linearly with GBS (e.g., LR = 1.6e-4 for GBS=512, so LR = 3.2e-4 for GBS=1024), with some schedules using square-root scaling for very large batches.
- **Exercise Steps**:
  1. Implement linear LR scaling: LR = base_LR × (GBS / base_GBS)
  2. Compare training with scaled vs. unscaled LR for different batch sizes
  3. Understand when to use linear vs. square-root scaling
  4. Practice calculating appropriate LR for a given GBS
  5. Visualize the relationship between GBS and optimal LR

### 5. Hyperparameter Tuning and Best Practices
- **Intuition**: Training hyperparameters (LR, warmup, decay schedule, GBS) interact and must be tuned together. Best practices come from empirical research and domain knowledge. This is required because suboptimal hyperparameters waste compute or prevent convergence. In practice, we start with proven configurations (e.g., from published papers), then tune based on validation metrics, with careful attention to learning rate and batch size relationship.
- **Exercise Steps**:
  1. Review hyperparameter configurations from published LLM training papers
  2. Set up a hyperparameter sweep: vary LR, warmup, GBS
  3. Compare training curves from different hyperparameter sets
  4. Practice reading training logs to diagnose hyperparameter issues (LR too high/low, insufficient warmup)
  5. Understand common pitfalls: LR too high (divergence), LR too low (slow convergence), insufficient warmup (instability)

