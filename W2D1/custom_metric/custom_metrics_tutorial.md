# Custom Metrics Logging in MegatronBridge → W&B

This tutorial shows two ways to add a custom metric to a MegatronBridge training run
and have it appear on the W&B dashboard. It is grounded in our actual production
codebase, not a toy example.

**Reference docs (from `megatron-bridge_yuli`):**
- `docs/training/callbacks.md` — callback system
- `docs/training/logging.md` — logger configuration
- `docs/nemo2-migration-guide.md` — custom forward_step / loss dict pattern
- `src/megatron/bridge/training/losses.py` — standard loss function (starting point)
- `src/megatron/bridge/training/gpt_step.py` — standard forward step

---

## How metrics reach W&B: the full pipeline

```
forward_step()
  └─ returns (output_tensor, loss_function)
       └─ loss_function(output_tensor)
            └─ returns (loss, num_tokens, {"lm loss": reporting_loss, ...})
                 └─ train_step() reduces across microbatches + DP ranks
                      └─ training_log() iterates over loss_dict keys
                           └─ wandb_writer.log({key: value}, iteration)
                                └─ metric visible in W&B dashboard
```

The key insight: **every key in the dict returned by the loss function** becomes a W&B
metric automatically. You don't call `wandb.log()` yourself — the training loop handles
that at `train_utils.py:553–557`:

```python
for key in loss_dict:
    if wandb_writer:
        wandb_writer.log({key: loss_dict[key]}, iteration)
```

The reporting format for each key is `torch.cat([value.view(1), num_tokens.view(1)])` —
a 2-element tensor. The training loop divides `value / num_tokens` after all-reducing
across data-parallel ranks, giving a properly token-weighted average.

---

## Approach 1: Custom Loss Function (loss dict method)

**When to use:** The metric is naturally computed inside the forward/loss pass — e.g.,
a per-token entropy, a vocabulary diversity score, or a secondary loss term. It needs
proper distributed reduction across microbatches and DP ranks.

**What changes:** Only the loss function returned by `forward_step`. No modifications
to the training loop, optimizer, or W&B configuration.

### Original code (`losses.py`, simplified)

```python
def masked_next_token_loss(loss_mask, output_tensor, ...):
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {"lm loss": reporting_loss})
    #                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                          Only one key → only "lm loss" appears in W&B
```

### Modified code: add `token_entropy` metric

```python
def masked_next_token_loss_with_entropy(loss_mask, output_tensor, ...):
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # --- custom metric: mean per-token entropy over non-padded positions ---
    # losses holds per-token cross-entropy values; exp(CE) = perplexity per token,
    # but CE itself is already the entropy of the true distribution under the model.
    # We compute the masked average as a separate logged quantity.
    masked_entropies = losses * loss_mask          # zero out padding positions
    entropy_sum = masked_entropies.sum()
    # Pack as [value_sum, token_count] — same format as "lm loss"
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_entropy = torch.cat([entropy_sum.clone().detach().view(1), num_tokens.view(1)])
    # --- end custom metric ---

    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])
    return (loss, num_tokens, {
        "lm loss": reporting_loss,
        "token entropy": reporting_entropy,   # ← new key → new W&B metric
    })
```

To wire it into the training run, patch `forward_step` in `gpt_step.py` to use this
function instead of `masked_next_token_loss`. See the companion script
`custom_metrics_example.py` for a self-contained version.

**Result on W&B:** A new metric called `token entropy` appears alongside `lm loss`
in every training step. It is properly averaged over tokens and synchronized across
data-parallel ranks — no extra code needed.

---

## Approach 2: Callback (side-channel method)

**When to use:** The metric is computed outside the loss function — e.g., from model
weights, optimizer state, or an external measurement — and doesn't need distributed
reduction via the microbatch pipeline. Also useful for integrating with third-party
systems (Slack alerts, custom dashboards) without touching framework code.

**What changes:** Nothing in the framework. You subclass `Callback` and pass it to
`pretrain()`.

### Example: log gradient-to-parameter ratio per step

```python
import torch
import torch.distributed as dist
from megatron.bridge.training.callbacks import Callback


class GradParamRatioCallback(Callback):
    """Log the ratio of gradient norm to parameter norm each step.

    This is a simple training health indicator: a very small ratio suggests
    vanishing gradients; a very large ratio suggests instability.
    W&B metric name: "grad_param_ratio"
    """

    def on_train_step_end(self, context):
        # Only log on the last rank — same rank where W&B is initialized
        if not dist.is_initialized() or dist.get_rank() == dist.get_world_size() - 1:
            wandb_writer = context.state.wandb_logger
            grad_norm = context.grad_norm
            if wandb_writer is None or grad_norm is None:
                return

            # Compute parameter norm from the first model chunk
            model_chunk = context.model[0]
            param_norm = sum(
                p.norm().item() ** 2
                for p in model_chunk.parameters()
                if p.requires_grad
            ) ** 0.5

            if param_norm > 0:
                ratio = grad_norm / param_norm
                step = context.state.train_state.step
                wandb_writer.log({"grad_param_ratio": ratio}, step)
```

Wire it in:

```python
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.gpt_step import forward_step

pretrain(cfg, forward_step, callbacks=[GradParamRatioCallback()])
```

In our production setup (`resolve_config.py`), `pretrain()` is called inside
`megatron.bridge.training.pretrain.pretrain(cfg, forward_step, ...)`. You would add
the callback at the same call site.

**Key design rules for callbacks:**

| Rule | Reason |
|------|--------|
| Guard with `get_rank() == get_world_size() - 1` | W&B is only initialized on the last rank |
| Don't raise exceptions inside callbacks | Uncaught exceptions stop training |
| Access `context.state.wandb_logger` — not `wandb` directly | This is the already-initialized module; direct `import wandb` calls may log from the wrong rank |
| Use `context.user_state` to accumulate across steps | It persists for the full training run |

---

## Approach 3: Custom metric from existing MTP pattern (reference)

MegatronBridge's own Multi-Token Prediction (MTP) loss provides a third pattern: a
dedicated helper class `MTPLossLoggingHelper` that writes to both TensorBoard and W&B
from inside `training_log()`. This is used when the metric lives in Megatron-Core
(not in the loss function return dict).

You can see this at `train_utils.py:645–647`:

```python
if config.model.mtp_num_layers is not None:
    mtp_loss_scale = 1 / get_num_microbatches()
    MTPLossLoggingHelper.track_mtp_metrics(mtp_loss_scale, iteration, writer, wandb_writer, total_loss_dict)
```

This approach requires modifying `training_log()` in `train_utils.py`. Only use it if
you are adding framework-level metrics that every run should track, not experiment-level
custom metrics. For experiment-level metrics, use Approach 1 or 2.

---

## Comparison

| | Approach 1: Loss dict | Approach 2: Callback | Approach 3: Helper in training_log |
|---|---|---|---|
| **Where computed** | Inside forward/loss pass | Anywhere in training step | Inside training_log() |
| **Distributed reduction** | Automatic (DP + microbatch) | Manual (rank guard needed) | Manual |
| **Files changed** | `losses.py` + `gpt_step.py` | None (pass callback to pretrain) | `train_utils.py` |
| **Log cadence** | Every `log_interval` steps | Every step (or you control it) | Every `log_interval` steps |
| **Best for** | Per-token metrics, auxiliary losses | Optimizer/weight stats, alerts | Core framework metrics |

---

## Checklist: verifying your metric appears in W&B

1. Run training for at least `log_interval` steps (default: 100)
2. In W&B, open the run using `Group = {SWEEP_NAME}` → `Name = {JOB_NAME}-{JOB_ID}`
3. In the Charts panel, search for your metric key name
4. If missing: check `<rank>_python_master.log` for Python errors in your custom code
5. If NaN: your `num_tokens` may be 0 (empty batch) — add a guard: `if num_tokens > 0`
6. For Approach 1: confirm the key appears in the console log line printed every `log_interval` steps (MegatronBridge prints all `loss_dict` keys there)
