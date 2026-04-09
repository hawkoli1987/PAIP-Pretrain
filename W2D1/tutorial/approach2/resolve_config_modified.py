#!/usr/bin/env python3
"""resolve_config_modified.py — resolve_config.py with a custom W&B metric
added via the Callback system.

Approach 2: Callback method.
The GradParamRatioCallback defined here logs "grad_param_ratio" directly to
the W&B writer each training step.  No framework files are modified; the
callback is passed as an argument to pretrain().

Reference: megatron-bridge_yuli/docs/training/callbacks.md
"""

# ---------------------------------------------------------------------------
# MODIFIED VERSION — changes vs resolve_config_original.py:
#   1. Import Callback from megatron.bridge.training.callbacks        # <-- ADDED
#   2. Define GradParamRatioCallback class                            # <-- ADDED
#   3. Pass callbacks=[GradParamRatioCallback()] to pretrain()        # <-- CHANGED
# ---------------------------------------------------------------------------

from __future__ import annotations

# ... (config-mapping boilerplate unchanged — see full source) ...

import torch.distributed as dist

from megatron.bridge.training.callbacks import Callback             # <-- ADDED
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.gpt_step import forward_step


# <-- ADDED: custom callback class ----------------------------------------

class GradParamRatioCallback(Callback):
    """Log the ratio of gradient norm to parameter norm each training step.

    W&B metric name: "grad_param_ratio"

    Why this metric?
        grad_norm / param_norm is a dimensionless training health indicator:
        - Healthy range:    ~1e-3 to ~1e-2
        - Very small ratio: possible vanishing gradients
        - Very large ratio: possible instability / exploding gradients

    Why use a callback instead of the loss dict (Approach 1)?
        This metric is derived from optimizer state (grad_norm) and model
        parameters — not from the forward pass.  It has no meaningful
        per-token weighting, so it doesn't belong in the loss function's
        reduction pipeline.  Callbacks are the right home for metrics that
        are side-channel observations on the training state.

    Design notes:
        - W&B is initialized ONLY on the last rank (get_world_size() - 1).
          All other ranks have wandb_logger = None.  We guard on rank to
          avoid unnecessary computation on worker ranks.
        - We skip skipped iterations (gradient overflow / loss spike rerun)
          to avoid misleading ratio values when no optimizer step occurred.
        - param_norm is computed from the first model chunk only.  In
          pipeline-parallel training each rank holds a subset of layers;
          the ratio is still meaningful as a per-rank health indicator.
    """

    def on_train_step_end(self, context):
        # Only run on the last rank where W&B is initialized
        if dist.is_initialized() and dist.get_rank() != dist.get_world_size() - 1:
            return

        wandb_writer = context.state.wandb_logger
        grad_norm = context.grad_norm
        if wandb_writer is None or grad_norm is None or context.skipped_iter:
            return

        model_chunk = context.model[0]
        param_norm_sq = sum(
            p.norm().item() ** 2
            for p in model_chunk.parameters()
            if p.requires_grad
        )
        param_norm = param_norm_sq ** 0.5

        if param_norm > 0:
            ratio = grad_norm / param_norm
            step = context.state.train_state.step
            wandb_writer.log({"grad_param_ratio": ratio}, step)

# --> end addition ----------------------------------------------------------


def main():
    # ... (argument parsing and config resolution unchanged) ...

    # -------------------------------------------------------------------------
    # Launch pretraining with custom callback
    # "grad_param_ratio" will now appear on the W&B dashboard alongside the
    # built-in metrics (lm loss, throughput, memory, grad-norm, etc.).
    # -------------------------------------------------------------------------
    pretrain(                                        # <-- CHANGED
        config=cfg,
        forward_step_func=forward_step,
        callbacks=[GradParamRatioCallback()],        # <-- ADDED
    )


if __name__ == "__main__":
    main()
