#!/usr/bin/env python3
"""resolve_config.py — Map a resolved flat training YAML onto a Megatron-Bridge
ConfigContainer and run pretraining.

Called by torchrun via the launcher-generated job script.  Reads a fully
resolved training config YAML (presets + derived values already applied by
launcher.py) and maps the flat keys onto the recipe's ConfigContainer.

Only runtime env vars set by the bash job template (CKPT_DIR, LOG_DIR,
WANDB_*, DATA_ROOT) are read from the environment; all training params
come from the YAML.

Usage:
    torchrun ... resolve_config.py --config /path/to/resolved.yaml
"""

# ---------------------------------------------------------------------------
# ORIGINAL SOURCE (unmodified copy — key sections)
# Source: megatron_bridge/scripts/resolve_config.py
#
# Note: the full file is ~291 lines of config-mapping boilerplate.  Only the
# imports and the final pretrain() call are shown here — those are the lines
# that change in the modified version.
# ---------------------------------------------------------------------------

from __future__ import annotations

# ... (config-mapping boilerplate omitted for brevity — see full source) ...

from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.gpt_step import forward_step  # standard forward step


def main():
    # ... (argument parsing and config resolution omitted) ...

    # -------------------------------------------------------------------------
    # Launch pretraining
    # No callbacks — only the built-in metrics (lm loss, throughput, memory,
    # grad-norm, etc.) appear on W&B.
    # -------------------------------------------------------------------------
    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
