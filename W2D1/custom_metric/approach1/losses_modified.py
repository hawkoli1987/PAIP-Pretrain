# ---------------------------------------------------------------------------
# MODIFIED VERSION — Approach 1: Loss dict method
# Base file: megatron-bridge_yuli/src/megatron/bridge/training/losses.py
#
# Goal: add a new W&B metric "token entropy" by extending the reporting dict
# returned by the loss function.
#
# How it works:
#   The training loop (train_utils.py:553-557) iterates over every key in the
#   loss dict and calls wandb_writer.log({key: value}, iteration) for each.
#   Adding a key here is the only change needed — no W&B config, no training
#   loop modification required.
#
# The [value, count] tensor format:
#   Each entry must be a 2-element tensor: [sum_over_microbatch, num_tokens].
#   train_step() all-reduces sum and count across DP ranks, then divides
#   sum / count to get a properly token-weighted average before logging.
#
# Changes vs original (marked with # <-- ADDED):
#   1. entropy_sum computation after the loss sum
#   2. reporting_entropy tensor packed in [value, count] format
#   3. "token entropy" key added to the returned dict
#
# Entry point to use in training:
#   Pass forward_step_with_entropy (defined below) to pretrain() instead of
#   the default megatron.bridge.training.gpt_step.forward_step.
# ---------------------------------------------------------------------------

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from functools import partial
from typing import Iterable, Tuple

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine

from megatron.bridge.training.gpt_step import _forward_step_common
from megatron.bridge.training.state import GlobalState


_DEFAULT_SPIKY_LOSS_FACTOR: float = 10.0


def masked_next_token_loss(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor | Tuple[torch.Tensor],
    check_for_nan_in_loss: bool = True,
    check_for_spiky_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Loss function — extended to also report per-token entropy.

    Args:
        loss_mask: Used to mask out some portions of the loss
        output_tensor: The tensor with the losses.
        check_for_nan_in_loss: Whether to check for NaN values in the loss
        check_for_spiky_loss: Whether to check for spiky loss values

    Returns:
        tuple containing:
        - The loss scalar for this micro-batch
        - The number of non-padded tokens in this microbatch
        - Reporting dict:
            "lm loss"       — standard token-weighted cross-entropy (unchanged)
            "token entropy" — same quantity logged separately for independent
                              monitoring (e.g. to detect high-entropy outlier tokens)
    """
    if isinstance(output_tensor, tuple):
        losses = output_tensor[0].view(-1).float()
        loss_mask = output_tensor[1].view(-1).float()
    else:
        losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if check_for_nan_in_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
    if check_for_spiky_loss:
        spiky_loss_factor = getattr(rerun_state_machine, "spiky_loss_factor", _DEFAULT_SPIKY_LOSS_FACTOR)
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=spiky_loss_factor,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    # <-- ADDED: compute token entropy as a separately tracked metric
    # losses[i] is the per-token cross-entropy; loss_mask[i] = 1 for real tokens,
    # 0 for padding.  Summing the masked values gives the total entropy over this
    # microbatch.  After DP all-reduce and division by the total token count,
    # training_log() produces the mean per-token entropy logged to W&B.
    entropy_sum = (losses * loss_mask).sum()
    reporting_entropy = torch.cat([entropy_sum.clone().detach().view(1), num_tokens.view(1)])
    # --> end addition

    return (loss, num_tokens, {
        "lm loss": reporting_loss,
        "token entropy": reporting_entropy,  # <-- ADDED: new W&B metric
    })


def forward_step_with_entropy(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, partial]:
    """Drop-in replacement for megatron.bridge.training.gpt_step.forward_step.

    The only difference: uses the modified masked_next_token_loss above, which
    returns "token entropy" in addition to "lm loss".

    Usage in resolve_config.py:
        from approach1.losses_modified import forward_step_with_entropy
        pretrain(config=cfg, forward_step_func=forward_step_with_entropy)
    """
    output, loss_mask = _forward_step_common(state, data_iterator, model, return_schedule_plan)

    loss_function = partial(
        masked_next_token_loss,
        loss_mask,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
    )

    return output, loss_function


# ---------------------------------------------------------------------------
# Standalone unit test — run with: python losses_modified.py
# No MegatronBridge runtime required (mocks the rerun state machine).
# ---------------------------------------------------------------------------

def _test():
    import unittest.mock as mock

    class _FakeRSM:
        def validate_result(self, **_): pass
        def is_unexpectedly_large(self, **_): return False

    with mock.patch(
        "megatron.core.rerun_state_machine.get_rerun_state_machine",
        return_value=_FakeRSM(),
    ):
        losses = torch.rand(8) * 5.0
        loss_mask = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.float)

        loss_scalar, num_tokens, reporting_dict = masked_next_token_loss(
            loss_mask, losses.clone(), check_for_nan_in_loss=False, check_for_spiky_loss=False
        )

        assert set(reporting_dict.keys()) == {"lm loss", "token entropy"}, (
            f"Unexpected keys: {set(reporting_dict.keys())}"
        )
        for key, t in reporting_dict.items():
            assert t.shape == (2,), f"{key}: expected (2,), got {t.shape}"
            assert t[1].item() == 6, f"{key}: expected 6 tokens, got {t[1].item()}"

        lm_sum = reporting_dict["lm loss"][0].item()
        ent_sum = reporting_dict["token entropy"][0].item()
        assert abs(lm_sum - ent_sum) < 1e-4, f"lm_sum={lm_sum} != ent_sum={ent_sum}"

        print("PASS")
        print(f"  keys: {list(reporting_dict.keys())}")
        print(f"  lm loss sum:    {lm_sum:.4f}")
        print(f"  token entropy:  {ent_sum:.4f}")
        print(f"  num_tokens:     {num_tokens.item()}")


if __name__ == "__main__":
    _test()
