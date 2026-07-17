"""Shared Hypothesis strategies for polgrad tests.

Module-specific strategies belong in the module's own test file; this file is not edited
by module implementations. Masked positions of every generated tensor are filled with a
sentinel junk value so that mask-invariance violations surface naturally in any test.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from hypothesis import strategies as st

MASKED_JUNK = 123.0


class LogprobBatch(NamedTuple):
    """A generated batch in polgrad's [B, T] right-padded convention (float64)."""

    logprobs: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    rollout_logprobs: torch.Tensor
    advantages: torch.Tensor
    response_mask: torch.Tensor


@st.composite
def padded_masks(
    draw: st.DrawFn, *, max_b: int = 8, max_t: int = 12, min_b: int = 1, min_t: int = 1
) -> torch.Tensor:
    """Right-padded boolean [B, T] masks with at least one true token per row."""
    b = draw(st.integers(min_b, max_b))
    t = draw(st.integers(min_t, max_t))
    lengths = [draw(st.integers(1, t)) for _ in range(b)]
    mask = torch.zeros((b, t), dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    return mask


def _fill(
    draw: st.DrawFn, mask: torch.Tensor, low: float, high: float, junk: float = MASKED_JUNK
) -> torch.Tensor:
    b, t = mask.shape
    vals = [
        draw(st.floats(low, high, allow_nan=False, allow_infinity=False, width=32))
        for _ in range(b * t)
    ]
    out = torch.tensor(vals, dtype=torch.float64).reshape(b, t)
    return torch.where(mask, out, torch.full_like(out, junk))


@st.composite
def logprob_batches(
    draw: st.DrawFn,
    *,
    max_b: int = 8,
    max_t: int = 12,
    max_gap: float = 2.0,
    seq_advantages: bool = False,
    max_abs_advantage: float = 3.0,
) -> LogprobBatch:
    """Batches of (logprobs, old, ref, rollout, advantages, mask).

    ``logprobs`` lie in [-8, -0.05]; the other streams are within ``max_gap`` of them so
    importance ratios stay in a numerically sane range (contract section 6). Advantages
    are [B] when ``seq_advantages`` else [B, T]. All tensors are float64; masked
    positions hold ``MASKED_JUNK``.
    """
    mask = draw(padded_masks(max_b=max_b, max_t=max_t))
    logprobs = _fill(draw, mask, -8.0, -0.05)

    def near(base: torch.Tensor) -> torch.Tensor:
        gap = _fill(draw, mask, -max_gap, max_gap, junk=0.0)
        return torch.where(mask, base + gap, torch.full_like(base, MASKED_JUNK))

    old_logprobs = near(logprobs)
    ref_logprobs = near(logprobs)
    rollout_logprobs = near(old_logprobs)
    if seq_advantages:
        b = mask.shape[0]
        vals = [
            draw(
                st.floats(
                    -max_abs_advantage,
                    max_abs_advantage,
                    allow_nan=False,
                    allow_infinity=False,
                    width=32,
                )
            )
            for _ in range(b)
        ]
        advantages = torch.tensor(vals, dtype=torch.float64)
    else:
        advantages = _fill(draw, mask, -max_abs_advantage, max_abs_advantage)
    return LogprobBatch(logprobs, old_logprobs, ref_logprobs, rollout_logprobs, advantages, mask)
