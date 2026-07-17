"""Machine-checked gradients: fp64 gradcheck of policy losses and derivation checks.

Two entry points. :func:`gradcheck_loss` runs ``torch.autograd.gradcheck`` on
:func:`polgrad.losses.policy_loss` over seeded random fp64 batches. Finite differences
see through ``.detach()``, so a loss with internal stop-gradients (CISPO's ``sg[ŵ]``,
GSPO-token's ``sg[s_i]``/``sg[r_t]``) cannot be finite-difference-checked directly;
gradcheck therefore runs on the sg-frozen equivalent derived in
``docs/derivations/losses.md`` (CISPO section), and the real config's autograd gradient
is asserted equal to the frozen equivalent's at the evaluation point.
:func:`check_gradient_formula` compares a hand-derived analytic gradient against
central finite differences of the loss value itself, which catches wrong derivations,
not just autograd inconsistency.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import torch
from torch import Tensor

from polgrad._validation import check_finite
from polgrad.kl import KLEstimator
from polgrad.losses import PolicyLossConfig, RatioKind, SurrogateKind, policy_loss

__all__ = ["check_gradient_formula", "gradcheck_loss"]

# Minimum distance of every ratio from every clip boundary (and of every KL delta from
# the |δ| kink), in ratio units; gradcheck perturbs logprobs by 1e-6, which moves ratios
# by a relative ~1e-6, so 1e-3 keeps every finite-difference sample on one branch.
_BRANCH_MARGIN = 1e-3
_MASKED_JUNK = 123.0
_MAX_DRAWS = 64


class _Batch(NamedTuple):
    logprobs: Tensor
    old_logprobs: Tensor
    advantages: Tensor
    response_mask: Tensor
    ref_logprobs: Tensor | None
    rollout_logprobs: Tensor | None


def _uniform(shape: tuple[int, ...], low: float, high: float, generator: torch.Generator) -> Tensor:
    return low + (high - low) * torch.rand(shape, generator=generator, dtype=torch.float64)


def _draw_mask(b: int, t: int, generator: torch.Generator, ragged: bool) -> Tensor:
    mask = torch.ones((b, t), dtype=torch.bool)
    if ragged and t > 1:
        lengths = torch.randint(1, t + 1, (b,), generator=generator)
        for i in range(b):
            mask[i, int(lengths[i]) :] = False
    return mask


def _surrogate_ratios(config: PolicyLossConfig, batch: _Batch) -> Tensor:
    """Flat tensor of the ratio values the surrogate branches on."""
    mask = batch.response_mask
    zero = torch.zeros((), dtype=torch.float64)
    log_ratio = torch.where(mask, batch.logprobs - batch.old_logprobs, zero)
    if config.ratio is RatioKind.TOKEN:
        return torch.exp(log_ratio[mask])
    lengths = mask.sum(dim=1, keepdim=True).to(torch.float64)
    return torch.exp(log_ratio.sum(dim=1, keepdim=True) / lengths).flatten()


def _branch_safe(config: PolicyLossConfig, batch: _Batch) -> bool:
    """True when no finite-difference perturbation can flip a clip branch or |δ| kink."""
    if config.surrogate is SurrogateKind.PG_CLIP and config.clip is not None:
        clip = config.clip
        ratios = _surrogate_ratios(config, batch)
        bounds = [
            1.0 - clip.eps_low if clip.eps_low is not None else None,
            1.0 + clip.eps_high if clip.eps_high is not None else None,
            clip.ratio_cap,
        ]
        for bound in bounds:
            if bound is not None and float((ratios - bound).abs().min()) < _BRANCH_MARGIN:
                return False
    if (
        config.kl is not None
        and config.kl.kind is KLEstimator.ABS
        and batch.ref_logprobs is not None
    ):
        delta = (batch.ref_logprobs - batch.logprobs)[batch.response_mask]
        if float(delta.abs().min()) < _BRANCH_MARGIN:
            return False
    return True


def _draw_batch(
    config: PolicyLossConfig, b: int, t: int, generator: torch.Generator, ragged: bool
) -> _Batch:
    """Seeded fp64 batch with junk at masked positions, resampled until branch-safe."""
    for _ in range(_MAX_DRAWS):
        mask = _draw_mask(b, t, generator, ragged)
        junk = torch.full((b, t), _MASKED_JUNK, dtype=torch.float64)
        logprobs = torch.where(mask, _uniform((b, t), -6.0, -0.1, generator), junk)
        old = torch.where(mask, logprobs + _uniform((b, t), -1.2, 1.2, generator), junk)
        advantages = torch.where(mask, _uniform((b, t), -3.0, 3.0, generator), junk)
        ref = (
            torch.where(mask, logprobs + _uniform((b, t), -1.2, 1.2, generator), junk)
            if config.kl is not None
            else None
        )
        rollout = (
            torch.where(mask, old + _uniform((b, t), -0.8, 0.8, generator), junk)
            if config.is_correction is not None
            else None
        )
        batch = _Batch(logprobs, old, advantages, mask, ref, rollout)
        if _branch_safe(config, batch):
            return batch
    raise RuntimeError(
        f"could not draw a branch-safe batch of shape {(b, t)} for {config!r} "
        f"after {_MAX_DRAWS} attempts"
    )


def _frozen_fn(config: PolicyLossConfig, batch: _Batch) -> Callable[[Tensor], Tensor]:
    """Return the sg-frozen gradcheck target for ``config`` at ``batch.logprobs``.

    CISPO becomes REINFORCE on advantages pre-scaled by the weight value ``ŵ(lp₀)``;
    GSPO-token becomes a TOKEN-ratio call against shifted ``old_logprobs`` chosen so
    ``exp(lp₀ - old') = s_i(lp₀)``. Configs without internal detach are returned as
    the plain :func:`polgrad.losses.policy_loss` call (docs/derivations/losses.md).
    """
    lp0, olp, adv, mask = batch.logprobs, batch.old_logprobs, batch.advantages, batch.response_mask
    ref, rollout = batch.ref_logprobs, batch.rollout_logprobs
    zero = torch.zeros((), dtype=torch.float64)
    z0 = torch.where(mask, lp0 - olp, zero)
    lengths = mask.sum(dim=1, keepdim=True).to(torch.float64)
    s0 = torch.exp(z0.sum(dim=1, keepdim=True) / lengths)
    if config.surrogate is SurrogateKind.CISPO:
        ratio0 = torch.exp(z0) if config.ratio is RatioKind.TOKEN else s0.expand_as(lp0)
        clip = config.clip
        assert clip is not None and clip.eps_high is not None  # validated by policy_loss
        high = 1.0 + clip.eps_high
        w0 = (
            ratio0.clamp(max=high)
            if clip.eps_low is None
            else ratio0.clamp(1.0 - clip.eps_low, high)
        )
        frozen = dataclasses.replace(
            config, surrogate=SurrogateKind.REINFORCE, ratio=RatioKind.TOKEN, clip=None
        )
        frozen_adv = w0 * torch.where(mask, adv, zero)

        def fn_cispo(x: Tensor) -> Tensor:
            return policy_loss(
                frozen,
                logprobs=x,
                old_logprobs=olp,
                advantages=frozen_adv,
                response_mask=mask,
                ref_logprobs=ref,
                rollout_logprobs=rollout,
            ).loss

        return fn_cispo
    if config.ratio is RatioKind.SEQUENCE_TOKEN:
        frozen = dataclasses.replace(config, ratio=RatioKind.TOKEN)
        shifted_old = lp0 - torch.log(s0)

        def fn_gspo_token(x: Tensor) -> Tensor:
            return policy_loss(
                frozen,
                logprobs=x,
                old_logprobs=shifted_old,
                advantages=adv,
                response_mask=mask,
                ref_logprobs=ref,
                rollout_logprobs=rollout,
            ).loss

        return fn_gspo_token

    def fn_plain(x: Tensor) -> Tensor:
        return policy_loss(
            config,
            logprobs=x,
            old_logprobs=olp,
            advantages=adv,
            response_mask=mask,
            ref_logprobs=ref,
            rollout_logprobs=rollout,
        ).loss

    return fn_plain


def gradcheck_loss(
    config: PolicyLossConfig,
    *,
    batch_shapes: Sequence[tuple[int, int]],
    generator: torch.Generator,
    ragged: bool = True,
) -> None:
    """Finite-difference-check ``policy_loss(config, ...)`` on seeded fp64 batches.

    For each ``(B, T)`` in ``batch_shapes``, draws a random batch (logprobs in
    ``[-6, -0.1]``, |logprob gaps| ≤ 1.2, advantages in ``[-3, 3]``, junk at masked
    positions, ratios kept ≥ 1e-3 from every clip boundary so finite differences never
    flip a branch), then (1) runs ``torch.autograd.gradcheck`` on the sg-frozen
    equivalent of the loss w.r.t. ``logprobs`` and (2) asserts the real config's
    autograd gradient equals the frozen equivalent's at the evaluation point
    (stop-gradient algebra in docs/derivations/losses.md). Returns ``None`` on success.

    Args:
        config: Loss specification; validated by :func:`polgrad.losses.policy_loss`.
        batch_shapes: Non-empty sequence of ``(B, T)`` with ``B, T >= 1``; keep
            ``B <= 8``, ``T <= 12`` (contract section 6 gradcheck bounds).
        generator: Explicit RNG; the same seed reproduces the same batches.
        ragged: Draw ragged right-padded masks when ``True``; all-true masks otherwise.

    Returns:
        ``None``; raises on any failure.

    Raises:
        ValueError: On empty or non-positive ``batch_shapes``, or any
            contract-section-4.3 config violation (propagated from ``policy_loss``).
        RuntimeError: From ``torch.autograd.gradcheck`` when autograd and finite
            differences disagree, or if no branch-safe batch is found.
        AssertionError: If the real config's gradient deviates from its
            stop-gradient-frozen equivalent at the evaluation point.

    References:
        docs/derivations/losses.md (sg-frozen gradcheck targets);
        docs/derivations/goldens.md;
        tests/test_verify.py::test_gradcheck_loss_passes_for_representative_configs.
    """
    shapes = tuple(batch_shapes)
    if not shapes:
        raise ValueError("batch_shapes must be a non-empty sequence of (B, T); got ()")
    for shape in shapes:
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
            raise ValueError(f"batch_shapes entries must be (B, T) with B, T >= 1; got {shape}")
    for b, t in shapes:
        batch = _draw_batch(config, b, t, generator, ragged)
        real_leaf = batch.logprobs.clone().requires_grad_(True)
        real = policy_loss(
            config,
            logprobs=real_leaf,
            old_logprobs=batch.old_logprobs,
            advantages=batch.advantages,
            response_mask=batch.response_mask,
            ref_logprobs=batch.ref_logprobs,
            rollout_logprobs=batch.rollout_logprobs,
        )
        (real_grad,) = torch.autograd.grad(real.loss, real_leaf)
        fn = _frozen_fn(config, batch)
        leaf = batch.logprobs.clone().requires_grad_(True)
        torch.autograd.gradcheck(fn, (leaf,))
        frozen_leaf = batch.logprobs.clone().requires_grad_(True)
        (frozen_grad,) = torch.autograd.grad(fn(frozen_leaf), frozen_leaf)
        if not torch.allclose(real_grad, frozen_grad, rtol=1e-10, atol=1e-10):
            max_diff = float((real_grad - frozen_grad).abs().max())
            raise AssertionError(
                f"policy_loss gradient deviates from its stop-gradient-frozen equivalent "
                f"for {config!r} at shape {(b, t)}: max |diff| = {max_diff:.3e}"
            )


def check_gradient_formula(
    fn: Callable[..., Tensor],
    analytic: Callable[..., Tensor],
    inputs: tuple[Tensor, ...],
    *,
    eps: float = 1e-6,
    atol: float,
    rtol: float,
) -> None:
    """Check a hand-derived gradient formula against central finite differences.

    Differentiates ``fn`` w.r.t. ``inputs[0]`` (the remaining inputs are held
    constant) with the central stencil ``g_i = (fn(x + ε e_i, ...) - fn(x - ε e_i,
    ...)) / (2ε)`` (error ``O(ε²)``), and compares elementwise against
    ``analytic(*inputs)``. Because the numerical side never consults autograd, this
    catches wrong derivations, not just autograd inconsistency. Both sides are
    upcast to ``float64`` explicitly for the comparison (docs/conventions.md); pass
    ``float64`` inputs so the stencil itself is accurate.

    Args:
        fn: Loss; ``fn(*inputs)`` must return a finite 0-dim tensor.
        analytic: Claimed gradient; ``analytic(*inputs)`` must return a finite tensor
            of ``inputs[0]``'s shape.
        inputs: Non-empty tuple of tensors; ``inputs[0]`` must be floating point.
        eps: Central-difference step; must be finite and positive.
        atol: Absolute tolerance of the elementwise comparison; finite, ``>= 0``.
        rtol: Relative tolerance of the elementwise comparison; finite, ``>= 0``.

    Returns:
        ``None``; raises on any failure.

    Raises:
        ValueError: On empty ``inputs``, a non-floating ``inputs[0]``, invalid
            ``eps``/``atol``/``rtol``, a non-scalar or non-finite ``fn`` output, or an
            ``analytic`` output whose shape differs from ``inputs[0]``.
        AssertionError: If the analytic gradient disagrees with the finite differences
            beyond ``atol``/``rtol``.

    References:
        docs/derivations/goldens.md;
        tests/test_verify.py::test_check_gradient_formula_accepts_correct_pg_derivation,
        tests/test_verify.py::test_check_gradient_formula_raises_on_wrong_derivation.
    """
    if not inputs:
        raise ValueError("inputs must be a non-empty tuple of tensors; got ()")
    x = inputs[0]
    if not torch.is_floating_point(x):
        raise ValueError(f"inputs[0] must be a floating-point tensor; got dtype {x.dtype}")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive; got {eps}")
    for name, tol in (("atol", atol), ("rtol", rtol)):
        if not math.isfinite(tol) or tol < 0.0:
            raise ValueError(f"{name} must be finite and non-negative; got {tol}")

    def scalar(out: Tensor, at: str) -> float:
        if out.dim() != 0:
            raise ValueError(
                f"fn must return a 0-dim (scalar) tensor; got shape {tuple(out.shape)} at {at}"
            )
        check_finite(f"fn output at {at}", out)
        return float(out)

    rest = inputs[1:]
    base = x.detach().clone()
    flat_base = base.reshape(-1)
    numeric = torch.zeros(flat_base.numel(), dtype=torch.float64)
    with torch.no_grad():
        for i in range(flat_base.numel()):
            plus = flat_base.clone()
            plus[i] += eps
            minus = flat_base.clone()
            minus[i] -= eps
            f_plus = scalar(fn(plus.reshape(x.shape), *rest), f"x + eps*e_{i}")
            f_minus = scalar(fn(minus.reshape(x.shape), *rest), f"x - eps*e_{i}")
            numeric[i] = (f_plus - f_minus) / (2.0 * eps)
    numeric_grad = numeric.reshape(x.shape)

    analytic_grad = analytic(*inputs)
    if analytic_grad.shape != x.shape:
        raise ValueError(
            f"analytic(*inputs) must have inputs[0]'s shape {tuple(x.shape)}; "
            f"got {tuple(analytic_grad.shape)}"
        )
    check_finite("analytic(*inputs)", analytic_grad)
    analytic64 = analytic_grad.detach().to(torch.float64)
    if not torch.allclose(analytic64, numeric_grad, rtol=rtol, atol=atol):
        diff = (analytic64 - numeric_grad).abs()
        idx = int(diff.argmax())
        raise AssertionError(
            f"analytic gradient disagrees with central finite differences: "
            f"max |analytic - numeric| = {float(diff.max()):.3e} at flat index {idx} "
            f"(analytic {float(analytic64.reshape(-1)[idx]):.6e}, "
            f"numeric {float(numeric_grad.reshape(-1)[idx]):.6e}; "
            f"eps={eps}, atol={atol}, rtol={rtol})"
        )
