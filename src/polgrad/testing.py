"""Public conformance-testing API: certify a framework loss against polgrad references.

:func:`assert_conforms` lets a framework maintainer pin a policy-loss implementation to
polgrad's reference semantics in a few lines::

    from polgrad.testing import assert_conforms

    def test_my_grpo_loss_conforms():
        assert_conforms(my_grpo_loss, "grpo")

What is certified: the scalar loss value and its gradient with respect to ``logprobs``
on the deterministic seeded random batches of :func:`random_batches`. What is not
certified: advantage estimation (upstream of the loss — batches carry already-computed
``[B]`` advantages), KL-in-reward shaping, and data handling (mask construction,
prompt/response splitting, batching, sampling). A reference config's as-loss KL term is
stripped before comparison; see :func:`assert_conforms`.

Loss callables follow the keyword-tensor convention of
:func:`polgrad.conformance.harness.compare_losses`: they are called with keyword tensors
``(logprobs, old_logprobs, advantages, response_mask)`` and must return a scalar loss
differentiable in ``logprobs``. Randomness enters only through explicit integer seeds
via ``torch.Generator`` (docs/conventions.md); Hypothesis is not involved.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace

import torch
from torch import Tensor

from polgrad import registry
from polgrad.conformance.harness import _REL_FLOOR, DeviationReport, _loss_and_grad
from polgrad.losses import PolicyLossConfig, policy_loss

__all__ = [
    "assert_conforms",
    "random_batches",
]

# Sampling bounds shared with tests/strategies.py: logprobs uniform in
# [-8, -0.0625] keep exp() of any log-ratio gap far from overflow and underflow.
_LOGPROB_LOW = -8.0
_LOGPROB_HIGH = -0.0625
_MAX_ABS_ADVANTAGE = 3.0


def _validated_shapes(shapes: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Validate a shapes sequence exactly as ``compare_losses`` does."""
    shape_list: list[tuple[int, int]] = []
    for shape in shapes:
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
            raise ValueError(f"every shape must be (B, T) with B, T >= 1; got {shape}")
        shape_list.append((int(shape[0]), int(shape[1])))
    if not shape_list:
        raise ValueError("shapes must be a non-empty sequence of (B, T) pairs")
    return shape_list


def _sample_batch(
    shape: tuple[int, int], generator: torch.Generator, max_gap: float, dtype: torch.dtype
) -> dict[str, Tensor]:
    """Draw one batch in polgrad's ``[B, T]`` right-padded convention.

    Masked positions hold values from the same ranges as response positions, so a
    tested loss that lets padding leak into its output deviates visibly (the same
    implicit mask-invariance stress as ``harness._sample_case``).
    """
    b, t = shape
    lengths = torch.randint(1, t + 1, (b,), generator=generator)
    response_mask = torch.arange(t).unsqueeze(0) < lengths.unsqueeze(1)
    span = _LOGPROB_HIGH - _LOGPROB_LOW
    logprobs = torch.rand((b, t), generator=generator, dtype=dtype) * span + _LOGPROB_LOW

    def near(base: Tensor) -> Tensor:
        gap = (torch.rand((b, t), generator=generator, dtype=dtype) * 2.0 - 1.0) * max_gap
        return base + gap

    old_logprobs = near(logprobs)
    ref_logprobs = near(logprobs)
    rollout_logprobs = near(old_logprobs)
    advantages = (torch.rand((b,), generator=generator, dtype=dtype) * 2.0 - 1.0) * (
        _MAX_ABS_ADVANTAGE
    )
    return {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "ref_logprobs": ref_logprobs,
        "rollout_logprobs": rollout_logprobs,
        "advantages": advantages,
        "response_mask": response_mask,
    }


def _seeded_batches(
    n_cases: int,
    shapes: Sequence[tuple[int, int]],
    *,
    seed: int,
    max_gap: float,
    dtype: torch.dtype,
) -> Iterator[tuple[int, dict[str, Tensor]]]:
    """Yield ``(case_seed, batch)`` pairs; validates arguments eagerly.

    Per-case seeds are drawn from a master ``torch.Generator`` seeded with ``seed``
    (the same derivation as ``compare_losses``), so :func:`assert_conforms` can record
    the worst case's seed in its :class:`DeviationReport`.
    """
    if n_cases < 1:
        raise ValueError(f"n_cases must be >= 1; got {n_cases}")
    shape_list = _validated_shapes(shapes)
    if not dtype.is_floating_point:
        raise ValueError(f"dtype must be floating point; got {dtype}")
    if max_gap < 0.0:
        raise ValueError(f"max_gap must be >= 0; got {max_gap}")

    def generate() -> Iterator[tuple[int, dict[str, Tensor]]]:
        master = torch.Generator().manual_seed(seed)
        for index in range(n_cases):
            case_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=master).item())
            case_generator = torch.Generator().manual_seed(case_seed)
            shape = shape_list[index % len(shape_list)]
            yield case_seed, _sample_batch(shape, case_generator, max_gap, dtype)

    return generate()


def random_batches(
    n_cases: int,
    shapes: Sequence[tuple[int, int]],
    *,
    seed: int,
    max_gap: float = 2.0,
    dtype: torch.dtype = torch.float64,
) -> Iterator[dict[str, Tensor]]:
    """Yield deterministic random batches in polgrad's ``[B, T]`` right-padded convention.

    Each batch is a keyword-tensor dict with keys ``logprobs``, ``old_logprobs``,
    ``ref_logprobs``, ``rollout_logprobs`` (all ``[B, T]``), ``advantages`` (``[B]``),
    and ``response_mask`` (``[B, T]`` bool) — the calling convention of
    :func:`polgrad.conformance.harness.compare_losses`. A loss callable takes the
    tensors it needs as keyword arguments and ignores the rest.

    Sampling mirrors the bounds of ``tests/strategies.py``: right-padded masks with at
    least one true token per row; ``logprobs`` uniform in ``[-8, -0.0625]``;
    ``old_logprobs`` and ``ref_logprobs`` within ``max_gap`` of ``logprobs`` and
    ``rollout_logprobs`` within ``max_gap`` of ``old_logprobs`` (importance ratios stay
    within ``e^±max_gap`` of 1); ``advantages`` uniform in ``[-3, 3]``. Masked positions
    hold values from the same ranges, so mask-invariance violations in a tested loss
    surface naturally. Shapes are cycled in order across cases.

    These batches exercise the loss only: advantages are precomputed constants, so
    nothing here certifies advantage estimators, KL-in-reward shaping, or data handling.

    Args:
        n_cases: Number of batches to yield; must be ``>= 1``.
        shapes: Non-empty sequence of ``(B, T)`` shapes, each with ``B, T >= 1``.
        seed: Master seed; equal arguments give bitwise-identical tensors. Randomness
            flows only through ``torch.Generator`` (docs/conventions.md); no Hypothesis.
        max_gap: Maximum absolute log-ratio gap between the logprob streams; ``>= 0``.
        dtype: Floating dtype of the sampled float tensors.

    Yields:
        One keyword-tensor dict per case, in a deterministic order.

    Raises:
        ValueError: If ``n_cases < 1``, ``shapes`` is empty or holds a non-positive
            dimension, ``max_gap < 0``, or ``dtype`` is not floating point; raised
            eagerly at call time, before the first batch is drawn.

    References:
        docs/conventions.md (shapes, masking, determinism);
        tests/test_testing_api.py::test_random_batches_same_seed_is_bitwise_identical,
        tests/test_testing_api.py::test_random_batches_masks_shapes_and_bounds.
    """
    cases = _seeded_batches(n_cases, shapes, seed=seed, max_gap=max_gap, dtype=dtype)
    return (batch for _, batch in cases)


def _resolve_reference(reference: PolicyLossConfig | str) -> tuple[PolicyLossConfig, str]:
    """Resolve ``reference`` to a config plus a provenance line for the report notes."""
    if isinstance(reference, str):
        spec = registry.get(reference)
        return spec.loss, f"ALGORITHMS[{reference!r}].loss"
    if isinstance(reference, PolicyLossConfig):
        return reference, "explicit PolicyLossConfig"
    raise TypeError(
        f"reference must be a PolicyLossConfig or an ALGORITHMS key; got {type(reference)!r}"
    )


def _policy_loss_fn(config: PolicyLossConfig) -> Callable[..., Tensor]:
    """Wrap a config as a loss callable under the harness keyword-tensor convention."""

    def reference_fn(
        *, logprobs: Tensor, old_logprobs: Tensor, advantages: Tensor, response_mask: Tensor
    ) -> Tensor:
        return policy_loss(
            config,
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            response_mask=response_mask,
        ).loss

    return reference_fn


def _case_metrics(
    loss_a: float, grad_a: Tensor, loss_b: float, grad_b: Tensor
) -> tuple[float, float, float]:
    """Per-case ``(loss rel diff, grad rel diff, grad cosine)``.

    Implements exactly the per-case definitions documented on
    :class:`~polgrad.conformance.harness.DeviationReport` (relative differences floored
    at ``1e-12``; cosine ``1.0`` when both gradients are zero, ``0.0`` when exactly one
    is), so reports from :func:`assert_conforms` and ``compare_losses`` are comparable.
    """
    loss_rel = abs(loss_a - loss_b) / max(abs(loss_a), abs(loss_b), _REL_FLOOR)
    norm_a = float(grad_a.norm())
    norm_b = float(grad_b.norm())
    grad_rel = float((grad_a - grad_b).norm()) / max(norm_a, norm_b, _REL_FLOOR)
    if norm_a <= _REL_FLOOR and norm_b <= _REL_FLOOR:
        cosine = 1.0
    elif norm_a <= _REL_FLOOR or norm_b <= _REL_FLOOR:
        cosine = 0.0
    else:
        cosine = float((grad_a * grad_b).sum()) / (norm_a * norm_b)
    return loss_rel, grad_rel, cosine


def assert_conforms(
    fn: Callable[..., Tensor],
    reference: PolicyLossConfig | str,
    *,
    n_cases: int = 64,
    shapes: Sequence[tuple[int, int]] = ((4, 8), (2, 12)),
    seed: int = 0,
    loss_rtol: float = 1e-9,
    grad_rtol: float = 1e-9,
) -> DeviationReport:
    """Assert that ``fn`` reproduces a polgrad reference loss on seeded random batches.

    Runs ``fn`` and the polgrad reference (``policy_loss(config, ...).loss``) on the
    same :func:`random_batches` cases and compares, per case, the scalar loss and its
    gradient with respect to ``logprobs``. ``fn`` is called with the keyword tensors
    ``(logprobs, old_logprobs, advantages, response_mask)`` only — the harness
    convention — and must return a scalar differentiable in ``logprobs``.

    Certified on success: the loss value (within ``loss_rtol`` relative difference) and
    the ``logprobs`` gradient (within ``grad_rtol`` relative L2 difference) on the
    seeded random batches. Not certified: advantage estimators — advantage estimation
    is upstream of this check, the batches carry precomputed ``[B]`` advantages —
    KL-in-reward shaping, and data handling (mask construction, prompt/response
    splitting, batching, sampling).

    KL handling: when the reference config carries an as-loss KL term (for example
    ``ALGORITHMS["grpo"].loss``, whose KL needs ``ref_logprobs``), the term is stripped
    with ``dataclasses.replace(config, kl=None)`` before comparison, and the report
    notes record the stripping. ``assert_conforms`` therefore certifies the clipped
    surrogate without the KL term unless the config already has ``kl=None``. Configs
    with a truncated importance-sampling correction (``is_correction``) are rejected,
    because the four-tensor convention carries no ``rollout_logprobs``; compare those
    with :func:`polgrad.conformance.harness.compare_losses` directly.

    Args:
        fn: Loss callable under test (keyword tensors → scalar loss).
        reference: A :class:`~polgrad.losses.PolicyLossConfig`, or an
            ``polgrad.ALGORITHMS`` key such as ``"grpo"`` (the entry's ``.loss`` is
            used; its advantage estimator and KL placement are not exercised).
        n_cases: Number of seeded comparison cases.
        shapes: ``(B, T)`` batch shapes, cycled across cases.
        seed: Master seed passed to :func:`random_batches`; equal arguments compare
            bitwise-identical batches.
        loss_rtol: Maximum allowed per-case loss relative difference.
        grad_rtol: Maximum allowed per-case gradient relative L2 difference.

    Returns:
        The :class:`~polgrad.conformance.harness.DeviationReport` on success, with
        notes recording the reference, the batch parameters, the certified scope, and
        any KL stripping. ``worst_case_seed`` is the per-case seed drawn from
        ``torch.Generator().manual_seed(seed)`` exactly as in ``compare_losses``;
        ``random_batches(n_cases, shapes, seed=seed)`` regenerates the full suite.

    Raises:
        AssertionError: If either tolerance is exceeded; the message embeds
            ``DeviationReport.summary()``.
        ValueError: If ``reference`` names no registered algorithm, the config sets
            ``is_correction``, the batch parameters are invalid, ``fn`` returns a
            non-scalar, or the config is incomplete for :func:`policy_loss` (for
            example Dr.GRPO's ``norm_len=None``).
        TypeError: If ``reference`` is neither a ``PolicyLossConfig`` nor a ``str``.

    References:
        docs/conventions.md (shapes, signs, determinism);
        tests/test_testing_api.py::test_assert_conforms_passes_for_correct_grpo_fn,
        tests/test_testing_api.py::test_assert_conforms_fails_for_wrong_aggregation,
        tests/test_testing_api.py::test_assert_conforms_registry_key_strips_kl_and_passes.
    """
    config, reference_line = _resolve_reference(reference)
    if config.is_correction is not None:
        raise ValueError(
            "reference config sets is_correction (truncated importance sampling), which "
            "needs rollout_logprobs at call time; assert_conforms passes only (logprobs, "
            "old_logprobs, advantages, response_mask) — use "
            "polgrad.conformance.harness.compare_losses directly for TIS losses"
        )
    notes: list[str] = [
        f"reference: {reference_line}",
        f"config: surrogate={config.surrogate.value}, ratio={config.ratio.value}, "
        f"aggregation={config.aggregation.value}, clip={config.clip}",
        f"batches: polgrad.testing.random_batches(n_cases={n_cases}, "
        f"shapes={tuple(tuple(s) for s in shapes)}, seed={seed})",
        "certifies: loss value and d loss / d logprobs only; not advantage estimation "
        "(upstream of this check), not KL-in-reward, not data handling",
    ]
    if config.kl is not None:
        # Stripped, not defaulted silently elsewhere: the four-tensor convention carries
        # no ref_logprobs, so the as-loss KL term cannot be evaluated here.
        config = replace(config, kl=None)
        notes.append(
            "kl: reference config carried an as-loss KL term; it was stripped "
            "(dataclasses.replace(config, kl=None)) — the surrogate is certified "
            "without the KL term"
        )
    reference_fn = _policy_loss_fn(config)

    max_loss_rel = 0.0
    max_grad_rel = 0.0
    cosine_min = 1.0
    worst_seed = 0
    worst_score = -1.0
    for case_seed, batch in _seeded_batches(
        n_cases, shapes, seed=seed, max_gap=2.0, dtype=torch.float64
    ):
        loss_a, grad_a = _loss_and_grad(fn, "fn", batch)
        loss_b, grad_b = _loss_and_grad(reference_fn, "polgrad reference", batch)
        loss_rel, grad_rel, cosine = _case_metrics(loss_a, grad_a, loss_b, grad_b)
        max_loss_rel = max(max_loss_rel, loss_rel)
        max_grad_rel = max(max_grad_rel, grad_rel)
        cosine_min = min(cosine_min, cosine)
        score = max(loss_rel, grad_rel)
        if score > worst_score:
            worst_score = score
            worst_seed = case_seed

    report = DeviationReport(
        max_loss_rel_diff=max_loss_rel,
        max_grad_rel_diff=max_grad_rel,
        grad_cosine_min=cosine_min,
        n_cases=n_cases,
        worst_case_seed=worst_seed,
        notes=tuple(notes),
    )
    if max_loss_rel > loss_rtol or max_grad_rel > grad_rtol:
        raise AssertionError(
            f"polgrad conformance violation: max loss rel diff {max_loss_rel:.3e} "
            f"(loss_rtol {loss_rtol:.1e}), max grad rel diff {max_grad_rel:.3e} "
            f"(grad_rtol {grad_rtol:.1e})\n{report.summary()}"
        )
    return report
