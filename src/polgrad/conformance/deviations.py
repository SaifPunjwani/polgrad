"""Registry of demonstrated framework deviations from paper/reference semantics.

Every :class:`Deviation` entry is backed by a test in ``tests/test_conformance.py``
that demonstrates the exact analytic discrepancy against the vendored framework code
(``polgrad.conformance._vendor``) or the labeled reimplementations in
``polgrad.conformance.harness``, at the pinned commit named in ``version``; entries
without a demonstrating test are not registered.

Semantics note: at the vendored pinned commit ``74a718a4``, verl's
``agg_loss("seq-mean-token-sum-norm")`` divides by ``global_batch_size`` (which
defaults to the number of rows with at least one response token, i.e. ``B`` for
polgrad-valid masks) and then by ``loss_scale_factor`` (default: the padded width
``loss_mask.shape[-1]``), so the loss is ``Σ_b(Σ_t m·x) / (B · T_padded)``. Relative
to polgrad ``TOKEN_SUM_NORM = Σ/(B·norm_len)`` (Dr.GRPO) the factor is therefore
``norm_len/T_padded``, and that is the factor the registered entry and its test
assert: the default divisor is the batch-dependent padded length, not Dr.GRPO's
fixed generation budget.

Scope note: the group-normalized advantage estimators of verl and TRL could not be
vendored as pure functions (see the drop lists in the ``_vendor`` file headers), so
degenerate-group behavior (polgrad raises ``ValueError`` for size-1 groups; see
docs/derivations/advantages.md, degenerate groups) has no demonstrating test here and
is not registered as a :class:`Deviation`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DEVIATIONS", "Deviation"]

_VERL_PIN = "commit 74a718a492092312f1004fe25369975137388849 (main, fetched 2026-07-16)"
_OPENRLHF_PIN = "commit bc71bb19464aca306b33080b2d2bb45d154e2f49 (main, fetched 2026-07-16)"


@dataclass(frozen=True)
class Deviation:
    """One demonstrated divergence between a framework loss and reference semantics.

    Attributes:
        framework: Framework name as registered in ``harness.VENDORED``.
        version: Pinned upstream commit (and fetch date) the demonstration ran against.
        component: The upstream function/branch that deviates.
        description: Neutral statement of the analytic difference.
        demonstrated_by: pytest node id in ``tests/test_conformance.py`` that asserts
            the exact factor or gap.

    References:
        docs/derivations/aggregation.md (the reference aggregation semantics);
        tests/test_conformance.py::test_deviations_registry_node_ids_exist.
    """

    framework: str
    version: str
    component: str
    description: str
    demonstrated_by: str


DEVIATIONS: tuple[Deviation, ...] = (
    Deviation(
        framework="verl",
        version=_VERL_PIN,
        component='agg_loss(loss_agg_mode="seq-mean-token-sum-norm")',
        description=(
            "With the default loss_scale_factor the loss is Σ_b(Σ_t m·x) / (B · T_padded), "
            "where T_padded = loss_mask.shape[-1] is the padded width of the current batch. "
            "Dr.GRPO (arXiv 2503.20783) normalizes by the fixed generation budget: "
            "Σ_b(Σ_t m·x) / (B · norm_len), polgrad Aggregation.TOKEN_SUM_NORM. The verl "
            "default therefore equals the Dr.GRPO loss times norm_len/T_padded, a factor "
            "that varies with batch padding; passing loss_scale_factor=norm_len recovers "
            "Dr.GRPO exactly."
        ),
        demonstrated_by=(
            "tests/test_conformance.py::"
            "test_verl_token_sum_norm_deviates_from_dr_grpo_by_norm_len_over_padded_len"
        ),
    ),
    Deviation(
        framework="verl",
        version=_VERL_PIN,
        component='agg_loss(loss_agg_mode="seq-mean-token-mean")',
        description=(
            "The per-sequence token mean divides by (L_b + 1e-8) instead of the exact token "
            "count L_b, so each row's contribution is scaled by L_b/(L_b + 1e-8) relative to "
            "polgrad Aggregation.SEQ_MEAN_TOKEN_MEAN — an O(1e-8/L_b) relative deflation that "
            "makes the loss differ from the exact sequence mean of token means whenever it is "
            "nonzero."
        ),
        demonstrated_by=(
            "tests/test_conformance.py::"
            "test_verl_seq_mean_token_mean_deviates_by_row_epsilon_factor"
        ),
    ),
    Deviation(
        framework="openrlhf",
        version=_OPENRLHF_PIN,
        component="aggregate_loss(token_level_loss=False)",
        description=(
            "The per-sample reduction divides each row's token sum by (L_b + 1e-8) instead of "
            "L_b before averaging over rows, so the loss equals polgrad "
            "Aggregation.SEQ_MEAN_TOKEN_MEAN with each row scaled by L_b/(L_b + 1e-8) — the "
            "same O(1e-8/L_b) relative deflation as verl's seq-mean-token-mean mode."
        ),
        demonstrated_by=(
            "tests/test_conformance.py::test_openrlhf_sample_level_deviates_by_row_epsilon_factor"
        ),
    ),
    Deviation(
        framework="verl",
        version=_VERL_PIN,
        component="compute_policy_loss_gspo",
        description=(
            "The importance ratio is the GSPO-token form of the GSPO paper's eq. 14 "
            "(arXiv 2507.18071), log s_{i,t} = sg[mean masked log-ratio] + log_prob - "
            "sg[log_prob]: the sequence weight is detached, so the gradient is token-local "
            "(sg[s_i] * grad logprob_t, polgrad RatioKind.SEQUENCE_TOKEN) rather than "
            "flowing through the length-normalized mean as in the paper's eq. 7 sequence "
            "ratio (polgrad RatioKind.SEQUENCE). The loss value equals the eq.-7 form for "
            "any advantages, and the gradients coincide when advantages are constant within "
            "each row (the per-sequence group-normalized advantages of eq. 6); for "
            "per-token advantages the gradients differ, and no verl config restores the "
            "eq.-7 gradient. The function also clamps the combined log ratio at max=10.0 "
            "and always aggregates with agg_loss('seq-mean-token-mean') (ignoring its "
            "loss_agg_mode parameter), thereby inheriting the (L_b + 1e-8) row deflation "
            "registered above for that mode."
        ),
        demonstrated_by=(
            "tests/test_conformance.py::test_verl_gspo_gradient_is_token_local_gspo_token_form"
        ),
    ),
)
