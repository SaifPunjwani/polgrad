"""Length-bias diagnostic: regress per-sequence objective weight on sequence length.

The aggregation mode decides how hard each sequence pulls on the scalar loss. The probe
forms the per-sequence absolute weighted-advantage mass

    y_i = Σ_t m_{i,t} · |A_{i,t} · w_{i,t}|,        x_i = L_i = Σ_t m_{i,t},

with ``w = effective_token_weights(response_mask, agg_mode, norm_len=norm_len)``, and
fits the ordinary-least-squares line ``y = β₀ + β₁·x`` with an HC3
heteroscedasticity-robust standard error for the slope, in closed-form torch (no scipy
at runtime). A slope compatible with zero means sequence length does not predict how
much loss mass a sequence carries; ``docs/diagnostics/length_bias.md`` derives the
estimator and reads the sign of the slope under each aggregation mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from polgrad._validation import broadcast_advantages, check_mask
from polgrad.aggregate import Aggregation, effective_token_weights

__all__ = ["LengthBiasReport", "length_bias_probe"]

# Two-sided 95% normal quantile: z with Φ(z) = 0.975, used for the CI endpoints.
_Z_975 = float(torch.special.ndtri(torch.tensor(0.975, dtype=torch.float64)))


@dataclass(frozen=True)
class LengthBiasReport:
    """OLS length-bias regression of per-sequence loss mass on sequence length.

    Attributes:
        slope: OLS slope ``β̂₁`` of ``y_i = Σ_t m·|A·w|`` on ``x_i = L_i``.
        slope_se: HC3 heteroscedasticity-robust standard error of the slope,
            ``sqrt(Σ_i (e_i/(1-h_i))² · (x_i-x̄)²) / Σ_i (x_i-x̄)²``.
        ci_low: ``slope - z·slope_se`` with ``z = Φ⁻¹(0.975)`` (95% normal-approx CI).
        ci_high: ``slope + z·slope_se``.
        intercept: OLS intercept ``β̂₀ = ȳ - β̂₁·x̄``.
        n: Number of sequences ``B`` entering the regression.
        per_seq_weight_sum: ``[B]`` float64 ``Σ_t w_{i,t}`` from
            :func:`polgrad.aggregate.effective_token_weights` — the total pull of each
            sequence on the aggregated loss.
        per_seq_length: ``[B]`` int64 response-token counts ``L_i = Σ_t m_{i,t}``.

    References:
        docs/diagnostics/length_bias.md; enforced by
        ``tests/test_diagnostics_length_bias.py::test_hc3_slope_se_matches_hand_computed_golden_case``.
    """

    slope: float
    slope_se: float
    ci_low: float
    ci_high: float
    intercept: float
    n: int
    per_seq_weight_sum: Tensor
    per_seq_length: Tensor

    def summary(self) -> str:
        """Return a compact human-readable multi-line description of the report."""
        return (
            f"length-bias probe: slope={self.slope:.4g} (HC3 se={self.slope_se:.4g},"
            f" 95% CI [{self.ci_low:.4g}, {self.ci_high:.4g}]),"
            f" intercept={self.intercept:.4g}, n={self.n}\n"
            f"per-seq weight sums in [{float(self.per_seq_weight_sum.min()):.4g},"
            f" {float(self.per_seq_weight_sum.max()):.4g}];"
            f" lengths in [{int(self.per_seq_length.min())}, {int(self.per_seq_length.max())}]"
        )


def _check_regressor_nondegenerate(lengths: Tensor) -> None:
    """Reject length patterns where the slope or its HC3 variance is undefined.

    Constant lengths give ``Σ(x_i - x̄)² = 0`` (no slope). Exactly one sequence at a
    distinct length among otherwise equal lengths gives that observation leverage
    ``h_i = 1`` (algebra in docs/diagnostics/length_bias.md), so HC3's
    ``1/(1 - h_i)²`` factor is undefined. Lengths are integers, so both conditions are
    checked exactly.
    """
    counts = torch.bincount(lengths)
    positive = counts[counts > 0]
    if positive.numel() == 1:
        raise ValueError(
            f"per_seq_length is constant (every sequence has {int(lengths[0])} response "
            "tokens); the length-bias slope is undefined"
        )
    if positive.numel() == 2 and int(positive.min()) == 1:
        raise ValueError(
            "HC3 is undefined when exactly one sequence has a distinct length "
            f"(leverage h = 1); got per-sequence lengths {lengths.tolist()}"
        )


def length_bias_probe(
    advantages: Tensor,
    response_mask: Tensor,
    *,
    agg_mode: Aggregation,
    norm_len: int | None = None,
) -> LengthBiasReport:
    """Regress per-sequence absolute loss mass on sequence length (OLS + HC3).

    With ``w = effective_token_weights(response_mask, agg_mode, norm_len=norm_len)``
    and ``[B]`` advantages broadcast across their row's tokens first, the probe fits

        y_i = Σ_t m_{i,t}·|A_{i,t}·w_{i,t}|,   x_i = L_i = Σ_t m_{i,t},
        β̂₁ = Σ_i (x_i-x̄)(y_i-ȳ) / S_xx,       S_xx = Σ_i (x_i-x̄)²,
        β̂₀ = ȳ - β̂₁·x̄,                        e_i = y_i - β̂₀ - β̂₁·x_i,
        h_i = 1/n + (x_i-x̄)²/S_xx,
        se(β̂₁) = sqrt( Σ_i (e_i/(1-h_i))² · (x_i-x̄)² ) / S_xx,
        CI = β̂₁ ± z·se(β̂₁),   z = Φ⁻¹(0.975),

    all in float64 torch. ``se(β̂₁)`` is the slope entry of the HC3 sandwich
    ``(XᵀX)⁻¹ Xᵀ diag(e²/(1-h)²) X (XᵀX)⁻¹`` reduced to closed form for a simple
    regression (docs/diagnostics/length_bias.md shows the algebra). A positive slope
    means longer sequences carry more absolute objective mass; what that implies per
    aggregation mode is derived on the docs page.

    Args:
        advantages: ``[B]`` per-sequence or ``[B, T]`` per-token advantages; masked
            positions of a ``[B, T]`` input are ignored and may hold any value.
        response_mask: ``[B, T]`` bool mask of response tokens.
        agg_mode: Aggregation mode whose effective token weights are probed.
        norm_len: Fixed generation budget; required iff ``agg_mode`` is
            ``Aggregation.TOKEN_SUM_NORM`` and ignored otherwise.

    Returns:
        A :class:`LengthBiasReport` with float64 statistics computed from detached
        inputs.

    Raises:
        ValueError: If the mask is invalid, ``advantages`` has the wrong shape or a
            non-finite response value, ``B < 3`` (no residual degrees of freedom with
            two parameters), all sequence lengths are equal (slope undefined), exactly
            one sequence has a distinct length (HC3 undefined at leverage ``h = 1``),
            or ``norm_len`` is missing while ``agg_mode`` is ``TOKEN_SUM_NORM``.

    References:
        docs/diagnostics/length_bias.md; enforced by
        ``tests/test_diagnostics_length_bias.py::test_hc3_slope_se_matches_hand_computed_golden_case``,
        ``tests/test_diagnostics_length_bias.py::test_ci_covers_known_slope_in_about_95_percent_of_runs``,
        ``tests/test_diagnostics_length_bias.py::test_mode_induced_weights_and_structural_slopes``.
    """
    check_mask(response_mask, like=response_mask)
    adv_tok = broadcast_advantages(
        advantages,
        response_mask,
        response_mask,
        like_name="response_mask",
        batch_mismatch_template="advantages has shape {adv_shape} but response_mask has B={b} rows",
        shape_mismatch_template="advantages must be [B] or [B, T] = {like_shape}; "
        "got shape {adv_shape}",
    )
    n = int(response_mask.shape[0])
    if n < 3:
        raise ValueError(
            "length_bias_probe needs at least 3 sequences (two regression parameters "
            f"leave n - 2 residual degrees of freedom); got B={n}"
        )
    lengths = response_mask.sum(dim=1)
    _check_regressor_nondegenerate(lengths)
    weights = effective_token_weights(response_mask, agg_mode, norm_len=norm_len)

    # Diagnostic only: the probe must never route gradients back into the advantages.
    adv64 = adv_tok.detach().to(torch.float64)
    zero = torch.zeros((), dtype=torch.float64)
    y = (torch.where(response_mask, adv64, zero).abs() * weights).sum(dim=1)
    x = lengths.to(torch.float64)

    x_bar = x.mean()
    y_bar = y.mean()
    dx = x - x_bar
    sxx = (dx * dx).sum()
    slope = (dx * (y - y_bar)).sum() / sxx
    intercept = y_bar - slope * x_bar
    resid = y - intercept - slope * x
    one_minus_h = 1.0 - 1.0 / n - dx * dx / sxx
    # HC3 residual weights e²/(1-h)²; leverage-one patterns were rejected above.
    omega = (resid / one_minus_h) ** 2
    se = float(torch.sqrt((omega * dx * dx).sum()) / sxx)
    slope_f = float(slope)
    half_width = _Z_975 * se
    return LengthBiasReport(
        slope=slope_f,
        slope_se=se,
        ci_low=slope_f - half_width,
        ci_high=slope_f + half_width,
        intercept=float(intercept),
        n=n,
        per_seq_weight_sum=weights.sum(dim=1),
        per_seq_length=lengths,
    )
