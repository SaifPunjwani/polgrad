# Effective sample size (`polgrad.diagnostics.ess`)

When the trainer policy $\pi_\theta$ drifts away from the sampling policy
$\pi_{\theta_\text{old}}$, an importance-weighted update behaves as if it had fewer samples
than the batch contains. `importance_ess` and `sliding_ess` measure that loss with the
standard importance-sampling effective sample size.

## Weights

With per-token gaps $g_{i,t} = \log\pi_\text{new}(y_{i,t}) - \log\pi_\text{old}(y_{i,t})$ and
response mask $m$:

- **sequence level** (default): $w_i = \exp\!\big(\sum_t m_{i,t}\, g_{i,t}\big)$, $n = B$.
  This is the exact, **unnormalized** sequence importance weight
  $\pi_\text{new}(y_i \mid x_i) / \pi_\text{old}(y_i \mid x_i)$. It is *not* the
  length-normalized `RatioKind.SEQUENCE` exponent used by GSPO, which divides the summed
  gap by $\sum_t m_{i,t}$ before exponentiating; equal per-token gaps on unequal lengths
  give **unequal** sequence weights here
  (`tests/test_diagnostics_ess.py::test_sequence_weights_are_unnormalized_sums`).
- **token level**: the per-token ratios $\exp(g_{i,t})$ pooled over all response tokens,
  $n = \sum_{b,t} m$.

## ESS

$$
\mathrm{ESS} = \frac{\big(\sum_i w_i\big)^2}{\sum_i w_i^2},
\qquad \texttt{ess\_ratio} = \frac{\mathrm{ESS}}{n} .
$$

This is the inverse sum of squared normalized weights: with
$\hat w_i = w_i / \sum_j w_j$,
$\;1/\sum_i \hat w_i^2 = (\sum_j w_j)^2 / \sum_i w_i^2 = \mathrm{ESS}$.

**Bounds.** By Cauchy–Schwarz, $(\sum_i w_i \cdot 1)^2 \le n \sum_i w_i^2$, so
$\mathrm{ESS} \le n$ with equality iff all $w_i$ are equal; $\mathrm{ESS} > 0$ since some
$w_i > 0$. Hence $\texttt{ess\_ratio} \in (0, 1]$
(`tests/test_diagnostics_ess.py::test_ess_ratio_bounds`).

**Scale invariance.** For any $c > 0$,

$$
\frac{(\sum_i c\,w_i)^2}{\sum_i (c\,w_i)^2}
= \frac{c^2 (\sum_i w_i)^2}{c^2 \sum_i w_i^2}
= \mathrm{ESS}(w).
$$

The implementation exploits this by shifting the log-weights by their maximum
($c = e^{-\max_i \log w_i}$) before exponentiating, so no weight exceeds $1$ and the sums
cannot overflow. Token-level ESS is therefore also invariant to a constant shift of all
log-weights (`tests/test_diagnostics_ess.py::test_token_level_shift_invariance`).

## Null calibration

1. **Identical policies.** All gaps are $0$, so every $w_i = e^0 = 1$ exactly and
   $\mathrm{ESS} = n^2 / n = n$, i.e. $\texttt{ess\_ratio} = 1$ with no floating-point
   slack (`tests/test_diagnostics_ess.py::test_identical_policies_ess_ratio_is_exactly_one`,
   `tests/test_diagnostics_ess.py::test_identical_policies_sliding_ess_is_exactly_one`).

2. **iid Normal log-weight noise.** Let $\log w_i = z_i \sim \mathcal N(0, \sigma^2)$ iid.
   Lognormal moments give $\mathbb E[w] = e^{\sigma^2/2}$ and
   $\mathbb E[w^2] = e^{2\sigma^2}$. Writing
   $\mathrm{ESS}/n = \bar w^2 \big/ \overline{w^2}$ with
   $\bar w = \frac1n \sum_i w_i$ and $\overline{w^2} = \frac1n \sum_i w_i^2$, the law of
   large numbers gives, as $n \to \infty$,

   $$
   \frac{\mathrm{ESS}}{n} \;\longrightarrow\;
   \frac{(\mathbb E[w])^2}{\mathbb E[w^2]}
   = \frac{e^{\sigma^2}}{e^{2\sigma^2}}
   = e^{-\sigma^2}.
   $$

   `tests/test_diagnostics_ess.py::test_mc_calibration_mean_ess_ratio_approaches_exp_neg_var`
   verifies this on 12 seeded runs of $n = 8192$ for $\sigma \in \{0.3, 0.6\}$
   ($e^{-0.09} \approx 0.913931$, $e^{-0.36} \approx 0.697676$) within absolute tolerance
   $0.01$, and
   `tests/test_diagnostics_ess.py::test_importance_ess_matches_direct_simulation`
   checks the report against the raw $(\sum w)^2/\sum w^2$ formula evaluated directly on
   the same seeded draws.

## Golden cases

Gaps $\text{new} - \text{old} = [[0.5, -0.5], [\ln 2, \text{junk}]]$ with mask
$[[T, T], [T, F]]$:

- **sequence**: $\log w = [0.5 - 0.5,\; \ln 2] = [0, \ln 2]$, so $w = [1, 2]$ and
  $\mathrm{ESS} = (1+2)^2 / (1^2 + 2^2) = 9/5 = 1.8$, $\texttt{ess\_ratio} = 0.9$.
  Log-weight stats: mean $\ln 2 / 2$, Bessel-corrected std
  $\sqrt{2 (\ln 2 / 2)^2 / 1} = \ln 2 / \sqrt 2$
  (`tests/test_diagnostics_ess.py::test_sequence_level_golden_case`).
- **token**: $w = [e^{0.5}, e^{-0.5}, 2]$, $\sum w = 4.2552519304$,
  $\sum w^2 = e + e^{-1} + 4 = 7.0861612696$,
  $\mathrm{ESS} = 4.2552519304^2 / 7.0861612696 = 2.5552860431$,
  $\texttt{ess\_ratio} = 0.8517620144$
  (`tests/test_diagnostics_ess.py::test_token_level_golden_case`).

## Sliding windows

`sliding_ess(..., window, step)` applies the sequence-level ESS to consecutive windows in
batch order (rollout-chronological): window $k$ covers rows
$[k\cdot\text{step},\, k\cdot\text{step} + \text{window})$ and reports
$\mathrm{ESS}_k / \text{window}$; the output has shape
$\lfloor (B - \text{window})/\text{step} \rfloor + 1$
(`tests/test_diagnostics_ess.py::test_sliding_ess_window_and_step_shapes`). A single
window covering the whole batch reproduces `importance_ess(...).ess_ratio`
(`tests/test_diagnostics_ess.py::test_sliding_ess_matches_importance_ess_full_window`).

Golden case: log-weights $[0, \ln 2, 0, \ln 2]$, i.e. weights $[1, 2, 1, 2]$:

- window 2, step 1: every window holds $\{1, 2\}$, so $\mathrm{ESS} = 9/5$ and
  $\mathrm{ESS}/2 = 0.9$ in all three windows;
- window 3, step 1: $\{1,2,1\}$ gives $16/6 = 8/3$, so $8/9$; $\{2,1,2\}$ gives $25/9$,
  so $25/27$;
- window 2, step 2: two windows, both $0.9$
  (`tests/test_diagnostics_ess.py::test_sliding_ess_golden_case`).

`window < 2`, `window > B`, and `step < 1` raise `ValueError`
(`tests/test_diagnostics_ess.py::test_window_validation_errors`).

## Masking and validation

Masked positions never affect any output: perturbing them leaves every report field
bitwise-equal (`tests/test_diagnostics_ess.py::test_mask_invariance_importance_ess`,
`tests/test_diagnostics_ess.py::test_mask_invariance_sliding_ess`). Finiteness is
enforced on response positions only
(`tests/test_diagnostics_ess.py::test_masked_positions_may_hold_nonfinite_junk`); shape,
mask, and `level` violations raise `ValueError`
(`tests/test_diagnostics_ess.py::test_input_validation_errors`).
