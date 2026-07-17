# Rollout-vs-trainer logprob mismatch (`polgrad.diagnostics.mismatch`)

`old_logprobs` recomputed by the trainer and `rollout_logprobs` reported by the inference
engine describe the same policy in exact arithmetic, but kernels, precision, and batching
make them differ in practice (`docs/conventions.md`). The per-token gap

$$
\Delta_{b,t} = \text{trainer\_logprobs}_{b,t} - \text{rollout\_logprobs}_{b,t}
$$

silently turns nominally on-policy training off-policy. `logprob_mismatch` reports its
size and shape.

## Report fields

All statistics pool the $n = \sum_{b,t} m$ response tokens; masked positions never
contribute (`tests/test_diagnostics_mismatch.py::test_mask_invariance_mismatch`).

| field | definition |
| --- | --- |
| `gap_mean`, `gap_std`, `gap_abs_max` | mean, Bessel-corrected std ($0.0$ when $n=1$), and $\max\lvert\Delta\rvert$ |
| `gap_quantiles` | linear-interpolation quantiles of $\Delta$ at (q01, q05, q25, q50, q75, q95, q99) |
| `kl_k1`, `kl_k2`, `kl_k3` | k1/k2/k3 estimates of $\mathrm{KL}(\text{trainer}\Vert\text{rollout})$ via `polgrad.kl.kl_estimate` |
| `kl_k3_reversed` | k3 estimate of $\mathrm{KL}(\text{rollout}\Vert\text{trainer})$ |
| `seq_log_ratio_*` | mean / std / $\max\lvert\cdot\rvert$ of the per-sequence sums $\sum_t m_{b,t}\Delta_{b,t}$ |
| `catastrophic_count`, `catastrophic_indices` | response tokens with $\lvert\Delta\rvert > \text{catastrophic\_gap}$ (strict); indices are a $[N, 2]$ long tensor of $(b, t)$ pairs in row-major order |
| `ppl_ratio` | $\exp(-\overline{\Delta})$, the trainer/rollout perplexity ratio (below) |

## KL estimates between the streams

`polgrad.kl` (see `docs/derivations/kl.md`) defines the estimators through
$\delta = \text{ref\_logprobs} - \text{logprobs}$. Passing
$\text{logprobs} = \text{trainer}$, $\text{ref} = \text{rollout}$ gives
$\delta = -\Delta$ and therefore per token

$$
k_1 = -\delta = \Delta,\qquad
k_2 = \tfrac{\delta^2}{2} = \tfrac{\Delta^2}{2},\qquad
k_3 = e^{\delta} - 1 - \delta = e^{-\Delta} - 1 + \Delta ,
$$

each averaged over response tokens. Swapping the roles gives
$\texttt{kl\_k3\_reversed} = \overline{e^{\Delta} - 1 - \Delta}$. In particular
$\texttt{kl\_k1} = \texttt{gap\_mean}$ always
(`tests/test_diagnostics_mismatch.py::test_kl_k1_equals_gap_mean_and_ppl_ratio_is_exp_neg`),
and $k_2, k_3 \ge 0$ pointwise, so both k3 fields are nonnegative
(`tests/test_diagnostics_mismatch.py::test_k3_estimates_nonnegative`).

One honesty caveat: the sampled tokens follow the *rollout* distribution, and the
k-estimators of `docs/derivations/kl.md` assume samples from their first argument. The
two streams score the same nominal policy, so the distinction vanishes as the gap does;
strictly, `kl_k3_reversed` is the direction whose sampling assumption holds exactly, and
the forward-direction fields evaluate the same formulas on rollout samples.

## Perplexity ratio

Perplexity of a stream on the sampled tokens is
$\mathrm{PPL} = \exp\!\big(-\frac1n \sum_{b,t} m\,\text{logprobs}\big)$, so

$$
\frac{\mathrm{PPL}_\text{trainer}}{\mathrm{PPL}_\text{rollout}}
= \exp\!\Big(-\tfrac1n \textstyle\sum m\,(\text{trainer} - \text{rollout})\Big)
= \exp\!\big(-\overline{\Delta}\big) = \texttt{ppl\_ratio}.
$$

`ppl_ratio` < 1 means the trainer assigns the sampled tokens *higher* likelihood than the
engine reported.

## Golden case (hand-derived)

Enforced by `tests/test_diagnostics_mismatch.py::test_golden_case_hand_derived`.
Inputs: trainer $[[-1.0, -2.0], [-1.5, \text{junk}]]$, rollout
$[[-1.2, -1.4], [-2.5, \text{junk}]]$, mask $[[T, T], [T, F]]$, catastrophic gap $0.7$.

- Gaps: $\Delta = [0.2, -0.6, 1.0]$, $n = 3$.
- $\texttt{gap\_mean} = (0.2 - 0.6 + 1.0)/3 = 0.2$.
- $\texttt{gap\_std}$: deviations $(0, -0.8, 0.8)$, $\sum d^2 = 1.28$,
  $1.28/(3-1) = 0.64$, $\sqrt{0.64} = 0.8$.
- $\texttt{gap\_abs\_max} = 1.0$.
- Quantiles: sorted $\Delta = [-0.6, 0.2, 1.0]$; linear interpolation at position
  $h = q\,(n-1) = 2q$:
  q01 $\to -0.6 + 0.02\cdot0.8 = -0.584$; q05 $\to -0.52$; q25 $\to -0.2$;
  q50 $\to 0.2$; q75 $\to 0.6$; q95 $\to 0.92$; q99 $\to 0.984$.
- $\texttt{kl\_k1} = 0.2$;
  $\texttt{kl\_k2} = (0.02 + 0.18 + 0.5)/3 = 0.7/3 \approx 0.2333333$.
- $\texttt{kl\_k3} = \big[(e^{-0.2} - 1 + 0.2) + (e^{0.6} - 1 - 0.6) + (e^{-1} - 1 + 1)\big]/3
  = (0.0187308 + 0.2221188 + 0.3678794)/3 = 0.2029097$.
- $\texttt{kl\_k3\_reversed} = \big[(e^{0.2} - 1 - 0.2) + (e^{-0.6} - 1 + 0.6) + (e^{1} - 2)\big]/3
  = (0.0214028 + 0.1488116 + 0.7182818)/3 = 0.2961654$.
- Sequence sums: $[0.2 - 0.6,\; 1.0] = [-0.4, 1.0]$; mean $0.3$; deviations $\mp 0.7$,
  std $\sqrt{0.98} = 0.9899495$; $\max\lvert\cdot\rvert = 1.0$.
- Catastrophic: only $(b{=}1, t{=}0)$ has $\lvert\Delta\rvert = 1.0 > 0.7$, so count $1$
  and indices $[[1, 0]]$.
- $\texttt{ppl\_ratio} = e^{-0.2} = 0.8187308$.

## Catastrophic tokens

A single rollout token whose recomputed logprob differs by more than
`catastrophic_gap` (default $5.0$, i.e. a factor $e^5 \approx 148$ in probability)
usually indicates a tokenization or kernel fault rather than smooth numeric noise; the
$(b, t)$ indices let callers inspect the offending positions directly. The comparison is
strict and the indices enumerate exactly the response positions that exceed it, verified
against a per-position oracle in
`tests/test_diagnostics_mismatch.py::test_catastrophic_indices_consistency`.

## Validation

Shape/mask violations, non-finite response positions, and
$\text{catastrophic\_gap} \le 0$ raise `ValueError`
(`tests/test_diagnostics_mismatch.py::test_validation_errors`); a single-token batch
reports $\texttt{gap\_std} = \texttt{seq\_log\_ratio\_std} = 0.0$, never NaN
(`tests/test_diagnostics_mismatch.py::test_single_token_stds_are_zero`).
