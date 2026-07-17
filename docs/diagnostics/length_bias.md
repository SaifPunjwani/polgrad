# Length-bias probe (`polgrad.diagnostics.length_bias`)

Aggregation decides how much of the scalar loss each sequence owns. `length_bias_probe`
measures whether that ownership depends on sequence length: it regresses each sequence's
absolute weighted-advantage mass on its response-token count and reports the OLS slope
with an HC3 heteroscedasticity-robust standard error, in closed-form torch (no scipy at
runtime).

## The probe

With mask $m \in \{0,1\}^{B \times T}$, advantages $A$ (a $[B]$ input is broadcast across
its row's tokens first;
`tests/test_diagnostics_length_bias.py::test_seq_advantages_equal_expanded_token_advantages`),
and $w = \texttt{effective\_token\_weights}(m, \text{agg\_mode}, \text{norm\_len})$:

$$
y_i = \sum_t m_{i,t}\,\bigl|A_{i,t}\,w_{i,t}\bigr|,
\qquad
x_i = L_i = \sum_t m_{i,t},
\qquad i = 1,\dots,n \;(= B).
$$

Masked positions are zeroed before the absolute value, so junk there never reaches any
output (bitwise mask invariance:
`tests/test_diagnostics_length_bias.py::test_mask_invariance`) and may even be
non-finite (`tests/test_diagnostics_length_bias.py::test_masked_positions_may_hold_nonfinite_junk`).
Advantages are detached on entry; the probe never feeds a gradient
(`tests/test_diagnostics_length_bias.py::test_probe_detaches_from_autograd`).
The report also carries the raw ingredients: `per_seq_length` $= L_i$ and
`per_seq_weight_sum` $= W_i = \sum_t w_{i,t}$
(`tests/test_diagnostics_length_bias.py::test_report_per_seq_fields_match_recomputation`).

## What each aggregation mode makes the probe see

From the effective-weight closed forms of `docs/derivations/aggregation.md`
($N = \sum_b L_b$, $\ell$ = `norm_len`):

| mode | per-token weight $w_{i,t}$ | per-sequence total $W_i$ |
| --- | --- | --- |
| `TOKEN_MEAN` | $m_{i,t}/N$ | $L_i/N$ |
| `SEQ_MEAN_TOKEN_MEAN` | $m_{i,t}/(B\,L_i)$ | $1/B$ |
| `SEQ_MEAN_TOKEN_SUM` | $m_{i,t}/B$ | $L_i/B$ |
| `TOKEN_SUM_NORM` | $m_{i,t}/(B\,\ell)$ | $L_i/(B\,\ell)$ |

The `SEQ_MEAN_TOKEN_MEAN` and `TOKEN_SUM_NORM` rows are asserted directly against
`polgrad.aggregate.effective_token_weights`: `SEQ_MEAN_TOKEN_MEAN` induces per-token
weight $\propto 1/L_i$ (exactly $m/(B L_i)$), while `TOKEN_SUM_NORM` induces the
constant $m/(B\,\ell)$
(`tests/test_diagnostics_length_bias.py::test_mode_induced_weights_and_structural_slopes`).

For a row-constant advantage magnitude $|a_i|$ the regressand is $y_i = |a_i|\,W_i$, so
an advantage stream with $|a|$ independent of length gives a **structural** slope set by
the aggregation alone. With $|A| \equiv 1$ (same test):

- `SEQ_MEAN_TOKEN_MEAN`: $y_i = 1/B$ for every row — slope $\approx 0$
  (asserted $< 10^{-12}$);
- `TOKEN_SUM_NORM`: $y_i = L_i/(B\,\ell)$ — slope exactly the per-token weight
  $1/(B\,\ell)$, zero residuals, se $\approx 0$;
- `TOKEN_MEAN`: slope $1/N$; `SEQ_MEAN_TOKEN_SUM`: slope $1/B$.

**Reading the sign.** A positive slope means longer sequences carry more absolute loss
mass per sequence; negative means shorter ones do.

- Under `SEQ_MEAN_TOKEN_MEAN` the structural slope is zero, so any significant slope is
  *data-level* length bias: $|A|$ itself correlates with length (positive: long
  responses get larger advantage magnitudes, e.g. length-favoring rewards). The flip
  side of the constant $W_i$ is the per-token dilution $w_{i,t} = 1/(B L_i)$: each token
  of a long response moves the loss less, so with negative advantages a long wrong
  answer is penalized less per token — the aggregation-side pathology that
  `TOKEN_SUM_NORM` (Dr.GRPO) removes by fixing the per-token weight.
- Under `TOKEN_MEAN`, `SEQ_MEAN_TOKEN_SUM`, and `TOKEN_SUM_NORM` a positive slope is
  *expected*: with per-token weight $c$ constant, $y_i = c\,L_i\,\overline{|A_i|}$, so
  length-independent token magnitudes with mean $\mu$ give slope $c\,\mu$ through the
  origin. Compare the measured slope against that baseline: a significantly larger
  slope means $|A|$ grows with length on top of the structural effect; a slope near
  zero (or negative) means $|A|$ decays with length fast enough to cancel it.

## OLS closed form

Minimizing $\sum_i (y_i - \beta_0 - \beta_1 x_i)^2$, the $\beta_0$ normal equation
$\sum_i (y_i - \beta_0 - \beta_1 x_i) = 0$ gives $\hat\beta_0 = \bar y - \hat\beta_1 \bar x$;
substituting into the $\beta_1$ equation $\sum_i x_i (y_i - \beta_0 - \beta_1 x_i) = 0$
gives

$$
\hat\beta_1 = \frac{S_{xy}}{S_{xx}},
\qquad
S_{xy} = \sum_i (x_i - \bar x)(y_i - \bar y),
\qquad
S_{xx} = \sum_i (x_i - \bar x)^2 .
$$

Slope and intercept agree with `scipy.stats.linregress` on 12 seeded cases across all
four modes (`tests/test_diagnostics_length_bias.py::test_ols_matches_scipy_linregress`).

## HC3 robust standard error

With design matrix $X = [\mathbf 1 \;\; x] \in \mathbb R^{n \times 2}$, residuals
$e_i = y_i - \hat\beta_0 - \hat\beta_1 x_i$, and hat values
$h_i = x_i^\top (X^\top X)^{-1} x_i$, the HC3 covariance estimator is the sandwich

$$
\widehat V = (X^\top X)^{-1}\, X^\top \operatorname{diag}\!\left(\frac{e_i^2}{(1-h_i)^2}\right) X\, (X^\top X)^{-1}.
$$

**Reduction to closed form.** For simple regression,

$$
X^\top X = \begin{pmatrix} n & \sum x \\ \sum x & \sum x^2 \end{pmatrix},
\qquad
\det = n \sum x^2 - \Bigl(\sum x\Bigr)^2 = n\,S_{xx},
$$

so

$$
(X^\top X)^{-1} = \frac{1}{n\,S_{xx}} \begin{pmatrix} \sum x^2 & -\sum x \\ -\sum x & n \end{pmatrix}.
$$

Hat values: using $\sum x^2 = S_{xx} + n\bar x^2$,

$$
h_i = \frac{\sum x^2 - 2 x_i \sum x + n x_i^2}{n\,S_{xx}}
= \frac{S_{xx} + n\,(x_i - \bar x)^2}{n\,S_{xx}}
= \frac{1}{n} + \frac{(x_i - \bar x)^2}{S_{xx}} .
$$

Slope entry: the second column of $(X^\top X)^{-1}$ is
$\tfrac{1}{n S_{xx}}(-\sum x,\; n)^\top = \tfrac{1}{S_{xx}}(-\bar x,\; 1)^\top$, so row
$i$ of $X (X^\top X)^{-1} e_2$ is

$$
c_i = \frac{x_i - \bar x}{S_{xx}},
\qquad\text{hence}\qquad
\widehat V_{\hat\beta_1} = \sum_i \frac{e_i^2}{(1-h_i)^2}\, c_i^2
= \frac{1}{S_{xx}^2} \sum_i \frac{e_i^2}{(1-h_i)^2}\,(x_i - \bar x)^2,
$$

and $\texttt{slope\_se} = \smash{\sqrt{\widehat V_{\hat\beta_1}}}$. The module computes
this reduced form; an independent full-matrix numpy sandwich agrees on 12 seeded cases
across all modes (`tests/test_diagnostics_length_bias.py::test_hc3_matches_full_numpy_sandwich`).

**Confidence interval.** $\texttt{ci\_low/high} = \hat\beta_1 \mp z \cdot \texttt{slope\_se}$
with $z = \Phi^{-1}(0.975) \approx 1.96$, computed at import via `torch.special.ndtri`
and cross-checked against `scipy.stats.norm.ppf(0.975)`
(`tests/test_diagnostics_length_bias.py::test_ci_uses_normal_975_quantile`).

## Hand-worked golden case

Lengths $[1, 2, 3]$ under `SEQ_MEAN_TOKEN_SUM` ($w = 1/3$ at every response token,
$W = [1/3,\, 2/3,\, 1]$), advantage rows $[3]$, $[3, 3]$, $[4, 4, 4]$:

$$
y = \Bigl[\tfrac33,\ \tfrac{3+3}{3},\ \tfrac{4+4+4}{3}\Bigr] = [1, 2, 4],
\qquad x = [1, 2, 3].
$$

- $\bar x = 2$, $\bar y = 7/3$, $S_{xx} = (-1)^2 + 0^2 + 1^2 = 2$;
- $S_{xy} = (-1)(1 - \tfrac73) + 0\cdot(2 - \tfrac73) + (1)(4 - \tfrac73) = \tfrac43 + \tfrac53 = 3$;
- $\hat\beta_1 = 3/2$, $\hat\beta_0 = \tfrac73 - \tfrac32\cdot 2 = -\tfrac23$;
- fitted $\hat y = [\tfrac56,\ \tfrac73,\ \tfrac{23}{6}]$, residuals
  $e = [\tfrac16,\ -\tfrac13,\ \tfrac16]$;
- $h = \tfrac13 + \tfrac{(x-2)^2}{2} = [\tfrac56,\ \tfrac13,\ \tfrac56]$,
  $1 - h = [\tfrac16,\ \tfrac23,\ \tfrac16]$;
- $\omega = e^2/(1-h)^2 = \bigl[\tfrac{1/36}{1/36},\ \tfrac{1/9}{4/9},\ \tfrac{1/36}{1/36}\bigr] = [1,\ \tfrac14,\ 1]$;
- $\widehat V_{\hat\beta_1} = \dfrac{1\cdot 1 + \tfrac14 \cdot 0 + 1 \cdot 1}{2^2} = \dfrac12$,
  so $\texttt{slope\_se} = 1/\sqrt2$.

`tests/test_diagnostics_length_bias.py::test_hc3_slope_se_matches_hand_computed_golden_case`
asserts every number above.

## Calibration: CI coverage of a known slope

Synthetic model with known slope: $L_i \sim \mathrm{Uniform}\{1,\dots,12\}$, $B = 200$,

$$
y_i^\star = 2 + 0.5\,L_i + \varepsilon_i,
\qquad \varepsilon_i \sim \mathcal N\bigl(0,\ (0.04\,L_i)^2\bigr)
\quad\text{(heteroscedastic by construction)} .
$$

Feeding $[B]$ advantages $a_i = B\,y_i^\star / L_i$ under `SEQ_MEAN_TOKEN_SUM` makes the
probe's regressand equal $y_i^\star$ exactly up to fp rounding (since
$y_i = |a_i| L_i / B$). Over 400 seeded runs the 95% HC3 CI covers the true slope $0.5$
at close to nominal rate — asserted within $[0.92, 0.98]$, where the binomial sd of a
coverage estimate at nominal $0.95$ with $400$ runs is
$\sqrt{0.95 \cdot 0.05 / 400} \approx 0.011$ — and the mean slope estimate recovers
$0.5$ within $2\times10^{-3}$
(`tests/test_diagnostics_length_bias.py::test_ci_covers_known_slope_in_about_95_percent_of_runs`).

## Degenerate inputs

All raise `ValueError` rather than emitting NaN (docs/conventions.md;
`tests/test_diagnostics_length_bias.py::test_degenerate_inputs_raise_value_errors`):

- **$B < 3$.** Two regression parameters leave $n - 2$ residual degrees of freedom;
  $n = 2$ fits every point exactly and HC3 has $h_i = 1$ for both rows.
- **Constant lengths.** $S_{xx} = 0$: the slope is undefined.
- **Exactly one sequence at a distinct length.** With $n-1$ rows at length $u$ and one
  at $v \ne u$: $\bar x = \frac{(n-1)u + v}{n}$, so $u - \bar x = \frac{u-v}{n}$ and
  $v - \bar x = \frac{(n-1)(v-u)}{n}$, giving

  $$
  S_{xx} = (n-1)\frac{(u-v)^2}{n^2} + \frac{(n-1)^2 (v-u)^2}{n^2}
  = \frac{(n-1)(u-v)^2}{n},
  $$

  $$
  h_v = \frac{1}{n} + \frac{(v - \bar x)^2}{S_{xx}}
  = \frac{1}{n} + \frac{(n-1)^2 (u-v)^2}{n^2} \cdot \frac{n}{(n-1)(u-v)^2}
  = \frac{1}{n} + \frac{n-1}{n} = 1 .
  $$

  That row's residual is identically zero, so HC3's $e_v^2/(1-h_v)^2$ is $0/0$;
  the probe rejects the pattern instead. Lengths are integers, so both length checks
  are exact (no tolerance).
- **`norm_len` missing with `TOKEN_SUM_NORM`.** Raised at call time by
  `effective_token_weights` (docs/derivations/aggregation.md;
  `tests/test_diagnostics_length_bias.py::test_norm_len_required_at_call_time_for_token_sum_norm`).
- Shape, dtype, mask, and response-position finiteness violations follow
  `polgrad._validation` (same test).

## Exact properties

- **Mask invariance.** Perturbing masked advantage positions leaves every report field
  bitwise-equal (`tests/test_diagnostics_length_bias.py::test_mask_invariance`).
- **Broadcast.** A $[B]$ advantage vector produces a report identical to its expanded
  $[B, T]$ form
  (`tests/test_diagnostics_length_bias.py::test_seq_advantages_equal_expanded_token_advantages`).
- **Scale equivariance.** $y$ is linear in $|A|$, so doubling the advantages doubles
  slope, se, intercept, and both CI endpoints exactly — multiplication by a power of
  two is lossless in floating point
  (`tests/test_diagnostics_length_bias.py::test_doubling_advantages_doubles_the_fit`).
