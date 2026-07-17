# Entropy diagnostics (`polgrad.diagnostics.entropy`)

Entropy collapse — the policy distribution sharpening until exploration dies — is a
standard RL post-training pathology. `token_entropy_estimate` produces a per-batch
entropy estimate from the sampled-token logprobs alone; `entropy_trend` watches the
resulting per-step series for drift (Theil–Sen slope) and sudden collapse (CUSUM
changepoint with a permutation-calibrated threshold).

## Sampled-token entropy estimate

For a distribution $\pi$, $H(\pi) = \mathbb E_{y\sim\pi}[-\log \pi(y)]$, so the sampled
token's $-\log \pi(y_t)$ is an unbiased one-sample Monte Carlo estimate of the
conditional entropy $H(\pi(\cdot \mid y_{<t}, x))$ **only when $y_t$ was sampled from the
same $\pi$ that is scored** (on-policy). Off-policy it estimates the cross-entropy of
$\pi$ under the sampling distribution instead. Pooling response tokens:

$$
\texttt{entropy\_estimate} = -\frac{\sum_{b,t} m_{b,t}\,\text{logprobs}_{b,t}}{\sum_{b,t} m_{b,t}},
\qquad
\texttt{per\_seq\_entropy}_b = -\frac{\sum_t m_{b,t}\,\text{logprobs}_{b,t}}{\sum_t m_{b,t}} .
$$

Writing $L_b = \sum_t m_{b,t}$, the two views are consistent by exchanging the order of
summation:

$$
\texttt{entropy\_estimate}
= \frac{\sum_b L_b \cdot \texttt{per\_seq\_entropy}_b}{\sum_b L_b}
$$

(`tests/test_diagnostics_entropy.py::test_pooled_entropy_is_length_weighted_mean_of_per_seq`).

Golden case (`tests/test_diagnostics_entropy.py::test_token_entropy_golden_case`):
logprobs $[[-1, -2], [-4, \text{junk}]]$ with mask $[[T, T], [T, F]]$ give
$\texttt{entropy\_estimate} = (1 + 2 + 4)/3 = 7/3$ and
$\texttt{per\_seq\_entropy} = [3/2, 4]$.

## Trend analysis

`entropy_trend(entropy_per_step, window=w, ...)` analyzes the trailing $w$ values
$x_0, \dots, x_{w-1}$ of the series
(`tests/test_diagnostics_entropy.py::test_trend_window_uses_trailing_values`).

### Theil–Sen slope

$$
\texttt{slope} = \operatorname{median}\Big\{ \frac{x_j - x_i}{j - i} \;:\; 0 \le i < j < w \Big\}.
$$

On an exactly linear window every pairwise slope equals the true slope, so the median
recovers it (`tests/test_diagnostics_entropy.py::test_theil_sen_recovers_exact_linear_slope`).
Corrupting $k$ of $w$ points contaminates $1 - \binom{w-k}{2}\big/\binom{w}{2}$ of the
pairs — for $k=3$, $w=30$ that is $1 - 351/435 \approx 19.3\%$, a minority, so the median
stays near the clean slope
(`tests/test_diagnostics_entropy.py::test_theil_sen_robust_to_minority_corruption`).

### CUSUM changepoint statistic

With $\bar x = \frac1w \sum_i x_i$ and partial sums $S_k = \sum_{i \le k} (x_i - \bar x)$:

$$
\texttt{cusum\_stat} = \max_k \lvert S_k \rvert .
$$

For a pure level shift — $x_i = a$ for $i < p$ and $x_i = b$ for $i \ge p$ — one has
$\bar x = \big(p\,a + (w-p)\,b\big)/w$, so for $k < p$

$$
S_k = (k+1)\,(a - \bar x) = (k+1)\,\frac{(w-p)(a-b)}{w},
$$

whose magnitude grows linearly in $k$ up to $k = p - 1$ and shrinks afterwards
(symmetrically, $S_k$ for $k \ge p$ decreases in magnitude toward $S_{w-1} = 0$). The
argmax is therefore $k = p - 1$, the last pre-change index, with peak
$\lvert S_{p-1}\rvert = p\,(w-p)\,\lvert a - b\rvert / w$. `changepoint_index` reports
that argmax as a global index into `entropy_per_step`. Example: 60 steps at levels
$3.0$ (first 30) then $1.0$, window 60: peak $30 \cdot 30 \cdot 2 / 60 = 30$ at index 29;
window 40 (10 high, 30 low): peak $10 \cdot 30 \cdot 2 / 40 = 15$, again at global
index 29
(`tests/test_diagnostics_entropy.py::test_cusum_detects_mean_shift_and_localizes_changepoint`).

### Permutation calibration

Under the no-change null the window values are exchangeable, so the observed statistic
and the statistics of $n_\text{perm}$ uniformly random permutations (drawn from the
explicit `torch.Generator`) are exchangeable too. With
$m = \lfloor \alpha\,(n_\text{perm} + 1) \rfloor$ and

$$
\texttt{threshold} = \text{the } m\text{-th largest of the } n_\text{perm}
\text{ permuted statistics},
$$

a changepoint is reported iff $\texttt{cusum\_stat} > \texttt{threshold}$
(`tests/test_diagnostics_entropy.py::test_no_changepoint_iff_stat_at_most_threshold`).
Rejection requires the observed value to rank in the top $m$ of the
$n_\text{perm} + 1$ exchangeable values, an event of probability

$$
P(\text{reject} \mid \text{null}) = \frac{m}{n_\text{perm} + 1} \le \alpha
$$

(equality when the statistics are almost surely distinct). This is why
$\lfloor \alpha (n_\text{perm} + 1) \rfloor \ge 1$ is required — otherwise the test could
never reject, and `entropy_trend` raises `ValueError` instead of silently reporting "no
changepoint" (`tests/test_diagnostics_entropy.py::test_trend_validation_errors`).

**MC false-positive rate.**
`tests/test_diagnostics_entropy.py::test_trend_false_positive_rate_calibrated` runs 500
seeded iid-Gaussian null windows with $\alpha = 0.05$, $n_\text{perm} = 199$ (so
$m = 10$ and the exact level is $10/200 = 0.05$) and verifies the empirical
false-positive rate is within four binomial standard errors
($4\sqrt{0.05 \cdot 0.95 / 500} \approx 0.039$) of $\alpha$.

Determinism: the permutations consume only the explicit generator, so a fixed seed
reproduces the report bit-for-bit
(`tests/test_diagnostics_entropy.py::test_trend_determinism_same_seed`), per the RNG rule
of `docs/conventions.md`.

## Masking and validation

Masked inputs never affect `token_entropy_estimate`
(`tests/test_diagnostics_entropy.py::test_mask_invariance_token_entropy`), and
`per_seq_entropy` preserves the input dtype
(`tests/test_diagnostics_entropy.py::test_per_seq_entropy_dtype_preserved`). Shape, mask,
finiteness, window-range, `alpha`, and `n_perm` violations raise `ValueError`
(`tests/test_diagnostics_entropy.py::test_entropy_validation_errors`,
`tests/test_diagnostics_entropy.py::test_trend_validation_errors`).
