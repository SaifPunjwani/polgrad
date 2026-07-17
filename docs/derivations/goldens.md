# Analytic goldens: softmax-bandit closed forms and hand-derived loss cases

This page derives every exact number shipped by `polgrad.verify.goldens` and the
machinery of `polgrad.verify.gradcheck`: the closed forms of the K-arm softmax bandit
(policy gradient, categorical KL, sampling identities) and the arithmetic of each
`GoldenCase`, worked line by line. Every claim names the pytest node that enforces it.

## Softmax bandit

Tabular policy over $K \ge 2$ arms with logits $\theta \in \mathbb{R}^K$ and fixed
per-arm rewards $r_k$:

$$
\pi_k = \frac{e^{\theta_k}}{Z}, \qquad Z = \sum_j e^{\theta_j}.
$$

Each sampled action is a one-token sequence in polgrad's $[B, T]$ convention
($T = 1$), so per-token and per-sequence semantics coincide and the closed forms below
certify full-pipeline losses.

### Softmax Jacobian

By the quotient rule, with $\partial Z / \partial \theta_j = e^{\theta_j}$:

$$
\frac{\partial \pi_k}{\partial \theta_j}
= \frac{\delta_{kj}\, e^{\theta_k}}{Z} - \frac{e^{\theta_k}\, e^{\theta_j}}{Z^2}
= \pi_k \delta_{kj} - \pi_k \pi_j
= \pi_k \big(\delta_{kj} - \pi_j\big).
$$

### Exact policy gradient

For per-arm advantages $A \in \mathbb{R}^K$, the expected objective is
$J(\theta) = \mathbb{E}_{a \sim \pi_\theta}[A_a] = \sum_k \pi_k A_k$. Applying the
Jacobian:

$$
\frac{\partial J}{\partial \theta_j}
= \sum_k A_k\, \pi_k \big(\delta_{kj} - \pi_j\big)
= \pi_j A_j - \pi_j \sum_k \pi_k A_k
= \pi_j \big(A_j - \bar A\big),
\qquad \bar A = \mathbb{E}_\pi[A].
$$

`SoftmaxBandit.exact_policy_gradient` returns this **ascent** gradient (polgrad losses
are its negation); it is verified against `torch.autograd` on
$J(\theta) = \sum_k \mathrm{softmax}(\theta)_k A_k$ itself by
`tests/test_verify.py::test_exact_policy_gradient_matches_autograd_on_expected_objective`.
Two corollaries serve as property tests
(`tests/test_verify.py::test_exact_policy_gradient_sums_to_zero_and_is_shift_invariant`):

- $\sum_j \pi_j (A_j - \bar A) = \bar A - \bar A = 0$ — the gradient sums to zero over
  arms (softmax is shift-invariant in $\theta$);
- replacing $A$ by $A + c$ leaves it unchanged, since $(A_j + c) - (\bar A + c) =
  A_j - \bar A$.

### Score-function identity

From $\log \pi_a = \theta_a - \log Z$ and $\partial \log Z / \partial \theta_j =
\pi_j$:

$$
\nabla_{\theta_j} \log \pi_a = \delta_{aj} - \pi_j ,
$$

so the REINFORCE estimator of $\nabla J$ has expectation

$$
\mathbb{E}_{a \sim \pi}\big[A_a \nabla_{\theta_j} \log \pi_a\big]
= \sum_a \pi_a A_a \big(\delta_{aj} - \pi_j\big)
= \pi_j A_j - \pi_j \bar A
= \frac{\partial J}{\partial \theta_j},
$$

the same closed form. MC-certified within 4 standard errors per coordinate on a seeded
run of $n = 16384$ samples per arm by
`tests/test_verify.py::test_exact_policy_gradient_matches_mc_score_function_estimate`,
using `verify.mc.mc_mean`.

### Categorical KL

For two logit vectors $\theta, \theta'$ with $\pi = \mathrm{softmax}(\theta)$,
$\pi' = \mathrm{softmax}(\theta')$:

$$
\mathrm{KL}\big(\pi \,\|\, \pi'\big)
= \sum_k \pi_k \big(\log \pi_k - \log \pi'_k\big)
= \sum_k \pi_k \big[(\theta_k - \log Z) - (\theta'_k - \log Z')\big]
\;\ge\; 0,
$$

with equality iff $\pi = \pi'$ (Gibbs' inequality). `SoftmaxBandit.exact_kl` evaluates
this via `log_softmax` in explicit `float64`; it is verified against the direct sum,
non-negativity, and exact self-KL $= 0$ by
`tests/test_verify.py::test_exact_kl_matches_direct_categorical_kl`.

### Sampling contract

`SoftmaxBandit.sample(n, generator)` draws $a_i \sim \pi_\theta$ and returns
$[n, 1]$ streams with $\texttt{logprobs}_{i,0} = \log \pi_{a_i}$ differentiable in
$\theta$. Because sampling is exact and on-policy with the reference equal to the
current policy, $\texttt{old\_logprobs} = \texttt{ref\_logprobs} =
\texttt{rollout\_logprobs} = \mathrm{sg}[\texttt{logprobs}]$. Two exact identities are
enforced by `tests/test_verify.py::test_softmax_bandit_sample_contract`:

- gradient of the summed stream: $\partial \big(\sum_i \log \pi_{a_i}\big) /
  \partial \theta_j = \sum_i (\delta_{a_i j} - \pi_j) = \mathrm{counts}_j - n \pi_j$;
- $\texttt{rewards}_i = r_{a_i}$ elementwise.

The sample-mean reward over $n = 32768$ seeded draws matches the closed form
$\sum_k \pi_k r_k$ within `clt_tolerance(σ, n)` for the exact reward standard
deviation $\sigma = \sqrt{\sum_k \pi_k r_k^2 - (\sum_k \pi_k r_k)^2}$
(`tests/test_verify.py::test_softmax_bandit_sample_reward_mean_matches_closed_form_mc`).

## Machine-checked gradients

`verify.gradcheck.check_gradient_formula` compares a hand-derived gradient against the
central stencil

$$
g_i = \frac{f(x + \varepsilon e_i) - f(x - \varepsilon e_i)}{2\varepsilon}
= \frac{\partial f}{\partial x_i} + O(\varepsilon^2),
$$

which never consults autograd and therefore catches wrong derivations. The PG/token
gradient $\partial L / \partial \texttt{lp}_t = -w_t A_t r_t$
([losses.md](losses.md)) passes
(`tests/test_verify.py::test_check_gradient_formula_accepts_correct_pg_derivation`,
`tests/test_verify.py::test_check_gradient_formula_pg_derivation_property`), while the
deliberately wrong derivation $-w_t A_t$ — the ratio factor dropped — is rejected
(`tests/test_verify.py::test_check_gradient_formula_raises_on_wrong_derivation`).

`verify.gradcheck.gradcheck_loss` runs `torch.autograd.gradcheck` on
`polgrad.losses.policy_loss` over seeded fp64 batches whose ratios are kept at least
$10^{-3}$ from every clip boundary (finite differences perturb logprobs by $10^{-6}$,
moving ratios by a relative $\sim 10^{-6}$, so no branch can flip). Losses with
internal stop-gradients are checked through their sg-frozen equivalents derived in
[losses.md](losses.md), and the real config's autograd gradient is asserted equal to
the frozen equivalent's at the evaluation point
(`tests/test_verify.py::test_gradcheck_loss_passes_for_representative_configs`).

## Golden loss cases

Shared setup for every case: `PG_CLIP` with $\varepsilon_{lo} = 0.2$ (lower bound
$lo = 0.8$) and $\varepsilon_{hi} = 0.3$ (upper bound $hi = 1.3$); token ratio
$r = e^{\texttt{lp} - \texttt{olp}}$; per-token loss $\ell = -\min\big(rA,\,
\mathrm{clip}(r, lo, hi)\, A\big)$ with the branch algebra of
[losses.md](losses.md); aggregation weights from [aggregation.md](aggregation.md); all
tensors `float64`. Every case is replayed by
`tests/test_verify.py::test_golden_cases_satisfied_by_policy_loss` with
$|\mathrm{loss} - \mathrm{expected}| \le 10^{-12}$ and gradient `atol` $10^{-12}$; the
1-token branch flags by
`tests/test_verify.py::test_golden_one_token_cases_report_expected_clip_branch`; the
required case inventory and these anchors by
`tests/test_verify.py::test_golden_cases_cover_contract_branches`; junk-at-masked
invariance by `tests/test_verify.py::test_golden_ragged_cases_are_mask_invariant`.

### pg_clip_inside_band

One token, `TOKEN_MEAN` with $N = 1$, weight $1$.

- $z = \texttt{lp} - \texttt{olp} = -0.7 - (-0.9) = 0.2$, so
  $r = e^{0.2} = 1.221403 \in (0.8, 1.3)$: the clip is inactive and both $\min$
  arguments equal $rA$.
- $A = 1.5$: objective $= 1.5\, e^{0.2} = 1.832104$, loss $= -1.5\, e^{0.2} =
  -1.832104$.
- Gradient: $\partial(-rA)/\partial \texttt{lp} = -A\, r = -1.5\, e^{0.2} =
  -1.832104$ (equal to the loss here only because $\partial r / \partial \texttt{lp}
  = r$ and the weight is $1$).
- No clip flag is set.

### pg_clip_clipped_high_positive_advantage

- $z = -0.2 - (-0.7) = 0.5$, so $r = e^{0.5} = 1.648721 > hi = 1.3$.
- $A = 2$: unclipped $= 2\, e^{0.5} = 3.297443$; clipped $= 1.3 \cdot 2 = 2.6$;
  $\min(3.297443,\, 2.6) = 2.6$.
- Loss $= -2.6$. The active branch is the constant $hi \cdot A$, so the gradient is
  exactly $0$; `clipped_high` is set.

### pg_clip_clipped_low_negative_advantage

- $z = -1.5 - (-1.0) = -0.5$, so $r = e^{-0.5} = 0.606531 < lo = 0.8$.
- $A = -1$: unclipped $= -e^{-0.5} = -0.606531$; clipped $= 0.8 \cdot (-1) = -0.8$;
  $\min(-0.606531,\, -0.8) = -0.8$.
- Loss $= 0.8$. Constant branch: gradient exactly $0$; `clipped_low` is set.

### dual_clip_negative_advantage_above_cap

`ratio_cap` $c = 2$; dual-clip objective for $A < 0$ is
$\max\big(\min(rA,\, \mathrm{clip}(r)A),\, cA\big)$ ([losses.md](losses.md)).

- $z = -0.4 - (-1.4) = 1.0$, so $r = e^{1} = 2.718282 > c = 2$.
- $A = -1.5$: unclipped $= -1.5\, e = -4.077423$; clipped $= 1.3 \cdot (-1.5) =
  -1.95$; $\min(-4.077423,\, -1.95) = -4.077423$.
- Dual-clip floor: $cA = 2 \cdot (-1.5) = -3$; $\max(-4.077423,\, -3) = -3$.
- Loss $= 3.0$. Constant branch: gradient exactly $0$; the binding cap is reported in
  `clipped_high`.

### pg_clip_two_token_ragged_token_mean

Mask $\begin{pmatrix}1 & 1\\ 1 & 0\end{pmatrix}$, so $N = 3$ and every response token
carries `TOKEN_MEAN` weight $w = 1/3$. The masked position $(1,1)$ holds junk
($123.0$) in every input tensor and contributes nothing.

| token | lp | olp | $z$ | $r = e^z$ | $A$ | branch | objective | per-token loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $(0,0)$ | $-0.3$ | $-0.7$ | $0.4$ | $1.491825 > 1.3$ | $1.0$ | clipped high | $\min(1.491825,\, 1.3) \cdot 1 = 1.3$ | $-1.3$ |
| $(0,1)$ | $-1.2$ | $-0.8$ | $-0.4$ | $0.670320 < 0.8$ | $-2.0$ | clipped low | $\min(-1.340640,\, -1.6) = -1.6$ | $1.6$ |
| $(1,0)$ | $-0.5$ | $-0.6$ | $0.1$ | $1.105171 \in (0.8, 1.3)$ | $-1.0$ | unclipped | $-e^{0.1} = -1.105171$ | $1.105171$ |

$$
\mathrm{loss} = \frac{-1.3 + 1.6 + e^{0.1}}{3} = \frac{0.3 + 1.105171}{3}
= \frac{1.405171}{3} = 0.468390 .
$$

Gradient: the two clipped tokens contribute $0$; at $(1,0)$,
$w \cdot (-A\, r) = \tfrac13 \cdot \big(1 \cdot e^{0.1}\big) = e^{0.1}/3 = 0.368390$;
masked $(1,1)$ is $0$. Expected gradient
$\begin{pmatrix}0 & 0\\ 0.368390 & 0\end{pmatrix}$.

### pg_clip_two_token_ragged_seq_mean_token_mean

Same input tensors under `SEQ_MEAN_TOKEN_MEAN`, whose weights are
$w = m / (B\, L_i)$ ([aggregation.md](aggregation.md)): row 0 ($L_0 = 2$) tokens carry
$1/(2 \cdot 2) = 1/4$; row 1 ($L_1 = 1$) carries $1/(2 \cdot 1) = 1/2$.

$$
\mathrm{loss} = \tfrac14(-1.3) + \tfrac14(1.6) + \tfrac12\, e^{0.1}
= \frac{0.3}{4} + \frac{e^{0.1}}{2}
= 0.075 + 0.552585 = 0.627585 .
$$

Gradient: $0$ at the clipped and masked tokens; at $(1,0)$,
$\tfrac12 \cdot e^{0.1} = 0.552585$. Expected gradient
$\begin{pmatrix}0 & 0\\ 0.552585 & 0\end{pmatrix}$.

Comparing the two ragged cases: the same per-token losses aggregate differently
because `SEQ_MEAN_TOKEN_MEAN` up-weights the short row ($1/2$ vs $1/3$ per token) and
down-weights the long row ($1/4$ vs $1/3$) — the length-bias mechanism quantified in
[aggregation.md](aggregation.md).
