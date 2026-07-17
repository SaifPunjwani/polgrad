# KL estimators: signs, expectations, and as-loss gradients

This page fixes the sign convention for `polgrad.kl` (stated once here; the code
follows it), derives the expectation and variance behavior of each estimator, and
derives what gradient each estimator actually produces when used *as a loss*. Every
claim names the pytest node that enforces it.

## Sign convention and notation

The target is the **reverse KL** of the current policy against a frozen reference,

$$
\mathrm{KL}(\pi \,\|\, \mathrm{ref}) = \mathbb{E}_{x \sim \pi}\big[\log \pi(x) - \log \mathrm{ref}(x)\big] \;\ge\; 0,
$$

estimated from samples drawn from $\pi$ (the rollouts). Following Schulman,
"Approximating KL Divergence" (http://joschu.net/blog/kl-approx.html), write

$$
r = \frac{\mathrm{ref}(x)}{\pi(x)}, \qquad
\delta_t = \texttt{ref\_logprobs}_t - \texttt{logprobs}_t = \log r ,
$$

so that $\mathbb{E}_\pi[\delta] = -\mathrm{KL}$. The four per-token estimators are

| kind | formula in $\delta$ | value at $\delta = 0$ |
| --- | --- | --- |
| `K1` | $k_1 = -\delta = \texttt{logprobs} - \texttt{ref\_logprobs}$ | $0$ |
| `K2` | $k_2 = \tfrac{1}{2}\delta^2$ | $0$ |
| `K3` | $k_3 = e^{\delta} - 1 - \delta$ | $0$ |
| `ABS` | $\lvert\delta\rvert$ | $0$ |

Hand-computed values at $\delta = -1$ and $\delta = 0.5$ are pinned in
`tests/test_kl.py::test_kl_estimate_golden_values`. Masked positions return exactly $0$
in both the value and the gradient
(`tests/test_kl.py::test_kl_estimate_masked_positions_are_zero_in_value_and_gradient`,
`tests/test_kl.py::test_kl_estimate_mask_invariance`).

Scope note: verl's `kl_penalty("full")` computes the exact per-position KL from
full-vocabulary logprobs; polgrad operates on sampled-token logprob streams only, so it
is out of scope (see the `polgrad.kl` module docstring).

## Expectations

**$\mathbb{E}[k_1] = \mathrm{KL}$ exactly** — it is the definition:
$\mathbb{E}_\pi[-\delta] = \mathbb{E}_\pi[\log \pi - \log \mathrm{ref}] = \mathrm{KL}$.

**$\mathbb{E}[k_3] = \mathrm{KL}$ exactly.** Since $\pi$ has full support (softmax
policies), $\mathbb{E}_\pi[r] = \sum_x \pi(x)\,\frac{\mathrm{ref}(x)}{\pi(x)} = \sum_x
\mathrm{ref}(x) = 1$, hence

$$
\mathbb{E}[k_3] = \mathbb{E}[e^{\delta}] - 1 - \mathbb{E}[\delta]
= 1 - 1 + \mathrm{KL} = \mathrm{KL}.
$$

Moreover $k_3 \ge 0$ pointwise because $e^{\delta} \ge 1 + \delta$ (the exponential lies
above its tangent at $0$), and $k_2, \lvert\delta\rvert \ge 0$ trivially
(`tests/test_kl.py::test_k2_k3_abs_are_pointwise_nonnegative`).

Both exact-expectation claims are MC-certified against closed-form categorical KL: with
$n = 300000$ seeded draws, the sample means of $k_1$ and $k_3$ land within
$4\hat\sigma/\sqrt{n}$ of $\sum_x p_x \log(p_x/q_x)$ (two-sided CLT miss probability
$\approx 6\times10^{-5}$ at $z = 4$;
`tests/test_kl.py::test_mc_k1_and_k3_match_closed_form_categorical_kl`).

**$\mathbb{E}[k_2]$ is biased, at third order in $\delta$.** From
$\mathbb{E}[e^{\delta}] = 1$, expanding the exponential (valid when the moments exist
and the series converges, e.g. bounded $\delta$):

$$
0 = \mathbb{E}[\delta] + \tfrac{1}{2}\mathbb{E}[\delta^2] + \tfrac{1}{6}\mathbb{E}[\delta^3] + \cdots
\;\Longrightarrow\;
\mathrm{KL} = -\mathbb{E}[\delta] = \tfrac{1}{2}\mathbb{E}[\delta^2] + \tfrac{1}{6}\mathbb{E}[\delta^3] + \cdots
$$

Since $\mathbb{E}[k_2] = \tfrac{1}{2}\mathbb{E}[\delta^2]$,

$$
\mathbb{E}[k_2] - \mathrm{KL} = -\tfrac{1}{6}\mathbb{E}[\delta^3] - \sum_{j \ge 4} \tfrac{1}{j!}\mathbb{E}[\delta^j],
$$

so the two agree to second order as $\delta \to 0$ but differ in general ($\mathbb{E}[k_2]
= \tfrac12 \mathbb{E}_\pi[(\log r)^2]$ is itself an $f$-divergence, with $f(u) =
\tfrac12 \log^2 u$, not the KL). Concrete instance: for $\pi = (0.4, 0.3, 0.2, 0.1)$ and
$q = (0.1, 0.2, 0.3, 0.4)$, exact enumeration gives $\mathrm{KL} = 0.4564$ while
$\mathbb{E}[k_2] = 0.5216$, a bias of $+0.0651$
(`tests/test_kl.py::test_k2_expected_value_bias_demonstrated_on_tabular_policy`).

**`ABS` is not an estimator of KL.** By Jensen, $\mathbb{E}\lvert\delta\rvert \ge
\lvert\mathbb{E}\delta\rvert = \mathrm{KL}$, with equality only when $\delta$ has
almost-surely constant sign. It is included solely for conformance with verl's
`kl_penalty("abs")` and is documented as such in the module docstring.

**Variance near $\pi \approx \mathrm{ref}$.** $\operatorname{Var}[k_1] =
\operatorname{Var}[\delta]$ is first order in the policy gap, while $k_3 =
\tfrac12\delta^2 + O(\delta^3)$ fluctuates only at second order. So for near-identical
policies $\operatorname{var}(k_3) < \operatorname{var}(k_1)$, verified on a seeded MC
draw (`tests/test_kl.py::test_var_k3_below_var_k1_for_near_identical_policies`). That
makes $k_3$ the lower-variance unbiased monitor of the two; the next section derives
why the gradient it produces *as a loss* is nevertheless biased.

## As-loss pathwise gradients

When an estimator is aggregated and back-propagated, autograd differentiates the
formula along the sampled tokens ("pathwise"), with
$\partial\delta/\partial\,\texttt{logprobs} = -1$:

| kind | per-token pathwise gradient | on-policy expectation |
| --- | --- | --- |
| `K1` | $\nabla k_1 = \nabla\,\texttt{logprobs}$ | $0$ |
| `K2` | $\nabla k_2 = (\texttt{logprobs} - \texttt{ref\_logprobs})\,\nabla\,\texttt{logprobs}$ | $\nabla\,\mathrm{KL}(\pi\|\mathrm{ref})$ |
| `K3` | $\nabla k_3 = (1 - e^{\delta})\,\nabla\,\texttt{logprobs}$ | $\pi - q$ on tabular policies $\ne \nabla\,\mathrm{KL}$ |

The per-token formulas themselves are checked against autograd in
`tests/test_kl.py::test_kl_estimate_gradients_match_analytic_formulas`, and
`tests/test_kl.py::test_fp64_gradcheck_kl_loss` runs `torch.autograd.gradcheck` on
`kl_loss` for every kind.

**k1 as a loss optimizes nothing.** $\nabla k_1 = \nabla \log \pi_\theta(x)$, and by the
score identity

$$
\mathbb{E}_\pi[\nabla \log \pi_\theta] = \sum_x \pi_\theta(x) \nabla \log \pi_\theta(x)
= \nabla \sum_x \pi_\theta(x) = \nabla 1 = 0 ,
$$

verified by exact enumeration on a tabular softmax policy
(`tests/test_kl.py::test_k1_as_loss_expected_gradient_is_zero`).

**k2 as a loss is the unbiased score-function gradient of the reverse KL.**
Differentiate $\mathrm{KL}(\theta) = \sum_x \pi_\theta(x)\big(\log \pi_\theta(x) - \log
q(x)\big)$:

$$
\nabla \mathrm{KL}
= \sum_x \nabla \pi_\theta(x) \log\frac{\pi_\theta(x)}{q(x)}
+ \underbrace{\sum_x \pi_\theta(x) \nabla \log \pi_\theta(x)}_{=\,0}
= \mathbb{E}_\pi\Big[\big(\log \pi_\theta - \log q\big)\,\nabla \log \pi_\theta\Big],
$$

which is exactly the per-sample $\nabla k_2$. Hence
`reverse_kl_grad_surrogate` $= \mathrm{agg}\big(\mathrm{sg}[\texttt{logprobs} -
\texttt{ref\_logprobs}]\cdot\texttt{logprobs}\big)$ and `kl_loss(K2, agg)` have the
same gradient — in polgrad the equality is exact (bitwise): both backward passes reduce
to the same weight-times-difference product, the intermediate factors differing only by
exact power-of-two scalings.
Enforced by `tests/test_kl.py::test_k2_as_loss_gradient_equals_reverse_kl_grad_surrogate`
(all four aggregations), with the unbiasedness itself checked by exact enumeration in
`tests/test_kl.py::test_k2_as_loss_expected_gradient_equals_analytic_grad_kl` and the
score-function form in
`tests/test_kl.py::test_reverse_kl_grad_surrogate_gradient_is_score_function_sample`.

**k3 as a loss is biased.** $\nabla k_3 = (1 - e^{\delta})\nabla\,\texttt{logprobs} =
\big(1 - \tfrac{q}{\pi}\big)\nabla \log \pi$. On a tabular softmax policy
($\nabla_{\theta_j} \log \pi(x) = \mathbf{1}[x = j] - \pi_j$):

$$
\mathbb{E}_\pi\big[\nabla_{\theta_j} k_3\big]
= \sum_x \pi_x \Big(1 - \frac{q_x}{\pi_x}\Big)\big(\mathbf{1}[x=j] - \pi_j\big)
= \sum_x (\pi_x - q_x)\big(\mathbf{1}[x=j] - \pi_j\big)
= (\pi_j - q_j) - \pi_j \underbrace{\textstyle\sum_x (\pi_x - q_x)}_{=\,0}
= \pi_j - q_j ,
$$

whereas the analytic gradient (from the k2 derivation, expanded at the softmax) is

$$
\nabla_{\theta_j} \mathrm{KL} = \pi_j \Big(\log\frac{\pi_j}{q_j} - \mathrm{KL}\Big).
$$

These differ in general. Worked instance $\pi = (0.4, 0.3, 0.2, 0.1)$,
$q = (0.1, 0.2, 0.3, 0.4)$ (KL $= 0.4564$):

$$
\pi - q = (0.3,\; 0.1,\; -0.1,\; -0.3), \qquad
\nabla \mathrm{KL} = (0.3719,\; -0.0153,\; -0.1724,\; -0.1843),
$$

a per-component gap of $(-0.0719,\; 0.1153,\; 0.0724,\; -0.1157)$ — note the sign flip
in component 2: descending the k3 loss moves $\theta_2$ in the *opposite* direction
from descending the true KL. Verified by exact enumeration
(`tests/test_kl.py::test_k3_as_loss_expected_gradient_is_pi_minus_q_not_grad_kl`) and by
MC on $10^6$ seeded samples, where the MC gradient of `kl_loss(K3, TOKEN_MEAN)` matches
$\pi - q$ within a per-component CLT tolerance while $\pi - q$ sits more than $10$
tolerances away from $\nabla \mathrm{KL}$
(`tests/test_kl.py::test_k3_as_loss_gradient_bias_mc_gap_vs_analytic_grad_kl`).
The pathologies of gradient-of-estimator KL regularization are studied in
"A Comedy of Estimators: On KL Regularization in RL Training of LLMs" (arXiv
2512.21852) and "Rethinking KL Regularization in RLHF: From Value Estimation to
Gradient Optimization" (arXiv 2510.01555).

## `kl_loss` and `KLLossConfig`

`kl_loss` is `aggregate(kl_estimate(...), aggregation)`, bitwise
(`tests/test_kl.py::test_kl_loss_equals_aggregate_of_kl_estimate`); the aggregation
weights and `norm_len` rules are derived in
[aggregation.md](aggregation.md). `KLLossConfig` is inert frozen data — constructing it
with `norm_len=None` is always legal and the `TOKEN_SUM_NORM` requirement fires at
`kl_loss` call time (`tests/test_kl.py::test_kl_loss_config_is_frozen_data`,
`tests/test_kl.py::test_reverse_kl_grad_surrogate_requires_norm_len_for_token_sum_norm`).

## KL in the reward

The reward placement (PPO-RLHF style) folds the penalty into the per-token rewards at
rollout time,

$$
r_t \;\leftarrow\; r_t - \beta\, k_t\big(\texttt{old\_logprobs},\, \texttt{ref\_logprobs}\big),
$$

using the **sampling policy's** logprobs — the penalty is a constant target for the
advantage estimator, not a gradient path, so `kl_in_reward` returns a detached tensor
(`tests/test_kl.py::test_kl_in_reward_golden_values`,
`tests/test_kl.py::test_kl_in_reward_is_detached`,
`tests/test_kl.py::test_kl_in_reward_mask_invariance`).
