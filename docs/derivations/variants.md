# Algorithm variants: paper equations, config mappings, and silent knobs

This page maps each entry of `polgrad.registry.ALGORITHMS` onto its paper equation,
derives why the chosen `PolicyLossConfig` fields reproduce that equation, and traces
every numeric constant to the paper or the paper's released code. Values a paper leaves
unstated are listed as **silent knobs** with the provenance of the shipped default.
Papers write maximization objectives; polgrad losses are the negated quantities to
minimize (`docs/conventions.md`). Aggregation weights $w_{i,t}$ are the closed forms of
[aggregation.md](aggregation.md); surrogate branch structure and stop-gradient algebra
are derived in [losses.md](losses.md); KL estimators in [kl.md](kl.md). Config-level
claims on this page are enforced by `tests/test_registry.py`; each section names the
exact nodes.

Notation: mask $m \in \{0,1\}^{B\times T}$, row lengths $|o_i| = L_i = \sum_t m_{i,t}$,
total tokens $N = \sum_i L_i$, per-token ratio $r_{i,t} =
\exp(\texttt{logprobs}_{i,t} - \texttt{old\_logprobs}_{i,t})$.

Sources checked (equation and section numbers refer to these versions): PPO
1707.06347v2 (full PDF), DeepSeekMath 2402.03300v3, RLOO 2402.14740v2 (full PDF),
REINFORCE++ 2501.03262v2, DAPO 2503.14476v2, Dr.GRPO 2503.20783v2, MiniMax-M1
2506.13585v1, GSPO 2507.18071v2; verl PR #2953 with its merged diff, verl
`trainer/config/ppo_trainer.yaml`, `trainer/config/actor/actor.yaml`, and
`trainer/ppo/core_algos.py` at `main` (2026-07-16); TRL `trainer/grpo_config.py` and
`trainer/grpo_trainer.py` at `main` (2026-07-16).

## `ppo` — PPO, clipped surrogate (arXiv 1707.06347)

Equation (7) of the paper, with $\varepsilon$ "a hyperparameter, say, $\varepsilon =
0.2$" (Sec. 3):

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat A_t,\;
\mathrm{clip}(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\,\hat A_t\big)\Big].
$$

Mapping: `SurrogateKind.PG_CLIP` on `RatioKind.TOKEN` with
`ClipConfig(eps_low=0.2, eps_high=0.2)`. $\hat{\mathbb{E}}_t$ is the empirical average
over every timestep collected in the batch (Algorithm 1 optimizes the surrogate on all
$NT$ timesteps), i.e. `Aggregation.TOKEN_MEAN` ($w = m/N$). Advantages are GAE, eq.
(11)-(12); the paper's own hyperparameter tables pin $\gamma = 0.99$, $\lambda = 0.95$
(Tables 3, 4, and 5), shipped as `GAEConfig(gamma=0.99, lam=0.95)`. Clipping at
$\varepsilon = 0.2$ is also the best clipping row of Table 1 (score 0.82).

**KL: none.** The clipped objective is the paper's replacement for a KL penalty; the
adaptive-KL variant (eq. 8) is a separate alternative that "performed worse than the
clipped surrogate objective" (Sec. 4). RLHF deployments re-add a KL-in-reward penalty
and set $\gamma = \lambda = 1$ — verl's `ppo_trainer.yaml` defaults are `gamma: 1.0`,
`lam: 1.0`, `kl_coef: 0.001` with `use_kl_in_reward: False` — none of which appears in
the paper, so none is shipped in this spec.

**Value side.** Eq. (9) combines $L^{CLIP}$ with an unclipped squared error $L^{VF} =
(V_\theta(s_t) - V_t^{\mathrm{targ}})^2$ weighted by $c_1$ ($c_1 = 1$, entropy $c_2 =
0.01$ in the Atari Table 5; $c_1$ irrelevant for MuJoCo where the networks are not
shared, Sec. 6.1). The $\tfrac12$ factor and value clipping in common PPO code are
implementation conventions, not paper equations; `polgrad.losses.value_loss` exposes
both through `clip_eps`.

**Silent knobs.** The paper predates LLM post-training: it specifies no token
aggregation for variable-length responses (the registry ships the paper-faithful
`TOKEN_MEAN`) and no advantage whitening (`polgrad.advantages.whiten` if wanted).

Enforced by `tests/test_registry.py::test_ppo_config_matches_paper_stated_values`;
branch semantics by `tests/test_losses.py::test_pg_clip_golden_one_token_inside_clip`
and neighbors; value loss by
`tests/test_losses.py::test_value_loss_golden_clip_branches`.

## `grpo` — GRPO (DeepSeekMath, arXiv 2402.03300)

Equation (3):

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}
\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(r_{i,t}\hat A_{i,t},\,
\mathrm{clip}(r_{i,t},\, 1-\varepsilon,\, 1+\varepsilon)\,\hat A_{i,t}\big)
- \beta\, \mathbb{D}_{KL}\big[\pi_\theta \,\|\, \pi_{ref}\big]\Big)\Bigg].
$$

The $\frac1G \sum_i \frac{1}{|o_i|} \sum_t$ structure is exactly
`Aggregation.SEQ_MEAN_TOKEN_MEAN` ($w = m/(B\,L_i)$). The KL term sits inside the same
double sum, i.e. it is aggregated identically — `KLLossConfig(aggregation=None)`
inherits the policy aggregation.

**KL estimator.** Eq. (4), "we estimate the KL divergence with the following unbiased
estimator":

$$
\mathbb{D}_{KL}\big[\pi_\theta \| \pi_{ref}\big] =
\frac{\pi_{ref}(o_{i,t}\mid q, o_{i,<t})}{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}
- \log\frac{\pi_{ref}(o_{i,t}\mid q, o_{i,<t})}{\pi_\theta(o_{i,t}\mid q, o_{i,<t})} - 1 .
$$

With $\delta_t = \texttt{ref\_logprobs}_t - \texttt{logprobs}_t$ this is $e^{\delta} -
\delta - 1 = k_3$ ([kl.md](kl.md), value form pinned by
`tests/test_kl.py::test_kl_estimate_golden_values`). Placement is **loss**, not reward
("different from the KL penalty term used in [...], we estimate the KL divergence
with..." — the term is subtracted inside the objective). $\beta = 0.04$: "The KL
coefficient is 0.04" (Sec. 4.2). Hence `KLLossConfig(kind=K3, coef=0.04)`.

**Advantage.** Outcome rewards "normalized by subtracting the group average and
dividing by the group standard deviation": $\hat A_{i,t} = (r_i -
\mathrm{mean}(\mathbf r))/\mathrm{std}(\mathbf r)$, i.e.
`grpo_advantages` with `GroupNormConfig(center=True, scale="std")`
(`tests/test_advantages.py::test_grpo_closed_form_single_group`).

**Silent knobs.**

- Clip $\varepsilon$: written in eq. (3), never valued anywhere in the paper. Shipped
  0.2 = the PPO paper's value, the TRL `GRPOConfig` default (`epsilon = 0.2`), and the
  Dr.GRPO reproduction's setting (arXiv 2503.20783 Table 6).
- Std guard and Bessel correction: unstated; `GroupNormConfig` defaults apply
  (`eps=1e-4` as in TRL; verl uses 1e-6, the released Dr.GRPO code 1e-8;
  `unbiased=True` matches the `torch.Tensor.std` default those codebases call).
- Group size: $G = 64$ outputs per question in the paper's run (Sec. 4.2); polgrad
  takes groups from `group_ids`, so this is data-side.
- Framework drift: current TRL defaults `beta = 0.0` (KL off), diverging from the
  paper.

Enforced by `tests/test_registry.py::test_grpo_config_matches_paper_stated_values`.

## `dr_grpo` — Dr.GRPO (arXiv 2503.20783)

Dr.GRPO modifies GRPO "by removing the length and std normalization terms":

$$
\mathcal{J}_{Dr.GRPO}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}
\sum_{t=1}^{|o_i|} \min\big(r_{i,t}\tilde A_{i,t},\,
\mathrm{clip}(r_{i,t},\, 1-\varepsilon,\, 1+\varepsilon)\,\tilde A_{i,t}\big)\Bigg],
\qquad \tilde A_{i} = r_i - \mathrm{mean}(\mathbf r).
$$

As written, $\frac1G\sum_i\sum_t$ is `SEQ_MEAN_TOKEN_SUM` ($w = m/B$). Their
implementation divides masked token sums by a **constant** instead ("we could replace
the `mask.sum(axis=dim)` with a constant value (e.g., generation budget)"; the code
divides by the global `MAX_TOKENS`), giving $w = m/(B\cdot\texttt{norm\_len})$ =
`Aggregation.TOKEN_SUM_NORM`. The two differ by the constant positive factor
$1/\texttt{norm\_len}$ (weight closed forms:
`tests/test_aggregate.py::test_effective_weights_golden_closed_forms`), which rescales
the gradient uniformly but — unlike GRPO's $1/L_i$ — identically for every sequence.
The registry ships the code-faithful `TOKEN_SUM_NORM` with **`norm_len=None`**, because
the budget is deployment-specific: apply
`dataclasses.replace(spec.loss, norm_len=<generation budget>)` before calling
`policy_loss`, which raises `ValueError` until then
(`tests/test_registry.py::test_dr_grpo_ships_norm_len_none_and_raises_until_replaced`).

**Pinned values.** Clip $\varepsilon = 0.2$ (Table 6, "Policy clipping parameter:
0.2"). KL: none — "We will assume $\beta = 0$ throughout this paper", and Table 6 lists
both KL coefficients as 0.0. Advantage: group centering only,
`GroupNormConfig(center=True, scale="none")`
(`tests/test_advantages.py::test_grpo_scale_none_equals_per_group_centering`). Their
runs: 3000-token response budget, 8 responses per question, temperature 1.0 (Table 6).

## `dapo` — DAPO (arXiv 2503.14476)

Equation (12), the token-level objective:

$$
\mathcal{J}_{DAPO}(\theta) = \mathbb{E}\Bigg[\frac{1}{\sum_{i=1}^{G}|o_i|}
\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\big(r_{i,t}\hat A_{i,t},\,
\mathrm{clip}(r_{i,t},\, 1-\varepsilon_{low},\, 1+\varepsilon_{high})\,\hat A_{i,t}\big)\Bigg].
$$

$\frac{1}{\sum_i |o_i|}\sum_i\sum_t = \frac1N\sum_{i,t}$ is exactly
`Aggregation.TOKEN_MEAN` (the full DAPO objective with Clip-Higher and the dynamic
sampling constraint is eq. (8); eq. (12) is its token-level loss). Clip-Higher: "we set
the clipping parameter $\varepsilon_{low}$ to 0.2 and $\varepsilon_{high}$ to 0.28",
i.e. `ClipConfig(eps_low=0.2, eps_high=0.28)`. KL: "we will exclude the KL term from
our proposed algorithm" — placement `none`. Advantage: eq. (9) keeps GRPO's group
normalization $(r_i - \mathrm{mean})/\mathrm{std}$ (silent on the $+\varepsilon$ guard
and Bessel correction; `GroupNormConfig` defaults apply).

**Outside the loss.** DAPO's dynamic sampling (filtering prompts whose group is all
correct/all wrong) and overlong reward shaping operate on the batch and reward,
upstream of the loss equation; they have no `PolicyLossConfig` representation and are
deliberately absent from the spec.

Enforced by `tests/test_registry.py::test_dapo_config_matches_paper_stated_values`.

## `gspo` and `gspo_token` — GSPO (arXiv 2507.18071)

Sequence-level importance ratio, eq. (7):

$$
s_i(\theta) = \Big(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{old}}(y_i\mid x)}\Big)^{1/|y_i|}
= \exp\Big(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}
\log\frac{\pi_\theta(y_{i,t}\mid x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}\mid x, y_{i,<t})}\Big)
\quad = \texttt{RatioKind.SEQUENCE},
$$

pinned by
`tests/test_losses.py::test_gspo_sequence_ratio_value_matches_masked_mean_exponent`.
Objective, eq. (5):

$$
\mathcal{J}_{GSPO}(\theta) = \mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G}
\min\big(s_i(\theta)\hat A_i,\, \mathrm{clip}(s_i(\theta),\, 1-\varepsilon_{left},\,
1+\varepsilon_{right})\,\hat A_i\big)\Big].
$$

polgrad broadcasts $s_i$ and $\hat A_i$ to the row's tokens, so every per-token value
of row $i$ equals the same constant $c_i$, and

$$
\mathrm{SEQ\_MEAN\_TOKEN\_MEAN} = \frac1B\sum_i \frac{1}{L_i}\sum_t m_{i,t}\, c_i
= \frac1B\sum_i c_i ,
$$

which is eq. (5) exactly. **Clip:** "we set the left and right clipping ranges in
Equation (5) to 3e-4 and 4e-4, respectively" — `ClipConfig(eps_low=3e-4,
eps_high=4e-4)`. These are two orders of magnitude tighter than token-level PPO
clipping because the $1/|y_i|$ exponent concentrates $s_i$ near 1; the same paper runs
its GRPO baseline at 0.2/0.27. **KL:** none in the objective ("we omit the KL
regularization term hereinafter"). **Advantage:** eq. (6), GRPO group normalization
$(r_i - \mathrm{mean})/\mathrm{std}$ (silent knobs as in the `grpo` entry).

**GSPO-token.** Eq. (13) restores the per-token double sum
($\frac1G\sum_i\frac{1}{|y_i|}\sum_t$ = `SEQ_MEAN_TOKEN_MEAN`) over the stop-gradient
ratio of eq. (14):

$$
s_{i,t}(\theta) = \mathrm{sg}[s_i(\theta)] \cdot
\frac{\pi_\theta(y_{i,t}\mid x, y_{i,<t})}{\mathrm{sg}[\pi_\theta(y_{i,t}\mid x, y_{i,<t})]}
\quad = \texttt{RatioKind.SEQUENCE\_TOKEN}.
$$

The value equals $s_i$ bitwise and the gradient is token-local
($\mathrm{sg}[s_i]\,\nabla\texttt{logprobs}_{i,t}$) — derived in
[losses.md](losses.md), pinned by
`tests/test_losses.py::test_gspo_sequence_token_value_equals_sequence_ratio_value` and
`::test_gspo_sequence_token_gradient_is_token_local`. With per-sequence advantages the
two variants produce identical losses and differ only in gradient decomposition
(`tests/test_losses.py::test_gspo_sequence_and_sequence_token_gradients_differ`); the
paper introduces the token variant to admit per-token advantage customization.

Enforced by `tests/test_registry.py::test_gspo_configs_match_paper_stated_values`.

## `cispo` — CISPO (MiniMax-M1, arXiv 2506.13585 eq. 4-5)

$$
\mathcal{J}_{CISPO}(\theta) = \mathbb{E}\Bigg[\frac{1}{\sum_{i=1}^{G}|o_i|}
\sum_{i=1}^{G}\sum_{t=1}^{|o_i|} \mathrm{sg}\big[\hat r_{i,t}(\theta)\big]\,
\hat A_{i,t}\, \log\pi_\theta(o_{i,t}\mid q, o_{i,<t})\Bigg],
\qquad
\hat r_{i,t}(\theta) = \mathrm{clip}\big(r_{i,t}(\theta),\,
1-\varepsilon^{IS}_{low},\, 1+\varepsilon^{IS}_{high}\big).
$$

Mapping: `SurrogateKind.CISPO` on `RatioKind.TOKEN`; the $\frac{1}{\sum_i|o_i|}$
normalizer is `Aggregation.TOKEN_MEAN`. The clipped weight is detached — gradients flow
only through the REINFORCE factor — which makes CISPO bitwise identical to REINFORCE on
the pre-scaled advantages $\mathrm{sg}[\hat r]\cdot \hat A$
(`tests/test_losses.py::test_cispo_gradient_equals_detached_weight_scaled_reinforce`).
KL: "There is no KL penalty term in CISPO". Advantage: group relative, adopted from
GRPO, $(R_i - \mathrm{mean})/\mathrm{std}$.

**The undisclosed clip bound.** The paper states the one-sided setting — "we did not
impose a lower bound on the IS weight by setting $\varepsilon^{IS}_{low}$ to a large
value; instead, we only tuned $\varepsilon^{IS}_{high}$" — hence
`ClipConfig(eps_low=None, ...)`. But **no numeric $\varepsilon^{IS}_{high}$ appears
anywhere in 2506.13585v1, appendices included**. The shipped `eps_high=0.2` is
therefore *not* a paper value: it is the upper bound of verl's
`compute_policy_loss_cispo`, whose `clip_ratio_high` falls back to the actor default
`clip_ratio: 0.2` (i.e. clamp at $1 + 0.2$; note verl defaults to a two-sided clamp).
Replace it to match a tuned setting. A parameterization caution: TRL's
`loss_type="cispo"` clamps the ratio at the **absolute** value `epsilon_high`
(`torch.clamp(coef_1, max=self.epsilon_high)`), not at $1+\varepsilon_{high}$ —
configs are not portable between the two conventions.

Enforced by
`tests/test_registry.py::test_cispo_config_matches_paper_and_flags_undisclosed_eps`.

## `rloo` — RLOO (arXiv 2402.14740)

The estimator, for $k$ samples of one prompt:

$$
\frac{1}{k}\sum_{i=1}^{k}\Big[R(y^{(i)}, x) - \frac{1}{k-1}\sum_{j\neq i}
R(y^{(j)}, x)\Big]\, \nabla\log\pi(y^{(i)}\mid x),
$$

with the whole completion "as a single action, as opposed to each token". The surrogate
whose gradient this is (for constant baselines) is $-\frac1k\sum_i A_i \log\pi(y^{(i)}
\mid x)$, and $\log\pi(y\mid x) = \sum_t \texttt{logprobs}_t$, so the polgrad form is
`SurrogateKind.REINFORCE` with `[B]` advantages under
`Aggregation.SEQ_MEAN_TOKEN_SUM` ($\frac1B\sum_i\sum_t$). No ratio and no clipping:
the paper removes PPO clipping entirely (Sec. 3.2, "Clipping is Rarely Necessary in
RLHF": clipping fired on "< 5% of the time per batch" and its removal "does not impact
learning meaningfully"), so `clip=None` and the `ratio` field is the mandated `TOKEN`
placeholder (unused by REINFORCE, contract 4.3).

**Advantage.** Leave-one-out baseline, `rloo_advantages` (no config). The two algebraic
forms $r_i - \mathrm{mean}_{j\neq i}(r_j) = \frac{G}{G-1}(r_i - \mathrm{mean})$ are
proved in [advantages.md](advantages.md) and pinned by
`tests/test_advantages.py::test_rloo_two_form_identity_exact_on_dyadic_inputs`.

**KL in the reward.** $R(x, y) = r_\phi(x, y) - \beta\log\frac{\pi_\theta(y\mid
x)}{\pi_{ref}(y\mid x)}$. The sequence log-ratio decomposes over tokens:
$\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} = \sum_t (\texttt{old\_logprobs}_t -
\texttt{ref\_logprobs}_t) = \sum_t k_1(t)$ evaluated at the sampling policy — exactly
`kl_in_reward(kind=K1)`, whose per-sequence sum reproduces the paper's penalty
(`tests/test_registry.py::test_kl_reward_specs_compose_with_kl_in_reward`).
$\beta = 0.03$ for TL;DR and $\beta = 0.10$ for Anthropic-HH ("We use a $\beta$ value
of 0.03" / "We use $\beta = 0.10$ for all Anthropic-HH experiments", Appendix C),
swept over $\{0.1, 0.25, 0.5, 1.0\}$ in Sec. 5.2.2. The spec ships the TL;DR value
0.03; it is dataset-dependent, replace to match. $k \in \{2, 4\}$ in the paper.

**Silent knobs.** No token-level aggregation exists in the single-action formulation
(the registry's `SEQ_MEAN_TOKEN_SUM` is the identity-preserving choice, not a paper
knob); no advantage std normalization.

Enforced by `tests/test_registry.py::test_rloo_config_matches_paper_stated_values`.

## `reinforce_pp` — REINFORCE++ (arXiv 2501.03262)

The objective keeps the PPO-clip surrogate, eq. (1):

$$
\mathcal{L}_{PPO}(\theta) = \mathbb{E}\Big[\frac{1}{|o|}\sum_t
\min\big(s_t(\theta)A_t,\, \mathrm{clip}(s_t(\theta),\, 1-\epsilon,\,
1+\epsilon)A_t\big)\Big],
$$

a per-sequence token mean under an expectation over sequences =
`Aggregation.SEQ_MEAN_TOKEN_MEAN`, with $\epsilon = 0.2$ ("The clipping parameter
($\epsilon$) is set to 0.2", Sec. 4.2.3).

**Token-level KL in the reward.** Eq. (8)-(9):

$$
A_{q, o_t} = r(o_{1:t}, q) - \beta\sum_{i=t}^{T}\mathrm{KL}(i),
\qquad
\mathrm{KL}(t) = \log\frac{\pi^{RL}_{\theta_{old}}(o_t\mid q, o_{<t})}
{\pi^{SFT}(o_t\mid q, o_{<t})},
$$

i.e. the $k_1$ estimator evaluated at the sampling policy against the frozen SFT
reference, folded into the per-token reward — `kl_in_reward(kind=K1, coef=0.001)`
($\beta = 0.001$, Sec. 4.2.3) followed by reward-to-go at $\gamma = 1.0$ (Sec. 4.2.3).

**Global advantage normalization.** Eq. (10): $A^{norm} = (A -
\mathrm{mean}(A))/\mathrm{std}(A)$ over the global batch. polgrad composes this as
`reinforce_pp_advantages(group_ids=None, batch_norm=True)`: the global-mean baseline
$r - \mathrm{mean}(r)$ followed by division by the batch std equals eq. (10) because
the std is shift-invariant
(`tests/test_advantages.py::test_reinforce_pp_global_baseline_closed_form`,
`::test_reinforce_pp_batch_norm_closed_form`). The REINFORCE++-baseline variant
(Appendix A) subtracts the per-group mean and then divides by the batch std — pass
`group_ids` (`tests/test_advantages.py::test_reinforce_pp_group_baseline_closed_form`).

**Silent knobs.** The paper does not state the $+\varepsilon$ guard on the std or the
Bessel convention; `ReinforcePPConfig(eps=1e-8)` is polgrad's division guard with the
Bessel-corrected std documented in [advantages.md](advantages.md).

Enforced by
`tests/test_registry.py::test_reinforce_pp_config_matches_paper_stated_values`.

## `grpo_tis` — GRPO + truncated importance sampling (verl PR #2953)

The rollout engine's logprobs differ numerically from the trainer's recomputed
`old_logprobs`; training on rollout samples with trainer gradients is therefore
silently off-policy. verl PR #2953 (merged 2025-08-26, linking the blog "Your Efficient
RL Framework Secretly Brings You Off-Policy RL Training") corrects the policy loss with
a truncated importance-sampling weight, token-level, in
`compute_policy_loss_vanilla`:

```python
tis_imp_ratio = torch.exp(old_log_prob - rollout_log_probs)
tis_imp_ratio = torch.clamp(tis_imp_ratio, max=config.tis_imp_ratio_cap)
pg_losses = pg_losses * tis_imp_ratio
```

which is

$$
\ell_t \leftarrow \mathrm{sg}[w_t]\cdot\ell_t, \qquad
w_t = \min\big(e^{\texttt{old\_logprobs}_t - \texttt{rollout\_logprobs}_t},\, C\big)
\quad = \texttt{ISCorrectionConfig(cap=}C\texttt{, level="token")} .
$$

(In verl neither logprob stream carries gradient; polgrad detaches explicitly, so the
weight is data in both.) The spec is the `grpo` entry plus this correction — enforced
field-for-field by
`tests/test_registry.py::test_grpo_tis_is_grpo_plus_verl_pr_2953_correction`; the
correction semantics are pinned by
`tests/test_losses.py::test_is_correction_weight_one_is_noop` and
`::test_is_correction_cap_binds`.

**The cap.** verl ships `tis_imp_ratio_cap: -1` — TIS disabled — and prescribes no
tuned value; the PR's usage example enables it with
`+actor_rollout_ref.actor.behav_imp_weight_cap=10.0`. The registry ships that example
value, $C = 10.0$; replace it to match your deployment.

## Full-pipeline GRPO vs Dr.GRPO: two independent factors

The `grpo` and `dr_grpo` specs differ in **two multiplicative places**, and matching
only one of them does not reproduce the other algorithm.

**1. Aggregation weights.** From the closed forms of
[aggregation.md](aggregation.md)
(`tests/test_aggregate.py::test_effective_weights_golden_closed_forms`):

$$
w^{GRPO}_{i,t} = \frac{m_{i,t}}{B\,L_i}
\qquad\text{vs}\qquad
w^{Dr.GRPO}_{i,t} = \frac{m_{i,t}}{B\cdot\texttt{norm\_len}} .
$$

With **identical advantages** fed to both losses, on a batch whose rows all have length
$L$ every weight ratio is the constant $\texttt{norm\_len}/L$, so

$$
\mathcal{L}_{SEQ\_MEAN\_TOKEN\_MEAN} = \frac{\texttt{norm\_len}}{L}\cdot
\mathcal{L}_{TOKEN\_SUM\_NORM}
$$

exactly (equal-length collapse:
`tests/test_aggregate.py::test_equal_length_token_mean_collapses_to_seq_mean_token_mean`,
`::test_equal_length_token_sum_norm_is_length_over_norm_len_times_token_mean`). On
ragged batches no scalar relates them: GRPO gives each token of sequence $i$ weight
$\propto 1/L_i$ — a fixed per-token objective contributes less per token the longer its
sequence — while Dr.GRPO weights every token equally. This is the length bias Dr.GRPO
removes on the loss side.

**2. The advantage std factor.** The pipelines also differ *before* the loss, in the
advantage estimator:

$$
A^{GRPO}_i = \frac{r_i - \mu_{g(i)}}{\sigma_{g(i)} + \varepsilon}
= \frac{A^{Dr.GRPO}_i}{\sigma_{g(i)} + \varepsilon} .
$$

Even with the aggregation matched, GRPO rescales every sequence of group $g$ by
$1/(\sigma_g + \varepsilon)$. Two groups with **identical centered rewards** but
different within-group spread receive gradients differing by exactly that per-group
factor — low-spread groups (prompts the policy nearly always solves or nearly always
fails) are up-weighted relative to high-spread ones. This is the question-level
difficulty bias of the Dr.GRPO paper, demonstrated with the exact
$1/(\sigma_g + \varepsilon)$ factor by
`tests/test_advantages.py::test_dr_grpo_difficulty_bias_exact_std_factor`.

A full-pipeline comparison of `ALGORITHMS["grpo"]` against `ALGORITHMS["dr_grpo"]`
therefore compounds both factors: per-sequence loss weights $1/L_i$ vs
$1/\texttt{norm\_len}$, and per-group advantage scales $1/(\sigma_g+\varepsilon)$ vs
$1$. Switching only `Aggregation` (or only `GroupNormConfig.scale`) produces a hybrid
that is neither paper's algorithm.
