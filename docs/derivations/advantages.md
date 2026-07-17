# Advantage estimators — derivations

Semantics and proofs for `polgrad.advantages`. Shapes, masking, and error rules follow
[docs/conventions.md](../conventions.md). Every claim names the test that enforces it;
all tests live in `tests/test_advantages.py`.

## Group normalization and the Dr.GRPO difficulty bias

Rows are partitioned into prompt groups by `group_ids`. For a group $g$ of size $G_g$
with sequence rewards $\{r_j\}$, define

$$
\mu_g = \frac{1}{G_g}\sum_{j \in g} r_j,
\qquad
\sigma_g = \sqrt{\frac{1}{G_g - c}\sum_{j \in g}(r_j - \mu_g)^2},
$$

with $c = 1$ (Bessel correction) when `GroupNormConfig.unbiased=True`, matching the
`torch.Tensor.std` default, and $c = 0$ otherwise. `grpo_advantages` computes

$$
A_i = \frac{r_i - \mu_{g(i)}}{\sigma_{g(i)} + \varepsilon}
\quad \text{(GRPO, scale="std")},
\qquad
A_i = r_i - \mu_{g(i)}
\quad \text{(Dr.GRPO, scale="none")},
$$

with `center=False` keeping $r_i$ in the numerator. The GRPO paper (DeepSeekMath,
arXiv 2402.03300) does not state $\varepsilon$; polgrad's default is $10^{-4}$
(framework choices are recorded in the `GroupNormConfig` docstring). Group labels are
arbitrary non-negative integers; only the induced row partition matters
(`test_grpo_group_labels_are_arbitrary_nonnegative_ints`).

Worked case (`test_grpo_closed_form_single_group`): $r = [1, 2, 3]$ in one group gives
$\mu = 2$, centered rewards $[-1, 0, 1]$, squared-deviation sum $2$, Bessel denominator
$2$, so $\sigma = 1$ and $A = [-1, 0, 1]/(1 + 10^{-4})$. With `unbiased=False` the same
batch gives $\sigma = \sqrt{2/3}$ (`test_grpo_unbiased_false_uses_population_std`);
with `center=False`, $A = [1, 2, 3]/(1 + 10^{-4})$
(`test_grpo_center_false_divides_raw_rewards`).

### Invariance properties

- **Shift invariance.** Under $r_j \mapsto r_j + c$: $\mu_g \mapsto \mu_g + c$, the
  deviations $r_j - \mu_g$ are unchanged, hence $\sigma_g$ and $A$ are unchanged for
  both scales (`test_grpo_shift_invariance`).
- **Scale behavior.** Under $r_j \mapsto a r_j$ with $a > 0$: deviations and
  $\sigma_g$ both scale by $a$, so

  $$
  A_i(a r) = \frac{a\,(r_i - \mu_g)}{a\,\sigma_g + \varepsilon},
  $$

  which equals $A_i(r)$ exactly only at $\varepsilon = 0$
  (`test_grpo_std_scale_is_scale_invariant_at_eps_zero`); with `scale="none"` the map
  is linear, $A(a r) = a\,A(r)$ (`test_grpo_scale_none_is_scale_equivariant`).
- **Dr.GRPO = plain centering.** `scale="none", center=True` is exactly per-group mean
  subtraction (`test_grpo_scale_none_equals_per_group_centering`).

### The difficulty bias

By construction the two estimators differ by the per-group factor

$$
A^{\text{GRPO}}_i = \frac{1}{\sigma_{g(i)} + \varepsilon}\, A^{\text{Dr.GRPO}}_i,
$$

so GRPO up-weights whole groups whose rewards have low spread ("too easy / too hard"
prompts have low $\sigma_g$) — the question-level difficulty bias identified by
Dr.GRPO (arXiv 2503.20783). Demonstration
(`test_dr_grpo_difficulty_bias_exact_std_factor`): take group $0 = [1, -1]$ and group
$1 = [1, -1, 1, -1]$. Both have $\mu_g = 0$, so the centered (Dr.GRPO) rewards are
elementwise identical, $\pm 1$. The stds differ through the group size alone:

$$
\sigma_0 = \sqrt{\tfrac{1^2 + 1^2}{2 - 1}} = \sqrt{2},
\qquad
\sigma_1 = \sqrt{\tfrac{1^2 + 1^2 + 1^2 + 1^2}{4 - 1}} = \sqrt{4/3},
$$

so GRPO assigns the two groups different advantage magnitudes,
$1/(\sqrt{2} + \varepsilon) \neq 1/(\sqrt{4/3} + \varepsilon)$, while Dr.GRPO treats
them identically. The test asserts the $1/(\sigma_g + \varepsilon)$ factor bitwise.

### Degenerate groups

A group of size 1 has an undefined standard deviation ($0/0$ under Bessel), so
`scale="std"` raises `ValueError` (`test_grpo_scale_std_group_of_one_raises`). verl at
the pinned commit special-cases singleton groups with mean $= 0$, std $= 1$, silently
emitting the raw *uncentered* reward $A = r_i/(1+\varepsilon)$ — and the same
special-case applies on its Dr.GRPO path (`norm_adv_by_std_in_grpo=False` yields
$r_i - 0$, the raw reward), so it does not match polgrad's `scale="none"` singleton
advantage of 0 either. polgrad raises instead. This is not registered as a Deviation
because the framework advantage estimators could not be vendored — see the scope note
in `polgrad/conformance/deviations.py`. Centering alone remains well defined — a
singleton's advantage is 0 (`test_grpo_scale_none_allows_group_of_one`).

## The RLOO identity

RLOO (arXiv 2402.14740) baselines each sequence with the mean reward of the *other*
members of its group. With $S = \sum_{j \in g} r_j = G\mu_g$:

$$
A_i \;=\; r_i - \frac{1}{G-1}\sum_{j \neq i} r_j
\;=\; r_i - \frac{S - r_i}{G-1}
\;=\; \frac{(G-1)\,r_i - S + r_i}{G-1}
\;=\; \frac{G\,r_i - S}{G-1}
\;=\; \frac{G}{G-1}\,\bigl(r_i - \mu_g\bigr).
$$

Corollary 1: $\sum_i A_i = \frac{G}{G-1}(S - G\mu_g) = 0$ within each group
(`test_rloo_group_advantages_sum_to_zero`).

Corollary 2: per-group mean centering (Dr.GRPO / REINFORCE++-baseline) equals
$\frac{G-1}{G}$ times RLOO — the LOO baseline removes the $\tfrac{1}{G} r_i$
self-contribution that plain centering leaves in.

Worked case (`test_rloo_closed_form_two_groups`): group $[1, 3]$ gives
$A = [1-3,\; 3-1] = [-2, 2]$; group $[2, 4, 6]$ gives
$A = [2 - \tfrac{4+6}{2},\; 4 - \tfrac{2+6}{2},\; 6 - \tfrac{2+4}{2}] = [-3, 0, 3]$,
equal to $\tfrac{3}{2}([2,4,6] - 4) = [-3, 0, 3]$ via the scaled form.

In floating point the arrangements round differently in general, so the identity is
tested two ways: on dyadic rewards $k/8$ with $G - 1 \in \{1, 2, 4\}$ every quantity in
$r_i - (S - r_i)/(G-1)$ and $(G r_i - S)/(G-1)$ is exactly representable and the test
asserts bitwise equality of both forms and the implementation
(`test_rloo_two_form_identity_exact_on_dyadic_inputs`); on generic floats agreement is
asserted to $10^{-10}$ (`test_rloo_two_form_identity_general_floats`).

A group of size 1 leaves the leave-one-out mean over an empty set undefined, so
`rloo_advantages` raises (`test_rloo_group_of_one_raises`).

## REINFORCE++ baseline variants

REINFORCE++ (arXiv 2501.03262) is critic-free REINFORCE with a batch baseline:

$$
A_i = r_i - b_i,
\qquad
b_i =
\begin{cases}
\frac{1}{B}\sum_j r_j & \text{group\_ids=None (REINFORCE++)}\\[4pt]
\mu_{g(i)} & \text{per-group (REINFORCE++-baseline)}
\end{cases}
$$

(`test_reinforce_pp_global_baseline_closed_form`,
`test_reinforce_pp_group_baseline_closed_form`). The per-group variant without
normalization coincides exactly with GRPO `scale="none"`
(`test_reinforce_pp_group_baseline_matches_grpo_scale_none`); a singleton group is
well defined and gets advantage 0 (`test_reinforce_pp_singleton_group_allowed`).

With `batch_norm=True` the centered advantages are divided by one **global**
Bessel-corrected standard deviation:

$$
A_i \;\leftarrow\; \frac{A_i}{\operatorname{std}(A) + \varepsilon},
\qquad
\operatorname{std}(A) = \sqrt{\frac{1}{B-1}\sum_j \bigl(A_j - \bar A\bigr)^2},
$$

where $\bar A = 0$ in exact arithmetic (each group's centered values sum to zero).
Because this is a single scalar for the whole batch, it rescales every advantage
uniformly — unlike GRPO's per-group $1/(\sigma_g + \varepsilon)$, it cannot reweight
groups against each other. Worked case
(`test_reinforce_pp_batch_norm_closed_form`): $r = [1, 2, 3, 6]$ gives
$A = [-2, -1, 0, 3]$, squared sum $4 + 1 + 0 + 9 = 14$, Bessel denominator $3$, so the
output is $A / (\sqrt{14/3} + 10^{-8})$. Fewer than 2 rewards make the Bessel std
undefined and raise (`test_reinforce_pp_batch_norm_single_reward_raises`).

## GAE over right-padded responses

For a row with $L$ real tokens $t = 0, \dots, L-1$, GAE (arXiv 1506.02438) with
terminal bootstrap $V_L := 0$ uses the TD residuals

$$
\delta_t = r_t + \gamma\, V_{t+1}\,\mathbf{1}[t+1 < L] - V_t,
\qquad
A_t = \sum_{l=0}^{L-1-t} (\gamma\lambda)^l\, \delta_{t+l},
\qquad
R_t = A_t + V_t .
$$

Splitting off the $l = 0$ term and reindexing $l = l' + 1$ gives the reverse
recursion computed by the implementation:

$$
A_t = \delta_t + \gamma\lambda \sum_{l'=0}^{L-2-t} (\gamma\lambda)^{l'}\,
\delta_{(t+1)+l'} = \delta_t + \gamma\lambda\, A_{t+1},
\qquad A_L = 0,
$$

a single $O(T)$ backward scan vectorized over the batch. The test oracle `_gae_slow`
evaluates the $O(T^2)$ sum directly from the definition; agreement on ragged batches
is asserted by `test_gae_matches_slow_oracle`.

**Masked right-padded handling.** Both occurrences of the successor index $t+1$ — the
bootstrap $\gamma V_{t+1}$ and the trace $\gamma\lambda A_{t+1}$ — are multiplied by
the mask column $m_{t+1}$. For a right-padded mask, $m_{t+1} = 0$ exactly at the last
real token, which simultaneously implements $V_L = 0$ and resets the recursion, so
values in the padding never reach a real position; the outputs are then zeroed at
masked positions. Since $0 \cdot x = 0$ exactly for finite $x$, mask invariance is
bitwise (`test_gae_mask_invariance_bitwise`). A mask with a real token *after* a
padded position has no unique $L$ and is rejected
(`test_gae_rejects_non_right_padded_mask`); other shape/dtype violations are covered
by `test_gae_rejects_shape_and_mask_violations`.

**Collapse at $\gamma = \lambda = 1$.** The residual sum telescopes:

$$
A_t = \sum_{s=t}^{L-1} \delta_s
= \sum_{s=t}^{L-1} r_s + \sum_{s=t}^{L-1}\bigl(V_{s+1}\mathbf{1}[s+1<L] - V_s\bigr)
= \sum_{s=t}^{L-1} r_s + \underbrace{V_L}_{0} - V_t,
$$

so advantages equal reward-to-go minus values and $R_t = \sum_{s \ge t} r_s$
(`test_gae_gamma_lambda_one_reduces_to_reward_to_go`).

Worked ragged case (`test_gae_closed_form_ragged_batch`), $\gamma = \lambda = 1/2$,
so $\gamma\lambda = 1/4$:

| row | $r$ | $V$ | $\delta$ | $A$ | $R$ |
| --- | --- | --- | --- | --- | --- |
| 0 ($L{=}3$) | $[1,2,3]$ | $[4,2,1]$ | $\delta_2 = 3-1 = 2$; $\delta_1 = 2 + \tfrac{1}{2}\cdot 1 - 2 = \tfrac12$; $\delta_0 = 1 + \tfrac{1}{2}\cdot 2 - 4 = -2$ | $A_2 = 2$; $A_1 = \tfrac12 + \tfrac14\cdot 2 = 1$; $A_0 = -2 + \tfrac14\cdot 1 = -1.75$ | $[2.25, 3, 3]$ |
| 1 ($L{=}2$) | $[1,1,\cdot]$ | $[1,1,\cdot]$ | $\delta_1 = 0$; $\delta_0 = 1 + \tfrac12 - 1 = \tfrac12$ | $A_1 = 0$; $A_0 = \tfrac12$; padding $\to 0$ | $[1.5, 1, 0]$ |

## Broadcasting and whitening

`broadcast_to_tokens` spreads a per-sequence value over its real tokens:
$\text{out}_{i,t} = a_i\, m_{i,t}$, exactly 0 in the padding
(`test_broadcast_to_tokens_closed_form`; input validation in
`test_broadcast_to_tokens_validation`).

`whiten` standardizes a per-token tensor over all real tokens, with masked mean and
Bessel-corrected masked variance ($n = \sum_{b,t} m_{b,t}$):

$$
\mu = \frac{\sum m \cdot x}{n},
\qquad
\sigma^2 = \frac{\sum m\,(x - \mu)^2}{n - 1},
\qquad
\text{out} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}}
\;\bigl(+\,\mu \text{ iff shift\_mean=False}\bigr).
$$

`shift_mean=False` restores the mean after scaling — the convention of the
masked-whitening helpers in common PPO implementations
(`test_whiten_shift_mean_false_restores_mean`). Worked case
(`test_whiten_closed_form`): $x = [1, 2, 3, 4]$ gives $\mu = 2.5$, squared-deviation
sum $2.25 + 0.25 + 0.25 + 2.25 = 5$, $\sigma^2 = 5/3$.

The whitened output has masked mean 0 and masked variance
$\sigma^2 / (\sigma^2 + \varepsilon)$, i.e. within
$\varepsilon / (\sigma^2 + \varepsilon)$ of 1
(`test_whiten_masked_moments`). Masked junk never enters $\mu$ or $\sigma^2$ (they are
computed through $m \cdot x$ products), so mask invariance is bitwise
(`test_whiten_mask_invariance_bitwise`). With fewer than 2 real tokens the Bessel
variance is undefined and `whiten` raises
(`test_whiten_fewer_than_two_tokens_raises`).

All functions preserve the input dtype (`test_dtype_preserved_float32`) and reject
malformed grouped inputs with `ValueError`s naming the argument
(`test_grouped_input_validation`).

## References

- GRPO: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
  Models", arXiv 2402.03300.
- Dr.GRPO: "Understanding R1-Zero-Like Training: A Critical Perspective",
  arXiv 2503.20783.
- RLOO: "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from
  Human Feedback in LLMs", arXiv 2402.14740.
- REINFORCE++: "REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global
  Advantage Normalization", arXiv 2501.03262.
- GAE: "High-Dimensional Continuous Control Using Generalized Advantage Estimation",
  arXiv 1506.02438.
