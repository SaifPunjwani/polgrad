# Policy-gradient surrogates: branches, stop-gradients, and exact gradients

This page derives the per-token objective and gradient of every surrogate in
`polgrad.losses`, the dual-clip branch structure, the GSPO sequence and sequence-token
gradients from their stop-gradient ($\mathrm{sg}[\cdot]$ = `.detach()`) algebra, CISPO's
stop-gradient semantics, the truncated importance-sampling (TIS) correction, and the
clipped value loss. Every claim names the pytest node that enforces it. All losses are
quantities to **minimize**; aggregation weights $w_{i,t}$ are the closed forms of
[aggregation.md](aggregation.md).

## Notation and ratio kinds

Per token $t$ of row $i$, with mask $m \in \{0,1\}^{B \times T}$, row lengths
$L_i = \sum_t m_{i,t}$, and log-ratio $z_t = \texttt{logprobs}_t -
\texttt{old\_logprobs}_t$ (zeroed at masked positions before any `exp`, so the masked
ratio is exactly $1$ and no padding junk reaches forward values or backward formulas —
`tests/test_losses.py::test_policy_loss_mask_invariance`,
`tests/test_losses.py::test_policy_loss_tolerates_non_finite_junk_at_masked_positions`):

| `RatioKind` | value | gradient w.r.t. $\texttt{logprobs}_{i,t}$ |
| --- | --- | --- |
| `TOKEN` | $r_t = e^{z_t}$ | $\partial r_t/\partial \texttt{lp}_{i,t} = r_t$ |
| `SEQUENCE` (GSPO) | $s_i = \exp\!\big(\tfrac{1}{L_i}\textstyle\sum_t m_{i,t} z_t\big)$, broadcast | $\partial s_i/\partial \texttt{lp}_{i,t} = s_i\, m_{i,t}/L_i$ |
| `SEQUENCE_TOKEN` (GSPO-token) | $s_{i,t} = \mathrm{sg}[s_i]\cdot r_t/\mathrm{sg}[r_t]$ | $\nabla s_{i,t} = \mathrm{sg}[s_i]\,\nabla \texttt{lp}_{i,t}$ |

The sequence forms follow GSPO (Zheng et al., "Group Sequence Policy Optimization",
arXiv 2507.18071). For `SEQUENCE_TOKEN`, $r_t/\mathrm{sg}[r_t] = 1$ exactly in IEEE
arithmetic ($x/x = 1$ for finite non-zero $x$), and polgrad computes
$\mathrm{sg}[s_i]\cdot(r_t/\mathrm{sg}[r_t])$ in that order, so the value equals the
`SEQUENCE` ratio **bitwise**
(`tests/test_losses.py::test_gspo_sequence_token_value_equals_sequence_ratio_value`);
the gradient chain rule gives $\mathrm{sg}[s_i]\cdot \nabla r_t / r_t =
\mathrm{sg}[s_i] \cdot r_t \nabla\texttt{lp}_t / r_t = \mathrm{sg}[s_i]\,
\nabla\texttt{lp}_t$, i.e. token-local. The `SEQUENCE` value construction is pinned by
`tests/test_losses.py::test_gspo_sequence_ratio_value_matches_masked_mean_exponent`.

## PG and REINFORCE

$$
\ell^{\mathrm{PG}}_t = -\rho_t A_t, \qquad
\ell^{\mathrm{REINFORCE}}_t = -A_t\,\texttt{logprobs}_t ,
$$

with $\rho$ any ratio kind for PG and no ratio at all for REINFORCE (`old_logprobs` is
ignored and `RatioKind.TOKEN` is required;
`tests/test_losses.py::test_reinforce_ignores_old_logprobs`). Token-ratio gradients:

$$
\nabla \ell^{\mathrm{PG}}_t = -A_t r_t \nabla \texttt{lp}_t, \qquad
\nabla \ell^{\mathrm{REINFORCE}}_t = -A_t \nabla \texttt{lp}_t .
$$

**On-policy collapse.** At $\texttt{old\_logprobs} = \mathrm{sg}[\texttt{logprobs}]$,
$z = 0$ and $r = e^0 = 1$ exactly, so $\nabla\ell^{\mathrm{PG}} = -A\nabla\texttt{lp} =
\nabla\ell^{\mathrm{REINFORCE}}$. PG_CLIP ties at $r = 1$ (inside any clip band), and a
tie of `torch.minimum` splits the incoming gradient in half between two branches whose
derivatives are identical there, recombining as $\tfrac12 x + \tfrac12 x = x$ exactly.
The three gradients (four with dual-clip) are therefore **bitwise** equal, for every
aggregation
(`tests/test_losses.py::test_on_policy_pg_clip_pg_reinforce_gradients_coincide`).

**Zero advantage.** Every surrogate is a multiple of $A_t$, so $A \equiv 0$ gives an
exactly zero surrogate gradient for every valid combination
(`tests/test_losses.py::test_zero_advantages_give_zero_surrogate_gradient`).

## PG_CLIP: branch structure

$$
\ell^{\mathrm{CLIP}}_t = -\min\big(\rho_t A_t,\; \mathrm{clip}(\rho_t,\, 1-\varepsilon_{lo},\, 1+\varepsilon_{hi})\, A_t\big)
$$

(Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 1707.06347). Write
$lo = 1-\varepsilon_{lo}$, $hi = 1+\varepsilon_{hi}$. Factoring the sign of $A_t$:

- $A_t > 0$: $\min(\rho, \mathrm{clip}(\rho))\,A = \min(\rho,\, hi)\,A$ — the lower
  bound can never win because $\mathrm{clip}(\rho) = lo > \rho$ there, and the $\min$
  rejects it.
- $A_t < 0$: $\min(\rho A, \mathrm{clip}(\rho) A) = \max(\rho,\, \mathrm{clip}(\rho))\,A
  = \max(\rho,\, lo)\,A$ — symmetric: the upper bound can never win.

| branch | condition | objective | gradient | reported as |
| --- | --- | --- | --- | --- |
| unclipped | $A>0,\ \rho \le hi$ or $A<0,\ \rho \ge lo$ | $\rho A$ | $-A\,\nabla\rho$ | — |
| clipped high | $A>0,\ \rho > hi$ | $hi\cdot A$ (const) | $0$ | `clipped_high` |
| clipped low | $A<0,\ \rho < lo$ | $lo\cdot A$ (const) | $0$ | `clipped_low` |
| zero | $A = 0$ | $0$ | $0$ | — |

Inside the band the two $\min$ arguments tie with identical derivatives, so the
gradient is $-A\nabla\rho$ there too (the tie-splitting argument above). The masks
report exactly the branch autograd took: on generic inputs with $A_t \ne 0$,
`clipped_low | clipped_high` coincides with the set of response tokens whose per-token
gradient is exactly $0$
(`tests/test_losses.py::test_pg_clip_masks_match_autograd_branch`).

Hand-derived goldens, $\varepsilon_{lo} = \varepsilon_{hi} = 0.2$, one token, weight
$1$:

- inside: $r = e^{0.1} = 1.10517$, $A = 2$: loss $= -2e^{0.1} = -2.21034$, gradient
  $-2e^{0.1}$ (`tests/test_losses.py::test_pg_clip_golden_one_token_inside_clip`);
- above, $A>0$: $r = e^{0.8} = 2.22554 > 1.2$, $A = 1.5$: $\min(1.5\,e^{0.8},
  1.5\cdot1.2) = \min(3.33831, 1.8) = 1.8$, loss $= -1.8$, gradient $0$
  (`tests/test_losses.py::test_pg_clip_golden_one_token_clipped_high_positive_advantage`);
- below, $A<0$: $r = e^{-0.8} = 0.44933 < 0.8$, $A = -1$: $\min(-e^{-0.8}, -0.8) =
  -0.8$, loss $= 0.8$, gradient $0$
  (`tests/test_losses.py::test_pg_clip_golden_one_token_clipped_low_negative_advantage`);
- 2-token ragged batch mixing all three branches under `TOKEN_MEAN`
  (`tests/test_losses.py::test_pg_clip_golden_two_token_ragged_mixed_branches`).

## Dual-clip (`ratio_cap`)

For $A_t < 0$ the pessimistic $\min$ lets $\rho A$ fall without bound as $\rho$ grows;
dual-clip PPO (Ye et al., "Mastering Complex Control in MOBA Games with Deep
Reinforcement Learning", arXiv 1912.09729) floors the objective with a constant
$c = \texttt{ratio\_cap} > 1$:

$$
\ell^{\mathrm{dual}}_t = -\max\big(\min(\rho A,\, \mathrm{clip}(\rho) A),\; cA\big)
\qquad (A_t < 0 \text{ only}).
$$

Since $A < 0$, $\max(\max(\rho, lo)A,\, cA) = \min(\max(\rho, lo),\, c)\,A$, giving the
$A<0$ branch table:

| condition ($A<0$) | objective | gradient | reported as |
| --- | --- | --- | --- |
| $\rho < lo$ | $lo\cdot A$ | $0$ | `clipped_low` |
| $lo \le \rho \le c$ | $\rho A$ | $-A\nabla\rho$ | — |
| $\rho > c$ | $c\cdot A$ (const) | $0$ | `clipped_high` |

The floor can only bind at $\rho > c$: for $\rho < lo$, $lo\,A > cA$ because
$lo < 1 < c$ and $A < 0$. Note the band $hi < \rho \le c$ where the ratio exceeds the
PPO bound but the gradient still flows — the upper clip is not the branch taken for
$A<0$. Goldens, $c = 3$:

- binding: $r = e^{1.5} = 4.48169 > 3$, $A = -2$:
  $\min(-2e^{1.5}, -2.4) = -8.96338$, then $\max(-8.96338, -6) = -6$: loss $= 6$,
  gradient $0$
  (`tests/test_losses.py::test_pg_clip_golden_dual_clip_negative_advantage_above_cap`);
- not binding: $r = e^{0.5} = 1.64872 \in (1.2, 3)$, $A = -1$: loss $= e^{0.5}$,
  gradient $+e^{0.5}$, no clip mask set
  (`tests/test_losses.py::test_pg_clip_golden_dual_clip_inactive_between_high_and_cap`).

`ratio_cap` must be finite and $> 1$, else `ValueError`
(`tests/test_losses.py::test_policy_loss_config_validation_errors`).

## GSPO gradients from the sg[] algebra

**GSPO-seq** (`RatioKind.SEQUENCE`, PG surrogate). With aggregation weights $w_{i,t}$,

$$
L = \sum_{i,\tau} w_{i,\tau}\,\big({-s_i A_{i,\tau}}\big)
\;\Longrightarrow\;
\frac{\partial L}{\partial \texttt{lp}_{i,t}}
= -\Big(\sum_{\tau} w_{i,\tau} A_{i,\tau}\Big)\,\frac{\partial s_i}{\partial \texttt{lp}_{i,t}}
= -\Big(\sum_{\tau} w_{i,\tau} A_{i,\tau}\Big)\, s_i\, \frac{m_{i,t}}{L_i}.
$$

Every token of a row moves with the same per-row coefficient — the whole row is coupled
through the length-normalized mean. With `SEQ_MEAN_TOKEN_MEAN` weights
$w = m/(B L_i)$ this is the GSPO paper's sequence-level update. Verified against
autograd for both `SEQ_MEAN_TOKEN_MEAN` and `TOKEN_MEAN`
(`tests/test_losses.py::test_gspo_sequence_gradient_matches_coupled_analytic_formula`).

**GSPO-token** (`RatioKind.SEQUENCE_TOKEN`). The stop-gradients place all sequence-level
structure in the value and none in the gradient:

$$
L = \sum_{i,t} w_{i,t}\Big({-\,\mathrm{sg}[s_i]\,\frac{r_t}{\mathrm{sg}[r_t]}\, A_{i,t}}\Big)
\;\Longrightarrow\;
\frac{\partial L}{\partial \texttt{lp}_{i,t}}
= -\,w_{i,t}\, A_{i,t}\, s_i ,
$$

token-local, because $\nabla(r_t/\mathrm{sg}[r_t]) = \nabla r_t / r_t =
\nabla \texttt{lp}_t$ and $\mathrm{sg}[s_i]$ contributes no path
(`tests/test_losses.py::test_gspo_sequence_token_gradient_is_token_local`).

The two decompositions agree in value bitwise yet differ in gradient. Worked instance
(`tests/test_losses.py::test_gspo_sequence_and_sequence_token_gradients_differ`): gaps
$z = (0.2, -0.2)$ give $s = e^0 = 1$; $A = (1, -2)$, `TOKEN_MEAN` ($w = 1/2$):

$$
\text{GSPO-seq: } -\Big(\tfrac12\cdot 1 + \tfrac12\cdot(-2)\Big)\cdot 1\cdot\tfrac12 = +0.25 \text{ at both tokens};
\qquad
\text{GSPO-token: } \big({-\tfrac12}\cdot 1,\; {-\tfrac12}\cdot(-2)\big) = (-0.5,\, +1.0).
$$

## CISPO: stop-gradient semantics

CISPO (MiniMax-M1, arXiv 2506.13585 eq. 4-5) clips the **weight**, not the objective,
and detaches it:

$$
\ell^{\mathrm{CISPO}}_t = -\,\mathrm{sg}[\hat w_t]\, A_t\, \texttt{logprobs}_t,
\qquad
\hat w_t = \min(\rho_t,\, 1+\varepsilon_{hi})
\;\;\text{or}\;\;
\mathrm{clip}(\rho_t,\, 1-\varepsilon_{lo},\, 1+\varepsilon_{hi}),
$$

one-sided ($\varepsilon_{lo}$ = `None`, MiniMax-M1's experimental setting) or two-sided
(the paper's general form). The gradient is $-\hat w_t A_t \nabla\texttt{lp}_t$: a
REINFORCE gradient scaled by a constant. Unlike PG_CLIP, a clipped token keeps a
bounded, non-zero gradient — clipping saturates the weight instead of killing the
update. Formally, CISPO is **identical** — bitwise in loss, per-token objective, and
gradient — to REINFORCE run on the pre-scaled advantages $\mathrm{sg}[\hat w]\cdot A$
(`tests/test_losses.py::test_cispo_gradient_equals_detached_weight_scaled_reinforce`).
`clipped_low`/`clipped_high` report where the weight saturated, independent of the sign
of $A$ (`tests/test_losses.py::test_cispo_clipped_masks_report_weight_clipping`).

Because finite differences see through `.detach()`, `torch.autograd.gradcheck` of a
stop-gradient loss runs on the sg-frozen equivalent (CISPO -> REINFORCE on
$\mathrm{sg}[\hat w] A$; GSPO-token -> TOKEN ratio against shifted `old_logprobs` with
$e^{\texttt{lp}_0 - \texttt{old}'} = s_i(\texttt{lp}_0)$), and the real config's
autograd gradient is asserted equal to the frozen equivalent's at the evaluation point.
Combos without internal detach are gradchecked directly. Every valid
`SurrogateKind` x `RatioKind` x `Aggregation` combination is covered
(`tests/test_losses.py::test_fp64_gradcheck_policy_loss_valid_combinations`,
`tests/test_losses.py::test_fp64_gradcheck_policy_loss_dual_clip`,
`tests/test_losses.py::test_fp64_gradcheck_policy_loss_with_is_correction_and_kl`).

## TIS correction (rollout/trainer mismatch)

The inference engine's `rollout_logprobs` and the trainer's recomputed `old_logprobs`
describe the same policy but differ numerically (see `docs/conventions.md`). Truncated
importance sampling (TIS; verl PR #2953) reweights the surrogate by the truncated
ratio of the two, **as data**:

$$
\ell_t \;\leftarrow\; \mathrm{sg}[w]\cdot \ell_t,
\qquad
w_t = \min\!\big(e^{\texttt{old}_t - \texttt{rollout}_t},\, \mathrm{cap}\big)
\quad\text{or}\quad
w_i = \min\!\Big(\exp\!\Big(\sum_t m_{i,t}(\texttt{old}_t - \texttt{rollout}_t)\Big),\, \mathrm{cap}\Big).
$$

The sequence exponent is the **unnormalized** masked sum — the exact sequence
importance weight — not the length-normalized `RatioKind.SEQUENCE` mean. Worked
instance: gap $0.5$ per token, $L = 2$, cap $1.9$: token weight
$\min(e^{0.5}, 1.9) = e^{0.5} = 1.64872$ (uncapped), sequence weight
$\min(e^{1.0}, 1.9) = \min(2.71828, 1.9) = 1.9$ (capped)
(`tests/test_losses.py::test_is_correction_sequence_level_uses_unnormalized_sum`).

Since $\mathrm{sg}[w]$ multiplies the surrogate elementwise: $w \equiv 1$ (i.e.
`rollout == old` and cap $\ge 1$) reproduces the uncorrected loss, objective, and
gradient bitwise (`tests/test_losses.py::test_is_correction_weight_one_is_noop`); a
uniformly binding cap scales every per-token objective by exactly `cap`
(`tests/test_losses.py::test_is_correction_cap_binds`).

## KL composition

With `PolicyLossConfig.kl` set,

$$
\mathrm{loss} = \mathrm{agg}\big(\ell_t\big) \;+\; \beta\cdot
\mathrm{kl\_loss}\big(\texttt{logprobs},\, \texttt{ref\_logprobs},\, \mathrm{kind},\, \mathrm{agg}_{KL}\big),
$$

where $\beta$ = `kl.coef`, $\mathrm{agg}_{KL}$ = `kl.aggregation` (inheriting the
policy aggregation when `None`) and likewise `kl.norm_len` inherits `config.norm_len`.
`PolicyLossResult.kl_loss` carries the **unscaled** KL scalar, so
`loss == aggregate(per_token_objective) + coef * kl_loss` holds bitwise
(`tests/test_losses.py::test_policy_loss_kl_term_composition`,
`tests/test_losses.py::test_policy_loss_kl_inherits_aggregation_and_norm_len`). The KL
estimators and their as-loss gradients are derived in [kl.md](kl.md).

## Value loss

$$
\ell^{V}_t = \tfrac12 \max\big((v_t - R_t)^2,\; (\mathrm{clip}(v_t,\, v^{old}_t - \varepsilon,\, v^{old}_t + \varepsilon) - R_t)^2\big),
$$

or $\tfrac12 (v_t - R_t)^2$ when `clip_eps` is `None` (PPO, arXiv 1707.06347). The
pessimistic $\max$ takes the clipped branch — a constant in $v_t$, killing the
gradient — exactly where the clipped squared error strictly exceeds the unclipped one;
`clipped_frac` counts those response tokens. Gradient per token: $w_t\,(v_t - R_t)$ on
the unclipped branch, $0$ on the clipped branch. Golden, $v = (1.5, 0.5)$,
$v^{old} = 1$, $R = 0$, $\varepsilon = 0.2$ (band $[0.8, 1.2]$):

$$
t_0:\ \max\big(\tfrac12 1.5^2,\, \tfrac12 1.2^2\big) = \max(1.125, 0.72) = 1.125
\ \text{(unclipped, grad } 0.5\cdot1.5 = 0.75\text{)};
$$

$$
t_1:\ \max\big(\tfrac12 0.5^2,\, \tfrac12 0.8^2\big) = \max(0.125, 0.32) = 0.32
\ \text{(clipped, grad } 0\text{)};
$$

`TOKEN_MEAN` loss $= (1.125 + 0.32)/2 = 0.7225$, `clipped_frac` $= 1/2$
(`tests/test_losses.py::test_value_loss_golden_clip_branches`,
`tests/test_losses.py::test_value_loss_golden_unclipped`,
`tests/test_losses.py::test_value_loss_clipped_frac_counts_clipped_branch_tokens`,
`tests/test_losses.py::test_value_loss_token_sum_norm_uses_norm_len`).

Implementation note: the clip is computed as `clamp(v, v_old - eps, v_old + eps)` and
not `v_old + clamp(v - v_old, -eps, eps)`, because `clamp` is comparison-based and
returns $v$ **bitwise** inside the band, while the additive form rounds
($v^{old} + (v - v^{old}) \ne v$ in general) and would let 1-ulp artifacts win the
strict clipped-branch comparison. Consequences: a band wider than every $|v - v^{old}|$
reproduces the unclipped loss bitwise
(`tests/test_losses.py::test_value_loss_wide_clip_equals_unclipped_bitwise`), and ties
inside the band split the $\max$ gradient between two branches with identical
derivatives, recombining exactly. fp64 gradcheck runs clipped and unclipped under every
aggregation (`tests/test_losses.py::test_fp64_gradcheck_value_loss`); mask invariance
and validation are pinned by `tests/test_losses.py::test_value_loss_mask_invariance`
and `tests/test_losses.py::test_value_loss_validation_errors`.

## Conventions enforced across all of the above

Masked positions are $0$ in `per_token_objective`, $1.0$ in `ratio`, `False` in
`clipped_*`, and carry exactly zero gradient
(`tests/test_losses.py::test_policy_loss_mask_invariance`,
`tests/test_losses.py::test_policy_loss_mask_invariance_with_is_correction_and_kl`,
`tests/test_losses.py::test_policy_loss_gradient_is_zero_at_masked_positions`).
`[B]` advantages are exactly their `[B, T]` broadcast
(`tests/test_losses.py::test_policy_loss_sequence_advantages_broadcast`). Input dtype
is preserved (`tests/test_losses.py::test_policy_loss_preserves_input_dtype`,
`tests/test_losses.py::test_value_loss_preserves_input_dtype_and_result_is_frozen`).
Configs are inert frozen data and every contract-4.3 validation rule raises
`ValueError` at call entry
(`tests/test_losses.py::test_configs_are_inert_frozen_data`,
`tests/test_losses.py::test_policy_loss_config_validation_errors`,
`tests/test_losses.py::test_policy_loss_call_time_validation_errors`).
