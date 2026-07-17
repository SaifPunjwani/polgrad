# PPO clip-band diagnostics (`polgrad.diagnostics.clipping`)

The PG_CLIP surrogate of `polgrad.losses` (see `docs/derivations/losses.md`) has the
per-token loss, for `RatioKind.TOKEN` where $r_t = e^{\text{logprobs}_t -
\text{old\_logprobs}_t}$,

$$
\ell_t = -\min\big(r_t A_t,\; \operatorname{clip}(r_t,\, 1-\varepsilon_{\mathrm{lo}},\,
1+\varepsilon_{\mathrm{hi}})\, A_t\big),
$$

and, when dual-clip is enabled with cap $c > 1$ (Ye et al., arXiv 1912.09729), for
$A_t < 0$ the flooring branch

$$
\ell_t = -\max\Big(\min\big(r_t A_t,\; \operatorname{clip}(r_t)\, A_t\big),\; c\,A_t\Big).
$$

Since $\partial r_t / \partial \text{logprobs}_t = r_t > 0$, whichever branch is active
determines whether the token contributes gradient at all. `clip_report` measures two
different partitions of the response tokens that are easy to conflate:

1. **band crossings** — where $r_t$ left the interval
   $(1-\varepsilon_{\mathrm{lo}},\, 1+\varepsilon_{\mathrm{hi}})$, split by advantage
   sign (the four quadrant fractions);
2. **killed gradients** — where $\partial \ell_t / \partial \text{logprobs}_t$ is
   exactly $0$ (`gradient_killed_mask`).

Two of the four quadrants kill the gradient; the other two do not.

## Quadrant fractions

Every fraction uses the same denominator $N = \sum_{b,t} m_{b,t}$ (all response
tokens), so the fields are directly comparable and no field divides by zero when one
advantage sign is absent from the batch.

| field | region | gradient |
| --- | --- | --- |
| `frac_pos_adv_clipped_high` | $A_t > 0$ and $r_t > 1+\varepsilon_{\mathrm{hi}}$ | killed |
| `frac_pos_adv_clipped_low` | $A_t > 0$ and $r_t < 1-\varepsilon_{\mathrm{lo}}$ | flows |
| `frac_neg_adv_clipped_high` | $A_t < 0$ and $r_t > 1+\varepsilon_{\mathrm{hi}}$ | flows, unless dual-clip and $r_t > c$ |
| `frac_neg_adv_clipped_low` | $A_t < 0$ and $r_t < 1-\varepsilon_{\mathrm{lo}}$ | killed |

The fractions count band crossings regardless of dual-clip; the dual-clip cap enters
only the killed-gradient census. Verified on hand-built tensors in
`tests/test_diagnostics_clipping.py::test_quadrant_fractions_constructed_case_dual_clip`
and against a per-token Python oracle on generic inputs in
`tests/test_diagnostics_clipping.py::test_quadrant_fractions_match_python_oracle`.

## The zero-gradient condition, branch by branch

Throughout, multiplying an inequality between ratios by $A_t$ preserves order for
$A_t > 0$ and reverses it for $A_t < 0$; that single fact decides every branch.
Ratios exactly on a bound are non-differentiable tie points, handled at the end.

**Case $A_t > 0$.**

- $r_t < 1-\varepsilon_{\mathrm{lo}}$: $\operatorname{clip}(r_t) =
  1-\varepsilon_{\mathrm{lo}}$, and $r_t A_t < (1-\varepsilon_{\mathrm{lo}}) A_t$, so
  the $\min$ keeps the unclipped $r_t A_t$. Gradient $-r_t A_t \ne 0$ — **flows**.
  Crossing the *low* bound with positive advantage clips nothing: PPO's $\min$ is
  one-sided pessimism.
- $1-\varepsilon_{\mathrm{lo}} \le r_t \le 1+\varepsilon_{\mathrm{hi}}$:
  $\operatorname{clip}$ is the identity, both arguments equal $r_t A_t$. Gradient
  $-r_t A_t \ne 0$ — **flows**.
- $r_t > 1+\varepsilon_{\mathrm{hi}}$: $r_t A_t > (1+\varepsilon_{\mathrm{hi}}) A_t$,
  so the $\min$ picks the constant $(1+\varepsilon_{\mathrm{hi}}) A_t$. Gradient $0$ —
  **killed**.

**Case $A_t < 0$, no dual-clip.**

- $r_t < 1-\varepsilon_{\mathrm{lo}}$: now $r_t A_t > (1-\varepsilon_{\mathrm{lo}})
  A_t$ (order reversed), so the $\min$ picks the constant
  $(1-\varepsilon_{\mathrm{lo}}) A_t$. Gradient $0$ — **killed**.
- $1-\varepsilon_{\mathrm{lo}} \le r_t \le 1+\varepsilon_{\mathrm{hi}}$: both
  arguments equal $r_t A_t$. Gradient $-r_t A_t = r_t \lvert A_t\rvert \ne 0$ —
  **flows**.
- $r_t > 1+\varepsilon_{\mathrm{hi}}$: $r_t A_t < (1+\varepsilon_{\mathrm{hi}}) A_t$,
  so the $\min$ keeps the unclipped $r_t A_t$. Gradient $r_t \lvert A_t\rvert \ne 0$ —
  **flows**, unbounded in $r_t$ (the pathology below).

**Case $A_t < 0$ with dual-clip cap $c$.** The inner $\min$ equals $r_t A_t$ for every
$r_t \ge 1-\varepsilon_{\mathrm{lo}}$ (both sub-cases above) and
$(1-\varepsilon_{\mathrm{lo}}) A_t$ below the band; the outer $\max$ compares it with
$c\,A_t$:

- $r_t > c$: $r_t A_t < c\,A_t$ (order reversed; note $c > 1 >
  1-\varepsilon_{\mathrm{lo}}$ so this branch is only reachable with the inner min at
  $r_t A_t$), so the $\max$ picks the constant $c\,A_t$. Gradient $0$ — **killed**.
- $r_t < 1-\varepsilon_{\mathrm{lo}}$: inner min is $(1-\varepsilon_{\mathrm{lo}})
  A_t$, and $c > 1-\varepsilon_{\mathrm{lo}}$ with $A_t < 0$ gives $c\,A_t <
  (1-\varepsilon_{\mathrm{lo}}) A_t$, so the $\max$ keeps the constant
  $(1-\varepsilon_{\mathrm{lo}}) A_t$ — still **killed**, unchanged from the no-cap
  case.
- $1-\varepsilon_{\mathrm{lo}} \le r_t < c$: $r_t A_t > c\,A_t$, the $\max$ keeps
  $r_t A_t$ — **flows**. Dual-clip shrinks the unbounded window from
  $(1+\varepsilon_{\mathrm{hi}}, \infty)$ to $(1+\varepsilon_{\mathrm{hi}}, c)$.

**Case $A_t = 0$.** $\ell_t \equiv 0$ for every $r_t$; gradient $0$ — **killed**.

Collecting the killed branches:

$$
\frac{\partial \ell_t}{\partial \text{logprobs}_t} = 0
\iff
(A_t > 0 \wedge r_t > 1+\varepsilon_{\mathrm{hi}})
\;\text{or}\;
(A_t < 0 \wedge r_t < 1-\varepsilon_{\mathrm{lo}})
\;\text{or}\;
(\text{dual-clip: } A_t < 0 \wedge r_t > c)
\;\text{or}\;
A_t = 0,
$$

and everywhere else the gradient is $-r_t A_t \ne 0$ because $r_t > 0$. Since every
`Aggregation` mode gives each response token a strictly positive weight $w_{b,t}$
(`docs/derivations/aggregation.md`), the same condition characterizes the zeros of the
aggregate loss gradient per token. This is verified — not just against the formula but
against autograd of `polgrad.losses.policy_loss` on generic inputs with tie points
excluded — in
`tests/test_diagnostics_clipping.py::test_gradient_killed_matches_policy_loss_autograd`.

## Why $A_t < 0$, $r_t \gg 1$ flows gradient without dual-clip

The $\min$ caps the objective from above only. For $A_t < 0$ the objective
$r_t A_t \to -\infty$ as $r_t$ grows, so the clipped argument is never the smaller one
and the surrogate stays at the unclipped $r_t A_t$: per-token loss
$r_t \lvert A_t\rvert$ with gradient magnitude $r_t \lvert A_t\rvert$ — **unbounded**,
and largest exactly where the current policy has already drifted furthest above the
sampling policy. A single strongly off-policy token with negative advantage can
dominate the batch gradient; this is the known PPO pathology that dual-clip
(arXiv 1912.09729) repairs by flooring the objective at $c\,A_t$, which zeroes the
gradient for $r_t > c$.

`tests/test_diagnostics_clipping.py::test_pathology_neg_adv_high_ratio_flows_gradient_without_dual_clip`
demonstrates both sides with exact numbers: band $(0.8, 1.2)$, $A = -1$, $r = 2.0$,
`TOKEN_MEAN` over $N = 2$ tokens. Without a cap the token is reported clipped-high yet
autograd returns gradient $w\,r\,\lvert A\rvert = \tfrac{1}{2}\cdot 2.0\cdot 1 = 1.0$;
with $c = 1.5 < r$ the same token's gradient is exactly $0$ and
`gradient_killed_mask` flips to `True`.

## Golden constructed case

Enforced by
`tests/test_diagnostics_clipping.py::test_quadrant_fractions_constructed_case_dual_clip`
and `::test_quadrant_fractions_constructed_case_without_dual_clip`. Band
$(1-0.2,\, 1+0.3) = (0.8, 1.3)$, cap $c = 3.0$, mask $[[T,T,T,T],[T,T,F,F]]$, so
$N = 6$:

| $(b,t)$ | $A_t$ | $r_t$ | quadrant | killed (cap $3.0$) | killed (no cap) |
| --- | --- | --- | --- | --- | --- |
| $(0,0)$ | $+1$ | $1.5$ | pos-high | yes | yes |
| $(0,1)$ | $+1$ | $0.5$ | pos-low | no | no |
| $(0,2)$ | $-1$ | $0.5$ | neg-low | yes | yes |
| $(0,3)$ | $-1$ | $1.5$ | neg-high | no ($1.5 < c$) | no |
| $(1,0)$ | $-1$ | $3.5$ | neg-high | yes ($3.5 > c$) | no |
| $(1,1)$ | $0$ | $1.0$ | none | yes ($A = 0$) | yes |

Fractions: pos-high $= 1/6$, pos-low $= 1/6$, neg-low $= 1/6$, neg-high $= 2/6$
(both $r = 1.5$ and $r = 3.5$ exceed $1.3$), identical with and without the cap;
`gradient_killed_frac` $= 4/6$ with the cap and $3/6$ without it — the difference is
exactly the pathological $(1,0)$ token.

## Tie points, scope, and null calibration

- The condition uses strict inequalities. At $r_t$ exactly on
  $1-\varepsilon_{\mathrm{lo}}$, $1+\varepsilon_{\mathrm{hi}}$, or $c$ the objective is
  non-differentiable and autograd's subgradient choice is an implementation detail;
  the autograd cross-test excludes these measure-zero ties, and `clip_report`
  classifies them by the strict inequalities above.
- The derivation is for `RatioKind.TOKEN`, where $\ell_t$ depends on
  $\text{logprobs}_t$ alone; sequence-level ratios couple all tokens of a row and are
  out of scope for this census.
- Masked positions are `False` in `gradient_killed_mask` and never affect any field,
  bitwise (`tests/test_diagnostics_clipping.py::test_mask_invariance_clipping`).
- Null calibration: identical policies give $r_t = 1$ strictly inside any band
  ($\varepsilon_{\mathrm{lo}}, \varepsilon_{\mathrm{hi}} > 0$), so every quadrant
  fraction is $0$ and the gradient is killed exactly where $A_t = 0$
  (`tests/test_diagnostics_clipping.py::test_identical_policies_nothing_clipped_killed_only_at_zero_advantage`);
  all-zero advantages kill every response token
  (`tests/test_diagnostics_clipping.py::test_zero_advantages_kill_every_response_token`).
- `[B]` advantages broadcast across each row exactly as in `polgrad.losses`
  (`tests/test_diagnostics_clipping.py::test_sequence_advantages_broadcast_matches_explicit_broadcast`);
  invalid clip configs, shapes, masks, non-finite response values, and non-positive
  response ratios raise `ValueError`
  (`tests/test_diagnostics_clipping.py::test_validation_errors`).

## References

- Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 1707.06347 — the
  clipped surrogate.
- Ye et al., "Mastering Complex Control in MOBA Games with Deep Reinforcement
  Learning", arXiv 1912.09729 — dual-clip PPO.
- `docs/derivations/losses.md` — the PG_CLIP branches and stop-gradient placements.
