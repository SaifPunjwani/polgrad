# Conventions

Every public function in polgrad follows the rules on this page. Tests enforce them; the
per-module derivation pages assume them.

## Shapes

- Per-token tensors are `[B, T]`: `B` response sequences, `T` the maximum response length,
  right-padded. Only **response** tokens appear; prompt tokens are excluded before polgrad
  is involved.
- Per-sequence tensors are `[B]`.
- `response_mask` is a `[B, T]` tensor of dtype `torch.bool` marking real response tokens.
  Every row must contain at least one true token; a violating mask raises `ValueError`.

## Masked positions

Values at masked positions never affect any output. Every `[B, T]` tensor returned by a
public function is exactly `0` at masked positions, with two exceptions:
`PolicyLossResult.ratio` is `1.0` (the neutral ratio), and boolean outputs
(`clipped_low`, `clipped_high`, `gradient_killed_mask`) are `False`. Mask-invariance tests
assert bitwise equality of full outputs when masked inputs are perturbed.

## Logprob streams

`logprobs[b, t] = log π(y_t | y_<t, x)` for the sampled token only. Up to four streams
appear, and their distinctness is load-bearing:

| stream | policy | gradient flows |
| --- | --- | --- |
| `logprobs` | current policy θ | yes |
| `old_logprobs` | behavior policy at sampling time (θ_old), recomputed by the trainer | no |
| `ref_logprobs` | frozen reference policy (KL anchor) | no |
| `rollout_logprobs` | what the inference engine reported during rollout | no |

`old_logprobs` and `rollout_logprobs` describe the same distribution in exact arithmetic;
in practice they differ (kernels, precision, batching), and that gap is a measurable
off-policy pathology — see `polgrad.diagnostics.mismatch` and the truncated
importance-sampling correction in `polgrad.losses`.

## Signs, ratios, dtypes

- All losses are quantities to **minimize**; the policy-gradient surrogate is already
  negated. `loss.backward()` ascends the objective.
- The importance ratio is computed as `r_t = exp(logprobs − old_logprobs)`, never as a
  quotient of exponentials.
- Functions preserve the input dtype; there are no silent casts. Verification helpers
  upcast to `float64` explicitly.
- Stop-gradient is written `sg[·]` in the derivation pages and implemented with
  `.detach()`. Every detach inside a loss carries a comment stating its semantic reason.

## Errors and determinism

- Invalid shapes, dtypes, or masks raise `ValueError` naming the argument and the
  offending shape. Degenerate inputs (all-masked rows) raise rather than emitting NaN.
- No randomness in the library except `polgrad.verify`, `polgrad.conformance.harness`,
  and `polgrad.diagnostics.entropy.entropy_trend`, all of which take an explicit
  `torch.Generator`.
