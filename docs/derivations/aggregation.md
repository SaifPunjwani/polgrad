# Aggregation: from per-token objectives to a scalar loss

This page derives the semantics of `polgrad.aggregate`: the effective-weight closed
forms for each `Aggregation` mode, the equal-length collapse theorems, and the
micro-batch (gradient-accumulation) weight algebra. Every claim names the pytest node
that enforces it.

## Notation

A batch holds $B$ response sequences right-padded to length $T$. The response mask is
$m \in \{0,1\}^{B \times T}$, the per-token objective is $x \in \mathbb{R}^{B \times T}$,
and

$$
L_b = \sum_{t} m_{b,t}, \qquad N = \sum_{b,t} m_{b,t} = \sum_b L_b .
$$

Every row satisfies $L_b \ge 1$ (masks with an empty row raise `ValueError`;
`tests/test_aggregate.py::test_aggregate_rejects_invalid_shapes_and_masks`).

## The four modes and their effective token weights

Each mode is a linear functional of $x$, so it can be written as an inner product
$\mathrm{agg}(x) = \sum_{b,t} w_{b,t}\, x_{b,t}$ with weights that depend only on the
mask. Reading the weight off each definition:

| mode | definition | effective weight $w_{b,t}$ |
| --- | --- | --- |
| `TOKEN_MEAN` | $\dfrac{\sum_{b,t} m_{b,t} x_{b,t}}{N}$ | $\dfrac{m_{b,t}}{N}$ |
| `SEQ_MEAN_TOKEN_MEAN` | $\dfrac{1}{B}\sum_b \dfrac{\sum_t m_{b,t} x_{b,t}}{L_b}$ | $\dfrac{m_{b,t}}{B\,L_b}$ |
| `SEQ_MEAN_TOKEN_SUM` | $\dfrac{1}{B}\sum_b \sum_t m_{b,t} x_{b,t}$ | $\dfrac{m_{b,t}}{B}$ |
| `TOKEN_SUM_NORM` | $\dfrac{\sum_{b,t} m_{b,t} x_{b,t}}{B\,\ell}$ | $\dfrac{m_{b,t}}{B\,\ell}$ |

Here $\ell$ = `norm_len` is Dr.GRPO's fixed generation budget; it must be supplied at
call time exactly when the mode is `TOKEN_SUM_NORM`
(`tests/test_aggregate.py::test_norm_len_required_for_token_sum_norm_at_call_time`,
`tests/test_aggregate.py::test_norm_len_ignored_for_other_modes`). Framework provenance:
`TOKEN_MEAN` is verl's `"token-mean"`; TRL adopted this global token-level normalization
in v0.16.0 (PR #2881; see issue #2995 for the ensuing debate — resolved for KL logging
only in PR #3004), and current TRL exposes it as `loss_type="bnpo"` (local batch), with
TRL's present default `loss_type="dapo"` being `TOKEN_MEAN` computed over the global
gradient-accumulated batch (the micro-batch weight algebra below formalizes the
distinction). `SEQ_MEAN_TOKEN_MEAN` is the GRPO paper equation and the TRL GRPOTrainer
default in v0.14-v0.15; `TOKEN_SUM_NORM` is Dr.GRPO.

`aggregate` is implemented *as* this inner product, so the identity

$$
\mathrm{aggregate}(x, m, \text{mode})
= \Big(\mathrm{effective\_token\_weights}(m, \text{mode}) \cdot x\Big).\mathrm{sum}()
$$

holds bitwise
(`tests/test_aggregate.py::test_aggregate_equals_effective_weights_inner_product`), and
because the functional is linear,

$$
\frac{\partial\, \mathrm{aggregate}(x)}{\partial x_{b,t}} = w_{b,t}
$$

exactly, verified against autograd
(`tests/test_aggregate.py::test_aggregate_gradient_equals_effective_weights`).
Hand-derived values for all four modes on the batch $x = [[1,2],[3,\cdot]]$,
$L = [2,1]$: `TOKEN_MEAN` $= 6/3 = 2$, `SEQ_MEAN_TOKEN_MEAN` $= (1.5+3)/2 = 2.25$,
`SEQ_MEAN_TOKEN_SUM` $= (3+3)/2 = 3$, `TOKEN_SUM_NORM` with $\ell = 4$: $6/8 = 0.75$
(`tests/test_aggregate.py::test_aggregate_golden_values`,
`tests/test_aggregate.py::test_effective_weights_golden_closed_forms`).

Two structural facts:

- Masked positions carry weight exactly $0$, so masked inputs can never reach the loss
  or its gradient
  (`tests/test_aggregate.py::test_effective_weights_zero_at_masked_positions`,
  `tests/test_aggregate.py::test_aggregate_mask_invariance`).
- `TOKEN_MEAN` and `SEQ_MEAN_TOKEN_MEAN` are convex combinations:
  $\sum_{b,t} w_{b,t} = N/N = 1$ and $\sum_b \sum_t \frac{m_{b,t}}{B L_b} = \sum_b
  \frac{L_b}{B L_b} = 1$
  (`tests/test_aggregate.py::test_effective_weights_of_mean_modes_sum_to_one`).

## Equal-length collapse theorems

Suppose every row has the same token count, $L_b = L$ for all $b$, so $N = BL$. Then:

1. **`TOKEN_MEAN` = `SEQ_MEAN_TOKEN_MEAN`.** The weights coincide:
   $\frac{m}{N} = \frac{m}{BL}$. In polgrad both divisors are computed as the same
   float, so the equality is bitwise
   (`tests/test_aggregate.py::test_equal_length_token_mean_collapses_to_seq_mean_token_mean`).
   The two modes differ only on ragged batches: `SEQ_MEAN_TOKEN_MEAN` weights a token
   by $1/(B L_b)$, i.e. inversely to its own sequence length, while `TOKEN_MEAN`
   weights all tokens equally.
2. **`SEQ_MEAN_TOKEN_SUM` = $L\,\cdot$ `TOKEN_MEAN`.** Weight ratio:
   $\frac{m/B}{m/(BL)} = L$
   (`tests/test_aggregate.py::test_equal_length_seq_mean_token_sum_is_length_times_token_mean`).
3. **`TOKEN_SUM_NORM` = $(L/\ell)\,\cdot$ `TOKEN_MEAN`.** Weight ratio:
   $\frac{m/(B\ell)}{m/(BL)} = L/\ell$
   (`tests/test_aggregate.py::test_equal_length_token_sum_norm_is_length_over_norm_len_times_token_mean`).

## Micro-batch weight algebra (gradient-accumulation inequivalence)

Split the batch into $K$ consecutive micro-batches $c = 1,\dots,K$ with row counts
$B_c$ ($\sum_c B_c = B$) and token counts $N_c = \sum_{b \in c} L_b$. Aggregating each
chunk with the same mode gives chunk-local weights $w^c$: the closed forms above with
$(N, B)$ replaced by $(N_c, B_c)$. Combining the $K$ chunk losses by

- **mean** (`loss_scale="mean"`, the gradient-accumulation/DDP default):
  $\mathcal{L} = \frac{1}{K} \sum_c \mathrm{agg}_c$, so a token in chunk $c$ carries
  weight $w^c_{b,t}/K$;
- **sum** (`loss_scale="sum"`): $\mathcal{L} = \sum_c \mathrm{agg}_c$, weight
  $w^c_{b,t}$ unchanged.

`microbatch_token_weights` returns exactly these weights; they match the autograd
gradient of an explicit micro-batch loop
(`tests/test_aggregate.py::test_microbatch_weights_match_simulated_loop_autograd`,
bitwise for the sum combine in
`tests/test_aggregate.py::test_microbatch_weights_sum_scale_match_simulated_loop_bitwise`,
hand-derived instance in
`tests/test_aggregate.py::test_microbatch_weights_golden_values`, masked-position zeros
in `tests/test_aggregate.py::test_microbatch_weights_zero_at_masked_positions`).

**When does micro-batched mean equal the full batch?** Compare per-token weights (a
token in row $b$ of chunk $c$):

| mode | micro-batched mean weight | full-batch weight | equal iff |
| --- | --- | --- | --- |
| `TOKEN_MEAN` | $\dfrac{m}{K\,N_c}$ | $\dfrac{m}{N}$ | $N_c = N/K$ for every $c$ |
| `SEQ_MEAN_TOKEN_MEAN` | $\dfrac{m}{K\,B_c\,L_b}$ | $\dfrac{m}{B\,L_b}$ | $B_c = B/K$ for every $c$ |
| `SEQ_MEAN_TOKEN_SUM` | $\dfrac{m}{K\,B_c}$ | $\dfrac{m}{B}$ | $B_c = B/K$ for every $c$ |
| `TOKEN_SUM_NORM` | $\dfrac{m}{K\,B_c\,\ell}$ | $\dfrac{m}{B\,\ell}$ | $B_c = B/K$ for every $c$ |

The equivalence conditions follow by setting the two weights equal and cancelling $m$
and the shared per-row factor ($L_b$ or $\ell$).

- For `TOKEN_MEAN` the condition is on **token** counts: micro-batching a ragged batch
  changes the loss unless every micro-batch happens to hold the same number of response
  tokens. This is the closed form of gradient-accumulation inequivalence. Worked
  instance with $L = [1, 3]$ and chunks of one row each: full-batch weight $1/4$ per
  token; micro-batched mean weights $\frac{1}{1 \cdot 2} = \frac12$ (chunk 1) and
  $\frac{1}{3 \cdot 2} = \frac16$ (chunk 2)
  (`tests/test_aggregate.py::test_token_mean_microbatch_mean_deviates_for_unequal_token_counts`).
  With equal token counts ($L = [2,2,2,2]$, chunks of 2 rows: $N_c = 4$, $K = 2$,
  $\frac{1}{4 \cdot 2} = \frac18 = \frac1N$) the equality is exact
  (`tests/test_aggregate.py::test_token_mean_microbatch_mean_matches_full_batch_for_equal_token_counts`).
- For the three per-sequence-denominator modes the condition is on **row** counts only:
  equal rows per micro-batch restores the full-batch loss even on ragged batches,
  because each row's denominator $B_c\,(\cdot)$ scales by exactly $K$
  (`tests/test_aggregate.py::test_per_sequence_modes_microbatch_mean_matches_full_batch_for_equal_row_counts`).

With `loss_scale="sum"` the combined objective is $\sum_c \mathrm{agg}_c$, which for
$K > 1$ is a different objective from the full batch under every mode (each chunk keeps
its own, smaller normalizer); polgrad reports its weights as-is rather than treating it
as an approximation of the full batch.

## dtype

`effective_token_weights` and `microbatch_token_weights` return `float64` (the mask has
no floating dtype to preserve; the weight identities are asserted at full precision).
`aggregate` casts the weights to the per-token dtype and preserves it
(`tests/test_aggregate.py::test_aggregate_preserves_input_dtype`).
