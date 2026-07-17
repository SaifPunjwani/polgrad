"""Registry of named RL post-training algorithms as exact polgrad configurations.

Each :class:`AlgorithmSpec` pins one published algorithm to a concrete
:class:`~polgrad.losses.PolicyLossConfig`, an advantage estimator, and a KL placement.
Every numeric constant (clip widths, KL coefficients, discount factors) is traced to the
paper or the paper's released code on ``docs/derivations/variants.md``; where a paper
leaves a knob unstated, the spec's ``notes`` say so and name the source of the shipped
value instead of presenting it as the paper's. ``registry.get`` is re-exported as
``polgrad.get_algorithm`` and ``registry.describe`` as ``polgrad.describe_algorithm``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from polgrad.advantages import GAEConfig, GroupNormConfig, ReinforcePPConfig
from polgrad.aggregate import Aggregation
from polgrad.kl import KLEstimator, KLLossConfig
from polgrad.losses import (
    ClipConfig,
    ISCorrectionConfig,
    PolicyLossConfig,
    RatioKind,
    SurrogateKind,
)

__all__ = [
    "ALGORITHMS",
    "AlgorithmSpec",
    "Citation",
    "describe",
    "get",
]


@dataclass(frozen=True)
class Citation:
    """A verified literature reference for an algorithm specification.

    Attributes:
        title: Exact title of the paper or upstream artifact.
        arxiv: arXiv identifier (e.g. ``"1707.06347"``), or ``None`` for non-arXiv
            sources such as a framework pull request.
        url: Canonical URL, or ``None``.
        notes: Where inside the source the load-bearing equations and numbers live.

    References:
        docs/derivations/variants.md;
        tests/test_registry.py::test_every_entry_is_a_complete_spec.
    """

    title: str
    arxiv: str | None
    url: str | None
    notes: str


@dataclass(frozen=True)
class AlgorithmSpec:
    """A published algorithm expressed as polgrad configuration plus provenance.

    Invariant (validated in ``__post_init__``): ``kl_placement == "loss"`` iff
    ``loss.kl is not None``, and ``kl_placement == "reward"`` iff
    ``kl_reward is not None``; consequently ``kl_placement == "none"`` carries no KL
    config on either side.

    Attributes:
        name: Registry key; equals the key in :data:`ALGORITHMS`.
        loss: Policy-loss configuration passed to :func:`polgrad.losses.policy_loss`.
        advantage: Which advantage estimator family the algorithm uses.
        advantage_config: Config for the estimator, or ``None`` where the estimator
            takes none (RLOO).
        kl_placement: ``"loss"`` adds the KL term to the loss, ``"reward"`` folds it
            into rewards via :func:`polgrad.kl.kl_in_reward`, ``"none"`` uses no KL.
        kl_reward: Estimator kind and coefficient for the in-reward placement.
        citation: Verified source of the equations and constants.
        notes: Aggregation, clip, KL, and advantage-normalization facts, including
            every knob the paper leaves unstated and where the shipped value comes
            from.

    References:
        docs/derivations/variants.md;
        tests/test_registry.py::test_spec_invariant_violations_raise.
    """

    name: str
    loss: PolicyLossConfig
    advantage: Literal["grpo", "rloo", "gae", "reinforce_pp"]
    advantage_config: GroupNormConfig | GAEConfig | ReinforcePPConfig | None
    kl_placement: Literal["reward", "loss", "none"]
    kl_reward: KLLossConfig | None
    citation: Citation
    notes: str

    def __post_init__(self) -> None:
        if (self.kl_placement == "loss") != (self.loss.kl is not None):
            raise ValueError(
                f"AlgorithmSpec {self.name!r}: kl_placement == 'loss' iff loss.kl is set; "
                f"got kl_placement={self.kl_placement!r} with loss.kl={self.loss.kl!r}"
            )
        if (self.kl_placement == "reward") != (self.kl_reward is not None):
            raise ValueError(
                f"AlgorithmSpec {self.name!r}: kl_placement == 'reward' iff kl_reward is set; "
                f"got kl_placement={self.kl_placement!r} with kl_reward={self.kl_reward!r}"
            )


_GRPO_LOSS = PolicyLossConfig(
    ratio=RatioKind.TOKEN,
    surrogate=SurrogateKind.PG_CLIP,
    clip=ClipConfig(eps_low=0.2, eps_high=0.2),
    aggregation=Aggregation.SEQ_MEAN_TOKEN_MEAN,
    kl=KLLossConfig(kind=KLEstimator.K3, coef=0.04),
)

_GRPO_CITATION = Citation(
    title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
    arxiv="2402.03300",
    url="https://arxiv.org/abs/2402.03300",
    notes="GRPO objective eq. 3; unbiased KL estimator eq. 4 (= Schulman's k3); Sec. 4.2 "
    "RL hyperparameters ('The KL coefficient is 0.04'; G = 64 outputs per question).",
)

_GRPO_NOTES = (
    "Aggregation: eq. 3 averages token means per output, then outputs per group "
    "((1/G) sum_i (1/|o_i|) sum_t) = SEQ_MEAN_TOKEN_MEAN. "
    "Clip: the paper writes eps in eq. 3 but never states its value; 0.2 shipped here is the PPO "
    "paper's value (arXiv 1707.06347 Sec. 3), the TRL GRPOConfig default (epsilon=0.2), and the "
    "Dr.GRPO reproduction's setting (arXiv 2503.20783 Table 6). "
    "KL: added directly to the loss (placement 'loss', not in the reward), estimator eq. 4 "
    "pi_ref/pi - log(pi_ref/pi) - 1 = k3, coef 0.04 (Sec. 4.2). "
    "Advantage: outcome rewards centered by the group mean and divided by the group std "
    "(A_i = (r_i - mean)/std); the paper does not specify the +eps guard on the std or the "
    "Bessel correction - GroupNormConfig defaults apply (eps=1e-4 as in TRL; verl uses 1e-6, "
    "the Dr.GRPO code 1e-8; unbiased=True matches torch .std() as used by those codebases). "
    "Current TRL defaults beta to 0.0 (KL off), diverging from the paper."
)


ALGORITHMS: dict[str, AlgorithmSpec] = {
    "ppo": AlgorithmSpec(
        name="ppo",
        loss=PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.PG_CLIP,
            clip=ClipConfig(eps_low=0.2, eps_high=0.2),
            aggregation=Aggregation.TOKEN_MEAN,
        ),
        advantage="gae",
        advantage_config=GAEConfig(gamma=0.99, lam=0.95),
        kl_placement="none",
        kl_reward=None,
        citation=Citation(
            title="Proximal Policy Optimization Algorithms",
            arxiv="1707.06347",
            url="https://arxiv.org/abs/1707.06347",
            notes="Clipped surrogate eq. 7 ('say, eps = 0.2'; best row of Table 1); GAE "
            "gamma=0.99, lam=0.95 in Tables 3-5; adaptive-KL alternative eq. 8 (Sec. 4).",
        ),
        notes=(
            "Aggregation: eq. 7's E_t is the empirical mean over every timestep of the batch = "
            "TOKEN_MEAN. "
            "Clip: eps_low = eps_high = 0.2 (Sec. 3 'say, eps = 0.2'; eps = 0.2 is the best "
            "clipping row of Table 1). "
            "KL: none - the clipped objective replaces the KL penalty; the paper's adaptive-KL "
            "variant (eq. 8) 'performed worse than the clipped surrogate objective' (Sec. 4). "
            "RLHF deployments commonly re-add a KL-in-reward penalty and set gamma = lam = 1 "
            "(verl ppo_trainer.yaml defaults gamma: 1.0, lam: 1.0, kl_coef: 0.001 with "
            "use_kl_in_reward: False); neither is part of the paper's objective, so neither is "
            "shipped here. "
            "Advantage: GAE (eq. 11-12) with gamma=0.99, lam=0.95 from the paper's MuJoCo/Atari "
            "tables (Tables 3-5); the paper predates LLM post-training and specifies no "
            "token-level aggregation or advantage whitening (see polgrad.advantages.whiten). "
            "Value side: pair with polgrad.losses.value_loss and this GAEConfig; the paper's "
            "L^VF (eq. 9) is an unclipped squared error (V - V^targ)^2 - value clipping and the "
            "1/2 factor are implementation conventions, exposed via value_loss(clip_eps=...)."
        ),
    ),
    "grpo": AlgorithmSpec(
        name="grpo",
        loss=_GRPO_LOSS,
        advantage="grpo",
        advantage_config=GroupNormConfig(center=True, scale="std"),
        kl_placement="loss",
        kl_reward=None,
        citation=_GRPO_CITATION,
        notes=_GRPO_NOTES,
    ),
    "dr_grpo": AlgorithmSpec(
        name="dr_grpo",
        loss=PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.PG_CLIP,
            clip=ClipConfig(eps_low=0.2, eps_high=0.2),
            aggregation=Aggregation.TOKEN_SUM_NORM,
            norm_len=None,
        ),
        advantage="grpo",
        advantage_config=GroupNormConfig(center=True, scale="none"),
        kl_placement="none",
        kl_reward=None,
        citation=Citation(
            title="Understanding R1-Zero-Like Training: A Critical Perspective",
            arxiv="2503.20783",
            url="https://arxiv.org/abs/2503.20783",
            notes="Dr.GRPO removes GRPO's 1/|o_i| and std terms; the code divides masked token "
            "sums by a constant generation budget (MAX_TOKENS); Table 6 settings.",
        ),
        notes=(
            "Aggregation: token sums divided by a constant generation budget instead of |o_i| "
            "(their masked-sum code divides by the global constant MAX_TOKENS) = TOKEN_SUM_NORM "
            "with norm_len = budget. This spec ships norm_len=None because the budget is "
            "deployment-specific: call dataclasses.replace(spec.loss, norm_len=<budget>) before "
            "policy_loss, which raises ValueError until then. Their runs use a 3000-token "
            "maximum response length and 8 responses per question (Table 6). "
            "Clip: eps_low = eps_high = 0.2 (Table 6 'Policy clipping parameter: 0.2'). "
            "KL: none ('We will assume beta = 0 throughout this paper'; Table 6 lists both KL "
            "coefficients as 0.0). "
            "Advantage: group-mean centering with the std division removed "
            "(GroupNormConfig(center=True, scale='none')); removing the per-group 1/(std+eps) "
            "factor is Dr.GRPO's difficulty-bias fix (docs/derivations/variants.md). "
            "The paper does not restate the sampling temperature/eps guards it inherits; its "
            "released code pins them (temperature 1.0, std eps 1e-8, unused here)."
        ),
    ),
    "dapo": AlgorithmSpec(
        name="dapo",
        loss=PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.PG_CLIP,
            clip=ClipConfig(eps_low=0.2, eps_high=0.28),
            aggregation=Aggregation.TOKEN_MEAN,
        ),
        advantage="grpo",
        advantage_config=GroupNormConfig(center=True, scale="std"),
        kl_placement="none",
        kl_reward=None,
        citation=Citation(
            title="DAPO: An Open-Source LLM Reinforcement Learning System at Scale",
            arxiv="2503.14476",
            url="https://arxiv.org/abs/2503.14476",
            notes="Token-level objective eq. 12 (main objective with Clip-Higher: eq. 8); 'we "
            "set the clipping parameter eps_low to 0.2 and eps_high to 0.28'; advantage "
            "normalization eq. 9.",
        ),
        notes=(
            "Aggregation: eq. 12 normalizes by the total token count "
            "((1/sum_i |o_i|) sum_i sum_t) = TOKEN_MEAN (the 'token-level policy gradient "
            "loss'). "
            "Clip: Clip-Higher, eps_low=0.2 and eps_high=0.28. "
            "KL: none ('we will exclude the KL term from our proposed algorithm'). "
            "Advantage: GRPO-style group normalization kept, eq. 9 (r_i - mean)/std; the paper "
            "is silent on the +eps guard and Bessel correction (GroupNormConfig defaults "
            "apply). "
            "Not representable in a loss config and therefore not shipped here: DAPO's dynamic "
            "sampling (batch filtering) and overlong reward shaping / soft punishment operate "
            "on the data and reward, upstream of the loss."
        ),
    ),
    "gspo": AlgorithmSpec(
        name="gspo",
        loss=PolicyLossConfig(
            ratio=RatioKind.SEQUENCE,
            surrogate=SurrogateKind.PG_CLIP,
            clip=ClipConfig(eps_low=3e-4, eps_high=4e-4),
            aggregation=Aggregation.SEQ_MEAN_TOKEN_MEAN,
        ),
        advantage="grpo",
        advantage_config=GroupNormConfig(center=True, scale="std"),
        kl_placement="none",
        kl_reward=None,
        citation=Citation(
            title="Group Sequence Policy Optimization",
            arxiv="2507.18071",
            url="https://arxiv.org/abs/2507.18071",
            notes="GSPO objective eq. 5; length-normalized sequence ratio eq. 7; clipping "
            "ranges in the experiments section ('3e-4 and 4e-4, respectively').",
        ),
        notes=(
            "Aggregation: eq. 5 is (1/G) sum_i over per-sequence terms; with the sequence ratio "
            "and advantage constant across a row, SEQ_MEAN_TOKEN_MEAN over the broadcast "
            "per-token values reproduces it exactly. "
            "Clip: 'we set the left and right clipping ranges in Equation (5) to 3e-4 and "
            "4e-4' - two orders of magnitude tighter than token-level PPO clipping because the "
            "length-normalized sequence ratio concentrates near 1 (the same paper runs its "
            "GRPO baseline at 0.2/0.27). "
            "KL: none in the objective ('we omit the KL regularization term hereinafter'). "
            "Advantage: GRPO group normalization, eq. 6 (r_i - mean)/std; silent on the +eps "
            "guard and Bessel correction (GroupNormConfig defaults apply). "
            "Ratio: RatioKind.SEQUENCE, s_i = exp((1/|y_i|) sum_t log-ratio), gradient flowing "
            "through every response token of the row (docs/derivations/losses.md)."
        ),
    ),
    "gspo_token": AlgorithmSpec(
        name="gspo_token",
        loss=PolicyLossConfig(
            ratio=RatioKind.SEQUENCE_TOKEN,
            surrogate=SurrogateKind.PG_CLIP,
            clip=ClipConfig(eps_low=3e-4, eps_high=4e-4),
            aggregation=Aggregation.SEQ_MEAN_TOKEN_MEAN,
        ),
        advantage="grpo",
        advantage_config=GroupNormConfig(center=True, scale="std"),
        kl_placement="none",
        kl_reward=None,
        citation=Citation(
            title="Group Sequence Policy Optimization",
            arxiv="2507.18071",
            url="https://arxiv.org/abs/2507.18071",
            notes="GSPO-token variant eq. 13 with the stop-gradient ratio eq. 14: "
            "s_{i,t} = sg[s_i] * pi_theta(y_{i,t}|..) / sg[pi_theta(y_{i,t}|..)].",
        ),
        notes=(
            "Aggregation: eq. 13 is (1/G) sum_i (1/|y_i|) sum_t = SEQ_MEAN_TOKEN_MEAN. "
            "Clip: same ranges as GSPO (3e-4 / 4e-4); the paper introduces no separate values "
            "for the token variant. "
            "KL: none, as in GSPO. "
            "Advantage: GRPO group normalization (eq. 6), with the variant existing to allow "
            "per-token advantage customization; when advantages are per-sequence constants the "
            "loss value and clipping state match GSPO exactly and only the gradient "
            "decomposition differs (token-local sg[s_i] * grad logprob_t; "
            "docs/derivations/losses.md). "
            "Silent knobs: identical to the gspo entry (std eps guard, Bessel correction)."
        ),
    ),
    "cispo": AlgorithmSpec(
        name="cispo",
        loss=PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.CISPO,
            clip=ClipConfig(eps_low=None, eps_high=0.2),
            aggregation=Aggregation.TOKEN_MEAN,
        ),
        advantage="grpo",
        advantage_config=GroupNormConfig(center=True, scale="std"),
        kl_placement="none",
        kl_reward=None,
        citation=Citation(
            title="MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention",
            arxiv="2506.13585",
            url="https://arxiv.org/abs/2506.13585",
            notes="CISPO objective eq. 4 with the clipped, stop-gradiented IS weight eq. 5; "
            "one-sided clipping stated in the experiments discussion.",
        ),
        notes=(
            "Aggregation: eq. 4 normalizes by the total token count "
            "((1/sum_i |o_i|) sum_i sum_t) = TOKEN_MEAN. "
            "Clip: the paper clips the importance-sampling weight, not the update, and runs "
            "one-sided ('we did not impose a lower bound on the IS weight ... instead, we only "
            "tuned eps_high^IS') = eps_low=None here. The paper does not disclose the tuned "
            "eps_high^IS value anywhere (verified against 2506.13585v1 including appendices); "
            "the shipped eps_high=0.2 comes from verl's compute_policy_loss_cispo default upper "
            "bound (1 + clip_ratio with clip_ratio: 0.2 in actor.yaml; note verl defaults to a "
            "two-sided clamp) - replace it to match a tuned setting. "
            "KL: none ('There is no KL penalty term in CISPO'). "
            "Advantage: GRPO group normalization (r_i - mean)/std, adopted from GRPO; silent on "
            "the +eps guard and Bessel correction (GroupNormConfig defaults apply). "
            "The clipped weight is detached (sg[w_hat]); gradients flow only through the "
            "REINFORCE factor logprobs (docs/derivations/losses.md)."
        ),
    ),
    "rloo": AlgorithmSpec(
        name="rloo",
        loss=PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.REINFORCE,
            clip=None,
            aggregation=Aggregation.SEQ_MEAN_TOKEN_SUM,
        ),
        advantage="rloo",
        advantage_config=None,
        kl_placement="reward",
        kl_reward=KLLossConfig(kind=KLEstimator.K1, coef=0.03),
        citation=Citation(
            title="Back to Basics: Revisiting REINFORCE Style Optimization for Learning "
            "from Human Feedback in LLMs",
            arxiv="2402.14740",
            url="https://arxiv.org/abs/2402.14740",
            notes="RLOO estimator in Sec. 2/3 (leave-one-out baseline over k samples, whole "
            "completion as one action); shaped reward R = r_phi - beta*log(pi/pi_ref); "
            "Appendix C beta values.",
        ),
        notes=(
            "Aggregation: the completion is a single action, so the surrogate is "
            "-A_i * log pi(y_i|x) = -A_i * sum_t logprobs, averaged over sequences = REINFORCE "
            "with [B] advantages under SEQ_MEAN_TOKEN_SUM. "
            "Clip: none - the paper removes PPO clipping entirely ('clipping is Rarely "
            "Necessary in RLHF') and uses the plain REINFORCE gradient. "
            "KL: in the reward, R(x, y) = r_phi(x, y) - beta*log(pi_theta(y|x)/pi_ref(y|x)); "
            "the sequence log-ratio equals the token sum of k1 evaluated at the sampling "
            "policy, i.e. polgrad's kl_in_reward with kind=K1. beta = 0.03 for TL;DR and 0.10 "
            "for Anthropic-HH (Appendix C), swept over {0.1, 0.25, 0.5, 1.0} in Sec. 5.2.2; "
            "0.03 is shipped - the value is dataset-dependent, replace to match. "
            "Advantage: leave-one-out baseline A_i = r_i - mean_{j != i}(r_j) over the k "
            "samples of the prompt (k in {2, 4} in the paper); rloo_advantages takes no config. "
            "The paper specifies no token-level aggregation (the single-action formulation has "
            "none) and no std normalization of advantages."
        ),
    ),
    "reinforce_pp": AlgorithmSpec(
        name="reinforce_pp",
        loss=PolicyLossConfig(
            ratio=RatioKind.TOKEN,
            surrogate=SurrogateKind.PG_CLIP,
            clip=ClipConfig(eps_low=0.2, eps_high=0.2),
            aggregation=Aggregation.SEQ_MEAN_TOKEN_MEAN,
        ),
        advantage="reinforce_pp",
        advantage_config=ReinforcePPConfig(batch_norm=True),
        kl_placement="reward",
        kl_reward=KLLossConfig(kind=KLEstimator.K1, coef=0.001),
        citation=Citation(
            title="REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global "
            "Advantage Normalization",
            arxiv="2501.03262",
            url="https://arxiv.org/abs/2501.03262",
            notes="PPO-clip objective eq. 1; token-level KL-in-reward eq. 8-9; global "
            "advantage normalization eq. 10; group-baseline variant in Appendix A; "
            "Sec. 4.2.3 hyperparameters (eps=0.2, beta=0.001, gamma=1.0). Numbering per v2.",
        ),
        notes=(
            "Aggregation: eq. 1 takes the token mean within a sequence under an expectation "
            "over sequences = SEQ_MEAN_TOKEN_MEAN. "
            "Clip: PPO clipping with eps_low = eps_high = 0.2 (Sec. 4.2.3 'The clipping "
            "parameter (eps) is set to 0.2'). "
            "KL: in the reward, token-level (eq. 8), with KL(t) = "
            "log(pi_theta_old(o_t|q,o_<t)/pi_SFT(o_t|q,o_<t)) (eq. 9) = k1 at the sampling "
            "policy, beta = 0.001 (Sec. 4.2.3); apply via kl_in_reward, then form returns with "
            "gamma = 1.0 (Sec. 4.2.3). "
            "Advantage: global batch z-normalization (A - mean)/std (eq. 10) = "
            "reinforce_pp_advantages(group_ids=None, batch_norm=True); the mean subtraction "
            "and std division compose to eq. 10 because the batch std is shift-invariant. The "
            "REINFORCE++-baseline variant (Appendix A) subtracts the per-group mean and then "
            "divides by the batch std = pass group_ids. "
            "The paper does not specify the +eps guard on the std; ReinforcePPConfig.eps=1e-8 "
            "is polgrad's division guard."
        ),
    ),
    "grpo_tis": AlgorithmSpec(
        name="grpo_tis",
        loss=replace(
            _GRPO_LOSS,
            is_correction=ISCorrectionConfig(cap=10.0, level="token"),
        ),
        advantage="grpo",
        advantage_config=GroupNormConfig(center=True, scale="std"),
        kl_placement="loss",
        kl_reward=None,
        citation=Citation(
            title="verl PR #2953: add Rollout-Training Mismatch Fix - Truncated "
            "importance sampling",
            arxiv=None,
            url="https://github.com/volcengine/verl/pull/2953",
            notes="Merged 2025-08-26. Token-level TIS weight in compute_policy_loss_vanilla: "
            "exp(old_log_prob - rollout_log_probs) clamped at tis_imp_ratio_cap, multiplied "
            "into pg_losses. GRPO base: arXiv 2402.03300 (see the 'grpo' entry).",
        ),
        notes=(
            "GRPO (see the 'grpo' entry for aggregation SEQ_MEAN_TOKEN_MEAN, clip 0.2/0.2, and "
            "the as-loss k3 KL with coef 0.04) plus the truncated importance-sampling "
            "correction of verl PR #2953 for the rollout/trainer logprob mismatch: "
            "w_t = min(exp(old_logprobs_t - rollout_logprobs_t), C) computed per token and "
            "multiplied into the per-token loss, applied as data (sg[w]; verl's inputs carry "
            "no gradient there, polgrad detaches explicitly). "
            "Cap: verl ships tis_imp_ratio_cap: -1 (TIS disabled) and prescribes no tuned "
            "value; the PR's usage example enables C = 10.0 "
            "(+actor_rollout_ref.actor.behav_imp_weight_cap=10.0), which is the cap shipped "
            "here - replace to match your setup. The PR links the motivating blog 'Your "
            "Efficient RL Framework Secretly Brings You Off-Policy RL Training'. "
            "Requires rollout_logprobs at call time; with rollout_logprobs == old_logprobs the "
            "correction is a no-op (tests/test_losses.py::test_is_correction_weight_one_is_noop)."
        ),
    ),
}


def get(name: str) -> AlgorithmSpec:
    """Look up an algorithm specification by registry key.

    ``get(name) is ALGORITHMS[name]``; exported at the package level as
    ``polgrad.get_algorithm``.

    Args:
        name: Registry key, one of the ten keys of :data:`ALGORITHMS`.

    Returns:
        The registered :class:`AlgorithmSpec`.

    Raises:
        ValueError: If ``name`` is not a registered algorithm; the message lists the
            known names.

    References:
        docs/derivations/variants.md;
        tests/test_registry.py::test_get_returns_registry_entries_and_rejects_unknown_names.
    """
    spec = ALGORITHMS.get(name)
    if spec is None:
        known = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"unknown algorithm name {name!r}; known names: {known}")
    return spec


def describe(name: str) -> str:
    """Render a plain-text summary of a registered algorithm specification.

    The output states, one fact per line: the surrogate and ratio kinds, the clip
    bounds, the aggregation mode (with the ``dataclasses.replace`` instruction when a
    ``TOKEN_SUM_NORM`` spec ships ``norm_len=None``), the KL placement with estimator
    and coefficient, any TIS correction, the advantage estimator and its config, the
    citation, and the provenance notes. Exported at the package level as
    ``polgrad.describe_algorithm``.

    Args:
        name: Registry key, one of the ten keys of :data:`ALGORITHMS`.

    Returns:
        A multi-line string; parseable by eye, not an API.

    Raises:
        ValueError: If ``name`` is not a registered algorithm.

    References:
        docs/derivations/variants.md;
        tests/test_registry.py::test_describe_states_aggregation_clip_and_kl_facts,
        tests/test_registry.py::test_dr_grpo_ships_norm_len_none_and_raises_until_replaced.
    """
    spec = get(name)
    loss = spec.loss
    lines = [f"{spec.name}: {loss.surrogate.value} surrogate on {loss.ratio.value} ratio"]
    if loss.clip is None:
        lines.append("clip: none")
    else:
        lines.append(
            f"clip: eps_low={loss.clip.eps_low}, eps_high={loss.clip.eps_high}, "
            f"ratio_cap={loss.clip.ratio_cap}"
        )
    aggregation = f"aggregation: {loss.aggregation.value}"
    if loss.aggregation is Aggregation.TOKEN_SUM_NORM:
        aggregation += f" (norm_len={loss.norm_len})"
    lines.append(aggregation)
    if loss.aggregation is Aggregation.TOKEN_SUM_NORM and loss.norm_len is None:
        lines.append(
            "norm_len is None: apply dataclasses.replace(spec.loss, norm_len=<generation "
            "budget>) before calling policy_loss, which raises ValueError until then"
        )
    if spec.kl_placement == "loss":
        kl = loss.kl
        assert kl is not None  # __post_init__ invariant
        if kl.aggregation is not None:
            kl_agg = kl.aggregation.value
        else:
            kl_agg = f"{loss.aggregation.value} (inherited)"
        lines.append(f"kl: as-loss, {kl.kind.value}, coef={kl.coef}, aggregation={kl_agg}")
    elif spec.kl_placement == "reward":
        kl = spec.kl_reward
        assert kl is not None  # __post_init__ invariant
        lines.append(
            f"kl: in-reward, {kl.kind.value}, coef={kl.coef} (apply via kl_in_reward before "
            "advantage estimation)"
        )
    else:
        lines.append("kl: none")
    if loss.is_correction is not None:
        lines.append(
            f"is_correction: TIS, {loss.is_correction.level}-level, cap={loss.is_correction.cap}"
        )
    lines.append(f"advantage: {spec.advantage} ({spec.advantage_config!r})")
    if spec.citation.arxiv is not None:
        source = f"arXiv {spec.citation.arxiv}"
    else:
        source = f"{spec.citation.url}"
    lines.append(f"citation: {spec.citation.title} ({source})")
    lines.append(f"notes: {spec.notes}")
    return "\n".join(lines)
