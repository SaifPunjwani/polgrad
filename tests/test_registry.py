"""Enforces the registry semantics of docs/derivations/variants.md: the AlgorithmSpec
KL-placement invariant, the presence and callability of all ten registry entries, the
describe() fact lines, dr_grpo's deferred norm_len, and spot-checks of shipped constants
against the values stated in the source papers (fetched and quoted on the docs page)."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from strategies import MASKED_JUNK, LogprobBatch, padded_masks

from polgrad.advantages import GAEConfig, GroupNormConfig, ReinforcePPConfig
from polgrad.aggregate import Aggregation
from polgrad.kl import KLEstimator, KLLossConfig, kl_in_reward
from polgrad.losses import (
    ClipConfig,
    ISCorrectionConfig,
    PolicyLossConfig,
    RatioKind,
    SurrogateKind,
    policy_loss,
)
from polgrad.registry import ALGORITHMS, AlgorithmSpec, Citation, describe, get

REGISTRY_KEYS = {
    "ppo",
    "grpo",
    "dr_grpo",
    "dapo",
    "gspo",
    "gspo_token",
    "cispo",
    "rloo",
    "reinforce_pp",
    "grpo_tis",
}

SMOKE_NORM_LEN = 8


@st.composite
def _registry_batches(draw: st.DrawFn) -> LogprobBatch:
    """Like strategies.logprob_batches(max_b=4, max_t=6, seq_advantages=True) but
    draws full-precision float64 values rather than width=32 floats."""
    mask = draw(padded_masks(max_b=4, max_t=6))
    b, t = mask.shape

    def fill(low: float, high: float) -> torch.Tensor:
        vals = [
            draw(st.floats(low, high, allow_nan=False, allow_infinity=False)) for _ in range(b * t)
        ]
        out = torch.tensor(vals, dtype=torch.float64).reshape(b, t)
        return torch.where(mask, out, torch.full_like(out, MASKED_JUNK))

    logprobs = fill(-8.0, -0.0625)

    def near(base: torch.Tensor) -> torch.Tensor:
        gap = fill(-2.0, 2.0)
        return torch.where(mask, base + gap, torch.full_like(base, MASKED_JUNK))

    old_logprobs = near(logprobs)
    ref_logprobs = near(logprobs)
    rollout_logprobs = near(old_logprobs)
    advantages = torch.tensor(
        [draw(st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False)) for _ in range(b)],
        dtype=torch.float64,
    )
    return LogprobBatch(logprobs, old_logprobs, ref_logprobs, rollout_logprobs, advantages, mask)


def _callable_loss(spec: AlgorithmSpec) -> PolicyLossConfig:
    """The spec's loss config with dr_grpo's deferred norm_len filled in."""
    if spec.loss.aggregation is Aggregation.TOKEN_SUM_NORM and spec.loss.norm_len is None:
        return dataclasses.replace(spec.loss, norm_len=SMOKE_NORM_LEN)
    return spec.loss


def _smoke_batch() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """A fixed ragged [2, 3] batch exercising every logprob stream."""
    mask = torch.tensor([[True, True, True], [True, True, False]])
    logprobs = torch.tensor([[-0.4, -1.1, -0.7], [-2.0, -0.3, 0.0]], dtype=torch.float64)
    old_logprobs = logprobs.detach() + torch.tensor(
        [[0.2, -0.1, 0.05], [-0.3, 0.15, 0.0]], dtype=torch.float64
    )
    ref_logprobs = logprobs.detach() + torch.tensor(
        [[-0.25, 0.1, 0.2], [0.05, -0.2, 0.0]], dtype=torch.float64
    )
    rollout_logprobs = old_logprobs + torch.tensor(
        [[0.03, -0.02, 0.01], [-0.04, 0.02, 0.0]], dtype=torch.float64
    )
    advantages = torch.tensor([1.5, -0.5], dtype=torch.float64)
    return logprobs, old_logprobs, ref_logprobs, rollout_logprobs, advantages, mask


def _minimal_citation() -> Citation:
    return Citation(title="synthetic", arxiv=None, url="https://example.invalid", notes="test")


def _minimal_loss(*, kl: KLLossConfig | None = None) -> PolicyLossConfig:
    return PolicyLossConfig(
        ratio=RatioKind.TOKEN,
        surrogate=SurrogateKind.PG,
        clip=None,
        aggregation=Aggregation.TOKEN_MEAN,
        kl=kl,
    )


def test_algorithms_has_exactly_the_ten_contract_keys() -> None:
    """ALGORITHMS carries the ten registry keys of docs/derivations/variants.md, no more,
    no fewer."""
    assert set(ALGORITHMS) == REGISTRY_KEYS


def test_every_entry_is_a_complete_spec() -> None:
    """Every entry constructs as a frozen AlgorithmSpec whose provenance fields are filled:
    name matches the key, the citation has a title and an arXiv id or URL, and the notes
    are non-empty (docs/derivations/variants.md: notes state every silent knob)."""
    for key, spec in ALGORITHMS.items():
        assert isinstance(spec, AlgorithmSpec)
        assert spec.name == key
        assert spec.citation.title
        assert spec.citation.arxiv is not None or spec.citation.url is not None
        assert spec.notes
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "mutated"  # type: ignore[misc]


def test_every_entry_policy_loss_runs_on_a_ragged_batch() -> None:
    """Every registry entry's loss config is accepted by policy_loss on a ragged batch
    (dr_grpo after the documented norm_len replacement) and yields a finite,
    differentiable scalar."""
    logprobs, old_logprobs, ref_logprobs, rollout_logprobs, advantages, mask = _smoke_batch()
    for spec in ALGORITHMS.values():
        lp = logprobs.clone().requires_grad_(True)
        result = policy_loss(
            _callable_loss(spec),
            logprobs=lp,
            old_logprobs=old_logprobs,
            advantages=advantages,
            response_mask=mask,
            ref_logprobs=ref_logprobs,
            rollout_logprobs=rollout_logprobs,
        )
        assert result.loss.shape == ()
        assert bool(torch.isfinite(result.loss))
        assert result.loss.requires_grad
        assert (spec.loss.kl is not None) == (result.kl_loss is not None)


def test_spec_invariant_violations_raise() -> None:
    """The AlgorithmSpec __post_init__ invariant rejects every mismatch between
    kl_placement and the kl configs: 'loss' iff loss.kl is set, 'reward' iff kl_reward
    is set."""
    kl = KLLossConfig(kind=KLEstimator.K1, coef=0.1)

    def build(
        placement: str, *, loss_kl: KLLossConfig | None, kl_reward: KLLossConfig | None
    ) -> AlgorithmSpec:
        return AlgorithmSpec(
            name="synthetic",
            loss=_minimal_loss(kl=loss_kl),
            advantage="rloo",
            advantage_config=None,
            kl_placement=placement,  # type: ignore[arg-type]
            kl_reward=kl_reward,
            citation=_minimal_citation(),
            notes="test",
        )

    with pytest.raises(ValueError, match=r"kl_placement == 'loss' iff loss\.kl"):
        build("loss", loss_kl=None, kl_reward=None)
    with pytest.raises(ValueError, match=r"kl_placement == 'loss' iff loss\.kl"):
        build("none", loss_kl=kl, kl_reward=None)
    with pytest.raises(ValueError, match=r"kl_placement == 'reward' iff kl_reward"):
        build("reward", loss_kl=None, kl_reward=None)
    with pytest.raises(ValueError, match=r"kl_placement == 'reward' iff kl_reward"):
        build("none", loss_kl=None, kl_reward=kl)
    with pytest.raises(ValueError, match=r"kl_placement == 'reward' iff kl_reward"):
        build("loss", loss_kl=kl, kl_reward=kl)
    assert build("loss", loss_kl=kl, kl_reward=None).kl_placement == "loss"
    assert build("reward", loss_kl=None, kl_reward=kl).kl_placement == "reward"
    assert build("none", loss_kl=None, kl_reward=None).kl_placement == "none"


def test_get_returns_registry_entries_and_rejects_unknown_names() -> None:
    """get(name) returns the identical registry object; an unknown name raises
    ValueError listing the known names."""
    for key, spec in ALGORITHMS.items():
        assert get(key) is spec
    with pytest.raises(ValueError, match="unknown algorithm name 'gspo2'") as excinfo:
        get("gspo2")
    for key in REGISTRY_KEYS:
        assert key in str(excinfo.value)


@pytest.mark.parametrize("name", sorted(REGISTRY_KEYS))
def test_describe_states_aggregation_clip_and_kl_facts(name: str) -> None:
    """describe() states the aggregation mode, the clip bounds (or their absence), the
    KL placement with estimator kind and coefficient, the advantage family, and the
    citation for every entry."""
    spec = ALGORITHMS[name]
    text = describe(name)
    assert spec.loss.aggregation.value in text
    if spec.loss.clip is None:
        assert "clip: none" in text
    else:
        assert f"eps_low={spec.loss.clip.eps_low}" in text
        assert f"eps_high={spec.loss.clip.eps_high}" in text
    if spec.kl_placement == "loss":
        assert spec.loss.kl is not None
        assert f"kl: as-loss, {spec.loss.kl.kind.value}, coef={spec.loss.kl.coef}" in text
    elif spec.kl_placement == "reward":
        assert spec.kl_reward is not None
        assert f"kl: in-reward, {spec.kl_reward.kind.value}, coef={spec.kl_reward.coef}" in text
    else:
        assert "kl: none" in text
    if spec.loss.is_correction is not None:
        assert f"cap={spec.loss.is_correction.cap}" in text
    assert f"advantage: {spec.advantage}" in text
    assert spec.citation.title in text
    assert spec.notes in text


def test_dr_grpo_ships_norm_len_none_and_raises_until_replaced() -> None:
    """dr_grpo ships loss.norm_len=None (docs/derivations/variants.md, dr_grpo);
    policy_loss raises ValueError at
    call time until the caller applies dataclasses.replace(spec.loss, norm_len=budget),
    and describe() states that instruction."""
    spec = ALGORITHMS["dr_grpo"]
    assert spec.loss.aggregation is Aggregation.TOKEN_SUM_NORM
    assert spec.loss.norm_len is None
    logprobs, old_logprobs, _, _, advantages, mask = _smoke_batch()
    with pytest.raises(ValueError, match="norm_len"):
        policy_loss(
            spec.loss,
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            response_mask=mask,
        )
    patched = dataclasses.replace(spec.loss, norm_len=SMOKE_NORM_LEN)
    result = policy_loss(
        patched,
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        response_mask=mask,
    )
    assert bool(torch.isfinite(result.loss))
    text = describe("dr_grpo")
    assert "norm_len=None" in text
    assert "dataclasses.replace" in text


def test_ppo_config_matches_paper_stated_values() -> None:
    """Spot-check 1 (arXiv 1707.06347): eps = 0.2 (Sec. 3 'say, eps = 0.2'; best clipping
    row of Table 1), GAE gamma = 0.99 and lam = 0.95 (Tables 3-5), token-mean E_t, no KL
    term in the clipped objective (the eq. 8 KL-penalty variant is a separate,
    worse-performing alternative per Sec. 4)."""
    spec = ALGORITHMS["ppo"]
    assert spec.loss.surrogate is SurrogateKind.PG_CLIP
    assert spec.loss.ratio is RatioKind.TOKEN
    assert spec.loss.clip == ClipConfig(eps_low=0.2, eps_high=0.2, ratio_cap=None)
    assert spec.loss.aggregation is Aggregation.TOKEN_MEAN
    assert spec.kl_placement == "none"
    assert spec.advantage == "gae"
    assert spec.advantage_config == GAEConfig(gamma=0.99, lam=0.95)


def test_grpo_config_matches_paper_stated_values() -> None:
    """Spot-check 2 (arXiv 2402.03300): eq. 3 aggregates (1/G) sum_i (1/|o_i|) sum_t =
    SEQ_MEAN_TOKEN_MEAN; the KL term is added to the loss with the eq. 4 unbiased
    estimator (= k3) at coef 0.04 (Sec. 4.2); advantages are group mean/std normalized.
    The paper never states the clip eps; the shipped 0.2 is documented as coming from
    PPO/TRL/Dr.GRPO in the notes."""
    spec = ALGORITHMS["grpo"]
    assert spec.loss.aggregation is Aggregation.SEQ_MEAN_TOKEN_MEAN
    assert spec.kl_placement == "loss"
    assert spec.loss.kl == KLLossConfig(kind=KLEstimator.K3, coef=0.04)
    assert spec.loss.clip == ClipConfig(eps_low=0.2, eps_high=0.2, ratio_cap=None)
    assert spec.advantage == "grpo"
    assert isinstance(spec.advantage_config, GroupNormConfig)
    assert spec.advantage_config.center is True
    assert spec.advantage_config.scale == "std"
    assert "never states its value" in spec.notes or "does not specify" in spec.notes


def test_dapo_config_matches_paper_stated_values() -> None:
    """Spot-check 3 (arXiv 2503.14476): Clip-Higher eps_low = 0.2 and eps_high = 0.28,
    token-level aggregation eq. 12 ((1/sum|o_i|) sum_i sum_t = TOKEN_MEAN), the KL term
    excluded, group mean/std advantages kept (eq. 9)."""
    spec = ALGORITHMS["dapo"]
    assert spec.loss.clip == ClipConfig(eps_low=0.2, eps_high=0.28, ratio_cap=None)
    assert spec.loss.aggregation is Aggregation.TOKEN_MEAN
    assert spec.kl_placement == "none"
    assert isinstance(spec.advantage_config, GroupNormConfig)
    assert spec.advantage_config.scale == "std"


def test_gspo_configs_match_paper_stated_values() -> None:
    """Spot-check 4 (arXiv 2507.18071): GSPO clips the length-normalized sequence ratio
    at left/right ranges 3e-4 / 4e-4; eq. 5 and eq. 13 both reduce to
    SEQ_MEAN_TOKEN_MEAN; gspo_token differs only in the stop-gradient ratio kind
    (eq. 14); no KL term."""
    gspo = ALGORITHMS["gspo"]
    gspo_token = ALGORITHMS["gspo_token"]
    assert gspo.loss.ratio is RatioKind.SEQUENCE
    assert gspo_token.loss.ratio is RatioKind.SEQUENCE_TOKEN
    for spec in (gspo, gspo_token):
        assert spec.loss.clip == ClipConfig(eps_low=3e-4, eps_high=4e-4, ratio_cap=None)
        assert spec.loss.aggregation is Aggregation.SEQ_MEAN_TOKEN_MEAN
        assert spec.kl_placement == "none"
    assert gspo_token.loss == dataclasses.replace(gspo.loss, ratio=RatioKind.SEQUENCE_TOKEN)


def test_cispo_config_matches_paper_and_flags_undisclosed_eps() -> None:
    """CISPO (arXiv 2506.13585 eq. 4-5): CISPO surrogate, token-mean aggregation, no KL
    term, one-sided weight clipping (eps_low=None per the paper's stated experimental
    setting). The paper does not disclose the tuned eps_high value; the notes must say
    so and name the source of the shipped 0.2 rather than presenting it as the
    paper's."""
    spec = ALGORITHMS["cispo"]
    assert spec.loss.surrogate is SurrogateKind.CISPO
    assert spec.loss.clip == ClipConfig(eps_low=None, eps_high=0.2, ratio_cap=None)
    assert spec.loss.aggregation is Aggregation.TOKEN_MEAN
    assert spec.kl_placement == "none"
    assert "does not disclose" in spec.notes
    assert "verl" in spec.notes


def test_rloo_config_matches_paper_stated_values() -> None:
    """RLOO (arXiv 2402.14740): unclipped REINFORCE on the whole completion, so
    SEQ_MEAN_TOKEN_SUM with [B] leave-one-out advantages and no clip; the KL penalty
    lives in the reward as the sequence log-ratio (k1), beta = 0.03 for TL;DR
    (Appendix C; 0.10 for Anthropic-HH, recorded in the notes)."""
    spec = ALGORITHMS["rloo"]
    assert spec.loss.surrogate is SurrogateKind.REINFORCE
    assert spec.loss.ratio is RatioKind.TOKEN
    assert spec.loss.clip is None
    assert spec.loss.aggregation is Aggregation.SEQ_MEAN_TOKEN_SUM
    assert spec.advantage == "rloo"
    assert spec.advantage_config is None
    assert spec.kl_placement == "reward"
    assert spec.kl_reward == KLLossConfig(kind=KLEstimator.K1, coef=0.03)
    assert "0.10" in spec.notes


def test_reinforce_pp_config_matches_paper_stated_values() -> None:
    """REINFORCE++ (arXiv 2501.03262): PPO-clip with eps = 0.2 and per-sequence token
    means (eq. 1), token-level k1 KL folded into the reward at beta = 0.001 (eq. 8-9,
    Sec. 4.2.3), and global batch advantage normalization (eq. 10)."""
    spec = ALGORITHMS["reinforce_pp"]
    assert spec.loss.surrogate is SurrogateKind.PG_CLIP
    assert spec.loss.clip == ClipConfig(eps_low=0.2, eps_high=0.2, ratio_cap=None)
    assert spec.loss.aggregation is Aggregation.SEQ_MEAN_TOKEN_MEAN
    assert spec.kl_placement == "reward"
    assert spec.kl_reward == KLLossConfig(kind=KLEstimator.K1, coef=0.001)
    assert spec.advantage == "reinforce_pp"
    assert spec.advantage_config == ReinforcePPConfig(batch_norm=True)


def test_grpo_tis_is_grpo_plus_verl_pr_2953_correction() -> None:
    """grpo_tis equals the grpo loss config plus the token-level truncated
    importance-sampling correction of verl PR #2953, shipped at the PR usage example's
    cap C = 10.0 (verl's own default, tis_imp_ratio_cap: -1, disables TIS; recorded in
    the notes)."""
    grpo = ALGORITHMS["grpo"]
    tis = ALGORITHMS["grpo_tis"]
    assert tis.loss.is_correction == ISCorrectionConfig(cap=10.0, level="token")
    assert dataclasses.replace(tis.loss, is_correction=None) == grpo.loss
    assert tis.kl_placement == "loss"
    assert "#2953" in tis.citation.title or "2953" in str(tis.citation.url)
    assert "-1" in tis.notes


def test_kl_reward_specs_compose_with_kl_in_reward() -> None:
    """For the in-reward specs (rloo, reinforce_pp), applying kl_reward via kl_in_reward
    subtracts coef * k1 per token, and the per-sequence sum of the k1 penalty equals the
    papers' sequence log-ratio log(pi_old(y|x)/pi_ref(y|x))."""
    logprobs, old_logprobs, ref_logprobs, _, _, mask = _smoke_batch()
    token_rewards = torch.zeros_like(logprobs)
    for name in ("rloo", "reinforce_pp"):
        spec = ALGORITHMS[name]
        assert spec.kl_reward is not None
        shaped = kl_in_reward(
            token_rewards,
            old_logprobs,
            ref_logprobs,
            spec.kl_reward.kind,
            spec.kl_reward.coef,
            response_mask=mask,
        )
        k1 = torch.where(mask, old_logprobs - ref_logprobs, torch.zeros_like(logprobs))
        torch.testing.assert_close(shaped, -spec.kl_reward.coef * k1)
        seq_log_ratio = k1.sum(dim=1)
        torch.testing.assert_close(shaped.sum(dim=1), -spec.kl_reward.coef * seq_log_ratio)


def test_registry_specs_policy_loss_is_mask_invariant() -> None:
    """Perturbing masked positions of every input stream changes no output of any
    registry entry's policy_loss, bitwise (docs/conventions.md masking rule)."""
    logprobs, old_logprobs, ref_logprobs, rollout_logprobs, advantages, mask = _smoke_batch()
    zero = torch.zeros((), dtype=torch.float64)
    junk = torch.where(mask, zero, torch.full((), 55.5, dtype=torch.float64))
    for spec in ALGORITHMS.values():
        config = _callable_loss(spec)
        results = []
        for perturb in (0.0, 1.0):
            results.append(
                policy_loss(
                    config,
                    logprobs=logprobs + perturb * junk,
                    old_logprobs=old_logprobs - perturb * junk,
                    advantages=advantages,
                    response_mask=mask,
                    ref_logprobs=ref_logprobs + perturb * junk,
                    rollout_logprobs=rollout_logprobs - perturb * junk,
                )
            )
        base, perturbed = results
        assert torch.equal(base.loss, perturbed.loss)
        assert torch.equal(base.per_token_objective, perturbed.per_token_objective)
        assert torch.equal(base.ratio, perturbed.ratio)
        assert torch.equal(base.clipped_low, perturbed.clipped_low)
        assert torch.equal(base.clipped_high, perturbed.clipped_high)


@given(batch=_registry_batches())
def test_all_specs_produce_finite_losses_on_generated_batches(batch: LogprobBatch) -> None:
    """Every registry entry's loss config yields a finite scalar on Hypothesis-generated
    ragged batches with [B] advantages (the shape every registered advantage estimator
    emits)."""
    for spec in ALGORITHMS.values():
        result = policy_loss(
            _callable_loss(spec),
            logprobs=batch.logprobs,
            old_logprobs=batch.old_logprobs,
            advantages=batch.advantages,
            response_mask=batch.response_mask,
            ref_logprobs=batch.ref_logprobs,
            rollout_logprobs=batch.rollout_logprobs,
        )
        assert result.loss.shape == ()
        assert bool(torch.isfinite(result.loss))
