"""Tests for the vendored conformance sources (contract section 4.8).

Verifies (a) self-integrity: the SHA256 recorded in each vendored file's header
matches the on-disk bytes below the header, and the whole-file SHA256s recorded
in NOTICE match the files as committed; (b) execution: every kept vendored
function runs on tiny tensors, including the dual-clip and correction branches.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import torch

from polgrad.conformance._vendor import openrlhf_loss, verl_core_algos

HEADER_END = "# === END VENDOR HEADER ===\n"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _body_sha256_check(module_file: str) -> tuple[str, str]:
    """Return (recorded, recomputed) SHA256 of the bytes below the vendor header."""
    text = Path(module_file).read_text()
    match = re.search(r"#   BODY_SHA256: ([0-9a-f]{64})\n", text)
    assert match is not None, "vendor header must record BODY_SHA256"
    head, sep, body = text.partition(HEADER_END)
    assert sep == HEADER_END, "vendor header must end with the END VENDOR HEADER line"
    assert match.start() < len(head), "BODY_SHA256 must be recorded inside the header"
    return match.group(1), hashlib.sha256(body.encode()).hexdigest()


def test_verl_vendored_body_sha256_matches_header() -> None:
    """The SHA256 recorded in verl_core_algos.py matches its on-disk vendored body."""
    recorded, recomputed = _body_sha256_check(verl_core_algos.__file__)
    assert recorded == recomputed


def test_openrlhf_vendored_body_sha256_matches_header() -> None:
    """The SHA256 recorded in openrlhf_loss.py matches its on-disk vendored body."""
    recorded, recomputed = _body_sha256_check(openrlhf_loss.__file__)
    assert recorded == recomputed


def test_notice_records_whole_file_sha256s() -> None:
    """NOTICE records the whole-file SHA256 of each vendored file as committed."""
    notice = (REPO_ROOT / "NOTICE").read_text()
    for module_file in (verl_core_algos.__file__, openrlhf_loss.__file__):
        path = Path(module_file)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.name in notice
        assert digest in notice, f"NOTICE must record the committed SHA256 of {path.name}"


def _tiny_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A fixed 2x3 batch (log_prob, old_log_prob, advantages, float mask) in float64."""
    log_prob = torch.tensor([[-1.0, -0.5, -2.0], [-0.7, -1.2, -0.3]], dtype=torch.float64)
    old_log_prob = torch.tensor([[-1.1, -0.6, -1.8], [-0.8, -1.0, -0.4]], dtype=torch.float64)
    advantages = torch.tensor([[0.5, -1.0, 0.2], [1.5, -0.4, 0.9]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    return log_prob, old_log_prob, advantages, mask


def test_verl_agg_loss_executes_all_modes() -> None:
    """verl agg_loss runs on every aggregation mode; token-mean equals the masked mean."""
    log_prob, _, _, mask = _tiny_batch()
    modes = ("token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm", "seq-mean-token-mean")
    for mode in modes:
        loss = verl_core_algos.agg_loss(loss_mat=log_prob, loss_mask=mask, loss_agg_mode=mode)
        assert loss.dim() == 0
        assert bool(torch.isfinite(loss))
    token_mean = verl_core_algos.agg_loss(
        loss_mat=log_prob, loss_mask=mask, loss_agg_mode="token-mean"
    )
    assert torch.allclose(token_mean, (log_prob * mask).sum() / mask.sum())
    with pytest.raises(ValueError, match="Invalid loss_agg_mode"):
        verl_core_algos.agg_loss(loss_mat=log_prob, loss_mask=mask, loss_agg_mode="bogus")


def test_verl_compute_policy_loss_executes_and_differentiates() -> None:
    """verl compute_policy_loss returns four finite scalars and backprops through log_prob."""
    log_prob, old_log_prob, advantages, mask = _tiny_batch()
    log_prob = log_prob.clone().requires_grad_(True)
    loss, clipfrac, ppo_kl, clipfrac_lower = verl_core_algos.compute_policy_loss(
        old_log_prob, log_prob, advantages, mask, cliprange=0.2
    )
    for value in (loss, clipfrac, ppo_kl, clipfrac_lower):
        assert value.dim() == 0
        assert bool(torch.isfinite(value))
    loss.backward()
    assert log_prob.grad is not None
    assert bool(torch.isfinite(log_prob.grad).all())


def test_verl_compute_policy_loss_dual_clip_branch() -> None:
    """verl compute_policy_loss dual-clip branch floors the A<0 objective at c·(-A).

    One token, A = -1, log-ratio 2 so ratio = e² ≈ 7.389 > c = 3:
    pg_losses1 = -A·ratio ≈ 7.389; pg_losses2 = -A·clip(ratio, 0.8, 1.2) = 1.2;
    clip1 = max = 7.389; pg_losses3 = -A·c = 3; loss = min(3, 7.389) = 3 exactly.
    """
    old_log_prob = torch.tensor([[-3.0]], dtype=torch.float64)
    log_prob = torch.tensor([[-1.0]], dtype=torch.float64)
    advantages = torch.tensor([[-1.0]], dtype=torch.float64)
    mask = torch.tensor([[1.0]], dtype=torch.float64)
    loss, _, _, clipfrac_lower = verl_core_algos.compute_policy_loss(
        old_log_prob, log_prob, advantages, mask, cliprange=0.2, clip_ratio_c=3.0
    )
    assert float(loss) == 3.0
    # masked_mean divides by mask.sum() + 1e-8, so the fraction is 1/(1 + 1e-8), not 1.
    assert float(clipfrac_lower) == pytest.approx(1.0, abs=1e-7)
    with pytest.raises(AssertionError, match="clip_ratio_c"):
        verl_core_algos.compute_policy_loss(
            old_log_prob, log_prob, advantages, mask, cliprange=0.2, clip_ratio_c=0.5
        )


def test_verl_kl_penalty_executes_all_kinds() -> None:
    """verl kl_penalty and kl_penalty_forward run for every supported estimator name."""
    log_prob, old_log_prob, _, _ = _tiny_batch()
    for kind in ("kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3", "k1+", "k3+"):
        out = verl_core_algos.kl_penalty(log_prob, old_log_prob, kind)
        assert out.shape == log_prob.shape
        assert bool(torch.isfinite(out).all())
    forward = verl_core_algos.kl_penalty_forward(log_prob, old_log_prob, "k1")
    assert torch.equal(forward, log_prob - old_log_prob)
    with pytest.raises(NotImplementedError):
        verl_core_algos.kl_penalty_forward(log_prob, old_log_prob, "full")


def test_verl_compute_value_loss_executes() -> None:
    """verl compute_value_loss returns a finite scalar loss and a clip fraction in [0, 1]."""
    values, _, returns, mask = _tiny_batch()
    vpreds = values + 0.3
    vf_loss, vf_clipfrac = verl_core_algos.compute_value_loss(
        vpreds=vpreds, returns=returns, values=values, response_mask=mask, cliprange_value=0.2
    )
    assert vf_loss.dim() == 0
    assert bool(torch.isfinite(vf_loss))
    assert 0.0 <= float(vf_clipfrac) <= 1.0


def test_verl_helper_functions_execute() -> None:
    """The vendored verl_F helpers (masked_sum, masked_mean, clip_by_value) run and agree."""
    x, _, _, mask = _tiny_batch()
    assert torch.allclose(verl_core_algos.masked_sum(x, mask), (x * mask).sum())
    assert torch.allclose(
        verl_core_algos.masked_mean(x, mask), (x * mask).sum() / (mask.sum() + 1e-8)
    )
    clipped = verl_core_algos.clip_by_value(x, x - 0.1, x + 0.1)
    assert torch.equal(clipped, x)


def test_openrlhf_aggregate_loss_executes_both_modes() -> None:
    """OpenRLHF aggregate_loss runs in token-level and sample-level modes."""
    loss_mat, _, _, mask = _tiny_batch()
    token_level = openrlhf_loss.aggregate_loss(loss_mat, mask, token_level_loss=True)
    sample_level = openrlhf_loss.aggregate_loss(loss_mat, mask, token_level_loss=False)
    for value in (token_level, sample_level):
        assert value.dim() == 0
        assert bool(torch.isfinite(value))
    assert torch.allclose(token_level, (loss_mat * mask).sum() / mask.sum())


def test_openrlhf_policy_loss_executes() -> None:
    """OpenRLHF PolicyLoss forward returns finite (loss, clip_ratio, ppo_kl, None) and backprops."""
    log_prob, old_log_prob, advantages, mask = _tiny_batch()
    log_prob = log_prob.clone().requires_grad_(True)
    module = openrlhf_loss.PolicyLoss(clip_eps_low=0.2, clip_eps_high=0.2)
    loss, clip_ratio, ppo_kl, vllm_kl = module(log_prob, old_log_prob, advantages, mask)
    for value in (loss, clip_ratio, ppo_kl):
        assert value.dim() == 0
        assert bool(torch.isfinite(value))
    assert vllm_kl is None
    loss.backward()
    assert log_prob.grad is not None


def test_openrlhf_policy_loss_dual_clip_matches_verl() -> None:
    """OpenRLHF's dual-clip branch gives the same 1-token floor c·(-A) = 3 as verl's.

    Same case as the verl dual-clip test: A = -1, ratio = e² > c = 3, so both
    implementations return exactly 3.0.
    """
    old_log_prob = torch.tensor([[-3.0]], dtype=torch.float64)
    log_prob = torch.tensor([[-1.0]], dtype=torch.float64)
    advantages = torch.tensor([[-1.0]], dtype=torch.float64)
    mask = torch.tensor([[1.0]], dtype=torch.float64)
    module = openrlhf_loss.PolicyLoss(clip_eps_low=0.2, clip_eps_high=0.2, dual_clip=3.0)
    loss, _, _, _ = module(log_prob, old_log_prob, advantages, mask)
    assert float(loss) == 3.0


def test_openrlhf_policy_loss_gspo_branch_executes() -> None:
    """OpenRLHF PolicyLoss with policy_loss_type='gspo' (sequence ratio) runs and backprops."""
    log_prob, old_log_prob, advantages, mask = _tiny_batch()
    log_prob = log_prob.clone().requires_grad_(True)
    module = openrlhf_loss.PolicyLoss(policy_loss_type="gspo")
    loss, _, _, _ = module(log_prob, old_log_prob, advantages, mask)
    assert loss.dim() == 0
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert log_prob.grad is not None


def test_openrlhf_policy_loss_is_correction_branches_execute() -> None:
    """OpenRLHF PolicyLoss vLLM IS-correction branches (tis, icepop, seq-mask-tis) run."""
    log_prob, old_log_prob, advantages, mask = _tiny_batch()
    rollout_log_prob = old_log_prob + 0.05
    for correction in ("tis", "icepop", "seq-mask-tis"):
        module = openrlhf_loss.PolicyLoss(
            enable_vllm_is_correction=True,
            vllm_is_truncated_threshold=[0.5, 2.0],
            vllm_is_correction_type=correction,
        )
        loss, _, _, vllm_kl = module(
            log_prob, old_log_prob, advantages, mask, rollout_log_probs=rollout_log_prob
        )
        assert bool(torch.isfinite(loss))
        assert vllm_kl is not None
        assert bool(torch.isfinite(vllm_kl))
    with pytest.raises(ValueError, match="vllm_is_correction_type"):
        openrlhf_loss.PolicyLoss(vllm_is_correction_type="bogus")


def test_openrlhf_value_loss_executes() -> None:
    """OpenRLHF ValueLoss runs with and without clipping and returns finite scalars."""
    values, old_values, returns, mask = _tiny_batch()
    clipped = openrlhf_loss.ValueLoss(clip_eps=0.2)(values, old_values, returns, mask)
    unclipped = openrlhf_loss.ValueLoss(clip_eps=None)(values, old_values, returns, mask)
    for value in (clipped, unclipped):
        assert value.dim() == 0
        assert bool(torch.isfinite(value))
    expected_unclipped = 0.5 * ((values - returns) ** 2 * mask).sum() / mask.sum()
    assert torch.allclose(unclipped, expected_unclipped)


def test_openrlhf_masked_mean_executes() -> None:
    """The vendored OpenRLHF masked_mean helper matches the direct masked mean."""
    x, _, _, mask = _tiny_batch()
    assert torch.allclose(openrlhf_loss.masked_mean(x, mask), (x * mask).sum() / mask.sum())
    assert torch.allclose(openrlhf_loss.masked_mean(x, None), x.mean())
