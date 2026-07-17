# =============================================================================
# VENDORED CODE - DO NOT EDIT. Used only by polgrad's conformance harness.
#
# Upstream repository: https://github.com/OpenRLHF/OpenRLHF
# Commit: bc71bb19464aca306b33080b2d2bb45d154e2f49
#   (HEAD of the default branch `main` at fetch time, 2026-07-16.)
# Source paths at that commit:
#   openrlhf/models/loss.py
#     upstream file SHA256: d1dea3b37cb4c647a1a4b2472fbaefb025ac7ffb50688408d33af4c66a957551
#   openrlhf/models/utils.py (one helper the kept losses call)
#     upstream file SHA256: afea248aa7bb2fb09d3eebce9d582cb98a8f904425ba1698c2ee38ca46e8dcb1
# Upstream license: Apache-2.0 (repository LICENSE at the pinned commit; the
#   upstream source files carry no per-file license header, and the upstream
#   LICENSE keeps the unmodified Apache-2.0 template copyright line; attribution
#   in the repo-root NOTICE file).
#
# Definitions kept (bodies byte-identical to upstream):
#   from openrlhf/models/utils.py:
#     masked_mean
#   from openrlhf/models/loss.py:
#     aggregate_loss - token-level vs sample-level (sequence-mean) aggregation
#     PolicyLoss     - PPO clipped objective with dual-clip branch, GSPO
#                      sequence-ratio branch, and the vLLM importance-sampling
#                      corrections (tis / icepop / seq-mask-tis)
#     ValueLoss      - clipped PPO value loss
#   PolicyLoss and ValueLoss subclass torch.nn.Module but hold only plain float /
#   str / bool hyperparameters and use no distributed or trainer state; their
#   forward passes are pure tensor functions. OpenRLHF has no separate GRPO loss
#   class: GRPO runs through PolicyLoss (its group normalization lives in the
#   trainer's advantage computation, out of scope for this file).
#
# Definitions dropped (not pure, or out of polgrad's policy/value-loss scope):
#   - GPTLMLoss: calls torch.distributed collectives (ring attention).
#   - SFTLoss: supervised fine-tuning loss, not an RL policy/value loss.
#   - PairWiseLoss, LogExpLoss, DPOLoss: reward-model / DPO losses, out of
#     scope for the conformance harness.
#
# Edits made (all of them; everything else is verbatim):
#   1. Reduced the upstream import block to the imports the kept definitions
#      use (`from typing import Optional`, `import torch`,
#      `import torch.nn as nn`); dropped `torch.distributed`,
#      `torch.nn.functional`, and `Tuple`, which only the dropped classes used.
#   2. Replaced `from .utils import masked_mean` with `masked_mean` vendored
#      verbatim from openrlhf/models/utils.py at the same commit.
#   3. Added the `# fmt: off` directive so formatters preserve upstream layout.
#
# Self-integrity: SHA256 of every byte of this file below the END-OF-HEADER
# line, exactly as committed (recomputed and checked by
# tests/test_vendor.py::test_openrlhf_vendored_body_sha256_matches_header):
#   BODY_SHA256: 4bcbb75a4f255110f4bb70e212e166aa0276b1a134f1d87683873632761d6ed4
# fmt: off
# === END VENDOR HEADER ===
from typing import Optional

import torch
import torch.nn as nn


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def aggregate_loss(
    loss: torch.Tensor,
    loss_mask: torch.Tensor,
    token_level_loss: bool = True,
    dp_size: int = 1,
    batch_num_tokens: Optional[float] = None,
    global_batch_size: Optional[float] = None,
) -> torch.Tensor:
    """Aggregate a per-token loss matrix into a scalar using one of two reduction modes:

    - ``token_level_loss=True``  -> per-token: masked-sum / global token count.
    - ``token_level_loss=False`` -> per-sample: sum of per-sequence token-means / global
      sample count.

    ``batch_num_tokens`` (token mode) and ``global_batch_size`` (sample mode) carry the
    *global* batch totals so the loss is invariant to data-parallel sharding; ``dp_size``
    compensates for the gradient averaging that DeepSpeed/DDP applies across DP ranks.
    """
    if token_level_loss:
        if batch_num_tokens is None:
            return masked_mean(loss, loss_mask, dim=None)
        return (loss * loss_mask).sum() / batch_num_tokens * dp_size

    token_counts = loss_mask.sum(dim=-1)
    seq_loss = (loss * loss_mask).sum(dim=-1) / (token_counts + 1e-8)
    seq_mask = (token_counts > 0).float()  # exclude fully masked sequences
    if global_batch_size is None:
        return masked_mean(seq_loss, seq_mask, dim=None)
    return (seq_loss * seq_mask).sum() / global_batch_size * dp_size


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        dual_clip: float = None,
        token_level_loss: bool = True,
        policy_loss_type: str = "ppo",
        enable_vllm_is_correction: bool = False,
        vllm_is_truncated_threshold: list = None,
        vllm_is_correction_type: str = "tis",
    ) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.token_level_loss = token_level_loss
        self.dual_clip = dual_clip
        self.policy_loss_type = policy_loss_type
        self.enable_vllm_is_correction = enable_vllm_is_correction
        self.vllm_is_truncated_threshold = vllm_is_truncated_threshold
        self.vllm_is_correction_type = vllm_is_correction_type

        # GSPO requires sequence-level loss (per-sample mean)
        if policy_loss_type == "gspo":
            self.token_level_loss = False

        # Dual-clip PPO: https://arxiv.org/pdf/1912.09729
        if dual_clip is not None:
            assert dual_clip > 1.0, f"dual_clip must be > 1.0, got {dual_clip}"

        if self.vllm_is_correction_type not in {"tis", "icepop", "seq-mask-tis"}:
            raise ValueError(
                f"Invalid vllm_is_correction_type: {self.vllm_is_correction_type}, must be one of tis/icepop/seq-mask-tis"
            )

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        rollout_log_probs: Optional[torch.Tensor] = None,
        dp_size: int = 1,
        batch_num_tokens: Optional[float] = None,
        global_batch_size: Optional[float] = None,
    ) -> torch.Tensor:
        raw_policy_log_ratio = log_probs - old_log_probs
        if self.policy_loss_type == "ppo":
            policy_log_ratio = raw_policy_log_ratio.clamp(min=-20.0, max=20.0)
            ratio = policy_log_ratio.exp()
        elif self.policy_loss_type == "gspo":
            # GSPO: https://arxiv.org/pdf/2507.18071
            if self.enable_vllm_is_correction:
                log_ratio = log_probs - rollout_log_probs
            else:
                log_ratio = raw_policy_log_ratio
            policy_log_ratio = raw_policy_log_ratio.clamp(min=-20.0, max=20.0)
            ratio = (log_ratio * action_mask).sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1)
            ratio = ratio.exp().unsqueeze(-1) * action_mask
        else:
            raise ValueError(f"Invalid policy loss type: {self.policy_loss_type}")

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages

        if self.dual_clip is None:
            # Standard PPO
            loss = -torch.min(surr1, surr2)
        else:
            # Standard PPO clipping
            clip1 = torch.min(surr1, surr2)
            # Dual-clip: additional lower bound for negative advantages
            clip2 = torch.max(clip1, self.dual_clip * advantages)
            # Apply dual-clip: use clip2 for negative advantages, clip1 for positive advantages
            loss = -torch.where(advantages < 0, clip2, clip1)

        # Your Efficient RL Framework Secretly Brings You Off-Policy RL Training: https://fengyao.notion.site/off-policy-rl
        vllm_kl = None
        if self.enable_vllm_is_correction and self.policy_loss_type == "ppo":
            low_threshold, high_threshold = self.vllm_is_truncated_threshold
            rollout_log_ratio = old_log_probs - rollout_log_probs
            if self.vllm_is_correction_type == "icepop":
                # ICEPOP: token-level filtering (set coefficients outside the interval to 0)
                vllm_is = torch.exp(rollout_log_ratio).detach()
                mask = (vllm_is >= low_threshold) & (vllm_is <= high_threshold)
                vllm_is = vllm_is * mask
                loss = vllm_is * loss
            elif self.vllm_is_correction_type == "seq-mask-tis":
                # seq-mask-tis: use sequence-level geometric mean only for filtering,
                # correction coefficients still use TIS (token-level clamp)
                seq_log_ratio = masked_mean(rollout_log_ratio, action_mask, dim=-1)
                seq_is = torch.exp(seq_log_ratio)
                seq_mask = (seq_is >= low_threshold) & (seq_is <= high_threshold)
                vllm_is = torch.exp(rollout_log_ratio).detach()
                loss = seq_mask.unsqueeze(-1) * vllm_is * loss
            else:
                # TIS: token-level clamp with low and high thresholds
                vllm_is = torch.exp(rollout_log_ratio).clamp(min=low_threshold, max=high_threshold).detach()
                loss = vllm_is * loss
            vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)

        loss = aggregate_loss(
            loss,
            action_mask,
            token_level_loss=self.token_level_loss,
            dp_size=dp_size,
            batch_num_tokens=batch_num_tokens,
            global_batch_size=global_batch_size,
        )
        clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
        ppo_kl = masked_mean(-raw_policy_log_ratio.detach(), action_mask, dim=None)
        return loss, clip_ratio, ppo_kl, vllm_kl


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None, token_level_loss: bool = True) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.token_level_loss = token_level_loss

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        dp_size: int = 1,
        batch_num_tokens: Optional[float] = None,
        global_batch_size: Optional[float] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = aggregate_loss(
            loss,
            action_mask,
            token_level_loss=self.token_level_loss,
            dp_size=dp_size,
            batch_num_tokens=batch_num_tokens,
            global_batch_size=global_batch_size,
        )
        return 0.5 * loss
