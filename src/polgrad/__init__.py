"""Reference semantics, conformance testing, and pathology diagnostics for LLM
policy-gradient post-training.

The core namespace exports the loss algebra, KL estimators, advantage estimators, and the
algorithm registry. ``polgrad.diagnostics``, ``polgrad.verify``, and
``polgrad.conformance`` are subpackages with their own exports. Conventions (shapes,
masking, signs, dtypes) are specified in ``docs/conventions.md`` and enforced by tests.
"""

from __future__ import annotations

from polgrad.advantages import (
    GAEConfig,
    GroupNormConfig,
    ReinforcePPConfig,
    broadcast_to_tokens,
    gae,
    grpo_advantages,
    reinforce_pp_advantages,
    rloo_advantages,
    whiten,
)
from polgrad.aggregate import (
    Aggregation,
    aggregate,
    effective_token_weights,
    microbatch_token_weights,
)
from polgrad.kl import (
    KLEstimator,
    KLLossConfig,
    kl_estimate,
    kl_in_reward,
    kl_loss,
    reverse_kl_grad_surrogate,
)
from polgrad.losses import (
    ClipConfig,
    ISCorrectionConfig,
    PolicyLossConfig,
    PolicyLossResult,
    RatioKind,
    SurrogateKind,
    ValueLossResult,
    policy_loss,
    value_loss,
)
from polgrad.registry import (
    ALGORITHMS,
    AlgorithmSpec,
    Citation,
)
from polgrad.registry import describe as describe_algorithm
from polgrad.registry import get as get_algorithm

__version__ = "0.2.0"

__all__ = [
    "ALGORITHMS",
    "Aggregation",
    "AlgorithmSpec",
    "Citation",
    "ClipConfig",
    "GAEConfig",
    "GroupNormConfig",
    "ISCorrectionConfig",
    "KLEstimator",
    "KLLossConfig",
    "PolicyLossConfig",
    "PolicyLossResult",
    "RatioKind",
    "ReinforcePPConfig",
    "SurrogateKind",
    "ValueLossResult",
    "__version__",
    "aggregate",
    "broadcast_to_tokens",
    "describe_algorithm",
    "effective_token_weights",
    "gae",
    "get_algorithm",
    "grpo_advantages",
    "kl_estimate",
    "kl_in_reward",
    "kl_loss",
    "microbatch_token_weights",
    "policy_loss",
    "reinforce_pp_advantages",
    "reverse_kl_grad_surrogate",
    "rloo_advantages",
    "value_loss",
    "whiten",
]
