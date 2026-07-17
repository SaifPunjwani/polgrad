"""Training-pathology diagnostics.

Every function takes plain tensors a training loop already has; every report is a frozen
dataclass with a ``summary()`` string; every threshold ships with a documented,
Monte-Carlo-tested null calibration (see ``docs/diagnostics/``).
"""

from __future__ import annotations

from polgrad.diagnostics.clipping import ClipReport, clip_report
from polgrad.diagnostics.entropy import (
    EntropyReport,
    TrendReport,
    entropy_trend,
    token_entropy_estimate,
)
from polgrad.diagnostics.ess import ESSReport, importance_ess, sliding_ess
from polgrad.diagnostics.length_bias import LengthBiasReport, length_bias_probe
from polgrad.diagnostics.mismatch import MismatchReport, logprob_mismatch

__all__ = [
    "ClipReport",
    "ESSReport",
    "EntropyReport",
    "LengthBiasReport",
    "MismatchReport",
    "TrendReport",
    "clip_report",
    "entropy_trend",
    "importance_ess",
    "length_bias_probe",
    "logprob_mismatch",
    "sliding_ess",
    "token_entropy_estimate",
]
