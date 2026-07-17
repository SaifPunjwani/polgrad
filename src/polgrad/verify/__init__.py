"""Verification harness as public API: gradcheck runners, finite-difference checks of
analytic gradient formulas, closed-form golden problems, and Monte Carlo tools."""

from __future__ import annotations

from polgrad.verify.goldens import (
    BanditBatch,
    GoldenCase,
    SoftmaxBandit,
    golden_cases,
)
from polgrad.verify.gradcheck import check_gradient_formula, gradcheck_loss
from polgrad.verify.mc import clt_tolerance, mc_mean

__all__ = [
    "BanditBatch",
    "GoldenCase",
    "SoftmaxBandit",
    "check_gradient_formula",
    "clt_tolerance",
    "golden_cases",
    "gradcheck_loss",
    "mc_mean",
]
