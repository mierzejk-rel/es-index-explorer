"""Tests for the frozen Segment 7 numerical execution contract."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_PLAN = PROJECT_ROOT / "reports" / "13-simple-mode-analysis-research-plan.md"
OPERATIONS = PROJECT_ROOT / "reports" / "22-segment-7-layer1-operations.md"


def test_research_plan_freezes_layer1_execution_choices() -> None:
    source = RESEARCH_PLAN.read_text(encoding="utf-8")

    for clause in (
        "**Layer 1 numerical execution contract.**",
        "**Inner Newton rule.**",
        "**Curvature acceptance.**",
        "**Three-start equivalence.**",
        "**Replenishment cap and terminal state.**",
        "**Global and local draw identity.**",
        "**Outer-batch MCSE.**",
        "**Importance-resampling purposive subset.**",
        "**Synthetic-recovery fixture.**",
        "**Exact ALR boundary operation.**",
    ):
        assert clause in source
    assert "LAYER1_REPLENISHMENT_EXHAUSTED" in source
    assert "NOT_COMPARABLE_SHARED_DRAWS" in source
    assert "Section 5.2 belongs to `fit-families`" in source


def test_operations_document_matches_frozen_layer1_constants() -> None:
    source = OPERATIONS.read_text(encoding="utf-8")

    for contract in (
        "`eps=0.5`",
        "`B_psi=500`",
        "`M_b=40`",
        "`M_cond=20,000`",
        "`gamma=0.90`",
        "`kappa=0.75`",
        "`10*B_target`",
        "`ESS/N>=0.10`",
        "`fit_layer1_runnable`",
        "`04-supplement-exploratory-p4-evidence.md`",
        "`partial_reports/05-suitability-estimates.md`",
    ):
        assert contract in source
