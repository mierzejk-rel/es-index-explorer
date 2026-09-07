"""Tests for the fail-closed DSL derivation gate."""

from typing import cast

import pytest
from pydantic import ValidationError

from es_index_explorer.question_analysis.contracts import DslGateStatus
from es_index_explorer.question_analysis.dsl_gate import (
    DSL_PROOF_OBLIGATIONS,
    DslGateEvidence,
    DslObligationEvidence,
    evaluate_dsl_gate,
    incomplete_dsl_evidence,
)

pytestmark = pytest.mark.unit


def test_incomplete_derivation_selects_gold_only_fallback() -> None:
    result = evaluate_dsl_gate(incomplete_dsl_evidence())

    assert result["status"] == DslGateStatus.NOT_ESTABLISHED
    assert result["unmet_obligations"] == list(DSL_PROOF_OBLIGATIONS)
    family_modes = cast(dict[str, str], result["family_inference_mode"])
    assert set(family_modes.values()) == {"gold_only_confirmatory"}
    assert result["full_population_surrogate_role"] == "exploratory_only"


def test_complete_hash_linked_evidence_establishes_derivation() -> None:
    evidence = DslGateEvidence(
        outcome_blind=True,
        obligations=tuple(
            DslObligationEvidence(
                obligation=obligation,
                satisfied=True,
                evidence_sha256=f"{index:064x}",
                rationale="Proof and synthetic evidence recorded.",
            )
            for index, obligation in enumerate(DSL_PROOF_OBLIGATIONS, start=1)
        ),
    )

    result = evaluate_dsl_gate(evidence)

    assert result["status"] == DslGateStatus.ESTABLISHED
    assert result["unmet_obligations"] == []
    family_modes = cast(dict[str, str], result["family_inference_mode"])
    assert set(family_modes.values()) == {"corrected_full_population"}


def test_dsl_evidence_requires_closed_inventory_and_hash_for_pass() -> None:
    with pytest.raises(ValidationError, match="evidence_sha256"):
        DslObligationEvidence(
            obligation=DSL_PROOF_OBLIGATIONS[0],
            satisfied=True,
            rationale="Claimed without evidence.",
        )

    with pytest.raises(ValidationError, match="frozen obligation inventory"):
        DslGateEvidence(
            outcome_blind=True,
            obligations=(
                DslObligationEvidence(
                    obligation=DSL_PROOF_OBLIGATIONS[0],
                    satisfied=False,
                    rationale="Incomplete.",
                ),
            ),
        )
