"""Fail-closed design-based surrogate-label derivation gate."""

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from es_index_explorer.question_analysis.contracts import DslGateStatus

DSL_PROOF_OBLIGATIONS = (
    "target_estimand_and_observed_data",
    "surrogate_error_and_missingness",
    "unequal_probability_gold_sampling",
    "two_way_clustered_score_mapping",
    "nuisance_estimation_and_cross_fitting",
    "influence_and_variance_derivation",
    "synthetic_calibration_bias_coverage",
)
P4_CONFIRMATORY_FAMILIES = ("F1", "F2", "F5", "F8", "F9")


class DslObligationEvidence(BaseModel):
    """Record evidence for one pre-outcome mathematical proof obligation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    obligation: str = Field(min_length=1)
    satisfied: bool
    evidence_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_satisfied_evidence(self) -> "DslObligationEvidence":
        """Require immutable evidence whenever an obligation is satisfied."""
        if self.satisfied and self.evidence_sha256 is None:
            raise ValueError("Satisfied DSL obligations require evidence_sha256")
        return self


class DslGateEvidence(BaseModel):
    """Record the complete outcome-blind DSL derivation evidence set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    outcome_blind: Literal[True]
    obligations: tuple[DslObligationEvidence, ...]

    @model_validator(mode="after")
    def validate_closed_inventory(self) -> "DslGateEvidence":
        """Require every frozen proof obligation exactly once."""
        observed = tuple(obligation.obligation for obligation in self.obligations)
        if observed != DSL_PROOF_OBLIGATIONS:
            raise ValueError("DSL evidence must follow the frozen obligation inventory")
        return self


def evaluate_dsl_gate(evidence: DslGateEvidence) -> dict[str, object]:
    """Evaluate the derivation gate and return a deterministic fallback mapping."""
    unmet = [
        obligation.obligation
        for obligation in evidence.obligations
        if not obligation.satisfied
    ]
    status = DslGateStatus.ESTABLISHED if not unmet else DslGateStatus.NOT_ESTABLISHED
    inference_mode = (
        "corrected_full_population"
        if status is DslGateStatus.ESTABLISHED
        else "gold_only_confirmatory"
    )
    return {
        "schema_version": 1,
        "status": status,
        "outcome_blind": evidence.outcome_blind,
        "proof_obligation_count": len(DSL_PROOF_OBLIGATIONS),
        "unmet_obligations": unmet,
        "family_inference_mode": {
            family: inference_mode for family in P4_CONFIRMATORY_FAMILIES
        },
        "full_population_surrogate_role": (
            "confirmatory_corrected"
            if status is DslGateStatus.ESTABLISHED
            else "exploratory_only"
        ),
    }


def incomplete_dsl_evidence(
    rationales: Mapping[str, str] | None = None,
) -> DslGateEvidence:
    """Create explicit fail-closed evidence for the currently unproved extension."""
    reasons = rationales or {}
    return DslGateEvidence(
        outcome_blind=True,
        obligations=tuple(
            DslObligationEvidence(
                obligation=obligation,
                satisfied=False,
                rationale=reasons.get(
                    obligation,
                    "No complete derivation and synthetic coverage evidence is recorded.",
                ),
            )
            for obligation in DSL_PROOF_OBLIGATIONS
        ),
    )


def validate_dsl_evidence_rows(
    rows: Sequence[Mapping[str, object]],
) -> DslGateEvidence:
    """Validate serialized obligation rows against the frozen inventory."""
    return DslGateEvidence(
        outcome_blind=True,
        obligations=tuple(DslObligationEvidence.model_validate(row) for row in rows),
    )
