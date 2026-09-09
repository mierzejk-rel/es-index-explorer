"""Benjamini-Hochberg and finite-support bracket adjudication."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from es_index_explorer.question_analysis.contracts import AnalysisFlag
from es_index_explorer.question_analysis.errors import MalformedInputError


@dataclass(frozen=True, slots=True)
class PValueBracket:
    """Store a point p-value or an uncertainty bracket."""

    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class BhFamilyDecision:
    """Store one deterministic family-level BH decision and diagnostic flag."""

    family_id: str
    rejected: bool | None
    analysis_flag: AnalysisFlag | None


@dataclass(frozen=True, slots=True)
class BhAdjudication:
    """Store lower/upper-corner BH results and indeterminate families."""

    lower_rejections: frozenset[str]
    upper_rejections: frozenset[str]
    definite_rejections: frozenset[str]
    definite_non_rejections: frozenset[str]
    indeterminate: frozenset[str]
    family_decisions: tuple[BhFamilyDecision, ...]


def benjamini_hochberg(
    p_values: Mapping[str, float], *, false_discovery_rate: float = 0.05
) -> frozenset[str]:
    """Apply the BH step-up rule to one eligible family set."""
    if not p_values:
        return frozenset()
    if not 0 < false_discovery_rate < 1:
        raise MalformedInputError("BH false-discovery rate must lie in (0, 1)")
    checked: list[tuple[str, float]] = []
    for family, p_value in p_values.items():
        value = float(p_value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise MalformedInputError(f"Invalid p-value for family {family}")
        checked.append((family, value))
    ordered = sorted(checked, key=lambda item: (item[1], item[0]))
    family_count = len(ordered)
    largest_rank = 0
    for rank, (_, p_value) in enumerate(ordered, start=1):
        if p_value <= rank * false_discovery_rate / family_count:
            largest_rank = rank
    return frozenset(family for family, _ in ordered[:largest_rank])


def adjudicate_bh_brackets(
    brackets: Mapping[str, PValueBracket],
    *,
    false_discovery_rate: float = 0.05,
) -> BhAdjudication:
    """Adjudicate all p-value brackets using the two monotone BH corners."""
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for family, bracket in brackets.items():
        if (
            not np.isfinite(bracket.lower)
            or not np.isfinite(bracket.upper)
            or not 0 <= bracket.lower <= bracket.upper <= 1
        ):
            raise MalformedInputError(f"Invalid p-value bracket for family {family}")
        lower[family] = bracket.lower
        upper[family] = bracket.upper
    lower_rejections = benjamini_hochberg(
        lower, false_discovery_rate=false_discovery_rate
    )
    upper_rejections = benjamini_hochberg(
        upper, false_discovery_rate=false_discovery_rate
    )
    indeterminate = lower_rejections.symmetric_difference(upper_rejections)
    families = frozenset(brackets)
    definite_rejections = lower_rejections & upper_rejections
    definite_non_rejections = families - (lower_rejections | upper_rejections)
    family_decisions = tuple(
        BhFamilyDecision(
            family_id=family,
            rejected=(
                None if family in indeterminate else family in definite_rejections
            ),
            analysis_flag=(
                AnalysisFlag.BH_INDETERMINATE if family in indeterminate else None
            ),
        )
        for family in sorted(families)
    )
    return BhAdjudication(
        lower_rejections=lower_rejections,
        upper_rejections=upper_rejections,
        definite_rejections=definite_rejections,
        definite_non_rejections=definite_non_rejections,
        indeterminate=indeterminate,
        family_decisions=family_decisions,
    )
