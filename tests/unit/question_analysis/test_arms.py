"""Tests for stage-neutral arm parsing."""

import pytest

from es_index_explorer.question_analysis.arms import parse_arm
from es_index_explorer.question_analysis.errors import MalformedInputError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "stage", "calls", "fetch", "context", "reasoning"),
    (
        (
            "S-A-hybrid-c2-rr-f30-g25-rnone-emc2_set1",
            "A",
            2,
            30,
            25,
            "none",
        ),
        (
            "S-B-bm25-dense-bm25-c3-union-f20-rnone-emc2_set2",
            "B",
            3,
            20,
            None,
            "none",
        ),
        (
            "S-C-bm25-c3-rr-f20-g20-rlow-mallinckrodt",
            "C",
            3,
            20,
            20,
            "low",
        ),
        (
            "S-C-bm25-dense-bm25-dense-c4-union-f15-rlow-emc2_set1",
            "C",
            4,
            15,
            None,
            "low",
        ),
    ),
)
def test_parse_arm_covers_every_stage(
    name: str,
    stage: str,
    calls: int,
    fetch: int,
    context: int | None,
    reasoning: str,
) -> None:
    identity = parse_arm(f"/experiments/{name}")

    assert identity.stage == stage
    assert identity.calls == calls
    assert identity.fetch == fetch
    assert identity.global_context == context
    assert identity.reasoning_effort == reasoning


def test_parse_arm_rejects_branch_count_mismatch() -> None:
    with pytest.raises(MalformedInputError, match="declares 3 calls"):
        parse_arm("S-B-bm25-dense-c3-union-f20-rnone-emc2_set1")


def test_parse_arm_rejects_unknown_stage() -> None:
    with pytest.raises(MalformedInputError, match="Cannot parse Simple Mode arm"):
        parse_arm("S-D-bm25-c1-rr-f10-g10-rnone-emc2_set1")
