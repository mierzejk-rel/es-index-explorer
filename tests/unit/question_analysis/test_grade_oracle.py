"""Tests for the source-generated ordinal-grade oracle."""

import pytest

from es_index_explorer.question_analysis.grade_oracle import GradeOracle

pytestmark = pytest.mark.unit


@pytest.fixture
def oracle() -> GradeOracle:
    return GradeOracle.load()


def test_fixture_covers_every_key_through_n_21(oracle: GradeOracle) -> None:
    assert oracle.max_n == 21
    assert len(oracle.grades) == 4046


def test_fixture_preserves_grade_edges_and_error_override(
    oracle: GradeOracle,
) -> None:
    assert oracle.lookup(0, 0, 4, False) == "Poor"
    assert oracle.lookup(3, 1, 0, False) == "Good"
    assert oracle.lookup(3, 1, 0, True) == "Critical Error"


@pytest.mark.parametrize(
    ("counts", "expected"),
    (
        ((0, 4, 0, False), 1),
        ((1, 3, 0, False), 1),
        ((2, 2, 0, False), 1),
        ((4, 0, 0, False), None),
        ((0, 0, 4, False), None),
        ((2, 2, 0, True), None),
    ),
)
def test_expectations_to_next_grade(
    oracle: GradeOracle,
    counts: tuple[int, int, int, bool],
    expected: int | None,
) -> None:
    assert oracle.expectations_to_next_grade(*counts) == expected
