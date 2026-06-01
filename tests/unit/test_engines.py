"""Unit tests for the dependency-free engine adapters."""

import pytest

from es_index_explorer.indexing.chunking import (
    Priority,
)
from es_index_explorer.indexing.engines import PunctuationClauseEngine

pytestmark = pytest.mark.unit


def test_punctuation_clause_engine_detects_marks() -> None:
    engine = PunctuationClauseEngine()
    text = "alpha (beta); gamma, delta"
    boundaries = engine.clause_boundaries(text, 0, len(text))
    priorities = {b.priority for b in boundaries}
    assert Priority.PARENTHETICAL in priorities
    assert Priority.SEMICOLON in priorities
    assert Priority.COMMA in priorities
    # boundaries sit immediately after each mark
    for b in boundaries:
        assert text[b.char_pos - 1] in {")", ";", ","}


def test_punctuation_clause_engine_digit_guard() -> None:
    engine = PunctuationClauseEngine()
    text = "pay 1,000 now, please"
    boundaries = engine.clause_boundaries(text, 0, len(text))
    comma_positions = [b.char_pos for b in boundaries if b.priority == Priority.COMMA]
    # the comma in "1,000" is NOT a boundary; the clause comma after "now" IS
    assert text[comma_positions[0] - 1] == ","
    assert all(text[pos - 2 : pos] != "0," for pos in comma_positions)
    assert len(comma_positions) == 1


def test_punctuation_clause_engine_respects_span_bounds() -> None:
    engine = PunctuationClauseEngine()
    text = "a; b; c"
    # only the region around the second semicolon
    boundaries = engine.clause_boundaries(text, 3, len(text))
    assert all(3 < b.char_pos < len(text) for b in boundaries)
