"""Shared fixtures for question-analysis annotation contract tests."""

from collections.abc import Callable

import pytest

QuestionPayloadFactory = Callable[..., dict[str, object]]


@pytest.fixture
def question_batch_items() -> list[dict[str, object]]:
    """Return one source-side question item addressed by a local annotation ID."""
    return [
        {
            "item_id": "source-question-001",
            "item_type": "question",
            "source_text": "Which records match?",
            "text_sha256": "a" * 64,
        }
    ]


@pytest.fixture
def question_payload_factory() -> QuestionPayloadFactory:
    """Build a valid applicable-QDMR response with optional field overrides."""

    def build(**overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "item_id": "item-001",
            "exhaustivity_requirement": "mention_some",
            "negative_conclusiveness": False,
            "presupposition_load": False,
            "qdmr_applicability": "applicable",
            "qdmr_step_count": 2,
            "qdmr_operator_set": ["FILTER", "SELECT"],
            "qdmr_normalized_question": None,
            "hop_structure": "atomic",
            "referring_form_type": "full_name_form",
            "referring_form_missingness": None,
            "recall_orientation": "precision_oriented",
            "cognitive_process_level": "remember",
        }
        payload.update(overrides)
        return payload

    return build
