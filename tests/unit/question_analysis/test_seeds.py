"""Tests for deterministic random streams."""

import numpy as np
import pytest
from pydantic import ValidationError

from es_index_explorer.question_analysis.contracts import (
    AnalysisLockManifest,
    FileFingerprint,
)
from es_index_explorer.question_analysis.seeds import (
    derive_stream_seed,
    named_stream_seeds,
    rng_for,
)

pytestmark = pytest.mark.unit

EXPECTED_STREAM_SEEDS = {
    "layer1_bootstrap": 14761339198039170915,
    "laplace_draws": 17455818688350029280,
    "glm_bootstrap": 5921207744115555729,
    "r_oracle": 801891564970050136,
    "gold_sampling": 13609121314617641732,
    "gold_recode_sampling": 14671333374540819100,
    "annotation_shuffle": 1545213380394893023,
    "diagnostic_resampling": 11705599034764626503,
    "oracle_offdiag": 10969513090860944516,
}


def test_named_stream_seeds_match_locked_values() -> None:
    assert named_stream_seeds() == EXPECTED_STREAM_SEEDS


def test_streams_are_independent_of_consumption_order() -> None:
    first = rng_for("gold_sampling").integers(0, 1_000_000, size=8)
    rng_for("annotation_shuffle").integers(0, 1_000_000, size=200)
    repeated = rng_for("gold_sampling").integers(0, 1_000_000, size=8)

    np.testing.assert_array_equal(first, repeated)


@pytest.mark.parametrize(("name", "seed"), EXPECTED_STREAM_SEEDS.items())
def test_individual_derivation_matches_manifest(name: str, seed: int) -> None:
    assert derive_stream_seed(name) == seed


@pytest.mark.parametrize(
    "stream_seeds",
    (
        {
            name: seed
            for name, seed in EXPECTED_STREAM_SEEDS.items()
            if name != "r_oracle"
        },
        {**EXPECTED_STREAM_SEEDS, "unexpected": 1},
        {**EXPECTED_STREAM_SEEDS, "gold_sampling": 1},
    ),
)
def test_manifest_rejects_noncanonical_stream_seeds(
    stream_seeds: dict[str, int],
) -> None:
    specification = FileFingerprint(
        path="specification.md", sha256="ab" * 32, size_bytes=1
    )

    with pytest.raises(
        ValidationError,
        match="stream_seeds must exactly match the frozen named streams",
    ):
        AnalysisLockManifest(
            specification=specification,
            stream_seeds=stream_seeds,
        )
