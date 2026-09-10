"""Immutable packed and multiplicity-compressed Layer 1 count representations."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.special import gammaln

from es_index_explorer.question_analysis.errors import MalformedInputError


@dataclass(frozen=True, slots=True)
class CompressedPfuCounts:
    """Store unique PFU patterns, exact multiplicities, and fixed log constants."""

    patterns: np.ndarray
    multiplicities: np.ndarray
    combinatorial_log_terms: np.ndarray

    @property
    def trace_count(self) -> int:
        """Return the represented number of raw traces."""
        return int(self.multiplicities.sum())


@dataclass(frozen=True, slots=True)
class PackedLayer1Data:
    """Store contiguous count arrays and deterministic hierarchy slices."""

    counts: np.ndarray
    variant_offsets: np.ndarray
    rubric_variant_offsets: np.ndarray
    variant_ids: tuple[str, ...]
    rubric_ids: tuple[str, ...]
    dataset_indices: np.ndarray
    expectation_counts: np.ndarray


class VariantDataLike(Protocol):
    """Describe the variant fields required by the packing boundary."""

    @property
    def variant_id(self) -> str:
        """Return the stable variant identifier."""
        ...

    @property
    def counts(self) -> np.ndarray:
        """Return the raw PFU count matrix."""
        ...


class RubricDataLike(Protocol):
    """Describe the rubric fields required by the packing boundary."""

    @property
    def rubric_id(self) -> str:
        """Return the stable rubric identifier."""
        ...

    @property
    def dataset(self) -> str:
        """Return the dataset identifier."""
        ...

    @property
    def expectation_count(self) -> int:
        """Return the rubric expectation count."""
        ...

    @property
    def variants(self) -> Sequence[VariantDataLike]:
        """Return the ordered variants."""
        ...


class Layer1DataLike(Protocol):
    """Describe the dataset fields required by the packing boundary."""

    @property
    def rubrics(self) -> Sequence[RubricDataLike]:
        """Return the ordered rubrics."""
        ...

    @property
    def dataset_levels(self) -> Sequence[str]:
        """Return the ordered dataset levels."""
        ...


def compress_pfu_counts(counts: np.ndarray) -> CompressedPfuCounts:
    """Compress repeated PFU rows without changing their mathematical weight."""
    values = np.asarray(counts)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or values.shape[0] == 0
        or not np.issubdtype(values.dtype, np.number)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values != np.floor(values))
    ):
        raise MalformedInputError("PFU counts must be a non-empty integer matrix")
    integer_values = np.ascontiguousarray(values, dtype=np.int64)
    patterns, multiplicities = np.unique(integer_values, axis=0, return_counts=True)
    totals = patterns.sum(axis=1)
    constants = gammaln(totals + 1.0) - gammaln(patterns + 1.0).sum(axis=1)
    return CompressedPfuCounts(
        patterns=np.ascontiguousarray(patterns),
        multiplicities=np.ascontiguousarray(multiplicities, dtype=np.int64),
        combinatorial_log_terms=np.ascontiguousarray(constants, dtype=float),
    )


def expand_pfu_counts(compressed: CompressedPfuCounts) -> np.ndarray:
    """Expand compressed PFU patterns in canonical pattern order."""
    _validate_compressed(compressed)
    return np.repeat(
        compressed.patterns,
        compressed.multiplicities,
        axis=0,
    )


def _validate_compressed(compressed: CompressedPfuCounts) -> None:
    patterns = np.asarray(compressed.patterns)
    multiplicities = np.asarray(compressed.multiplicities)
    constants = np.asarray(compressed.combinatorial_log_terms)
    if (
        patterns.ndim != 2
        or patterns.shape[1] != 3
        or multiplicities.shape != (len(patterns),)
        or constants.shape != (len(patterns),)
        or np.any(multiplicities <= 0)
        or not np.isfinite(patterns).all()
        or not np.isfinite(constants).all()
    ):
        raise MalformedInputError("Compressed PFU count contract is invalid")


def pack_layer1_dataset(data: Layer1DataLike) -> PackedLayer1Data:
    """Pack a Layer1Dataset-like object into contiguous numeric arrays."""
    rubrics = tuple(data.rubrics)
    counts: list[np.ndarray] = []
    variant_offsets = [0]
    rubric_variant_offsets = [0]
    variant_ids: list[str] = []
    rubric_ids: list[str] = []
    dataset_levels = tuple(data.dataset_levels)
    dataset_lookup = {dataset: index for index, dataset in enumerate(dataset_levels)}
    dataset_indices: list[int] = []
    expectation_counts: list[int] = []
    for rubric in rubrics:
        rubric_ids.append(str(rubric.rubric_id))
        dataset_indices.append(dataset_lookup[str(rubric.dataset)])
        expectation_counts.append(int(rubric.expectation_count))
        for variant in rubric.variants:
            current = np.ascontiguousarray(variant.counts, dtype=np.int64)
            counts.append(current)
            variant_ids.append(str(variant.variant_id))
            variant_offsets.append(variant_offsets[-1] + len(current))
        rubric_variant_offsets.append(len(variant_ids))
    return PackedLayer1Data(
        counts=(
            np.ascontiguousarray(np.vstack(counts))
            if counts
            else np.empty((0, 3), dtype=np.int64)
        ),
        variant_offsets=np.asarray(variant_offsets, dtype=np.int64),
        rubric_variant_offsets=np.asarray(rubric_variant_offsets, dtype=np.int64),
        variant_ids=tuple(variant_ids),
        rubric_ids=tuple(rubric_ids),
        dataset_indices=np.asarray(dataset_indices, dtype=np.int64),
        expectation_counts=np.asarray(expectation_counts, dtype=np.int64),
    )
