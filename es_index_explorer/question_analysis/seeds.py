"""Deterministic named random streams."""

from hashlib import sha256
from typing import TYPE_CHECKING

from es_index_explorer.question_analysis.contracts import MASTER_SEED

if TYPE_CHECKING:
    from numpy.random import Generator

STREAM_NAMES = (
    "layer1_bootstrap",
    "laplace_draws",
    "glm_bootstrap",
    "r_oracle",
    "gold_sampling",
    "annotation_shuffle",
    "diagnostic_resampling",
)


def derive_stream_seed(stream_name: str, master_seed: int = MASTER_SEED) -> int:
    """Derive one stable unsigned 64-bit stream seed."""
    if not stream_name:
        raise ValueError("stream_name must be non-empty")
    if master_seed < 0:
        raise ValueError("master_seed must be non-negative")
    digest = sha256(f"{master_seed}:{stream_name}".encode("utf-8")).digest()  # noqa: UP012
    return int.from_bytes(digest[:8], byteorder="big")


def named_stream_seeds(master_seed: int = MASTER_SEED) -> dict[str, int]:
    """Return every frozen named stream seed."""
    return {name: derive_stream_seed(name, master_seed) for name in STREAM_NAMES}


def rng_for(stream_name: str, master_seed: int = MASTER_SEED) -> "Generator":
    """Create an independent NumPy generator for a named stream."""
    from numpy.random import default_rng

    return default_rng(derive_stream_seed(stream_name, master_seed))
