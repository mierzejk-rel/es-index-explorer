"""Freeze the outcome-blind production-shaped Layer 1 calibration design."""

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from es_index_explorer.question_analysis.layer1_model import inverse_alr_batch
from es_index_explorer.question_analysis.storage import atomic_write_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    PROJECT_ROOT / "artifacts" / "question_analysis" / "simplemode-v1-segment7"
)
SOURCE_TABLE = SOURCE_ROOT / "tables" / "trace_pfu_table.parquet"
CONTRACT = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "question_analysis"
    / "layer1-synthetic-contract.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "question_analysis"
    / "layer1-synthetic-production-design.json"
)
REFERENCE_DRAW_COUNT = 100_000


def _reference_events(
    rubrics: list[dict[str, object]],
    contract: dict[str, object],
) -> dict[str, float]:
    mu0 = np.asarray(cast(list[float], contract["mu0"]))
    offsets = cast(dict[str, list[float]], contract["dataset_offsets"])
    sigma_between = np.asarray(cast(list[list[float]], contract["sigma_between"]))
    sigma_within = np.asarray(cast(list[list[float]], contract["sigma_within"]))
    rng = np.random.default_rng(cast(int, contract["reference_probability_seed"]))
    totals = {
        f"{kind}_{suffix}": 0.0
        for kind in ("minimum_ge", "proportion_ge")
        for suffix in ("0_75", "0_60", "0_50")
    }
    for rubric in rubrics:
        dataset = str(rubric["dataset"])
        variants = cast(list[dict[str, object]], rubric["variants"])
        variant_count = len(variants)
        rubric_latents = rng.multivariate_normal(
            mu0 + np.asarray(offsets[dataset]),
            sigma_between,
            size=REFERENCE_DRAW_COUNT,
        )
        values = np.empty((REFERENCE_DRAW_COUNT, variant_count))
        for variant_index in range(variant_count):
            eta = rubric_latents + rng.multivariate_normal(
                np.zeros(2),
                sigma_within,
                size=REFERENCE_DRAW_COUNT,
            )
            probabilities = inverse_alr_batch(eta)
            values[:, variant_index] = probabilities[:, 0] + 0.5 * probabilities[:, 2]
        binding_count = int(np.ceil(0.75 * variant_count))
        for floor, suffix in (
            (0.75, "0_75"),
            (0.60, "0_60"),
            (0.50, "0_50"),
        ):
            totals[f"minimum_ge_{suffix}"] += float(
                np.mean(np.min(values, axis=1) >= floor)
            )
            totals[f"proportion_ge_{suffix}"] += float(
                np.mean(np.count_nonzero(values >= floor, axis=1) >= binding_count)
            )
    return {name: value / len(rubrics) for name, value in sorted(totals.items())}


def main() -> int:
    """Generate the immutable structural calibration design and references."""
    frame = pd.read_parquet(SOURCE_TABLE)
    structural = frame.loc[
        frame["eligible"].astype(bool),
        [
            "rubric_id",
            "rubric_order",
            "eval_dataset",
            "variant_id",
            "variant_index",
            "arm_id",
            "stage",
            "N_r",
        ],
    ].sort_values(["rubric_order", "variant_index", "arm_id"])
    rubrics: list[dict[str, object]] = []
    grouped = structural.groupby(
        ["rubric_id", "rubric_order", "eval_dataset", "N_r"],
        sort=False,
    )
    for (rubric_id, rubric_order, dataset, expectation_count), rubric in grouped:
        variants = [
            {
                "variant_id": str(variant_id),
                "variant_index": int(variant["variant_index"].iloc[0]),
                "arm_ids": list(variant["arm_id"].astype(str)),
                "stages": list(variant["stage"].astype(str)),
            }
            for variant_id, variant in rubric.groupby("variant_id", sort=False)
        ]
        rubrics.append(
            {
                "rubric_id": str(rubric_id),
                "rubric_order": int(rubric_order),
                "dataset": str(dataset),
                "expectation_count": int(expectation_count),
                "variants": variants,
            }
        )
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source_sha256 = hashlib.sha256(SOURCE_TABLE.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "source_root": SOURCE_ROOT.name,
        "source_table": "tables/trace_pfu_table.parquet",
        "source_table_sha256": source_sha256,
        "outcome_columns_used": [],
        "rubric_count": len(rubrics),
        "variant_count": sum(
            len(cast(list[object], rubric["variants"])) for rubric in rubrics
        ),
        "reference_probability_seed": contract["reference_probability_seed"],
        "reference_draw_count": REFERENCE_DRAW_COUNT,
        "reference_event_probabilities": _reference_events(rubrics, contract),
        "rubrics": rubrics,
    }
    atomic_write_bytes(
        OUTPUT,
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
