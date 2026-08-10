"""Stage B adapter: reproducible ``current_union`` (heterogeneous) comparisons.

This module supplies the Stage B specific pieces (tool-multiset arm identity,
pairwise comparison rules, within-trace BM25/dense overlap, and the Stage B
report) on top of the generic, stage-neutral core in
:mod:`es_index_explorer.mlflow_analysis.experiment_arms`. See
``README-mlflow-rubric-analysis.md`` for the adapter pattern.
"""

import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from es_index_explorer.mlflow_analysis import experiment_arms as core

GRADE_ORDER = core.GRADE_ORDER
PASSING_GRADES = core.PASSING_GRADES
PERCENTILE_METHOD = core.PERCENTILE_METHOD

_DATASET_SUFFIXES = ("emc2_set1", "emc2_set2", "mallinckrodt")
_ARM_PATTERN = re.compile(
    r"^S-B-(?P<branches>(?:bm25|dense)(?:-(?:bm25|dense))*)-c(?P<calls>\d+)"
    r"-union-f(?P<fetch>\d+)-rnone$"
)
_DIMENSION_COLUMNS = [
    "tool_multiset",
    "calls",
    "fetch",
    "homogeneous",
]
_MODE_TOOL_NAMES = {
    "bm25": "get_relevant_documents_bm25",
    "dense": "get_relevant_documents_dense",
}
_EXPECTED_STAGE_B_ARMS = {
    "S-B-bm25-dense-c2-union-f15-rnone": ("bm25", "dense"),
    "S-B-bm25-bm25-c2-union-f20-rnone": ("bm25", "bm25"),
    "S-B-bm25-dense-bm25-c3-union-f15-rnone": ("bm25", "dense", "bm25"),
    "S-B-bm25-dense-bm25-c3-union-f20-rnone": ("bm25", "dense", "bm25"),
    "S-B-bm25-dense-bm25-dense-c4-union-f15-rnone": (
        "bm25",
        "dense",
        "bm25",
        "dense",
    ),
    "S-B-bm25-dense-c2-union-f20-rnone": ("bm25", "dense"),
}
_EXPECTED_ARM_COUNT = len(_EXPECTED_STAGE_B_ARMS)
_EXPECTED_DATASET_COUNT = len(_DATASET_SUFFIXES)
_EXPECTED_RUN_COUNT = _EXPECTED_ARM_COUNT * _EXPECTED_DATASET_COUNT

# Re-exported so tests can import helpers by their stable private names.
_pareto_sets = core.pareto_sets


def analyze_stage_b_snapshot(
    *,
    snapshot_dir: Path,
    report_path: Path,
) -> Path:
    """Analyze Stage B runs and write auditable local outputs plus a report.

    Parameters
    ----------
    snapshot_dir
        Completed sanitized snapshot containing the Stage B
        (6 arms x 3 datasets = 18) runs.
    report_path
        Markdown report receiving the evidence-backed Stage B write-up.

    Returns
    -------
    Path
        Directory containing auditable CSV and JSON analysis artifacts.
    """
    pandas = core.load_pandas()
    snapshot_dir = snapshot_dir.expanduser().resolve()
    analysis_dir = snapshot_dir / "stage_b_analysis"
    analysis_dir.mkdir(exist_ok=True)

    manifest = core.read_json(snapshot_dir / "manifest.json")
    experiments = pandas.read_parquet(snapshot_dir / "experiments.parquet")
    runs = pandas.read_parquet(snapshot_dir / "runs.parquet")
    metrics = pandas.read_parquet(snapshot_dir / "run_metrics.parquet")
    quality = pandas.read_parquet(snapshot_dir / "trace_quality.parquet")
    retrieval = pandas.read_parquet(snapshot_dir / "trace_retrieval.parquet")
    timings = pandas.read_parquet(snapshot_dir / "span_timings.parquet")

    runs = _augment_runs(runs)
    stage_contract_validation = _stage_b_contract_validation(runs, retrieval)
    trace_level, validation = core.build_trace_level(
        pandas=pandas,
        runs=runs,
        quality=quality,
        retrieval=retrieval,
        timings=timings,
        dimension_columns=_DIMENSION_COLUMNS,
    )
    run_level = core.build_run_level(
        pandas=pandas,
        runs=runs,
        metrics=metrics,
        trace_level=trace_level,
        dimension_columns=_DIMENSION_COLUMNS,
    )
    validation["max_pass_rate_crosscheck_absolute_difference"] = float(
        (run_level["good_acceptable_rate"] - run_level["reported_total_pass_rate"])
        .abs()
        .max()
    )
    validation["max_p50_latency_crosscheck_absolute_difference_s"] = float(
        (run_level["latency_p50_s"] - run_level["reported_execution_time_p50_s"])
        .abs()
        .max()
    )
    quality_summary = core.quality_summary(pandas, run_level)
    latency_summary = core.latency_summary(pandas, run_level)
    pairwise, pairwise_details = _pairwise_comparisons(pandas, trace_level, runs)
    overlap = _retrieval_overlap(pandas, retrieval, timings, runs)
    pareto = core.pareto_sets(pandas, run_level)
    use_case = core.use_case_summary(pandas, trace_level)
    operation_timings = core.operation_timing_summary(pandas, timings, runs)
    grade_distribution = core.grade_distribution(pandas, trace_level)

    outputs = {
        "trace_level.csv": trace_level,
        "run_level.csv": run_level,
        "quality_by_arm_dataset.csv": quality_summary,
        "latency_by_arm_dataset.csv": latency_summary,
        "pairwise_comparisons.csv": pairwise,
        "pairwise_trace_deltas.csv": pairwise_details,
        "within_trace_bm25_dense_overlap.csv": overlap,
        "pareto_by_dataset.csv": pareto,
        "quality_by_use_case.csv": use_case,
        "operation_timings.csv": operation_timings,
        "grade_distribution.csv": grade_distribution,
    }
    for filename, dataframe in outputs.items():
        dataframe.to_csv(analysis_dir / filename, index=False)

    validation.update(
        core.population_validation(
            snapshot_dir=snapshot_dir,
            manifest=manifest,
            experiments=experiments,
            runs=runs,
            expected_experiment_count=_EXPECTED_RUN_COUNT,
            expected_run_count=_EXPECTED_RUN_COUNT,
            expected_arm_count=_EXPECTED_ARM_COUNT,
            expected_dataset_count=_EXPECTED_DATASET_COUNT,
        )
    )
    validation.update(stage_contract_validation)
    validation["valid_for_stage_b_analysis"] = all(
        [
            validation["manifest_experiment_count"] == _EXPECTED_RUN_COUNT,
            validation["manifest_run_count"] == _EXPECTED_RUN_COUNT,
            validation["observed_experiment_count"] == _EXPECTED_RUN_COUNT,
            validation["observed_run_count"] == _EXPECTED_RUN_COUNT,
            validation["observed_arm_count"] == _EXPECTED_ARM_COUNT,
            validation["observed_dataset_count"] == _EXPECTED_DATASET_COUNT,
            validation["stage_b_arm_dataset_matrix_matches"],
            validation["runtime_configuration_matches"],
            validation["retrieval_mode_contract_matches"],
            validation["snapshot_file_checksums_match"],
            validation["conflicting_duplicate_assessment_count"] == 0,
            validation["trace_without_rubric_key_count"] == 0,
            validation["trace_without_grade_count"] == 0,
            validation["trace_without_latency_count"] == 0,
            validation["max_pass_rate_crosscheck_absolute_difference"] == 0,
            validation["max_p50_latency_crosscheck_absolute_difference_s"] < 0.001,
        ]
    )
    core.write_json(analysis_dir / "validation.json", validation)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(
            snapshot_dir=snapshot_dir,
            validation=validation,
            run_level=run_level,
            pairwise=pairwise,
            overlap=overlap,
            pareto=pareto,
            operation_timings=operation_timings,
        ),
        encoding="utf-8",
    )
    _write_calculation_manifest(
        analysis_dir=analysis_dir,
        snapshot_dir=snapshot_dir,
        validation=validation,
    )
    return analysis_dir


def _augment_runs(runs: Any) -> Any:
    """Parse dataset segment and retrieval-tool multiset identity from names."""
    parsed_rows: list[dict[str, object]] = []
    for row in runs.to_dict(orient="records"):
        short_name = str(row["experiment_name"]).rstrip("/").rsplit("/", maxsplit=1)[-1]
        arm_id, dataset_segment = core.split_arm_and_dataset(
            short_name, _DATASET_SUFFIXES
        )
        match = _ARM_PATTERN.fullmatch(arm_id)
        if match is None:
            raise ValueError(f"Cannot parse Stage B experiment identity {arm_id!r}.")
        branches = tuple(match.group("branches").split("-"))
        parsed_rows.append(
            {
                **row,
                "arm_id": arm_id,
                "dataset_segment": dataset_segment,
                "tool_multiset": _tool_multiset_label(branches),
                "calls": int(match.group("calls")),
                "fetch": int(match.group("fetch")),
                "homogeneous": len(set(branches)) == 1,
            }
        )
    return runs.__class__(parsed_rows)


def _tool_multiset_label(modes: tuple[str, ...]) -> str:
    """Return a deterministic display label for a retrieval-mode multiset."""
    counts = Counter(modes)
    ordered_modes = [
        *(mode for mode in _MODE_TOOL_NAMES if counts[mode]),
        *sorted(set(counts) - set(_MODE_TOOL_NAMES)),
    ]
    return " + ".join(
        f"{mode}×{counts[mode]}" for mode in ordered_modes
    )


def _expected_tool_multiset(arm_id: str) -> Counter[str] | None:
    """Return the configured MCP-tool multiset expected for one Stage B arm."""
    modes = _EXPECTED_STAGE_B_ARMS.get(arm_id)
    if modes is None:
        return None
    return Counter(_MODE_TOOL_NAMES[mode] for mode in modes)


def _parse_tool_multiset(value: object) -> Counter[str] | None:
    """Decode an exported JSON tool multiset without accepting malformed data."""
    if not isinstance(value, str):
        return None
    try:
        tools = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
        return None
    return Counter(tools)


def _as_int(value: object) -> int | None:
    """Coerce an exported numeric parameter to an integer when possible."""
    if value is None or core.is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_absent(value: object) -> bool:
    """Return whether an optional exported parameter is absent."""
    return value is None or core.is_missing(value)


def _stage_b_contract_validation(runs: Any, retrieval: Any) -> dict[str, object]:
    """Validate Stage B's exact arm matrix and exported runtime configuration."""
    expected_cells = {
        (arm_id, dataset)
        for arm_id in _EXPECTED_STAGE_B_ARMS
        for dataset in _DATASET_SUFFIXES
    }
    observed_cells = [
        (str(row["arm_id"]), str(row["dataset_segment"]))
        for row in runs.to_dict(orient="records")
    ]
    observed_cell_counts = Counter(observed_cells)
    observed_cell_set = set(observed_cells)
    expected_arm_ids = set(_EXPECTED_STAGE_B_ARMS)
    observed_arm_ids = set(runs["arm_id"])

    configuration_mismatches: list[dict[str, object]] = []
    for row in runs.to_dict(orient="records"):
        arm_id = str(row["arm_id"])
        expected_tools = _expected_tool_multiset(arm_id)
        reasons: list[str] = []
        if expected_tools is None:
            reasons.append("unexpected Stage B arm ID")
        else:
            if _parse_tool_multiset(row["simple_required_tools"]) != expected_tools:
                reasons.append("required tool multiset")
            if _as_int(row["configured_retrieval_call_count"]) != sum(
                expected_tools.values()
            ):
                reasons.append("configured retrieval call count")
            if row["simple_merge_policy"] != "current_union":
                reasons.append("merge policy")
            if _as_int(row["simple_per_call_fetch_count"]) != int(row["fetch"]):
                reasons.append("per-call fetch count")
            if not _is_absent(row["simple_global_context_chunk_count"]):
                reasons.append("global context chunk count")
            if row["reasoning_effort"] != "none":
                reasons.append("reasoning effort")
            if _as_int(row["invocation_concurrency"]) != 1:
                reasons.append("invocation concurrency")
        if reasons:
            configuration_mismatches.append(
                {
                    "run_id": row["run_id"],
                    "arm_id": arm_id,
                    "dataset_segment": row["dataset_segment"],
                    "mismatched_fields": ", ".join(reasons),
                }
            )

    retrieval_mode_violations: list[dict[str, object]] = []
    mode_rows = retrieval[retrieval["retrieval_mode"].notna()].merge(
        runs[["run_id", "arm_id", "dataset_segment"]],
        on="run_id",
        how="left",
        validate="many_to_one",
    )
    for (run_id, trace_id), group in mode_rows.groupby(["run_id", "trace_id"]):
        arm_id = str(group.iloc[0]["arm_id"])
        expected_tools = _expected_tool_multiset(arm_id)
        if expected_tools is None:
            continue
        expected_modes = Counter(
            mode
            for mode, tool_name in _MODE_TOOL_NAMES.items()
            for _ in range(expected_tools[tool_name])
        )
        actual_modes = Counter(group["retrieval_mode"])
        if actual_modes != expected_modes:
            retrieval_mode_violations.append(
                {
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "arm_id": arm_id,
                    "expected_retrieval_modes": _tool_multiset_label(
                        tuple(
                            mode
                            for mode, count in expected_modes.items()
                            for _ in range(count)
                        )
                    ),
                    "actual_retrieval_modes": _tool_multiset_label(
                        tuple(
                            mode
                            for mode, count in actual_modes.items()
                            for _ in range(count)
                        )
                    ),
                }
            )

    duplicate_cells = sorted(
        f"{arm_id}|{dataset}"
        for (arm_id, dataset), count in observed_cell_counts.items()
        if count > 1
    )
    return {
        "expected_stage_b_arm_ids": sorted(expected_arm_ids),
        "observed_stage_b_arm_ids": sorted(observed_arm_ids),
        "missing_stage_b_arm_ids": sorted(expected_arm_ids - observed_arm_ids),
        "unexpected_stage_b_arm_ids": sorted(observed_arm_ids - expected_arm_ids),
        "missing_stage_b_arm_dataset_cells": sorted(
            f"{arm_id}|{dataset}" for arm_id, dataset in expected_cells - observed_cell_set
        ),
        "unexpected_stage_b_arm_dataset_cells": sorted(
            f"{arm_id}|{dataset}" for arm_id, dataset in observed_cell_set - expected_cells
        ),
        "duplicate_stage_b_arm_dataset_cells": duplicate_cells,
        "stage_b_arm_dataset_matrix_matches": (
            observed_cell_set == expected_cells and not duplicate_cells
        ),
        "runtime_configuration_mismatch_count": len(configuration_mismatches),
        "runtime_configuration_mismatches": configuration_mismatches,
        "runtime_configuration_matches": not configuration_mismatches,
        "retrieval_mode_contract_violation_count": len(retrieval_mode_violations),
        "retrieval_mode_contract_violations": retrieval_mode_violations,
        "retrieval_mode_contract_matches": not retrieval_mode_violations,
    }


def _pairwise_comparisons(pandas: Any, trace_level: Any, runs: Any) -> tuple[Any, Any]:
    """Compute Stage B paired, directly inspectable experiment effects."""
    return core.pairwise_comparisons(
        pandas,
        trace_level,
        runs,
        dimension_columns=_DIMENSION_COLUMNS,
        comparison_type_fn=_comparison_type,
        orient_fn=_orient_comparison,
    )


def _comparison_type(left: dict[str, object], right: dict[str, object]) -> str | None:
    """Classify a controlled Stage B pair or return no comparison.

    Only single-dimension-changed pairs are controlled comparisons. Pairs
    that additionally differ in call count (for example B3/B4 versus B5) mix
    call count with tool composition and are intentionally excluded.
    """
    if left["tool_multiset"] == right["tool_multiset"] and left["fetch"] != right["fetch"]:
        return "fetch_at_matched_tool_multiset"
    if (
        left["calls"] == right["calls"]
        and left["fetch"] == right["fetch"]
        and left["tool_multiset"] != right["tool_multiset"]
        and left["homogeneous"] != right["homogeneous"]
    ):
        return "tool_composition_at_matched_calls_fetch"
    return None


def _orient_comparison(
    left: dict[str, object],
    right: dict[str, object],
    comparison_type: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Orient pairwise deltas from the simpler/baseline arm to the alternative."""
    if comparison_type == "fetch_at_matched_tool_multiset" and int(
        left["fetch"]
    ) > int(right["fetch"]):
        return right, left
    if comparison_type == "tool_composition_at_matched_calls_fetch":
        if bool(left["homogeneous"]) and not bool(right["homogeneous"]):
            return left, right
        if bool(right["homogeneous"]) and not bool(left["homogeneous"]):
            return right, left
    return left, right


def _retrieval_overlap(pandas: Any, retrieval: Any, timings: Any, runs: Any) -> Any:
    """Measure within-trace BM25/dense ranked-chunk overlap for mixed arms."""
    return core.within_trace_family_overlap(
        pandas,
        retrieval,
        timings,
        runs,
        left_family="bm25",
        right_family="dense",
        eligible_run_ids=set(runs[~runs["homogeneous"]]["run_id"]),
    )


def _render_report(
    *,
    snapshot_dir: Path,
    validation: dict[str, object],
    run_level: Any,
    pairwise: Any,
    overlap: Any,
    pareto: Any,
    operation_timings: Any,
) -> str:
    """Render the evidence-backed Stage B report with reproducible formulas."""
    best_rows = (
        run_level.sort_values(
            ["dataset_segment", "good_acceptable_rate", "latency_p50_s"],
            ascending=[True, False, True],
        )
        .groupby("dataset_segment")
        .head(6)
    )
    pareto_rows = pareto[
        pareto[["pareto_pass_p50", "pareto_good_p50", "pareto_rubric_p50"]].any(axis=1)
    ]
    arm_matrix = run_level[
        ["dataset_segment", "arm_id", "tool_multiset", "calls", "fetch", "homogeneous"]
    ].drop_duplicates(["arm_id", "tool_multiset", "calls", "fetch", "homogeneous"])
    overlap_summary = (
        overlap.groupby(["dataset_segment", "arm_id"])
        .agg(
            matched_traces=("rubric_key", "count"),
            median_jaccard=("jaccard", "median"),
            median_bm25_unique=("bm25_unique_count", "median"),
            median_dense_unique=("dense_unique_count", "median"),
        )
        .reset_index()
        if len(overlap)
        else overlap
    )
    fetch_effects = pairwise[
        pairwise["comparison_type"] == "fetch_at_matched_tool_multiset"
    ].sort_values("right_minus_left_pass_rate", ascending=False)
    composition_effects = pairwise[
        pairwise["comparison_type"] == "tool_composition_at_matched_calls_fetch"
    ].sort_values("right_minus_left_pass_rate", ascending=False)
    operation_summary = (
        operation_timings.groupby("simple_operation")
        .agg(
            run_arm_observations=("arm_id", "count"),
            median_of_run_p50_ms=("observed_duration_p50_ms", "median"),
            median_llm_retry_rate=("llm_retry_rate", "median"),
            median_es_retry_rate=("es_retry_rate", "median"),
        )
        .reset_index()
        .sort_values("median_of_run_p50_ms", ascending=False)
    )
    start_times = run_level["start_time_utc"].sort_values()
    return f"""# Stage B results

Generated from the immutable sanitized snapshot at `{snapshot_dir}` on
{datetime.now(UTC).isoformat()}.

## Arm matrix observed

{core.markdown_table(arm_matrix.sort_values(["calls", "fetch", "tool_multiset"]))}

## Validation and integrity

- Snapshot checksum validation: **{validation["snapshot_file_checksums_match"]}**
- Runs / experiments / arms / datasets: **{validation["observed_run_count"]} /
  {validation["observed_experiment_count"]} / {validation["observed_arm_count"]} /
  {validation["observed_dataset_count"]}**
- Unique traces: **{validation["unique_trace_count"]}**
- Exact duplicate assessment rows removed: **{validation["identical_duplicate_assessment_row_count"]}**
- Conflicting duplicate assessments: **{validation["conflicting_duplicate_assessment_count"]}**
- Missing grade / latency / derived rubric key: **{validation["trace_without_grade_count"]} /
  {validation["trace_without_latency_count"]} / {validation["trace_without_rubric_key_count"]}**
- Exact expected arm × dataset matrix: **{validation["stage_b_arm_dataset_matrix_matches"]}**
- Runtime configuration contract: **{validation["runtime_configuration_matches"]}**
- Runtime retrieval-mode contract violations: **{validation["retrieval_mode_contract_violation_count"]}**
- Valid for analysis: **{validation["valid_for_stage_b_analysis"]}**

Pairing uses the unique root `invoke_*` span name (`rubric_key`) within each
dataset segment, exactly as in Stage A.

## Exact calculations

- `Good+Acceptable rate = (Good count + Acceptable count) / unique trace count`.
- `Good rate = Good count / unique trace count`.
- `Critical grade rate = Critical Error count / unique trace count`.
- `RubricV2 mean = arithmetic mean of the one RubricV2 assessment per trace`.
- End-to-end latency is the `observed_duration_ms / 1000` of the unique
  `UNKNOWN` root span whose name begins with `invoke_`.
- p50/p90/p95 use pandas quantiles with `{PERCENTILE_METHOD}` interpolation.
- Two single-dimension pairwise comparisons are controlled: `fetch` at a
  matched retrieval-tool multiset, and tool composition (homogeneous vs.
  heterogeneous) at a matched call count and fetch. Pairs that also differ in
  call count (for example the c3 vs. c4 arms) are intentionally excluded as
  uncontrolled.
- Within-trace BM25/dense overlap unions ranked chunk identities produced by
  each retrieval mode's calls in the same trace and computes Jaccard overlap;
  it is only defined for traces where both modes contributed at least one
  chunk (that is, heterogeneous arms).
- Pareto membership is computed independently per dataset. An arm is dominated
  if another arm has quality greater than or equal and p50 latency less than
  or equal, with at least one strict improvement.

## Top observed arms by dataset

{core.markdown_table(best_rows[["dataset_segment", "arm_id", "trace_count", "good_acceptable_rate", "rubric_v2_mean", "latency_p50_s", "latency_p95_s", "avg_total_tokens"]])}

## Pareto sensitivity

The following arms are Pareto-efficient under at least one of Good+Acceptable,
Good-only, or RubricV2 quality, always against p50 latency:

{core.markdown_table(pareto_rows[["dataset_segment", "arm_id", "good_acceptable_rate", "good_rate", "rubric_v2_mean", "latency_p50_s", "pareto_pass_p50", "pareto_good_p50", "pareto_rubric_p50"]])}

## Fetch effect at matched retrieval-tool multiset

{core.markdown_table(fetch_effects[["dataset_segment", "left_arm", "right_arm", "matched_trace_count", "right_grade_wins", "ties", "right_grade_losses", "right_minus_left_pass_rate", "right_minus_left_rubric_v2_mean", "right_minus_left_latency_median_s"]]) if len(fetch_effects) else "No matched fetch pairs were observed in this snapshot."}

## Tool composition effect at matched calls/fetch

{core.markdown_table(composition_effects[["dataset_segment", "left_arm", "right_arm", "matched_trace_count", "right_grade_wins", "ties", "right_grade_losses", "right_minus_left_pass_rate", "right_minus_left_rubric_v2_mean", "right_minus_left_latency_median_s"]]) if len(composition_effects) else "No matched composition pairs were observed in this snapshot."}

## Within-trace BM25/dense overlap (heterogeneous arms)

{core.markdown_table(overlap_summary) if len(overlap_summary) else "No heterogeneous-arm traces with both signals present were observed."}

## Operation timing and retries

{core.markdown_table(operation_summary)}

## Reproducibility and caveats

- Do not pool raw traces across datasets; all headline comparisons are
  dataset-specific.
- Latency is observational. `latency_by_arm_dataset.csv` publishes UTC run
  timestamps and hours so time-of-day confounding can be checked. Runs span
  `{start_times.iloc[0]}` through `{start_times.iloc[-1]}` and were not
  randomized into common time windows, so small latency differences must not
  be attributed causally to arm parameters.
- Run-level MLflow metrics are retained as cross-checks, while headline quality
  and latency are recalculated from trace-level rows.
- The snapshot excludes messages, answers, query text, chunk content, XML, and
  scorer rationales by design.
- This report states observed effects only. It intentionally does not
  prescribe a Stage C design; that decision is made separately once Stage B
  results are reviewed.

## Audit files

All calculations are under `{snapshot_dir / "stage_b_analysis"}`:

- `trace_level.csv`: one row per run/trace, the primary recomputation source.
- `run_level.csv`: per-run raw counts and calculated metrics.
- `quality_by_arm_dataset.csv` and `latency_by_arm_dataset.csv`.
- `pairwise_comparisons.csv`, `within_trace_bm25_dense_overlap.csv`, and
  `quality_by_use_case.csv`.
- `pairwise_trace_deltas.csv` contains every exact paired rubric cell;
  `operation_timings.csv` contains operation/retry diagnostics; and
  `grade_distribution.csv` contains every grade numerator and denominator.
- `pareto_by_dataset.csv`, `validation.json`, and
  `calculation_manifest.json`.
"""


def _write_calculation_manifest(
    *,
    analysis_dir: Path,
    snapshot_dir: Path,
    validation: dict[str, object],
) -> None:
    """Write formulas, policies, input hashes, and output hashes."""
    output_files = sorted(
        path
        for path in analysis_dir.iterdir()
        if path.is_file() and path.name != "calculation_manifest.json"
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_dir": str(snapshot_dir),
        "input_files": core.file_hashes(
            [
                snapshot_dir / "manifest.json",
                snapshot_dir / "experiments.parquet",
                snapshot_dir / "runs.parquet",
                snapshot_dir / "run_metrics.parquet",
                snapshot_dir / "trace_quality.parquet",
                snapshot_dir / "trace_retrieval.parquet",
                snapshot_dir / "span_timings.parquet",
            ]
        ),
        "output_files": core.file_hashes(output_files),
        "formulas": {
            "good_acceptable_rate": "(Good + Acceptable) / unique traces",
            "good_rate": "Good / unique traces",
            "critical_grade_rate": "Critical Error / unique traces",
            "rubric_v2_mean": "arithmetic mean over unique trace RubricV2",
            "latency_seconds": "root invoke_* UNKNOWN observed_duration_ms / 1000",
            "percentile_method": PERCENTILE_METHOD,
            "pairing_key": "dataset_segment + rubric_key",
            "jaccard": "intersection(chunk identities) / union(chunk identities), within trace",
            "pareto": "maximize quality and minimize p50 latency per dataset",
        },
        "grade_order": GRADE_ORDER,
        "passing_grades": sorted(PASSING_GRADES),
        "validation": validation,
    }
    core.write_json(analysis_dir / "calculation_manifest.json", payload)
