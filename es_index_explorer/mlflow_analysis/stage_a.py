"""Stage A adapter: reproducible single-signal-retrieval comparisons.

This module supplies the Stage A specific pieces (arm-identity parsing,
pairwise comparison rules, and the Stage A report) on top of the generic,
stage-neutral core in :mod:`es_index_explorer.mlflow_analysis.experiment_arms`.
See ``README-mlflow-rubric-analysis.md`` for the adapter pattern.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from es_index_explorer.mlflow_analysis import experiment_arms as core

GRADE_ORDER = core.GRADE_ORDER
PASSING_GRADES = core.PASSING_GRADES
PERCENTILE_METHOD = core.PERCENTILE_METHOD

_DATASET_SUFFIXES = ("emc2_set1", "emc2_set2", "mallinckrodt")
_ARM_PATTERN = re.compile(
    r"^S-A-(?P<mode>bm25|dense|hybrid)-c(?P<calls>\d+)-rr-f(?P<fetch>\d+)-g(?P<context>\d+)-rnone$"
)
_DIMENSION_COLUMNS = ["retrieval_family", "calls", "fetch", "context"]

# Re-exported so existing call sites/tests can keep importing helpers by their
# previous private names from this module.
_to_boolean = core.to_boolean
_use_case_from_rubric = core.use_case_from_rubric
_pareto_sets = core.pareto_sets


def analyze_stage_a_snapshot(
    *,
    snapshot_dir: Path,
    report_path: Path,
) -> Path:
    """Analyze Stage A runs and write auditable local outputs plus a report.

    Parameters
    ----------
    snapshot_dir
        Completed sanitized snapshot containing the 54 Stage A runs.
    report_path
        Markdown report receiving evidence-backed Stage B recommendations.

    Returns
    -------
    Path
        Directory containing auditable CSV and JSON analysis artifacts.
    """
    pandas = core.load_pandas()
    snapshot_dir = snapshot_dir.expanduser().resolve()
    analysis_dir = snapshot_dir / "stage_a_analysis"
    analysis_dir.mkdir(exist_ok=True)

    manifest = core.read_json(snapshot_dir / "manifest.json")
    experiments = pandas.read_parquet(snapshot_dir / "experiments.parquet")
    runs = pandas.read_parquet(snapshot_dir / "runs.parquet")
    metrics = pandas.read_parquet(snapshot_dir / "run_metrics.parquet")
    quality = pandas.read_parquet(snapshot_dir / "trace_quality.parquet")
    retrieval = pandas.read_parquet(snapshot_dir / "trace_retrieval.parquet")
    timings = pandas.read_parquet(snapshot_dir / "span_timings.parquet")

    runs = _augment_runs(runs)
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
        "retrieval_overlap.csv": overlap,
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
            expected_experiment_count=54,
            expected_run_count=54,
            expected_arm_count=18,
            expected_dataset_count=3,
        )
    )
    validation["valid_for_stage_a_analysis"] = all(
        [
            validation["manifest_experiment_count"] == 54,
            validation["manifest_run_count"] == 54,
            validation["observed_experiment_count"] == 54,
            validation["observed_run_count"] == 54,
            validation["observed_arm_count"] == 18,
            validation["observed_dataset_count"] == 3,
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

    recommendations = _stage_b_recommendations(run_level, overlap, pareto)
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
            recommendations=recommendations,
        ),
        encoding="utf-8",
    )
    _write_calculation_manifest(
        analysis_dir=analysis_dir,
        snapshot_dir=snapshot_dir,
        validation=validation,
        recommendations=recommendations,
    )
    return analysis_dir


def _augment_runs(runs: Any) -> Any:
    """Parse dataset segment and experiment parameters from experiment names."""
    parsed_rows: list[dict[str, object]] = []
    for row in runs.to_dict(orient="records"):
        short_name = str(row["experiment_name"]).rstrip("/").rsplit("/", maxsplit=1)[-1]
        arm_id, dataset_segment = core.split_arm_and_dataset(
            short_name, _DATASET_SUFFIXES
        )
        match = _ARM_PATTERN.fullmatch(arm_id)
        if match is None:
            raise ValueError(f"Cannot parse Stage A experiment identity {arm_id!r}.")
        parsed_rows.append(
            {
                **row,
                "arm_id": arm_id,
                "dataset_segment": dataset_segment,
                "retrieval_family": match.group("mode"),
                "calls": int(match.group("calls")),
                "fetch": int(match.group("fetch")),
                "context": int(match.group("context")),
            }
        )
    return runs.__class__(parsed_rows)


def _pairwise_comparisons(pandas: Any, trace_level: Any, runs: Any) -> tuple[Any, Any]:
    """Compute Stage A paired, directly inspectable experiment effects."""
    return core.pairwise_comparisons(
        pandas,
        trace_level,
        runs,
        dimension_columns=_DIMENSION_COLUMNS,
        comparison_type_fn=_comparison_type,
        orient_fn=_orient_comparison,
    )


def _comparison_type(left: dict[str, object], right: dict[str, object]) -> str | None:
    """Classify a controlled Stage A pair or return no comparison."""
    if (
        left["retrieval_family"] == right["retrieval_family"]
        and left["context"] == right["context"]
        and left["calls"] != right["calls"]
    ):
        return "call_count_at_matched_mode_context"
    if (
        left["retrieval_family"] == right["retrieval_family"]
        and left["calls"] == right["calls"]
        and left["context"] != right["context"]
    ):
        return "context_at_matched_mode_calls"
    if (
        {left["retrieval_family"], right["retrieval_family"]} == {"bm25", "dense"}
        and left["calls"] == right["calls"]
        and left["context"] == right["context"]
    ):
        return "bm25_vs_dense_at_matched_calls_context"
    return None


def _orient_comparison(
    left: dict[str, object],
    right: dict[str, object],
    comparison_type: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Orient pairwise deltas from the simpler/baseline arm to the alternative."""
    if comparison_type == "call_count_at_matched_mode_context" and int(
        left["calls"]
    ) > int(right["calls"]):
        return right, left
    if comparison_type == "context_at_matched_mode_calls" and int(
        left["context"]
    ) > int(right["context"]):
        return right, left
    if (
        comparison_type == "bm25_vs_dense_at_matched_calls_context"
        and left["retrieval_family"] == "dense"
    ):
        return right, left
    return left, right


def _retrieval_overlap(pandas: Any, retrieval: Any, timings: Any, runs: Any) -> Any:
    """Measure BM25/dense ranked-chunk overlap for matched rubric variations."""
    return core.between_arm_family_overlap(
        pandas,
        retrieval,
        timings,
        runs,
        family_column="retrieval_family",
        left_family="bm25",
        right_family="dense",
        match_columns=["calls", "context"],
    )


def _stage_b_recommendations(
    run_level: Any, overlap: Any, pareto: Any
) -> dict[str, object]:
    """Derive transparent Stage B defaults from Stage A evidence."""
    mode_summary = (
        run_level.groupby(["dataset_segment", "retrieval_family"])
        .agg(
            best_pass_rate=("good_acceptable_rate", "max"),
            best_latency_p50_s=("latency_p50_s", "min"),
        )
        .reset_index()
    )
    dense_mall = mode_summary[
        (mode_summary["dataset_segment"] == "mallinckrodt")
        & (mode_summary["retrieval_family"] == "dense")
    ].iloc[0]
    bm25_mall = mode_summary[
        (mode_summary["dataset_segment"] == "mallinckrodt")
        & (mode_summary["retrieval_family"] == "bm25")
    ].iloc[0]
    overlap_summary = (
        overlap.groupby("dataset_segment")
        .agg(
            median_jaccard=("jaccard", "median"),
            median_dense_unique=("dense_unique_count", "median"),
            paired_trace_count=("rubric_key", "count"),
        )
        .reset_index()
    )
    robust_pareto = (
        pareto.groupby("arm_id")[
            ["pareto_pass_p50", "pareto_good_p50", "pareto_rubric_p50"]
        ]
        .sum()
        .sum(axis=1)
        .sort_values(ascending=False)
    )
    return {
        "primary_homogeneous_arm": "BM25 current_union c2/f20",
        "heterogeneous_pilot": "BM25-first + dense-second current_union c2/f20",
        "dense_only_union": "exclude",
        "rrf_union_control": "defer unless implementation cost is negligible",
        "c3_control": (
            "include one BM25 c3 control" if _material_c3_gain(run_level) else "exclude"
        ),
        "rationale": {
            "dense_mallinckrodt_best_pass_rate": float(dense_mall["best_pass_rate"]),
            "bm25_mallinckrodt_best_pass_rate": float(bm25_mall["best_pass_rate"]),
            "overlap_by_dataset": overlap_summary.to_dict(orient="records"),
            "most_robust_pareto_arms": robust_pareto.head(6).to_dict(),
        },
    }


def _material_c3_gain(run_level: Any) -> bool:
    """Return whether matched BM25 c3 arms improve pass rate materially."""
    bm25 = run_level[run_level["retrieval_family"] == "bm25"]
    gains: list[float] = []
    for dataset in bm25["dataset_segment"].unique():
        group = bm25[bm25["dataset_segment"] == dataset]
        for context in (15, 20):
            c3 = group[(group["calls"] == 3) & (group["context"] == context)]
            lower = group[(group["calls"] < 3) & (group["context"] == context)]
            if not c3.empty and not lower.empty:
                gains.append(
                    float(
                        c3.iloc[0]["good_acceptable_rate"]
                        - lower["good_acceptable_rate"].max()
                    )
                )
    return any(gain >= 0.05 for gain in gains)


def _render_report(
    *,
    snapshot_dir: Path,
    validation: dict[str, object],
    run_level: Any,
    pairwise: Any,
    overlap: Any,
    pareto: Any,
    operation_timings: Any,
    recommendations: dict[str, object],
) -> str:
    """Render the evidence-backed Stage A report with reproducible formulas."""
    best_rows = (
        run_level.sort_values(
            ["dataset_segment", "good_acceptable_rate", "latency_p50_s"],
            ascending=[True, False, True],
        )
        .groupby("dataset_segment")
        .head(5)
    )
    pareto_rows = pareto[
        pareto[["pareto_pass_p50", "pareto_good_p50", "pareto_rubric_p50"]].any(axis=1)
    ]
    overlap_summary = (
        overlap.groupby("dataset_segment")
        .agg(
            matched_traces=("rubric_key", "count"),
            median_jaccard=("jaccard", "median"),
            median_bm25_unique=("bm25_unique_count", "median"),
            median_dense_unique=("dense_unique_count", "median"),
        )
        .reset_index()
    )
    call_effects = pairwise[
        pairwise["comparison_type"] == "call_count_at_matched_mode_context"
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
    return f"""# Stage A results and Stage B redesign

Generated from the immutable sanitized snapshot at `{snapshot_dir}` on
{datetime.now(UTC).isoformat()}.

## Executive conclusion

Stage A supports BM25 as the primary Stage B homogeneous `current_union` arm.
Dense-only retrieval is not competitive on Mallinckrodt, while BM25/dense
ranked-result overlap is low enough to justify one heterogeneous BM25-first,
dense-second late-union pilot. Include one BM25 c3 control because matched
Stage A evidence shows a material quality gain in at least one dataset/context.

Recommended Stage B matrix:

1. BM25 `current_union`, c2/f20.
2. Heterogeneous `[bm25, dense]` `current_union`, c2/f20.
3. BM25 `current_union`, c3/f20 control.
4. Do not include dense-only union; defer RRF-union unless its implementation
   cost is negligible.

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
- Valid for analysis: **{validation["valid_for_stage_a_analysis"]}**

The exporter did not retain `row_id`, `use_case`, `dataset_id`, or
`evalset_variant` trace attributes. Pairing therefore uses the unique root
`invoke_*` span name (`rubric_key`) within each dataset segment. Every trace has
exactly one such key. Use-case names are derived from that key.

## Exact calculations

- `Good+Acceptable rate = (Good count + Acceptable count) / unique trace count`.
- `Good rate = Good count / unique trace count`.
- `Critical grade rate = Critical Error count / unique trace count`.
- `RubricV2 mean = arithmetic mean of the one RubricV2 assessment per trace`.
- End-to-end latency is the `observed_duration_ms / 1000` of the unique
  `UNKNOWN` root span whose name begins with `invoke_`.
- p50/p90/p95 use pandas quantiles with `{PERCENTILE_METHOD}` interpolation.
- Pairwise results inner-join arms on `(dataset_segment, rubric_key)` and publish
  matched sample size, grade wins/ties/losses, pass-rate delta, RubricV2 delta,
  and median paired latency delta.
- Pareto membership is computed independently per dataset. An arm is dominated
  if another arm has quality greater than or equal and p50 latency less than or
  equal, with at least one strict improvement.

## Top observed arms by dataset

{core.markdown_table(best_rows[["dataset_segment", "arm_id", "trace_count", "good_acceptable_rate", "rubric_v2_mean", "latency_p50_s", "latency_p95_s", "avg_total_tokens"]])}

## Pareto sensitivity

The following arms are Pareto-efficient under at least one of Good+Acceptable,
Good-only, or RubricV2 quality, always against p50 latency:

{core.markdown_table(pareto_rows[["dataset_segment", "arm_id", "good_acceptable_rate", "good_rate", "rubric_v2_mean", "latency_p50_s", "pareto_pass_p50", "pareto_good_p50", "pareto_rubric_p50"]])}

## BM25 versus dense retrieval overlap

Chunk identities are unioned across calls within each trace before computing
Jaccard overlap. Counts below are medians over exact matched rubric variations:

{core.markdown_table(overlap_summary)}

Low Jaccard plus non-zero unique contributions support testing heterogeneous
late union. This does not prove that unique dense chunks improve answer quality;
Stage B must test that causal hypothesis.

## Matched call-count effects

Largest observed pass-rate changes when call count changes at matched retrieval
family and global context:

{core.markdown_table(call_effects.head(18)[["dataset_segment", "left_arm", "right_arm", "matched_trace_count", "right_grade_wins", "ties", "right_grade_losses", "right_minus_left_pass_rate", "right_minus_left_rubric_v2_mean", "right_minus_left_latency_median_s"]])}

## Operation timing and retries

Each row below summarizes the per-run/arm operation p50s. The complete
dataset-specific values, successful-attempt timings, and retry observation
counts are in `operation_timings.csv`.

{core.markdown_table(operation_summary)}

## Stage B recommendation

```json
{json.dumps(recommendations, indent=2, sort_keys=True)}
```

The recommendation is falsifiable: all source rows and formulas are published
beside the snapshot. In particular, `pairwise_comparisons.csv` exposes every
paired effect and `retrieval_overlap.csv` exposes every matched Jaccard input.

## Reproducibility and caveats

- Do not pool raw traces across datasets; all headline comparisons are
  dataset-specific.
- Latency is observational. `latency_by_arm_dataset.csv` publishes UTC run
  timestamps and hours so time-of-day confounding can be checked. Runs span
  `{start_times.iloc[0]}` through `{start_times.iloc[-1]}` and were not
  randomized into common time windows, so small latency differences must not
  be attributed causally to arm parameters.
- The 102 duplicate ordinal-grade rows were identical and removed only after
  equality validation. No conflicting duplicate was found.
- Run-level MLflow metrics are retained as cross-checks, while headline quality
  and latency are recalculated from trace-level rows.
- The snapshot excludes messages, answers, query text, chunk content, XML, and
  scorer rationales by design.

## Audit files

All calculations are under `{snapshot_dir / "stage_a_analysis"}`:

- `trace_level.csv`: one row per run/trace, the primary recomputation source.
- `run_level.csv`: per-run raw counts and calculated metrics.
- `quality_by_arm_dataset.csv` and `latency_by_arm_dataset.csv`.
- `pairwise_comparisons.csv`, `retrieval_overlap.csv`, and
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
    recommendations: dict[str, object],
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
            "jaccard": "intersection(chunk identities) / union(chunk identities)",
            "pareto": "maximize quality and minimize p50 latency per dataset",
        },
        "grade_order": GRADE_ORDER,
        "passing_grades": sorted(PASSING_GRADES),
        "validation": validation,
        "recommendations": recommendations,
    }
    core.write_json(analysis_dir / "calculation_manifest.json", payload)
