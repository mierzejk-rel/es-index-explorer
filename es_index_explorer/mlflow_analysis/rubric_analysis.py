"""Join sanitized MLflow outcomes to versioned Air Assist rubric specifications."""

import hashlib
import json
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from es_index_explorer.mlflow_analysis.snapshot import decode_quality_value


def analyze_rubrics(
    *,
    snapshot_dir: Path,
    rubric_root: Path,
    task_path: Path,
) -> Path:
    """Create validated rubric, question, expectation, and use-case analyses.

    Parameters
    ----------
    snapshot_dir : Path
        Completed enriched MLflow snapshot directory.
    rubric_root : Path
        Local root containing Air Assist ``*.rubric.toml`` source files.
    task_path : Path
        Air Assist task TOML defining the use-case taxonomy.

    Returns
    -------
    Path
        Directory containing auditable joined analysis tables and discrepancies.
    """
    pandas = _load_pandas()
    snapshot_dir = snapshot_dir.expanduser().resolve()
    rubric_root = rubric_root.expanduser().resolve()
    task_path = task_path.expanduser().resolve()
    analysis_dir = snapshot_dir / "rubric_analysis"
    analysis_dir.mkdir(exist_ok=True)
    source_manifest = _read_snapshot_manifest(snapshot_dir)

    catalogue, expectations, source_discrepancies = _build_catalogue(
        pandas=pandas,
        rubric_root=rubric_root,
        task_path=task_path,
    )
    invocations = pandas.read_parquet(snapshot_dir / "trace_invocations.parquet")
    run_rubrics = pandas.read_parquet(snapshot_dir / "run_rubrics.parquet")
    criteria = pandas.read_parquet(snapshot_dir / "trace_criteria.parquet")
    quality = pandas.read_parquet(snapshot_dir / "trace_quality.parquet")

    joins, join_discrepancies = _join_invocations(
        pandas=pandas,
        invocations=invocations,
        run_rubrics=run_rubrics,
        catalogue=catalogue,
    )
    discrepancies = pandas.concat(
        [source_discrepancies, join_discrepancies],
        ignore_index=True,
    )
    eligible = joins[joins["join_status"] == "eligible"].copy()
    outcomes = _build_trace_outcomes(
        quality=quality,
        eligible=eligible,
    )
    assessments = _build_trace_assessments(
        quality=quality,
        eligible=eligible,
    )
    criteria_outcomes = _build_criteria_outcomes(
        pandas=pandas,
        criteria=criteria,
        expectations=expectations,
        eligible=eligible,
    )

    catalogue.to_csv(analysis_dir / "rubric_catalogue.csv", index=False)
    expectations.to_csv(analysis_dir / "rubric_expectations.csv", index=False)
    joins.to_csv(analysis_dir / "trace_rubric_join.csv", index=False)
    discrepancies.to_csv(analysis_dir / "rubric_join_discrepancies.csv", index=False)
    outcomes.to_csv(analysis_dir / "trace_rubric_outcomes.csv", index=False)
    assessments.to_csv(analysis_dir / "trace_rubric_assessments.csv", index=False)
    criteria_outcomes.to_csv(
        analysis_dir / "trace_expectation_outcomes.csv", index=False
    )
    _summarize_outcomes(pandas, outcomes, analysis_dir)
    _summarize_criteria(pandas, criteria_outcomes, analysis_dir)

    coverage = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_dir": str(snapshot_dir),
        "rubric_root": str(rubric_root),
        "task_path": str(task_path),
        "catalogue_rubric_count": len(catalogue),
        "catalogue_expectation_count": len(expectations),
        "invocation_count": len(invocations),
        "eligible_invocation_count": len(eligible),
        "excluded_invocation_count": int(len(invocations) - len(eligible)),
        "discrepancy_count": len(discrepancies),
        "discrepancy_counts": (
            discrepancies["discrepancy_type"].value_counts().to_dict()
            if not discrepancies.empty
            else {}
        ),
        "catalogue_sha256": _dataframe_sha256(catalogue),
        "source_snapshot_manifest": source_manifest,
    }
    (analysis_dir / "rubric_analysis_coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return analysis_dir


def _build_catalogue(
    *,
    pandas: Any,
    rubric_root: Path,
    task_path: Path,
) -> tuple[Any, Any, Any]:
    """Parse source rubrics while excluding legacy weight/category fields."""
    use_case_taxonomy = _load_use_case_taxonomy(task_path)
    rubric_rows: list[dict[str, object]] = []
    expectation_rows: list[dict[str, object]] = []
    discrepancy_rows: list[dict[str, object]] = []
    for path in sorted(rubric_root.rglob("*.rubric.toml")):
        relative_path = path.relative_to(rubric_root)
        if "deprecated" in relative_path.parts:
            continue
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            discrepancy_rows.append(
                _discrepancy(
                    "catalogue_parse_error",
                    detail=f"{type(exc).__name__}",
                    source_path=str(relative_path),
                )
            )
            continue
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            discrepancy_rows.append(
                _discrepancy(
                    "catalogue_missing_meta",
                    detail="Rubric has no [meta] table.",
                    source_path=str(relative_path),
                )
            )
            continue
        question = _rubric_question(payload, meta)
        if question is None:
            discrepancy_rows.append(
                _discrepancy(
                    "catalogue_missing_question",
                    detail="No meta.question or user input message.",
                    source_path=str(relative_path),
                )
            )
            continue
        use_cases = _use_cases(meta.get("use_case"))
        content_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        catalogue_id = hashlib.sha256(
            f"{relative_path}:{content_sha256}".encode()
        ).hexdigest()
        schema_version = str(meta.get("schema_version", ""))
        expectations_raw = payload.get("expectations")
        if not isinstance(expectations_raw, list):
            expectations_raw = []
        material_values = [
            item.get("material", True)
            for item in expectations_raw
            if isinstance(item, dict)
        ]
        rubric_rows.append(
            {
                "catalogue_id": catalogue_id,
                "source_path": str(relative_path),
                "filename": path.name,
                "content_sha256": content_sha256,
                "schema_version": schema_version,
                "dataset_id": _scalar(meta.get("dataset_id")),
                "task_name": _scalar(meta.get("task_name")),
                "use_cases": json.dumps(use_cases, sort_keys=True),
                "use_case_descriptions": json.dumps(
                    {
                        use_case: use_case_taxonomy[use_case]
                        for use_case in use_cases
                        if use_case in use_case_taxonomy
                    },
                    sort_keys=True,
                ),
                "question": question,
                "expectation_count": len(expectations_raw),
                "material_is_uniform": len(set(material_values)) <= 1,
                "author": _scalar(meta.get("author")),
                "version": _scalar(meta.get("version")),
                "comment": _scalar(meta.get("comment")),
            }
        )
        for index, expectation in enumerate(expectations_raw):
            if not isinstance(expectation, dict):
                continue
            name = expectation.get("name")
            description = expectation.get("description")
            if not isinstance(name, str) or not isinstance(description, str):
                discrepancy_rows.append(
                    _discrepancy(
                        "catalogue_invalid_expectation",
                        detail="Expectation is missing name or description.",
                        source_path=str(relative_path),
                    )
                )
                continue
            expectation_rows.append(
                {
                    "catalogue_id": catalogue_id,
                    "source_path": str(relative_path),
                    "expectation_name": name,
                    "description": description,
                    "material": bool(expectation.get("material", True)),
                    "document_ids": json.dumps(
                        expectation.get("document_ids", []),
                        sort_keys=True,
                    ),
                    "positive_indicators": json.dumps(
                        expectation.get("positive_indicators", []),
                        sort_keys=True,
                    ),
                    "negative_indicators": json.dumps(
                        expectation.get("negative_indicators", []),
                        sort_keys=True,
                    ),
                    "detection_hints": json.dumps(
                        expectation.get("detection_hints", []),
                        sort_keys=True,
                    ),
                    "ordinal_index": index,
                }
            )
        for use_case in use_cases:
            if use_case not in use_case_taxonomy:
                discrepancy_rows.append(
                    _discrepancy(
                        "catalogue_unknown_use_case",
                        detail=use_case,
                        source_path=str(relative_path),
                    )
                )
    return (
        pandas.DataFrame(rubric_rows),
        pandas.DataFrame(expectation_rows),
        pandas.DataFrame(discrepancy_rows),
    )


def _join_invocations(
    *,
    pandas: Any,
    invocations: Any,
    run_rubrics: Any,
    catalogue: Any,
) -> tuple[Any, Any]:
    """Resolve every invocation to one source rubric or emit a discrepancy."""
    catalogue_by_path = {
        row["source_path"]: row for row in catalogue.to_dict(orient="records")
    }
    roster = {
        (
            str(row["run_id"]),
            _to_int(row["rubric_index"]),
            _to_int(row.get("variant_index")),
        ): row
        for row in run_rubrics.to_dict(orient="records")
        if row.get("artifact_status") == "ok"
    }
    join_rows: list[dict[str, object]] = []
    discrepancy_rows: list[dict[str, object]] = []
    for invocation in invocations.to_dict(orient="records"):
        direct_path = _scalar(invocation.get("rubric_file_path"))
        direct_source_path, direct_resolution = _resolve_source_path(
            direct_path,
            catalogue_by_path,
        )
        rubric_index = _to_int(invocation.get("rubric_index"))
        variant_index = _to_int(invocation.get("variant_index"))
        if rubric_index is None or variant_index is None:
            discrepancy_rows.append(
                _discrepancy(
                    "unparsable_span_identity",
                    detail="Rubric key does not encode rubric and variant indices.",
                    run_id=invocation["run_id"],
                    trace_id=invocation["trace_id"],
                    rubric_key=invocation.get("rubric_key"),
                )
            )
        roster_row = (
            roster.get((str(invocation["run_id"]), rubric_index, variant_index))
            if rubric_index is not None and variant_index is not None
            else None
        )
        if roster_row is None:
            discrepancy_rows.append(
                _discrepancy(
                    "missing_run_rubric_artifact",
                    detail="No matching run rubric roster entry.",
                    run_id=invocation["run_id"],
                    trace_id=invocation["trace_id"],
                    rubric_key=invocation.get("rubric_key"),
                )
            )
        roster_path = (
            _scalar(roster_row.get("rubric_file_path")) if roster_row else None
        )
        roster_source_path, roster_resolution = _resolve_source_path(
            roster_path,
            catalogue_by_path,
        )
        if direct_source_path is not None:
            source_path = direct_source_path
            join_method = "direct_rubric_v2_file_path"
            if roster_source_path is not None and roster_source_path != source_path:
                discrepancy_rows.append(
                    _discrepancy(
                        "identity_mismatch",
                        detail="RubricV2 file path and run artifact resolve to different sources.",
                        run_id=invocation["run_id"],
                        trace_id=invocation["trace_id"],
                        rubric_key=invocation.get("rubric_key"),
                        source_path=source_path,
                    )
                )
                join_rows.append(
                    {
                        **invocation,
                        "join_status": "excluded",
                        "join_method": join_method,
                        "catalogue_id": None,
                    }
                )
                continue
        else:
            source_path = roster_source_path
            join_method = "positional_run_artifact_fallback"
            if source_path is not None:
                discrepancy_rows.append(
                    _discrepancy(
                        "positional_fallback",
                        detail="Resolved through run rubric index and variant position.",
                        run_id=invocation["run_id"],
                        trace_id=invocation["trace_id"],
                        rubric_key=invocation.get("rubric_key"),
                        source_path=source_path,
                    )
                )
        if source_path is None:
            discrepancy_type = (
                "path_ambiguity"
                if "ambiguous" in {direct_resolution, roster_resolution}
                else "source_miss"
            )
            discrepancy_rows.append(
                _discrepancy(
                    discrepancy_type,
                    detail="No unique catalogue source path.",
                    run_id=invocation["run_id"],
                    trace_id=invocation["trace_id"],
                    rubric_key=invocation.get("rubric_key"),
                )
            )
            join_rows.append(
                {
                    **invocation,
                    "join_status": "excluded",
                    "join_method": join_method,
                    "catalogue_id": None,
                }
            )
            continue
        rubric = catalogue_by_path[source_path]
        comparison_questions = (
            invocation.get("question"),
            rubric.get("question"),
        )
        if roster_row is not None:
            comparison_questions += (roster_row.get("question"),)
        if not all(isinstance(value, str) for value in comparison_questions):
            discrepancy_rows.append(
                _discrepancy(
                    "missing_question_identity",
                    detail="Invocation, artifact, or TOML question is absent.",
                    run_id=invocation["run_id"],
                    trace_id=invocation["trace_id"],
                    rubric_key=invocation.get("rubric_key"),
                    source_path=source_path,
                )
            )
            join_rows.append(
                {
                    **invocation,
                    "join_status": "excluded",
                    "join_method": join_method,
                    "catalogue_id": rubric["catalogue_id"],
                }
            )
            continue
        if len({_normalize_text(value) for value in comparison_questions}) > 1:
            discrepancy_rows.append(
                _discrepancy(
                    "question_mismatch",
                    detail="Invocation, run artifact, and TOML questions differ.",
                    run_id=invocation["run_id"],
                    trace_id=invocation["trace_id"],
                    rubric_key=invocation.get("rubric_key"),
                    source_path=source_path,
                )
            )
            join_rows.append(
                {
                    **invocation,
                    "join_status": "excluded",
                    "join_method": join_method,
                    "catalogue_id": rubric["catalogue_id"],
                }
            )
            continue
        dataset_values = [
            _normalize_dataset(value)
            for value in (
                invocation.get("dataset_id"),
                rubric.get("dataset_id"),
            )
            if _scalar(value)
        ]
        if roster_row is not None and _scalar(roster_row.get("dataset_id")):
            dataset_values.append(_normalize_dataset(roster_row["dataset_id"]))
        if len(set(dataset_values)) > 1:
            discrepancy_rows.append(
                _discrepancy(
                    "dataset_mismatch",
                    detail="Invocation, run artifact, and TOML datasets differ.",
                    run_id=invocation["run_id"],
                    trace_id=invocation["trace_id"],
                    rubric_key=invocation.get("rubric_key"),
                    source_path=source_path,
                )
            )
            join_rows.append(
                {
                    **invocation,
                    "join_status": "excluded",
                    "join_method": join_method,
                    "catalogue_id": rubric["catalogue_id"],
                }
            )
            continue
        join_rows.append(
            {
                **invocation,
                "join_status": "eligible",
                "join_method": join_method,
                "catalogue_id": rubric["catalogue_id"],
                "source_path": source_path,
                "canonical_question": rubric["question"],
                "use_cases": rubric["use_cases"],
                "use_case_descriptions": rubric["use_case_descriptions"],
            }
        )
    return pandas.DataFrame(join_rows), pandas.DataFrame(discrepancy_rows)


def _build_trace_outcomes(*, quality: Any, eligible: Any) -> Any:
    """Join typed assessment values to eligible rubric traces."""
    quality = quality.copy()
    quality["assessment_value"] = quality.apply(decode_quality_value, axis=1)
    rubric_scores = quality[quality["assessment_name"] == "RubricV2"][
        [
            "run_id",
            "trace_id",
            "assessment_value",
            "ordinal_grade",
            "detected_error_modes",
        ]
    ].rename(columns={"assessment_value": "rubric_v2_score"})
    return eligible.merge(
        rubric_scores,
        on=["run_id", "trace_id"],
        how="left",
        validate="one_to_one",
    )


def _build_trace_assessments(*, quality: Any, eligible: Any) -> Any:
    """Join every raw typed assessment value to eligible rubric traces."""
    assessments = quality.copy()
    assessments["assessment_value"] = assessments.apply(decode_quality_value, axis=1)
    return assessments.merge(
        eligible[
            [
                "run_id",
                "trace_id",
                "catalogue_id",
                "source_path",
                "canonical_question",
                "use_cases",
            ]
        ],
        on=["run_id", "trace_id"],
        how="inner",
        validate="many_to_one",
    )


def _build_criteria_outcomes(
    *, pandas: Any, criteria: Any, expectations: Any, eligible: Any
) -> Any:
    """Join per-expectation criterion states to eligible traces and expectations."""
    expected = expectations[
        ["catalogue_id", "expectation_name", "description", "material"]
    ]
    trace_catalogue = eligible[["run_id", "trace_id", "catalogue_id"]]
    return criteria.merge(
        trace_catalogue, on=["run_id", "trace_id"], how="inner"
    ).merge(expected, on=["catalogue_id", "expectation_name"], how="left")


def _summarize_outcomes(pandas: Any, outcomes: Any, analysis_dir: Path) -> None:
    """Write rubric, question, use-case, and error-mode summaries."""
    groupings = {
        "rubric_summary.csv": ["catalogue_id", "source_path"],
        "question_summary.csv": ["catalogue_id", "canonical_question"],
    }
    for filename, columns in groupings.items():
        summary = (
            outcomes.groupby(columns, dropna=False)
            .agg(
                trace_count=("trace_id", "count"),
                rubric_v2_mean=("rubric_v2_score", "mean"),
                good_count=("ordinal_grade", lambda values: (values == "Good").sum()),
                acceptable_count=(
                    "ordinal_grade",
                    lambda values: (values == "Acceptable").sum(),
                ),
                partial_count=(
                    "ordinal_grade",
                    lambda values: (values == "Partial").sum(),
                ),
                poor_count=("ordinal_grade", lambda values: (values == "Poor").sum()),
                critical_count=(
                    "ordinal_grade",
                    lambda values: (values == "Critical Error").sum(),
                ),
            )
            .reset_index()
        )
        summary.to_csv(analysis_dir / filename, index=False)
    exploded = outcomes.assign(
        use_case_key=outcomes["use_cases"].map(_json_list)
    ).explode("use_case_key")
    exploded["use_case_label"] = exploded["use_case_key"].map(_display_label)
    exploded["use_case_description"] = exploded.apply(
        lambda row: _json_mapping(row["use_case_descriptions"]).get(
            row["use_case_key"]
        ),
        axis=1,
    )
    (
        exploded.groupby(
            ["use_case_key", "use_case_label", "use_case_description"],
            dropna=False,
        )
        .agg(
            trace_count=("trace_id", "count"),
            rubric_v2_mean=("rubric_v2_score", "mean"),
        )
        .reset_index()
        .to_csv(analysis_dir / "use_case_summary.csv", index=False)
    )
    error_rows: list[dict[str, object]] = []
    for row in outcomes.to_dict(orient="records"):
        for error_mode in _json_list(row.get("detected_error_modes")):
            if isinstance(error_mode, dict) and error_mode.get("detected"):
                error_rows.append(
                    {
                        "catalogue_id": row["catalogue_id"],
                        "error_mode": error_mode.get("name"),
                        "run_id": row["run_id"],
                        "trace_id": row["trace_id"],
                    }
                )
    error_outcomes = pandas.DataFrame(error_rows)
    error_outcomes.to_csv(analysis_dir / "error_mode_outcomes.csv", index=False)
    if error_outcomes.empty:
        pandas.DataFrame(
            columns=["catalogue_id", "error_mode", "count", "rate"]
        ).to_csv(analysis_dir / "error_mode_summary.csv", index=False)
        return
    error_summary = (
        error_outcomes.groupby(["catalogue_id", "error_mode"])
        .size()
        .reset_index(name="count")
    )
    trace_counts = outcomes.groupby("catalogue_id")["trace_id"].nunique()
    error_summary["rate"] = error_summary.apply(
        lambda row: row["count"] / trace_counts[row["catalogue_id"]],
        axis=1,
    )
    error_summary.to_csv(analysis_dir / "error_mode_summary.csv", index=False)


def _summarize_criteria(
    pandas: Any, criteria_outcomes: Any, analysis_dir: Path
) -> None:
    """Write expectation state counts and rates."""
    if criteria_outcomes.empty:
        pandas.DataFrame(
            columns=["catalogue_id", "expectation_name", "state", "count", "rate"]
        ).to_csv(analysis_dir / "expectation_summary.csv", index=False)
        return
    counts = (
        criteria_outcomes.groupby(
            ["catalogue_id", "expectation_name", "description", "state"]
        )
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(["catalogue_id", "expectation_name"])["count"].transform(
        "sum"
    )
    counts["rate"] = counts["count"] / totals
    counts.to_csv(analysis_dir / "expectation_summary.csv", index=False)


def _load_use_case_taxonomy(task_path: Path) -> dict[str, str]:
    """Load task use-case descriptions keyed by technical name."""
    payload = tomllib.loads(task_path.read_text(encoding="utf-8"))
    specification = payload.get("specification", {})
    use_cases = (
        specification.get("use_cases", []) if isinstance(specification, dict) else []
    )
    return {
        item["name"]: item.get("description", "")
        for item in use_cases
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def _read_snapshot_manifest(snapshot_dir: Path) -> dict[str, object]:
    """Load the full source snapshot manifest for coverage provenance."""
    path = snapshot_dir / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Snapshot manifest at {path} is not an object.")
    return payload


def _rubric_question(payload: dict[str, object], meta: dict[str, object]) -> str | None:
    """Return meta question or first user message from a rubric TOML."""
    question = meta.get("question")
    if isinstance(question, str):
        return question
    inputs = payload.get("input")
    input_rows = inputs if isinstance(inputs, list) else [inputs]
    for input_row in input_rows:
        if not isinstance(input_row, dict):
            continue
        question = _extract_question(input_row)
        if question:
            return question
    return None


def _extract_question(value: object) -> str | None:
    """Return first user message content from a structured input object."""
    if not isinstance(value, dict):
        return None
    messages = value.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return (
                message.get("content")
                if isinstance(message.get("content"), str)
                else None
            )
    return None


def _use_cases(value: object) -> list[str]:
    """Normalize a scalar/list use-case value to technical keys."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _resolve_source_path(
    path: str | None,
    catalogue_by_path: dict[str, dict[str, object]],
) -> tuple[str | None, str]:
    """Resolve a logged source path to one local rubric-root relative path."""
    if not path:
        return None, "missing"
    path_parts = Path(path).parts
    for index, part in enumerate(path_parts):
        if part == "rubric_data":
            candidate = str(Path(*path_parts[index + 1 :]))
            if candidate in catalogue_by_path:
                return candidate, "exact"
    matches = [
        relative
        for relative in catalogue_by_path
        if relative.endswith(path) or Path(relative).name == Path(path).name
    ]
    if len(matches) == 1:
        return matches[0], "suffix"
    return None, "ambiguous" if matches else "missing"


def _normalize_text(value: str) -> str:
    """Normalize question text for source cross-checking."""
    return " ".join(value.casefold().split())


def _normalize_dataset(value: object) -> str:
    """Normalize dataset aliases for conservative equality checks."""
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _scalar(value: object) -> str | None:
    """Convert scalar metadata to text without serializing nested objects."""
    return value if isinstance(value, str) else None


def _to_int(value: object) -> int | None:
    """Coerce a numeric scalar to an integer."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _json_list(value: object) -> list[object]:
    """Decode a JSON list or return an empty list."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _json_mapping(value: object) -> dict[str, str]:
    """Decode a JSON object containing string values."""
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return (
        {key: item for key, item in parsed.items() if isinstance(item, str)}
        if isinstance(parsed, dict)
        else {}
    )


def _display_label(value: object) -> str | None:
    """Return a deterministic human-readable use-case label."""
    return value.replace("_", " ").title() if isinstance(value, str) else None


def _discrepancy(
    discrepancy_type: str, *, detail: str, **fields: object
) -> dict[str, object]:
    """Build one audit-friendly discrepancy row."""
    return {"discrepancy_type": discrepancy_type, "detail": detail, **fields}


def _dataframe_sha256(dataframe: Any) -> str:
    """Return a stable hash for an analysis input dataframe."""
    payload = json.dumps(
        dataframe.fillna("").to_dict(orient="records"),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_pandas() -> Any:
    """Load pandas only when rubric analysis is executed."""
    import pandas

    return pandas
