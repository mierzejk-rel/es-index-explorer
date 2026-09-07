"""Catalogue-to-snapshot join and structural verification."""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from es_index_explorer.mlflow_analysis.snapshot import (
    _EXPERIMENT_COLUMNS,
    _METRIC_COLUMNS,
    _RETRIEVAL_COLUMNS,
    _RUN_COLUMNS,
    _RUN_RUBRIC_COLUMNS,
    _SNAPSHOT_PARQUET_FILES,
    _SPAN_TIMING_COLUMNS,
    _TRACE_CRITERIA_COLUMNS,
    _TRACE_FAILURE_COLUMNS,
    _TRACE_INVOCATION_COLUMNS,
    _TRACE_QUALITY_COLUMNS,
    decode_quality_value,
)
from es_index_explorer.question_analysis.arms import parse_arm
from es_index_explorer.question_analysis.catalogue import (
    Catalogue,
    build_catalogue,
    stable_id,
)
from es_index_explorer.question_analysis.contracts import (
    TRACE_PFU_TABLE_COLUMNS,
    ArtifactMetadata,
    WorkflowCommand,
)
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    MalformedInputError,
)
from es_index_explorer.question_analysis.grade_oracle import (
    DEFAULT_GRADE_ORACLE,
    DEFAULT_GRADE_ORACLE_PROVENANCE,
    GradeOracle,
)
from es_index_explorer.question_analysis.storage import sha256_file, versioned_frame
from es_index_explorer.question_analysis.workspace import (
    PROJECT_ROOT,
    AnalysisWorkspace,
)

DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "artifacts" / "mlflow" / "simplemode-stage-v3"
DEFAULT_RUBRIC_ROOT = (
    PROJECT_ROOT.parent
    / "r1-evals-new"
    / "src"
    / "r1_evals"
    / "rubrics"
    / "rubric_data"
)
DEFAULT_TASK_PATH = (
    PROJECT_ROOT.parent / "r1-evals-new" / "rubrics" / "tasks" / "air_assist.toml"
)

EXPECTED_INVENTORY = {
    "emc2_set1": {
        "rubrics": 21,
        "variants": 102,
        "expectations": 120,
        "traces": 2856,
    },
    "emc2_set2": {
        "rubrics": 20,
        "variants": 79,
        "expectations": 98,
        "traces": 2212,
    },
    "mallinckrodt": {
        "rubrics": 22,
        "variants": 82,
        "expectations": 96,
        "traces": 2296,
    },
}
SNAPSHOT_COLUMNS = {
    "experiments": _EXPERIMENT_COLUMNS,
    "runs": _RUN_COLUMNS,
    "run_metrics": _METRIC_COLUMNS,
    "trace_quality": _TRACE_QUALITY_COLUMNS,
    "trace_invocations": _TRACE_INVOCATION_COLUMNS,
    "trace_criteria": _TRACE_CRITERIA_COLUMNS,
    "run_rubrics": _RUN_RUBRIC_COLUMNS,
    "trace_failures": _TRACE_FAILURE_COLUMNS,
    "trace_retrieval": _RETRIEVAL_COLUMNS,
    "span_timings": _SPAN_TIMING_COLUMNS,
}

DISCREPANCY_COLUMNS = [
    "discrepancy_id",
    "severity",
    "discrepancy_type",
    "eval_dataset",
    "run_id",
    "trace_id",
    "rubric_id",
    "variant_index",
    "arm_id",
    "expectation_index",
    "expected",
    "actual",
]


@dataclass(frozen=True, slots=True)
class JoinConfig:
    """Input locations for the structural join."""

    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR
    rubric_root: Path = DEFAULT_RUBRIC_ROOT
    task_path: Path = DEFAULT_TASK_PATH
    grade_fixture: Path = DEFAULT_GRADE_ORACLE
    grade_provenance: Path = DEFAULT_GRADE_ORACLE_PROVENANCE


@dataclass(frozen=True, slots=True)
class JoinResult:
    """All Segment 2 tables and verification products."""

    catalogue: Catalogue
    criterion_table: pd.DataFrame
    trace_pfu_table: pd.DataFrame
    discrepancies: pd.DataFrame
    structural_verification: dict[str, object]
    f3_design_rank: dict[str, object]
    report: str


def run_join(
    workspace: AnalysisWorkspace, config: JoinConfig | None = None
) -> tuple[ArtifactMetadata, ...]:
    """Run Segment 2 and persist every load-bearing join artifact."""
    config = _resolved_config(config or JoinConfig())
    _register_inputs(workspace, config)
    result = build_join(config)
    artifacts = (
        workspace.store.write_parquet(
            "tables/rubric_catalogue.parquet",
            _versioned(result.catalogue.rubrics),
            created_by=WorkflowCommand.JOIN,
        ),
        workspace.store.write_parquet(
            "tables/variant_catalogue.parquet",
            _versioned(result.catalogue.variants),
            created_by=WorkflowCommand.JOIN,
        ),
        workspace.store.write_parquet(
            "tables/expectation_catalogue.parquet",
            _versioned(result.catalogue.expectations),
            created_by=WorkflowCommand.JOIN,
        ),
        workspace.store.write_parquet(
            "tables/criterion_table.parquet",
            _versioned(result.criterion_table),
            created_by=WorkflowCommand.JOIN,
        ),
        workspace.store.write_parquet(
            "tables/trace_pfu_table.parquet",
            _versioned(result.trace_pfu_table),
            created_by=WorkflowCommand.JOIN,
        ),
        workspace.store.write_parquet(
            "tables/join_discrepancies.parquet",
            _versioned(result.discrepancies),
            created_by=WorkflowCommand.JOIN,
        ),
        workspace.store.write_json(
            "structural_verification.json",
            result.structural_verification,
            created_by=WorkflowCommand.JOIN,
        ),
        workspace.store.write_json(
            "f3_design_rank.json",
            result.f3_design_rank,
            created_by=WorkflowCommand.JOIN,
        ),
        workspace.store.write_bytes(
            "partial_reports/01-data-contract.md",
            result.report.encode("utf-8"),
            created_by=WorkflowCommand.JOIN,
        ),
    )
    if not bool(result.structural_verification["passed"]):
        failures = result.structural_verification["blocking_failures"]
        raise GateFailureError(f"Structural verification failed: {failures}")
    return artifacts


def build_join(config: JoinConfig | None = None) -> JoinResult:
    """Build and verify the Segment 2 population without persisting it."""
    config = _resolved_config(config or JoinConfig())
    snapshot_manifest = _verify_snapshot(config.snapshot_dir)
    catalogue = build_catalogue(config.rubric_root, config.task_path)
    grade_oracle = GradeOracle.load(config.grade_fixture, config.grade_provenance)
    tables = cast(
        dict[str, pd.DataFrame],
        {
            name.removesuffix(".parquet"): pd.read_parquet(config.snapshot_dir / name)
            for name in _SNAPSHOT_PARQUET_FILES
        },
    )
    _validate_snapshot_columns(tables)

    discrepancies = _catalogue_discrepancies(catalogue)
    runs = _build_runs(tables["runs"])
    trace_identity = _build_trace_identity(
        tables["trace_invocations"],
        tables["run_rubrics"],
        runs,
        catalogue,
        tables["trace_failures"],
        discrepancies,
    )
    criterion_table = _build_criterion_table(
        tables["trace_criteria"],
        trace_identity,
        catalogue,
        discrepancies,
    )
    trace_pfu_table = _build_trace_pfu_table(
        criterion_table,
        trace_identity,
        tables["trace_quality"],
        grade_oracle,
        discrepancies,
    )
    f3_design_rank = _f3_design_rank(criterion_table)
    structural_verification = _structural_verification(
        snapshot_manifest,
        catalogue,
        runs,
        trace_identity,
        criterion_table,
        trace_pfu_table,
        discrepancies,
        f3_design_rank,
    )
    discrepancy_frame = pd.DataFrame(discrepancies, columns=DISCREPANCY_COLUMNS)
    if not discrepancy_frame.empty:
        discrepancy_frame = discrepancy_frame.sort_values(
            ["severity", "discrepancy_type", "eval_dataset", "run_id", "trace_id"],
            kind="stable",
        ).reset_index(drop=True)
    return JoinResult(
        catalogue=_sorted_catalogue(catalogue),
        criterion_table=criterion_table,
        trace_pfu_table=trace_pfu_table,
        discrepancies=discrepancy_frame,
        structural_verification=structural_verification,
        f3_design_rank=f3_design_rank,
        report=_render_report(structural_verification, f3_design_rank),
    )


def _resolved_config(config: JoinConfig) -> JoinConfig:
    return JoinConfig(
        snapshot_dir=config.snapshot_dir.resolve(),
        rubric_root=config.rubric_root.resolve(),
        task_path=config.task_path.resolve(),
        grade_fixture=config.grade_fixture.resolve(),
        grade_provenance=config.grade_provenance.resolve(),
    )


def _register_inputs(workspace: AnalysisWorkspace, config: JoinConfig) -> None:
    workspace.register_input(
        "snapshot/manifest.json", config.snapshot_dir / "manifest.json"
    )
    for filename in _SNAPSHOT_PARQUET_FILES:
        workspace.register_input(f"snapshot/{filename}", config.snapshot_dir / filename)
    catalogue = build_catalogue(config.rubric_root, config.task_path)
    for path in catalogue.input_paths:
        if path == config.task_path:
            name = "rubrics/task.toml"
        else:
            name = f"rubrics/{path.relative_to(config.rubric_root).as_posix()}"
        workspace.register_input(name, path)
    workspace.register_input("grade_oracle/fixture", config.grade_fixture)
    workspace.register_input("grade_oracle/provenance", config.grade_provenance)


def _verify_snapshot(snapshot_dir: Path) -> dict[str, object]:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise MalformedInputError(f"Snapshot manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MalformedInputError(f"Invalid snapshot manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 3:
        raise MalformedInputError(
            "Simple Mode analysis requires snapshot schema version 3"
        )
    file_rows = manifest.get("files")
    if not isinstance(file_rows, list):
        raise MalformedInputError("Snapshot manifest has no file inventory")
    rows_by_name = {
        str(row["name"]): row
        for row in file_rows
        if isinstance(row, dict) and "name" in row
    }
    if set(rows_by_name) != set(_SNAPSHOT_PARQUET_FILES):
        raise MalformedInputError(
            "Snapshot manifest does not contain the ten frozen tables"
        )
    for filename in _SNAPSHOT_PARQUET_FILES:
        path = snapshot_dir / filename
        row = rows_by_name[filename]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise MalformedInputError(f"Snapshot checksum mismatch: {path}")
    return manifest


def _validate_snapshot_columns(tables: Mapping[str, pd.DataFrame]) -> None:
    for name, expected_columns in SNAPSHOT_COLUMNS.items():
        actual_columns = tables[name].columns.tolist()
        if actual_columns != expected_columns:
            raise MalformedInputError(
                f"Snapshot table {name}.parquet schema differs from schema v3: "
                f"expected {expected_columns}, found {actual_columns}"
            )


def _catalogue_discrepancies(catalogue: Catalogue) -> list[dict[str, object]]:
    discrepancies: list[dict[str, object]] = []
    for row in catalogue.rubrics.to_dict(orient="records"):
        duplicate_count = int(row["duplicate_expectation_name_count"])
        if duplicate_count:
            _add_discrepancy(
                discrepancies,
                severity="warning",
                discrepancy_type="duplicate_expectation_name_resolved_by_occurrence",
                eval_dataset=str(row["eval_dataset"]),
                rubric_id=str(row["rubric_id"]),
                expected="unique expectation names",
                actual=f"{duplicate_count} duplicate occurrence(s)",
            )
    normalized = catalogue.expectations[
        catalogue.expectations["expectation_name"]
        != catalogue.expectations["source_expectation_name"]
    ]
    for row in normalized.to_dict(orient="records"):
        _add_discrepancy(
            discrepancies,
            severity="warning",
            discrepancy_type="expectation_name_outer_whitespace_normalized",
            eval_dataset=str(row["eval_dataset"]),
            rubric_id=str(row["rubric_id"]),
            expectation_index=int(row["expectation_index"]),
            expected=str(row["source_expectation_name"]),
            actual=str(row["expectation_name"]),
        )
    return discrepancies


def _build_runs(runs: pd.DataFrame) -> pd.DataFrame:
    _require_unique(runs, ["run_id"], "runs")
    rows: list[dict[str, object]] = []
    for row in runs.to_dict(orient="records"):
        identity = parse_arm(str(row["experiment_name"]))
        rows.append(
            {
                "experiment_id": str(row["experiment_id"]),
                "run_id": str(row["run_id"]),
                "experiment_name": str(row["experiment_name"]),
                **asdict(identity),
            }
        )
    result = pd.DataFrame(rows)
    if result["arm_id"].nunique() != 28:
        raise GateFailureError(
            f"Expected 28 distinct arm IDs, found {result['arm_id'].nunique()}"
        )
    return result


def _build_trace_identity(
    invocations: pd.DataFrame,
    run_rubrics: pd.DataFrame,
    runs: pd.DataFrame,
    catalogue: Catalogue,
    trace_failures: pd.DataFrame,
    discrepancies: list[dict[str, object]],
) -> pd.DataFrame:
    invocation_key = ["run_id", "rubric_index", "variant_index"]
    _require_unique(invocations, ["run_id", "trace_id"], "trace_invocations")
    _require_unique(invocations, invocation_key, "trace_invocations")
    _require_unique(run_rubrics, invocation_key, "run_rubrics")
    if set(map(tuple, invocations[invocation_key].to_numpy())) != set(
        map(tuple, run_rubrics[invocation_key].to_numpy())
    ):
        raise GateFailureError("Invocation and run-rubric roster keys differ")

    roster_columns = [
        *invocation_key,
        "rubric_file_path",
        "question",
        "dataset_id",
        "artifact_sha256",
        "artifact_status",
        "input_variant_count",
    ]
    trace = invocations.merge(
        run_rubrics[roster_columns],
        on=invocation_key,
        how="inner",
        suffixes=("_invocation", "_roster"),
        validate="one_to_one",
    )
    trace.rename(
        columns={
            "rubric_file_path_invocation": "invocation_rubric_file_path",
            "rubric_file_path_roster": "rubric_file_path",
        },
        inplace=True,
    )
    trace = trace.merge(
        runs,
        on=["run_id", "experiment_id"],
        how="inner",
        validate="many_to_one",
    )
    if len(trace) != len(invocations):
        raise GateFailureError("Invocation-to-run join lost rows")
    trace = _attach_catalogue_source_paths(trace, catalogue)

    variants = catalogue.variants[
        [
            "variant_id",
            "rubric_id",
            "rubric_order",
            "eval_dataset",
            "source_path",
            "variant_index",
            "question",
            "variant_count",
            "expectation_count",
        ]
    ].rename(columns={"question": "question_catalogue"})
    trace = trace.merge(
        variants,
        on=["eval_dataset", "source_path", "variant_index"],
        how="left",
        suffixes=("", "_catalogue"),
        validate="many_to_one",
        indicator=True,
    )
    trace["identity_joined"] = trace["_merge"].eq("both")
    trace.drop(columns="_merge", inplace=True)
    trace["question_match"] = trace["question_invocation"].eq(
        trace["question_roster"]
    ) & trace["question_invocation"].eq(trace["question_catalogue"])
    trace["roster_variant_count_match"] = trace["input_variant_count"].eq(
        trace["variant_count"]
    )
    trace["status_eligible"] = trace["trace_status"].eq("TraceStatus.OK") & trace[
        "criteria_parse_status"
    ].eq("parsed")
    failure_keys = set(
        map(tuple, trace_failures[["run_id", "trace_id"]].drop_duplicates().to_numpy())
    )
    trace["failure_free"] = [
        (run_id, trace_id) not in failure_keys
        for run_id, trace_id in zip(trace["run_id"], trace["trace_id"], strict=True)
    ]
    trace["eligible"] = (
        trace["identity_joined"]
        & trace["question_match"]
        & trace["roster_variant_count_match"]
        & trace["status_eligible"]
        & trace["failure_free"]
        & trace["artifact_status"].eq("ok")
    )

    for row in trace[~trace["eligible"]].to_dict(orient="records"):
        failed_checks = [
            name
            for name in (
                "identity_joined",
                "question_match",
                "roster_variant_count_match",
                "status_eligible",
                "failure_free",
            )
            if not bool(row[name])
        ]
        _add_discrepancy(
            discrepancies,
            severity="blocking",
            discrepancy_type="trace_identity_or_status_mismatch",
            eval_dataset=str(row["eval_dataset"]),
            run_id=str(row["run_id"]),
            trace_id=str(row["trace_id"]),
            rubric_id=_nullable_string(row.get("rubric_id")),
            variant_index=int(row["variant_index"]),
            arm_id=str(row["arm_id"]),
            expected="all identity and status checks pass",
            actual=", ".join(failed_checks),
        )
    return trace


def _attach_catalogue_source_paths(
    trace: pd.DataFrame, catalogue: Catalogue
) -> pd.DataFrame:
    source_paths = catalogue.rubrics[
        ["eval_dataset", "rubric_file_name", "source_path"]
    ]
    _require_unique(
        source_paths,
        ["eval_dataset", "rubric_file_name"],
        "catalogue rubric filenames",
    )
    _require_unique(
        source_paths,
        ["eval_dataset", "source_path"],
        "catalogue rubric source paths",
    )
    result = trace.merge(
        source_paths,
        left_on=["eval_dataset", "rubric_file_path"],
        right_on=["eval_dataset", "rubric_file_name"],
        how="left",
        validate="many_to_one",
        indicator="_source_path_merge",
    )
    missing_count = int(result["_source_path_merge"].ne("both").sum())
    if missing_count:
        raise GateFailureError(
            f"Snapshot roster has {missing_count} rubric filename rows without a "
            "unique catalogue source path"
        )
    return result.drop(columns="_source_path_merge")


def _build_criterion_table(
    criteria: pd.DataFrame,
    trace_identity: pd.DataFrame,
    catalogue: Catalogue,
    discrepancies: list[dict[str, object]],
) -> pd.DataFrame:
    criteria = criteria.copy()
    criteria.rename(
        columns={"rubric_file_path": "criteria_rubric_file_path"}, inplace=True
    )
    criteria["expectation_index"] = criteria.groupby(
        ["run_id", "trace_id"], sort=False
    ).cumcount()
    trace_columns = [
        "run_id",
        "trace_id",
        "experiment_id",
        "variant_id",
        "rubric_id",
        "rubric_order",
        "eval_dataset",
        "rubric_file_path",
        "variant_index",
        "arm_id",
        "stage",
        "retrieval_family",
        "retrieval_modes",
        "calls",
        "merge_policy",
        "fetch",
        "global_context",
        "reasoning_effort",
        "expectation_count",
        "eligible",
    ]
    joined = criteria.merge(
        trace_identity[trace_columns],
        on=["run_id", "trace_id", "experiment_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    joined["trace_joined"] = joined["_merge"].eq("both")
    joined.drop(columns="_merge", inplace=True)
    expectation_columns = [
        "expectation_id",
        "rubric_id",
        "expectation_index",
        "expectation_name",
        "description",
        "material",
        "document_ids",
        "expectation_document_count",
    ]
    joined = joined.merge(
        catalogue.expectations[expectation_columns],
        on=["rubric_id", "expectation_index"],
        how="left",
        suffixes=("_observed", "_catalogue"),
        validate="many_to_one",
        indicator=True,
    )
    joined["expectation_joined"] = joined["_merge"].eq("both")
    joined.drop(columns="_merge", inplace=True)
    joined["observed_expectation_name"] = joined[
        "expectation_name_observed"
    ].str.strip()
    joined["expectation_name_match"] = joined["observed_expectation_name"].eq(
        joined["expectation_name_catalogue"]
    )
    joined["material_match"] = joined["material_observed"].eq(
        joined["material_catalogue"]
    )
    whitespace_normalized = joined[
        joined["expectation_name_observed"] != joined["observed_expectation_name"]
    ].drop_duplicates(["rubric_id", "expectation_index", "expectation_name_observed"])
    for row in whitespace_normalized.to_dict(orient="records"):
        _add_discrepancy(
            discrepancies,
            severity="warning",
            discrepancy_type="assessment_expectation_name_outer_whitespace_normalized",
            eval_dataset=_nullable_string(row.get("eval_dataset")),
            rubric_id=_nullable_string(row.get("rubric_id")),
            expectation_index=int(row["expectation_index"]),
            expected=_nullable_string(row.get("expectation_name_catalogue")),
            actual=str(row["expectation_name_observed"]),
        )
    joined["eligible"] = (
        joined["eligible"]
        & joined["trace_joined"]
        & joined["expectation_joined"]
        & joined["expectation_name_match"]
        & joined["material_match"]
    )
    joined["criterion_observation_id"] = [
        stable_id("criterion", trace_id, expectation_id)
        for trace_id, expectation_id in zip(
            joined["trace_id"], joined["expectation_id"], strict=True
        )
    ]

    observed_counts = joined.groupby(["run_id", "trace_id"]).size()
    expected_counts = trace_identity.set_index(["run_id", "trace_id"])[
        "expectation_count"
    ]
    count_match = observed_counts.reindex(expected_counts.index, fill_value=0).eq(
        expected_counts
    )
    mismatched_keys = count_match[~count_match].index
    if len(mismatched_keys):
        trace_key_index = trace_identity.set_index(["run_id", "trace_id"]).index
        trace_identity.loc[trace_key_index.isin(mismatched_keys), "eligible"] = False
        criterion_key_index = pd.MultiIndex.from_frame(joined[["run_id", "trace_id"]])
        joined.loc[criterion_key_index.isin(mismatched_keys), "eligible"] = False
        identity_by_key = trace_identity.set_index(["run_id", "trace_id"])
        for run_id, trace_id in mismatched_keys:
            identity = identity_by_key.loc[(run_id, trace_id)]
            _add_discrepancy(
                discrepancies,
                severity="blocking",
                discrepancy_type="criterion_count_mismatch",
                eval_dataset=str(identity["eval_dataset"]),
                run_id=str(run_id),
                trace_id=str(trace_id),
                rubric_id=str(identity["rubric_id"]),
                variant_index=int(identity["variant_index"]),
                arm_id=str(identity["arm_id"]),
                expected=str(int(expected_counts.loc[(run_id, trace_id)])),
                actual=str(int(observed_counts.get((run_id, trace_id), 0))),
            )
    for row in joined[~joined["eligible"]].to_dict(orient="records"):
        failed_checks = [
            name
            for name in (
                "trace_joined",
                "expectation_joined",
                "expectation_name_match",
                "material_match",
            )
            if not bool(row[name])
        ]
        _add_discrepancy(
            discrepancies,
            severity="blocking",
            discrepancy_type="criterion_identity_mismatch",
            eval_dataset=_nullable_string(row.get("eval_dataset")),
            run_id=str(row["run_id"]),
            trace_id=str(row["trace_id"]),
            rubric_id=_nullable_string(row.get("rubric_id")),
            variant_index=_nullable_int(row.get("variant_index")),
            arm_id=_nullable_string(row.get("arm_id")),
            expectation_index=int(row["expectation_index"]),
            expected=_nullable_string(row.get("expectation_name_catalogue")),
            actual=f"{row.get('expectation_name_observed')}; {failed_checks}",
        )

    columns = [
        "criterion_observation_id",
        "expectation_id",
        "rubric_id",
        "rubric_order",
        "variant_id",
        "eval_dataset",
        "rubric_file_path",
        "variant_index",
        "arm_id",
        "stage",
        "retrieval_family",
        "retrieval_modes",
        "calls",
        "merge_policy",
        "fetch",
        "global_context",
        "reasoning_effort",
        "run_id",
        "trace_id",
        "expectation_index",
        "expectation_name_observed",
        "observed_expectation_name",
        "expectation_name_catalogue",
        "description",
        "document_ids",
        "expectation_document_count",
        "material_catalogue",
        "state",
        "eligible",
        "expectation_name_match",
        "material_match",
    ]
    result = joined[columns].rename(
        columns={
            "expectation_name_observed": "assessment_expectation_name",
            "expectation_name_catalogue": "expectation_name",
            "material_catalogue": "material",
        }
    )
    return result.sort_values(
        [
            "rubric_order",
            "variant_index",
            "arm_id",
            "trace_id",
            "expectation_index",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _build_trace_pfu_table(
    criterion_table: pd.DataFrame,
    trace_identity: pd.DataFrame,
    quality: pd.DataFrame,
    grade_oracle: GradeOracle,
    discrepancies: list[dict[str, object]],
) -> pd.DataFrame:
    state_counts = (
        criterion_table.pivot_table(
            index=["run_id", "trace_id"],
            columns="state",
            values="expectation_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=["PASS", "FAIL", "UNDETERMINED"], fill_value=0)
        .rename(columns={"PASS": "P", "FAIL": "F", "UNDETERMINED": "U"})
        .reset_index()
    )
    identity_columns = [
        "run_id",
        "trace_id",
        "rubric_id",
        "rubric_order",
        "variant_id",
        "eval_dataset",
        "rubric_file_path",
        "variant_index",
        "arm_id",
        "stage",
        "expectation_count",
        "eligible",
    ]
    result = trace_identity[identity_columns].merge(
        state_counts,
        on=["run_id", "trace_id"],
        how="left",
        validate="one_to_one",
    )
    result[["P", "F", "U"]] = result[["P", "F", "U"]].fillna(0).astype(int)
    result["N_r"] = result["P"] + result["F"] + result["U"]
    result["criterion_count_match"] = result["N_r"].eq(result["expectation_count"])
    result["eligible"] = result["eligible"] & result["criterion_count_match"]
    result["rubric_v2"] = (result["P"] + 0.5 * result["U"]) / result["N_r"]

    quality = quality.copy()
    quality["decoded_value"] = [
        decode_quality_value(row) for row in quality.to_dict(orient="records")
    ]
    rubric_quality = quality[quality["assessment_name"].eq("RubricV2")].copy()
    _require_unique(rubric_quality, ["run_id", "trace_id"], "RubricV2 assessments")
    rubric_quality["logged_rubric_v2"] = rubric_quality["decoded_value"].astype(float)
    rubric_quality["detected_error_override"] = [
        _detected_error_override(value)
        for value in rubric_quality["detected_error_modes"]
    ]

    error_quality = quality[quality["assessment_name"].str.startswith("errors_")].copy()
    error_quality["error_truthy"] = [
        _error_truthy(value) for value in error_quality["decoded_value"]
    ]
    error_summary = (
        error_quality.groupby(["run_id", "trace_id"])
        .agg(
            error_scorer_judgement_count=("assessment_name", "nunique"),
            error_scorer_override=("error_truthy", "any"),
        )
        .reset_index()
    )
    result = result.merge(
        rubric_quality[
            [
                "run_id",
                "trace_id",
                "logged_rubric_v2",
                "ordinal_grade",
                "detected_error_override",
            ]
        ],
        on=["run_id", "trace_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        error_summary,
        on=["run_id", "trace_id"],
        how="left",
        validate="one_to_one",
    )
    result.rename(columns={"ordinal_grade": "logged_ordinal_grade"}, inplace=True)
    result["error_scorer_judgement_count"] = (
        result["error_scorer_judgement_count"].fillna(0).astype(int)
    )
    result["error_override"] = result["detected_error_override"].fillna(False) | result[
        "error_scorer_override"
    ].fillna(False)
    result["recomputed_ordinal_grade"] = [
        grade_oracle.lookup(int(p), int(f), int(u), bool(error_override))
        for p, f, u, error_override in zip(
            result["P"],
            result["F"],
            result["U"],
            result["error_override"],
            strict=True,
        )
    ]
    result["expectations_to_next_grade"] = [
        grade_oracle.expectations_to_next_grade(
            int(p), int(f), int(u), bool(error_override)
        )
        for p, f, u, error_override in zip(
            result["P"],
            result["F"],
            result["U"],
            result["error_override"],
            strict=True,
        )
    ]
    result["rubric_v2_match"] = np.isclose(
        result["rubric_v2"],
        result["logged_rubric_v2"],
        rtol=0,
        atol=1e-12,
        equal_nan=False,
    )
    result["ordinal_grade_match"] = result["recomputed_ordinal_grade"].eq(
        result["logged_ordinal_grade"]
    )

    integrity_failures = result[
        ~result["criterion_count_match"]
        | ~result["rubric_v2_match"]
        | ~result["ordinal_grade_match"]
    ]
    for row in integrity_failures.to_dict(orient="records"):
        _add_discrepancy(
            discrepancies,
            severity="blocking",
            discrepancy_type="trace_outcome_integrity_mismatch",
            eval_dataset=str(row["eval_dataset"]),
            run_id=str(row["run_id"]),
            trace_id=str(row["trace_id"]),
            rubric_id=str(row["rubric_id"]),
            variant_index=int(row["variant_index"]),
            arm_id=str(row["arm_id"]),
            expected=(
                f"N={row['expectation_count']}; RubricV2={row['rubric_v2']}; "
                f"grade={row['recomputed_ordinal_grade']}"
            ),
            actual=(
                f"N={row['N_r']}; RubricV2={row['logged_rubric_v2']}; "
                f"grade={row['logged_ordinal_grade']}"
            ),
        )

    return (
        result[list(TRACE_PFU_TABLE_COLUMNS)]
        .sort_values(
            ["rubric_order", "variant_index", "arm_id", "trace_id"], kind="stable"
        )
        .reset_index(drop=True)
    )


def _f3_design_rank(criterion_table: pd.DataFrame) -> dict[str, object]:
    rows = criterion_table[
        criterion_table["eligible"]
        & criterion_table["stage"].eq("A")
        & criterion_table["retrieval_family"].eq("hybrid")
    ].copy()
    rows["expectation_count"] = rows.groupby("rubric_id")["expectation_id"].transform(
        "nunique"
    )
    rows["g"] = rows["global_context"].astype(float)
    rows["c"] = rows["calls"].astype(float)
    rows["expectation_count_x_g"] = rows["expectation_count"] * rows["g"]
    predictor_columns = ["expectation_count", "g", "expectation_count_x_g", "c"]
    predictors = rows[predictor_columns].to_numpy(dtype=float)
    matrix = np.column_stack((np.ones(len(rows)), predictors))
    column_names = ["intercept", *predictor_columns]
    rank = int(np.linalg.matrix_rank(matrix))
    full_rank = rank == matrix.shape[1]
    interaction_column_index = column_names.index("expectation_count_x_g")
    matrix_without_interaction = np.delete(matrix, interaction_column_index, axis=1)
    rank_without_interaction = int(np.linalg.matrix_rank(matrix_without_interaction))
    interaction_identified = rank > rank_without_interaction

    scaled = predictors.copy()
    scaled -= scaled.mean(axis=0)
    standard_deviation = scaled.std(axis=0)
    scaled[:, standard_deviation > 0] /= standard_deviation[standard_deviation > 0]
    scaled_matrix = np.column_stack((np.ones(len(rows)), scaled))
    condition_number = float(np.linalg.cond(scaled_matrix))
    vif = {
        column: _vif(predictors, index)
        for index, column in enumerate(predictor_columns)
    }
    cluster_ids = sorted(rows["arm_id"].unique().tolist())
    return {
        "schema_version": 1,
        "population": "eligible Stage A hybrid criterion observations",
        "row_count": len(rows),
        "row_ids": rows["criterion_observation_id"].tolist(),
        "columns": column_names,
        "rank": rank,
        "column_count": matrix.shape[1],
        "full_rank": full_rank,
        "rank_without_interaction": rank_without_interaction,
        "column_count_without_interaction": matrix_without_interaction.shape[1],
        "interaction_identified": interaction_identified,
        "condition_number_standardized": condition_number,
        "vif": vif,
        "interaction_status": (
            "confirmatory" if interaction_identified else "exploratory"
        ),
        "H_F3": len(cluster_ids),
        "hybrid_cluster_ids": cluster_ids,
    }


def _vif(matrix: np.ndarray, index: int) -> float | None:
    target = matrix[:, index]
    others = np.delete(matrix, index, axis=1)
    design = np.column_stack((np.ones(len(matrix)), others))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    total = target - target.mean()
    denominator = float(total @ total)
    if denominator == 0:
        return None
    r_squared = 1 - float(residual @ residual) / denominator
    return None if r_squared >= 1 else float(1 / (1 - r_squared))


def _structural_verification(
    snapshot_manifest: dict[str, object],
    catalogue: Catalogue,
    runs: pd.DataFrame,
    trace_identity: pd.DataFrame,
    criterion_table: pd.DataFrame,
    trace_pfu_table: pd.DataFrame,
    discrepancies: list[dict[str, object]],
    f3_design_rank: dict[str, object],
) -> dict[str, object]:
    segment_results: dict[str, dict[str, object]] = {}
    blocking_failures: list[str] = []
    for eval_dataset, expected in EXPECTED_INVENTORY.items():
        rubric_rows = catalogue.rubrics[
            catalogue.rubrics["eval_dataset"].eq(eval_dataset)
        ]
        variant_rows = catalogue.variants[
            catalogue.variants["eval_dataset"].eq(eval_dataset)
        ]
        expectation_rows = catalogue.expectations[
            catalogue.expectations["eval_dataset"].eq(eval_dataset)
        ]
        trace_rows = trace_pfu_table[
            trace_pfu_table["eval_dataset"].eq(eval_dataset)
            & trace_pfu_table["eligible"]
        ]
        actual = {
            "rubrics": len(rubric_rows),
            "variants": len(variant_rows),
            "expectations": len(expectation_rows),
            "traces": len(trace_rows),
        }
        passed = actual == expected
        if not passed:
            blocking_failures.append(f"{eval_dataset}_inventory")
        segment_results[eval_dataset] = {
            "expected": expected,
            "actual": actual,
            "passed": passed,
        }

    expected_criterion_count = int(
        (
            catalogue.rubrics["variant_count"]
            * catalogue.rubrics["expectation_count"]
            * 28
        ).sum()
    )
    actual_criterion_count = int(criterion_table["eligible"].sum())
    if expected_criterion_count != actual_criterion_count:
        blocking_failures.append("criterion_observation_count")

    arm_balance_failure_count = _arm_balance_failure_count(trace_pfu_table)
    if arm_balance_failure_count:
        blocking_failures.append("arm_balance")

    materiality = _materiality_verification(catalogue)
    if materiality["passed"] is not True:
        blocking_failures.append("materiality_invariant")

    run_variant_counts = (
        trace_identity[trace_identity["eligible"]]
        .groupby(["run_id", "eval_dataset"])
        .size()
    )
    expected_variants = {
        dataset: values["variants"] for dataset, values in EXPECTED_INVENTORY.items()
    }
    run_variant_failure_count = sum(
        count != expected_variants[dataset]
        for (_, dataset), count in run_variant_counts.items()
    )
    if run_variant_failure_count:
        blocking_failures.append("run_variant_population")

    grade_mismatch_count = int((~trace_pfu_table["ordinal_grade_match"]).sum())
    grade_mismatch_rate = grade_mismatch_count / len(trace_pfu_table)
    if grade_mismatch_rate > 0.01:
        blocking_failures.append("grade_integrity")
    rubric_v2_mismatch_count = int((~trace_pfu_table["rubric_v2_match"]).sum())
    if rubric_v2_mismatch_count:
        blocking_failures.append("rubric_v2_integrity")
    ineligible_trace_count = int((~trace_pfu_table["eligible"]).sum())
    if ineligible_trace_count:
        blocking_failures.append("ineligible_traces")
    blocking_discrepancy_count = sum(
        row["severity"] == "blocking" for row in discrepancies
    )
    if blocking_discrepancy_count:
        blocking_failures.append("join_discrepancies")
    warning_discrepancy_count = sum(
        row["severity"] == "warning" for row in discrepancies
    )

    variant_counts = catalogue.rubrics["variant_count"]
    min_variant_count = int(variant_counts.min())
    max_variant_count = int(variant_counts.max())
    error_judgement_missing_count = int(
        (trace_pfu_table["error_scorer_judgement_count"] != 10).sum()
    )

    return {
        "schema_version": 1,
        "passed": not blocking_failures,
        "blocking_failures": blocking_failures,
        "snapshot": {
            "schema_version": snapshot_manifest["schema_version"],
            "experiment_count": snapshot_manifest["experiment_count"],
            "run_count": snapshot_manifest["run_count"],
            "manifest_sha256": sha256(
                json.dumps(snapshot_manifest, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "checksums_verified": True,
        },
        "per_segment": segment_results,
        "derived_totals": {
            "rubrics": len(catalogue.rubrics),
            "variants": len(catalogue.variants),
            "expectations": len(catalogue.expectations),
            "traces": len(trace_pfu_table),
            "expected_criterion_observations": expected_criterion_count,
            "actual_eligible_criterion_observations": actual_criterion_count,
        },
        "variant_range": {
            "min_V_r": min_variant_count,
            "max_V_r": max_variant_count,
            "min_V_r_at_least_2": min_variant_count >= 2,
            "V_r_equals_1_rubric_count": int((variant_counts == 1).sum()),
            "blocking": False,
        },
        "coverage": {
            "arm_balance_failure_count": arm_balance_failure_count,
            "run_variant_failure_count": run_variant_failure_count,
            "ineligible_trace_count": ineligible_trace_count,
            "blocking_discrepancy_count": blocking_discrepancy_count,
            "warning_discrepancy_count": warning_discrepancy_count,
        },
        "materiality": materiality,
        "integrity": {
            "rubric_v2_mismatch_count": rubric_v2_mismatch_count,
            "grade_mismatch_count": grade_mismatch_count,
            "grade_mismatch_rate": grade_mismatch_rate,
            "grade_block_threshold": 0.01,
        },
        "degenerate_case_census": {
            "P_zero_trace_count": int((trace_pfu_table["P"] == 0).sum()),
            "F_zero_trace_count": int((trace_pfu_table["F"] == 0).sum()),
            "U_zero_trace_count": int((trace_pfu_table["U"] == 0).sum()),
            "all_undetermined_trace_count": int(
                (trace_pfu_table["U"] == trace_pfu_table["N_r"]).sum()
            ),
            "single_expectation_rubric_count": int(
                (catalogue.rubrics["expectation_count"] == 1).sum()
            ),
            "single_expectation_trace_count": int((trace_pfu_table["N_r"] == 1).sum()),
            "missing_error_judgement_trace_count": error_judgement_missing_count,
            "error_override_trace_count": int(trace_pfu_table["error_override"].sum()),
            "zero_component_smoothing_required_trace_count": int(
                ((trace_pfu_table[["P", "F", "U"]] == 0).any(axis=1)).sum()
            ),
        },
        "deferred_degenerate_cases": [
            "outer_mml_nonconvergence",
            "laplace_mode_or_hessian_failure",
            "importance_resampling_low_ess",
            "perfect_or_quasi_perfect_separation",
            "singular_observed_restricted_covariance",
            "singular_or_nonfinite_bootstrap_replicate",
            "empty_proportional_odds_cut_point",
            "degenerate_gold_validation",
        ],
        "f3_design_rank": {
            "full_rank": f3_design_rank["full_rank"],
            "interaction_status": f3_design_rank["interaction_status"],
            "H_F3": f3_design_rank["H_F3"],
        },
        "run_count": len(runs),
    }


def _arm_balance_failure_count(trace_pfu_table: pd.DataFrame) -> int:
    arm_balance = (
        trace_pfu_table[trace_pfu_table["eligible"]]
        .groupby(["rubric_id", "variant_index"])
        .agg(row_count=("arm_id", "size"), distinct_arm_count=("arm_id", "nunique"))
    )
    return int(arm_balance.ne(28).any(axis="columns").sum())


def _materiality_verification(catalogue: Catalogue) -> dict[str, int | bool]:
    expectation_count = len(catalogue.expectations)
    material_expectation_count = int(catalogue.expectations["material"].eq(True).sum())
    non_material_expectation_count = expectation_count - material_expectation_count
    rubric_mismatch_count = int(
        catalogue.rubrics["material_expectation_count"]
        .ne(catalogue.rubrics["expectation_count"])
        .sum()
    )
    return {
        "expectation_count": expectation_count,
        "material_expectation_count": material_expectation_count,
        "non_material_expectation_count": non_material_expectation_count,
        "rubric_mismatch_count": rubric_mismatch_count,
        "passed": non_material_expectation_count == 0 and rubric_mismatch_count == 0,
    }


def _render_report(
    verification: dict[str, object], f3_design_rank: dict[str, object]
) -> str:
    totals = cast(dict[str, int], verification["derived_totals"])
    coverage = cast(dict[str, int], verification["coverage"])
    materiality = cast(dict[str, int | bool], verification["materiality"])
    integrity = cast(dict[str, int | float], verification["integrity"])
    variant_range = cast(dict[str, int | bool], verification["variant_range"])
    census = cast(dict[str, int], verification["degenerate_case_census"])
    per_segment = cast(dict[str, dict[str, object]], verification["per_segment"])
    lines = [
        "# Partial report 01 — Data contract",
        "",
        f"Structural verification: **{'PASS' if verification['passed'] else 'FAIL'}**.",
        "",
        "## Analysis population",
        "",
    ]
    for dataset, result in per_segment.items():
        actual = cast(dict[str, int], result["actual"])
        lines.append(
            f"- `{dataset}`: {actual['rubrics']} rubrics, {actual['variants']} variants, "
            f"{actual['expectations']} expectations, {actual['traces']} traces."
        )
    lines.extend(
        [
            (
                f"- Total: {totals['rubrics']} rubrics, {totals['variants']} variants, "
                f"{totals['expectations']} expectations, {totals['traces']} traces."
            ),
            (
                f"- Criterion observations: "
                f"{totals['actual_eligible_criterion_observations']} "
                f"(expected {totals['expected_criterion_observations']})."
            ),
            (
                f"- Designed variants per rubric: "
                f"{variant_range['min_V_r']}–{variant_range['max_V_r']}."
            ),
            "",
            "## Gates",
            "",
            f"- Arm-balance failures: {coverage['arm_balance_failure_count']}.",
            f"- Ineligible traces: {coverage['ineligible_trace_count']}.",
            (
                f"- Non-blocking join warnings: "
                f"{coverage['warning_discrepancy_count']} "
                f"(see `tables/join_discrepancies.parquet`)."
            ),
            (
                f"- Material expectations: "
                f"{materiality['material_expectation_count']}/"
                f"{materiality['expectation_count']} "
                f"(non-material: {materiality['non_material_expectation_count']})."
            ),
            f"- RubricV2 mismatches: {integrity['rubric_v2_mismatch_count']}.",
            (
                f"- Grade mismatches: {integrity['grade_mismatch_count']} "
                f"({integrity['grade_mismatch_rate']:.4%})."
            ),
            (
                f"- F3 interaction: {f3_design_rank['interaction_status']} "
                f"(rank {f3_design_rank['rank']}/{f3_design_rank['column_count']}, "
                f"H_F3={f3_design_rank['H_F3']})."
            ),
            "",
            "## Join-time degenerate-case census",
            "",
        ]
    )
    lines.extend(f"- `{name}`: {value}." for name, value in census.items())
    lines.extend(
        [
            "",
            "All counts are structural diagnostics. No feature–outcome association was computed.",
            "",
        ]
    )
    return "\n".join(lines)


def _sorted_catalogue(catalogue: Catalogue) -> Catalogue:
    return Catalogue(
        rubrics=catalogue.rubrics.sort_values(
            "rubric_order", kind="stable"
        ).reset_index(drop=True),
        variants=catalogue.variants.sort_values(
            ["rubric_order", "variant_index"], kind="stable"
        ).reset_index(drop=True),
        expectations=catalogue.expectations.sort_values(
            ["rubric_order", "expectation_index"], kind="stable"
        ).reset_index(drop=True),
        input_paths=catalogue.input_paths,
    )


_versioned = versioned_frame


def _require_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    duplicates = frame.duplicated(columns, keep=False)
    if duplicates.any():
        raise GateFailureError(
            f"{label} has {int(duplicates.sum())} rows with duplicate keys {columns}"
        )


def _detected_error_override(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        rows = json.loads(value)
    except json.JSONDecodeError:
        return False
    return isinstance(rows, list) and any(
        isinstance(row, dict) and row.get("detected") is True for row in rows
    )


def _error_truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.casefold() == "true"
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value >= 1
    return value is True


def _add_discrepancy(
    rows: list[dict[str, object]],
    *,
    severity: str,
    discrepancy_type: str,
    eval_dataset: str | None = None,
    run_id: str | None = None,
    trace_id: str | None = None,
    rubric_id: str | None = None,
    variant_index: int | None = None,
    arm_id: str | None = None,
    expectation_index: int | None = None,
    expected: str | None = None,
    actual: str | None = None,
) -> None:
    discrepancy_id = stable_id(
        "discrepancy",
        discrepancy_type,
        eval_dataset,
        run_id,
        trace_id,
        rubric_id,
        variant_index,
        arm_id,
        expectation_index,
    )
    rows.append(
        {
            "discrepancy_id": discrepancy_id,
            "severity": severity,
            "discrepancy_type": discrepancy_type,
            "eval_dataset": eval_dataset,
            "run_id": run_id,
            "trace_id": trace_id,
            "rubric_id": rubric_id,
            "variant_index": variant_index,
            "arm_id": arm_id,
            "expectation_index": expectation_index,
            "expected": expected,
            "actual": actual,
        }
    )


def _nullable_string(value: object) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


def _nullable_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(cast(str | int | float, value))
