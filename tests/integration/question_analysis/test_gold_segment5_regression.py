"""Regression coverage for the canonical Segment 5 sampling frame."""

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from es_index_explorer.question_analysis.contracts import (
    AnnotatorKind,
    StepRecord,
    StepStatus,
    WorkflowCommand,
)
from es_index_explorer.question_analysis.gold import (
    InitialHumanGoldProvenance,
    run_gold_ingest_initial,
    run_gold_sample,
)
from es_index_explorer.question_analysis.storage import (
    canonical_json_bytes,
    sha256_file,
)
from es_index_explorer.question_analysis.workspace import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_SPECIFICATION,
    AnalysisWorkspace,
)

pytestmark = pytest.mark.integration


def test_canonical_frame_round_trips_all_feature_long_initial_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "analysis"
    workspace = AnalysisWorkspace.initialize(root, DEFAULT_SPECIFICATION)
    for name in ("annotation_agreement.parquet", "features_deterministic.parquet"):
        pd.read_parquet(DEFAULT_ANALYSIS_ROOT / "tables" / name).to_parquet(
            root / "tables" / name,
            index=False,
        )
    started_at = datetime(2026, 9, 8, 12, tzinfo=UTC)
    state = workspace.load_state()
    steps = dict(state.steps)
    for command in (
        WorkflowCommand.JOIN,
        WorkflowCommand.FEATURES,
        WorkflowCommand.ANNOTATE_EMIT,
        WorkflowCommand.ANNOTATE_RUN,
        WorkflowCommand.ANNOTATE_INGEST,
    ):
        steps[command] = StepRecord(
            status=StepStatus.COMPLETED,
            attempts=1,
            started_at=started_at - timedelta(hours=2),
            completed_at=started_at - timedelta(hours=1),
        )
    workspace.save_state(state.model_copy(update={"steps": steps}))
    workspace.run_step(WorkflowCommand.GOLD_SAMPLE, run_gold_sample, now=started_at)

    sample = pd.read_parquet(root / "tables" / "gold_sample.parquet")
    assert sample["item_id"].nunique() == 189
    assert sample["stratum_id"].nunique() == 18

    template_path = root / "gold" / "adjudication.csv"
    completed_path = tmp_path / "completed-initial.csv"
    with (
        template_path.open(encoding="utf-8", newline="") as source,
        completed_path.open("w", encoding="utf-8", newline="") as destination,
    ):
        reader = csv.DictReader(source)
        assert reader.fieldnames is not None
        writer = csv.DictWriter(
            destination,
            fieldnames=reader.fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        row_count = 0
        for row in reader:
            row["human_value_json"] = row["first_value_json"]
            writer.writerow(row)
            row_count += 1
    assert row_count == 1_680

    manifest_path = root / "gold" / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = InitialHumanGoldProvenance(
        annotator_kind=AnnotatorKind.HUMAN,
        annotator_id="integration-fixture",
        annotator_role="non-scientific regression fixture",
        expertise="Not scientific evidence.",
        qualification="Automated integration fixture.",
        annotation_environment="Ephemeral pytest workspace.",
        bundle_manifest_sha256=sha256_file(manifest_path),
        completed_adjudication_sha256=sha256_file(completed_path),
        schema_sha256=manifest["schema_sha256"],
        codebook_sha256=manifest["codebook_sha256"],
        decision_ledger_sha256=manifest["decision_ledger_sha256"],
        started_at=started_at,
        completed_at=started_at + timedelta(hours=1),
        outcome_blindness_confirmed=True,
    )
    provenance_path = tmp_path / "initial-provenance.json"
    provenance_path.write_bytes(canonical_json_bytes(provenance))
    workspace.run_step(
        WorkflowCommand.GOLD_INGEST_INITIAL,
        lambda current: run_gold_ingest_initial(
            current,
            completed_path,
            provenance_path,
        ),
        now=started_at + timedelta(hours=2),
    )

    labels = pd.read_parquet(root / "tables" / "gold_initial_labels.parquet")
    assert len(labels) == 1_680
    assert not labels.duplicated(["local_item_id", "feature"]).any()
    assert not workspace.load_state().outcome_modeling_unlocked
