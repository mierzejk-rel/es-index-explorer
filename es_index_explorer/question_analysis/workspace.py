"""Analysis-root initialization, integrity checks, and resumable execution."""

import platform
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import BaseModel

from es_index_explorer.question_analysis.contracts import (
    ANALYSIS_ID,
    AnalysisLockManifest,
    ArtifactMetadata,
    StepStatus,
    WorkflowCommand,
    WorkflowState,
)
from es_index_explorer.question_analysis.errors import (
    AnalysisError,
    MalformedInputError,
)
from es_index_explorer.question_analysis.seeds import named_stream_seeds
from es_index_explorer.question_analysis.storage import (
    ArtifactStore,
    atomic_write_bytes,
    canonical_json_bytes,
    fingerprint_file,
    sha256_file,
)
from es_index_explorer.question_analysis.workflow import (
    complete_step,
    fail_step,
    start_step,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "artifacts" / "question_analysis" / ANALYSIS_ID
DEFAULT_SPECIFICATION = (
    PROJECT_ROOT / "reports" / "13-simple-mode-analysis-research-plan.md"
)
DEFAULT_RESOURCES = {
    "r_oracle_dockerfile": PROJECT_ROOT
    / "tests"
    / "oracles"
    / "fwildclusterboot"
    / "Dockerfile",
    "stanza_en_resource_manifest": PROJECT_ROOT
    / "es_index_explorer"
    / "question_analysis"
    / "resources"
    / "stanza-en-resource-manifest.json",
}

StepAction = Callable[["AnalysisWorkspace"], Iterable[ArtifactMetadata] | None]


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _tool_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "es-index-explorer": _installed_version("es-index-explorer"),
    }
    for distribution in (
        "cursor-sdk",
        "matplotlib",
        "numpy",
        "pandas",
        "pyarrow",
        "pydantic",
        "scipy",
        "seaborn",
        "spacy",
        "stanza",
        "statsmodels",
    ):
        versions[distribution] = _installed_version(distribution)
    return versions


@dataclass(frozen=True, slots=True)
class AnalysisWorkspace:
    """One initialized, integrity-checked analysis root."""

    root: Path
    store: ArtifactStore

    @classmethod
    def initialize(
        cls,
        root: Path = DEFAULT_ANALYSIS_ROOT,
        specification: Path = DEFAULT_SPECIFICATION,
    ) -> "AnalysisWorkspace":
        workspace = cls(root=root.resolve(), store=ArtifactStore(root))
        specification_fingerprint = fingerprint_file(specification.resolve())
        resource_fingerprints = {
            name: fingerprint_file(path) for name, path in DEFAULT_RESOURCES.items()
        }
        tool_versions = _tool_versions()
        workspace.store.prepare()
        manifest_path = workspace.root / "manifest.json"
        state_path = workspace.root / "state.json"

        if manifest_path.exists():
            manifest = workspace.load_manifest()
            if manifest.specification != specification_fingerprint:
                raise MalformedInputError(
                    "Specification fingerprint differs from the initialized analysis lock"
                )
            if manifest.resources != resource_fingerprints:
                raise MalformedInputError(
                    "Resource fingerprint differs from the initialized analysis lock"
                )
            if manifest.tool_versions != tool_versions:
                raise MalformedInputError(
                    "Tool versions differ from the initialized analysis lock"
                )
        else:
            if state_path.exists():
                raise MalformedInputError("state.json exists without manifest.json")
            manifest = AnalysisLockManifest(
                specification=specification_fingerprint,
                stream_seeds=named_stream_seeds(),
                tool_versions=tool_versions,
                resources=resource_fingerprints,
            )
            workspace._write_root_model("manifest.json", manifest)

        if not state_path.exists():
            workspace.save_state(WorkflowState.initial())

        workspace.verify_manifest_inputs(manifest)
        workspace.verify_recorded_artifacts(workspace.load_state())
        return workspace

    @classmethod
    def open_existing(cls, root: Path) -> "AnalysisWorkspace":
        workspace = cls(root=root.resolve(), store=ArtifactStore(root))
        if (
            not (workspace.root / "manifest.json").is_file()
            or not (workspace.root / "state.json").is_file()
        ):
            raise MalformedInputError(
                f"Analysis root is not initialized: {workspace.root}"
            )
        manifest = workspace.load_manifest()
        state = workspace.load_state()
        workspace.verify_manifest_inputs(manifest)
        workspace.verify_recorded_artifacts(state)
        return workspace

    def load_manifest(self) -> AnalysisLockManifest:
        """Load the immutable analysis-lock manifest."""
        return self.store.read_model("manifest.json", AnalysisLockManifest)

    def load_state(self) -> WorkflowState:
        """Load persistent workflow state."""
        return self.store.read_model("state.json", WorkflowState)

    def save_state(self, state: WorkflowState) -> None:
        """Atomically persist workflow state."""
        self._write_root_model("state.json", state)

    def register_input(self, name: str, path: Path) -> AnalysisLockManifest:
        """Add or verify an immutable input before ``join`` starts."""
        if not name:
            raise MalformedInputError("Input name must be non-empty")
        manifest = self.load_manifest()
        fingerprint = fingerprint_file(path.resolve())
        existing = manifest.inputs.get(name)
        if existing is not None and existing != fingerprint:
            raise MalformedInputError(
                f"Input {name!r} differs from its locked fingerprint"
            )
        if existing is not None:
            return manifest
        state = self.load_state()
        if state.steps[WorkflowCommand.JOIN].status not in (
            StepStatus.PENDING,
            StepStatus.RUNNING,
        ):
            raise MalformedInputError(
                "New inputs cannot be added after join has finished an attempt"
            )
        inputs = dict(manifest.inputs)
        inputs[name] = fingerprint
        manifest = manifest.model_copy(update={"inputs": inputs})
        self._write_root_model("manifest.json", manifest)
        return manifest

    def run_step(self, command: WorkflowCommand, action: StepAction) -> bool:
        """Run a gated action, persisting running/failure/completion states."""
        state, should_run = start_step(self.load_state(), command)
        if not should_run:
            return False
        self.save_state(state)
        try:
            artifacts = tuple(action(self) or ())
        except AnalysisError as error:
            self.save_state(fail_step(state, command, error))
            raise
        completed = complete_step(state, command, artifacts)
        self.save_state(completed)
        return True

    def verify_manifest_inputs(
        self, manifest: AnalysisLockManifest | None = None
    ) -> None:
        """Verify every locked input before work resumes."""
        current = manifest or self.load_manifest()
        for fingerprint in (
            current.specification,
            *current.inputs.values(),
            *current.resources.values(),
        ):
            path = Path(fingerprint.path)
            if not path.is_file():
                raise MalformedInputError(f"Locked input is missing: {path}")
            if (
                path.stat().st_size != fingerprint.size_bytes
                or sha256_file(path) != fingerprint.sha256
            ):
                raise MalformedInputError(f"Locked input changed: {path}")

    def verify_recorded_artifacts(self, state: WorkflowState | None = None) -> None:
        """Verify every recorded output before work resumes."""
        current = state or self.load_state()
        for metadata in current.artifacts.values():
            path = self.store.path_for(metadata.path)
            if not path.is_file():
                raise MalformedInputError(f"Recorded artifact is missing: {path}")
            if (
                path.stat().st_size != metadata.size_bytes
                or sha256_file(path) != metadata.sha256
            ):
                raise MalformedInputError(f"Recorded artifact changed: {path}")

    def _write_root_model(self, name: str, value: BaseModel) -> None:
        atomic_write_bytes(self.root / name, canonical_json_bytes(value))
