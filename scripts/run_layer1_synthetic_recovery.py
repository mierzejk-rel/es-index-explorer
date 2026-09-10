"""Run the frozen checkpointed 24-rubric Layer 1 synthetic recovery gate."""

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import cast

import numpy as np

from es_index_explorer.question_analysis.layer1_checkpoint import (
    Layer1CheckpointCompatibilityIdentity,
    Layer1CheckpointSession,
    Layer1ExecutionProvenance,
)
from es_index_explorer.question_analysis.layer1_execution import (
    Layer1ExecutionConfig,
    ProgressEvent,
    RateLimitedProgressWriter,
    ShutdownToken,
    SpawnProcessExecutor,
    resolve_worker_count,
    signal_shutdown_context,
)
from es_index_explorer.question_analysis.layer1_model import (
    Layer1MmlFit,
    Layer1MomentStart,
    Layer1StartFit,
    fit_layer1_mml,
    method_of_moments_start,
)
from es_index_explorer.question_analysis.layer1_pipeline import (
    _auto_tune_model_executor,
    _execution_provenance,
    _fit_payload,
    _start_from_payload,
    _start_payload,
)
from es_index_explorer.question_analysis.storage import atomic_write_bytes
from scripts.profile_layer1 import _synthetic_design

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "question_analysis"
    / "layer1-synthetic-contract.json"
)
RESULT_ROOT = PROJECT_ROOT / "TestResults" / "layer1-performance"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(
    contract_path: Path,
) -> Layer1CheckpointCompatibilityIdentity:
    return Layer1CheckpointCompatibilityIdentity(
        specification_sha256=_sha256(
            PROJECT_ROOT / "reports" / "13-simple-mode-analysis-research-plan.md"
        ),
        input_sha256s={"synthetic_contract": _sha256(contract_path)},
        resource_sha256s={
            name: _sha256(
                PROJECT_ROOT / "es_index_explorer" / "question_analysis" / name
            )
            for name in (
                "layer1_model.py",
                "layer1_laplace.py",
                "layer1_packed.py",
                "layer1_execution.py",
            )
        },
        numerical_contract_version="layer1-v1",
        statistical_run_config={"fixture": "24-rubric-synthetic-v1"},
        seed_contract={"stream": "diagnostic_resampling:layer1-synthetic-recovery"},
        numerical_backend="adaptive-finite-difference-v1",
        architecture=platform.machine(),
        float_abi=f"{np.dtype(np.float64).str}:{sys.byteorder}",
        python_version=platform.python_version(),
        numpy_version=version("numpy"),
        scipy_version=version("scipy"),
        blas_lapack_identity=json.dumps(
            getattr(np.__config__, "CONFIG", {}),
            sort_keys=True,
            default=str,
        ),
    )


def _recovery_payload(
    fit: Layer1MmlFit,
    contract: dict[str, object],
) -> dict[str, object]:
    truth = json.loads(json.dumps(contract))
    expected_phi = float(cast(float, truth["phi"]))
    expected_mu0 = np.asarray(
        cast(list[float], truth["mu0"]),
        dtype=float,
    )
    expected_within = np.asarray(
        cast(list[list[float]], truth["sigma_within"]),
        dtype=float,
    )
    expected_between = np.asarray(
        cast(list[list[float]], truth["sigma_between"]),
        dtype=float,
    )
    recovered = fit.hyperparameters
    recovery = {
        "phi_relative_error": abs(recovered.phi - expected_phi) / expected_phi,
        "mu0_maximum_absolute_error": float(
            np.max(np.abs(recovered.mu0 - expected_mu0))
        ),
        "sigma_within_relative_frobenius_error": float(
            np.linalg.norm(recovered.sigma_within - expected_within)
            / np.linalg.norm(expected_within)
        ),
        "sigma_between_relative_frobenius_error": float(
            np.linalg.norm(recovered.sigma_between - expected_between)
            / np.linalg.norm(expected_between)
        ),
    }
    acceptance = cast(dict[str, float], truth["acceptance"])
    passed = bool(
        recovery["phi_relative_error"] <= acceptance["phi_relative_error"]
        and recovery["mu0_maximum_absolute_error"]
        <= acceptance["dataset_mean_alr_max_abs_error"]
        and recovery["sigma_within_relative_frobenius_error"]
        <= acceptance["covariance_relative_frobenius_error"]
        and recovery["sigma_between_relative_frobenius_error"]
        <= acceptance["covariance_relative_frobenius_error"]
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "fit": dict(_fit_payload(fit)),
        "recovery": recovery,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run or resume the synthetic three-start fit with visible progress."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=5.0)
    arguments = parser.parse_args(argv)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    stream_contract = str(contract["generation_child_stream"])
    stream_name, child_name = stream_contract.split(":", maxsplit=1)
    if stream_name != "diagnostic_resampling":
        raise ValueError("Synthetic recovery stream contract is invalid")
    data, _ = _synthetic_design(contract, child_name=child_name)
    moments: Layer1MomentStart = method_of_moments_start(data)
    worker_count = None if arguments.workers == "auto" else int(arguments.workers)
    execution_config = Layer1ExecutionConfig(
        worker_mode="auto" if worker_count is None else "process",
        worker_count=worker_count,
        progress_interval_seconds=arguments.progress_interval,
    )
    auto_tune = (
        _auto_tune_model_executor(data, moments, execution_config)
        if worker_count is None
        else None
    )
    selected_workers = resolve_worker_count(
        execution_config,
        auto_tune_result=auto_tune,
    )
    provenance: Layer1ExecutionProvenance = _execution_provenance(
        execution_config,
        selected_workers,
        auto_tune,
    )
    session = Layer1CheckpointSession.open(
        RESULT_ROOT / "synthetic-recovery",
        _identity(CONTRACT),
        provenance,
        resume=not arguments.fresh,
        fresh=arguments.fresh,
    )
    progress = RateLimitedProgressWriter(
        jsonl_path=RESULT_ROOT / "synthetic-recovery-progress.jsonl",
        interval_seconds=arguments.progress_interval,
    )
    shutdown = ShutdownToken()
    completed: dict[float, Layer1StartFit] = {}
    for multiplier in (0.5, 1.0, 2.0):
        payload = session.read_json_state(
            f"primary/start_{str(multiplier).replace('.', '_')}.json"
        )
        if payload is not None:
            completed[multiplier] = _start_from_payload(payload)

    def iteration(
        multiplier: float,
        iteration_index: int,
        objective: float,
        gradient: float,
    ) -> None:
        progress.write(
            ProgressEvent(
                phase="synthetic_primary_mml",
                work_unit_id=f"start:{multiplier}",
                iteration=iteration_index,
                objective=objective,
                gradient_maximum=gradient,
                checkpoint_path=session.root.as_posix(),
            )
        )
        if shutdown.interruption is not None:
            session.mark_interrupted(shutdown.interruption.exit_code)
            raise SystemExit(shutdown.interruption.exit_code)

    def completed_start(start: Layer1StartFit) -> None:
        session.commit_json_state(
            f"primary/start_{str(start.phi_multiplier).replace('.', '_')}.json",
            _start_payload(start),
            task_id=f"synthetic:start:{start.phi_multiplier}",
        )
        progress.write(
            ProgressEvent(
                phase="synthetic_start_complete",
                work_unit_id=f"start:{start.phi_multiplier}",
                checkpoint_path=session.root.as_posix(),
            ),
            force=True,
        )

    with (
        signal_shutdown_context(shutdown),
        SpawnProcessExecutor(selected_workers) as executor,
    ):
        fit = fit_layer1_mml(
            data,
            moments,
            executor=executor,
            completed_starts=completed,
            start_callback=completed_start,
            iteration_callback=iteration,
        )
    result = _recovery_payload(fit, contract)
    atomic_write_bytes(
        RESULT_ROOT / "synthetic-recovery.json",
        (json.dumps(result, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    session.mark_completed()
    progress.write(
        ProgressEvent(
            phase="synthetic_recovery_complete",
            completed=3,
            total=3,
            checkpoint_path=session.root.as_posix(),
        ),
        force=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
