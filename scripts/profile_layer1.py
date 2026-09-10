"""Profile bounded Layer 1 work units without running full propagation."""

import argparse
import cProfile
import json
import platform
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Self, cast

import numpy as np
import psutil

from es_index_explorer.question_analysis.layer1_execution import SpawnProcessExecutor
from es_index_explorer.question_analysis.layer1_model import (
    Layer1Dataset,
    Layer1Hyperparameters,
    Layer1RubricData,
    Layer1VariantData,
    fit_mml_start,
    laplace_log_marginal,
    method_of_moments_start,
    pack_hyperparameters,
    unpack_hyperparameters,
)
from es_index_explorer.question_analysis.layer1_pipeline import load_layer1_dataset
from es_index_explorer.question_analysis.layer1_propagation import (
    simulate_layer1_dataset,
)
from es_index_explorer.question_analysis.seeds import derive_child_seed
from es_index_explorer.question_analysis.storage import (
    ArtifactStore,
    atomic_write_bytes,
)
from es_index_explorer.question_analysis.workspace import AnalysisWorkspace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "TestResults" / "layer1-performance" / "baseline.json"
DEFAULT_PROFILE = PROJECT_ROOT / "TestResults" / "layer1-performance" / "baseline.prof"
DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "question_analysis"
    / "layer1-synthetic-contract.json"
)
DEFAULT_PRODUCTION_ROOT = (
    PROJECT_ROOT / "artifacts" / "question_analysis" / "simplemode-v1-segment7"
)


@dataclass(frozen=True, slots=True)
class Measurement:
    """Store one bounded profiling measurement."""

    label: str
    elapsed_seconds: float
    aggregate_peak_rss_bytes: int
    status: str
    error: str | None


class AggregateMemorySampler:
    """Sample aggregate parent and recursive-child resident memory."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stopped.is_set():
            processes = (process, *process.children(recursive=True))
            total = 0
            for current in processes:
                try:
                    total += current.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak_bytes = max(self.peak_bytes, total)
            self._stopped.wait(self.interval_seconds)

    def __enter__(self) -> Self:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: object,
    ) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join()


def _measure(label: str, operation: Callable[[], object]) -> Measurement:
    started = time.perf_counter()
    status = "passed"
    error: str | None = None
    with AggregateMemorySampler() as memory:
        try:
            operation()
        except Exception as caught:  # noqa: BLE001 - profiler records bounded failure class.
            status = "failed"
            error = f"{type(caught).__name__}: {caught}"
    return Measurement(
        label=label,
        elapsed_seconds=time.perf_counter() - started,
        aggregate_peak_rss_bytes=memory.peak_bytes,
        status=status,
        error=error,
    )


def _synthetic_template(
    contract: dict[str, object],
) -> tuple[Layer1Dataset, Layer1Hyperparameters]:
    offset_payload = cast(
        dict[str, list[float]],
        contract["dataset_offsets"],
    )
    datasets = tuple(offset_payload)
    offsets = {
        str(dataset): np.asarray(values, dtype=float)
        for dataset, values in offset_payload.items()
    }
    hyperparameters = Layer1Hyperparameters(
        phi=float(cast(float, contract["phi"])),
        mu0=np.asarray(cast(list[float], contract["mu0"]), dtype=float),
        dataset_offsets=offsets,
        sigma_within=np.asarray(
            cast(list[list[float]], contract["sigma_within"]),
            dtype=float,
        ),
        sigma_between=np.asarray(
            cast(list[list[float]], contract["sigma_between"]),
            dtype=float,
        ),
    )
    rubrics: list[Layer1RubricData] = []
    rubrics_per_dataset = cast(int, contract["rubrics_per_dataset"])
    variants_per_rubric = cast(int, contract["variant_count_per_rubric"])
    traces_per_variant = cast(int, contract["trace_count_per_variant"])
    expectation_count = cast(int, contract["expectation_count"])
    for dataset in datasets:
        for rubric_index in range(rubrics_per_dataset):
            variants = tuple(
                Layer1VariantData(
                    variant_id=f"{dataset}-r{rubric_index}-v{variant_index}",
                    counts=np.tile(
                        [[expectation_count // 3] * 3],
                        (traces_per_variant, 1),
                    ),
                    arm_ids=tuple(
                        f"a{index:02d}" for index in range(traces_per_variant)
                    ),
                )
                for variant_index in range(variants_per_rubric)
            )
            rubrics.append(
                Layer1RubricData(
                    rubric_id=f"{dataset}-r{rubric_index}",
                    rubric_order=len(rubrics),
                    dataset=dataset,
                    expectation_count=expectation_count,
                    variants=variants,
                )
            )
    design = Layer1Dataset(rubrics=tuple(rubrics), dataset_levels=datasets)
    return design, hyperparameters


def _synthetic_design(
    contract: dict[str, object],
    *,
    child_name: str = "layer1-profile-baseline",
) -> tuple[Layer1Dataset, Layer1Hyperparameters]:
    design, hyperparameters = _synthetic_template(contract)
    rng = np.random.default_rng(derive_child_seed("diagnostic_resampling", child_name))
    return simulate_layer1_dataset(design, hyperparameters, rng=rng), hyperparameters


def _balanced_subset(data: Layer1Dataset, per_dataset: int) -> Layer1Dataset:
    selected = tuple(
        rubric
        for dataset in data.dataset_levels
        for rubric in [
            current for current in data.rubrics if current.dataset == dataset
        ][:per_dataset]
    )
    return Layer1Dataset(rubrics=selected, dataset_levels=data.dataset_levels)


def _profile(
    contract_path: Path,
    production_root: Path,
    profile_path: Path,
    *,
    worker_count: int,
) -> list[Measurement]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    synthetic, _ = _synthetic_design(contract)
    moments = method_of_moments_start(synthetic)
    measurements: list[Measurement] = []
    for rubric_count in (1, 4, 8, 24):
        subset = Layer1Dataset(
            rubrics=synthetic.rubrics[:rubric_count],
            dataset_levels=synthetic.dataset_levels,
        )
        measurements.append(
            _measure(
                f"synthetic_marginal_{rubric_count}_rubrics",
                lambda subset=subset: laplace_log_marginal(
                    subset,
                    moments.hyperparameters,
                    moments.rubric_latent_starts,
                ),
            )
        )
    optimizer_data = _balanced_subset(synthetic, 2)
    optimizer_moments = method_of_moments_start(optimizer_data)
    if worker_count == 1:
        measurements.append(
            _measure(
                "synthetic_optimizer_start_6_rubrics",
                lambda: fit_mml_start(
                    optimizer_data,
                    optimizer_moments,
                    phi_multiplier=1.0,
                ),
            )
        )
    else:
        with SpawnProcessExecutor(worker_count) as executor:
            measurements.append(
                _measure(
                    "synthetic_optimizer_start_6_rubrics",
                    lambda: fit_mml_start(
                        optimizer_data,
                        optimizer_moments,
                        phi_multiplier=1.0,
                        executor=executor,
                    ),
                )
            )
    if (production_root / "tables" / "trace_pfu_table.parquet").is_file():
        workspace = AnalysisWorkspace(
            root=production_root,
            store=ArtifactStore(production_root),
        )
        production = load_layer1_dataset(workspace)
        production_moments = method_of_moments_start(production)
        measurements.append(
            _measure(
                "production_marginal_63_rubrics",
                lambda: laplace_log_marginal(
                    production,
                    production_moments.hyperparameters,
                    production_moments.rubric_latent_starts,
                ),
            )
        )
        vector = pack_hyperparameters(
            production_moments.hyperparameters, production.dataset_levels
        )
        step = 1e-5 * max(1.0, abs(float(vector[0])))

        def production_component() -> tuple[float, float]:
            direction = np.zeros_like(vector)
            direction[0] = step
            return (
                laplace_log_marginal(
                    production,
                    unpack_hyperparameters(
                        vector + direction, production.dataset_levels
                    ),
                    production_moments.rubric_latent_starts,
                )[0],
                laplace_log_marginal(
                    production,
                    unpack_hyperparameters(
                        vector - direction, production.dataset_levels
                    ),
                    production_moments.rubric_latent_starts,
                )[0],
            )

        measurements.append(_measure("production_fd_component_0", production_component))
    profiler = cProfile.Profile()
    profiler.runcall(
        laplace_log_marginal,
        Layer1Dataset(
            rubrics=synthetic.rubrics[:8],
            dataset_levels=synthetic.dataset_levels,
        ),
        moments.hyperparameters,
        moments.rubric_latent_starts,
    )
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(profile_path)
    return measurements


def main(argv: Sequence[str] | None = None) -> int:
    """Run bounded Layer 1 baseline profiling and persist canonical JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--production-root", type=Path, default=DEFAULT_PRODUCTION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args(argv)
    measurements = _profile(
        arguments.contract,
        arguments.production_root,
        arguments.profile,
        worker_count=arguments.workers,
    )
    payload = {
        "schema_version": 1,
        "kind": (
            "pre_optimization_baseline"
            if arguments.workers == 1
            else "optimized_parallel_profile"
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "numpy_version": version("numpy"),
        "scipy_version": version("scipy"),
        "psutil_version": version("psutil"),
        "worker_count": arguments.workers,
        "memory_measurement": "max sampled aggregate parent+recursive-child RSS at 0.1s cadence",
        "measurements": [
            {
                "label": measurement.label,
                "elapsed_seconds": measurement.elapsed_seconds,
                "aggregate_peak_rss_bytes": measurement.aggregate_peak_rss_bytes,
                "status": measurement.status,
                "error": measurement.error,
            }
            for measurement in measurements
        ],
        "cprofile_path": arguments.profile.as_posix(),
    }
    atomic_write_bytes(
        arguments.output,
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
