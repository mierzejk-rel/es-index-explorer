"""Run or resume the 12-replicate Layer 1 synthetic recovery calibration."""

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

from es_index_explorer.question_analysis.layer1_checkpoint import (
    Layer1CheckpointSession,
)
from es_index_explorer.question_analysis.layer1_execution import (
    Layer1ExecutionConfig,
    ProgressEvent,
    RateLimitedProgressWriter,
    SpawnProcessExecutor,
    resolve_worker_count,
)
from es_index_explorer.question_analysis.layer1_laplace import conditional_mode
from es_index_explorer.question_analysis.layer1_model import (
    Layer1Dataset,
    Layer1Hyperparameters,
    Layer1MmlFit,
    Layer1RubricData,
    Layer1StartFit,
    Layer1VariantData,
    fit_layer1_mml,
    inverse_alr_batch,
    method_of_moments_start,
)
from es_index_explorer.question_analysis.layer1_pipeline import (
    _auto_tune_model_executor,
    _execution_provenance,
    _fit_payload,
    _start_from_payload,
    _start_payload,
)
from es_index_explorer.question_analysis.layer1_propagation import (
    simulate_layer1_dataset_with_truth,
)
from es_index_explorer.question_analysis.seeds import derive_child_seed
from es_index_explorer.question_analysis.storage import atomic_write_bytes
from scripts.profile_layer1 import _synthetic_template
from scripts.run_layer1_synthetic_recovery import (
    CONTRACT,
    RESULT_ROOT,
    _identity,
)

CALIBRATION_ROOT = RESULT_ROOT / "synthetic-calibration"
CALIBRATION_RESULT = RESULT_ROOT / "synthetic-calibration.json"
PRODUCTION_DESIGN = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "question_analysis"
    / "layer1-synthetic-production-design.json"
)


def _production_template(
    design_payload: dict[str, object],
    contract: dict[str, object],
) -> tuple[Layer1Dataset, Layer1Hyperparameters]:
    _, hyperparameters = _synthetic_template(contract)
    rubrics = tuple(
        Layer1RubricData(
            rubric_id=str(rubric["rubric_id"]),
            rubric_order=cast(int, rubric["rubric_order"]),
            dataset=str(rubric["dataset"]),
            expectation_count=cast(int, rubric["expectation_count"]),
            variants=tuple(
                Layer1VariantData(
                    variant_id=str(variant["variant_id"]),
                    counts=np.tile(
                        [[cast(int, rubric["expectation_count"]), 0, 0]],
                        (len(cast(list[str], variant["arm_ids"])), 1),
                    ),
                    arm_ids=tuple(cast(list[str], variant["arm_ids"])),
                    stages=tuple(cast(list[str], variant["stages"])),
                )
                for variant in cast(list[dict[str, object]], rubric["variants"])
            ),
        )
        for rubric in cast(list[dict[str, object]], design_payload["rubrics"])
    )
    return (
        Layer1Dataset(
            rubrics=rubrics,
            dataset_levels=tuple(
                cast(dict[str, list[float]], contract["dataset_offsets"])
            ),
        ),
        hyperparameters,
    )


def _mean_hyperparameters(
    fits: Sequence[Layer1MmlFit],
) -> dict[str, object]:
    datasets = tuple(sorted(fits[0].hyperparameters.dataset_offsets))
    return {
        "phi": float(np.mean([fit.hyperparameters.phi for fit in fits])),
        "mu0": np.mean([fit.hyperparameters.mu0 for fit in fits], axis=0).tolist(),
        "dataset_offsets": {
            dataset: np.mean(
                [fit.hyperparameters.dataset_offsets[dataset] for fit in fits],
                axis=0,
            ).tolist()
            for dataset in datasets
        },
        "sigma_within": np.mean(
            [fit.hyperparameters.sigma_within for fit in fits],
            axis=0,
        ).tolist(),
        "sigma_between": np.mean(
            [fit.hyperparameters.sigma_between for fit in fits],
            axis=0,
        ).tolist(),
    }


def _aggregate_recovery(
    fits: Sequence[Layer1MmlFit],
    contract: dict[str, object],
    latent_mean_rmses: Sequence[float],
    event_probabilities: Sequence[dict[str, float]],
    event_reference: dict[str, float],
) -> dict[str, object]:
    mean = _mean_hyperparameters(fits)
    truth_phi = float(cast(float, contract["phi"]))
    truth_mu0 = np.asarray(cast(list[float], contract["mu0"]))
    truth_offsets = cast(dict[str, list[float]], contract["dataset_offsets"])
    truth_within = np.asarray(cast(list[list[float]], contract["sigma_within"]))
    truth_between = np.asarray(cast(list[list[float]], contract["sigma_between"]))
    mean_offsets = cast(dict[str, list[float]], mean["dataset_offsets"])
    identified_errors = []
    for dataset, truth_offset in truth_offsets.items():
        truth_mean = truth_mu0 + np.asarray(truth_offset)
        fitted_mean = np.asarray(cast(list[float], mean["mu0"])) + np.asarray(
            mean_offsets[dataset]
        )
        identified_errors.append(float(np.max(np.abs(fitted_mean - truth_mean))))
    errors = {
        "dataset_mean_alr_max_abs_error": max(identified_errors),
        "phi_relative_error": abs(cast(float, mean["phi"]) - truth_phi) / truth_phi,
        "sigma_within_relative_frobenius_error": float(
            np.linalg.norm(np.asarray(mean["sigma_within"]) - truth_within)
            / np.linalg.norm(truth_within)
        ),
        "sigma_between_relative_frobenius_error": float(
            np.linalg.norm(np.asarray(mean["sigma_between"]) - truth_between)
            / np.linalg.norm(truth_between)
        ),
        "latent_mean_rmse": float(np.mean(latent_mean_rmses)),
    }
    mean_events = {
        name: float(
            np.mean([probabilities[name] for probabilities in event_probabilities])
        )
        for name in event_reference
    }
    errors["decision_probability_max_abs_error"] = max(
        abs(mean_events[name] - reference)
        for name, reference in event_reference.items()
    )
    acceptance = cast(dict[str, float], contract["acceptance"])
    passed = bool(
        errors["dataset_mean_alr_max_abs_error"]
        <= acceptance["dataset_mean_alr_max_abs_error"]
        and errors["phi_relative_error"] <= acceptance["phi_relative_error"]
        and errors["sigma_within_relative_frobenius_error"]
        <= acceptance["covariance_relative_frobenius_error"]
        and errors["sigma_between_relative_frobenius_error"]
        <= acceptance["covariance_relative_frobenius_error"]
        and errors["latent_mean_rmse"] <= acceptance["latent_mean_rmse"]
        and errors["decision_probability_max_abs_error"]
        <= acceptance["decision_probability_max_abs_error"]
    )
    return {
        "passed": passed,
        "mean_hyperparameters": mean,
        "errors": errors,
        "mean_event_probabilities": mean_events,
    }


def _fitted_event_probabilities(
    fit: Layer1MmlFit,
    *,
    replicate: int,
    design: Layer1Dataset,
) -> dict[str, float]:
    rng = np.random.default_rng(
        derive_child_seed(
            "diagnostic_resampling",
            f"layer1-calibration-events:{replicate}",
        )
    )
    draw_count = 20_000
    hyperparameters = fit.hyperparameters
    totals = {
        f"{kind}_{suffix}": 0.0
        for kind in ("minimum_ge", "proportion_ge")
        for suffix in ("0_75", "0_60", "0_50")
    }
    for rubric in design.rubrics:
        rubric_latents = rng.multivariate_normal(
            hyperparameters.mu0 + hyperparameters.dataset_offsets[rubric.dataset],
            hyperparameters.sigma_between,
            size=draw_count,
        )
        values = np.empty((draw_count, len(rubric.variants)))
        for variant_index in range(len(rubric.variants)):
            eta = rubric_latents + rng.multivariate_normal(
                np.zeros(2),
                hyperparameters.sigma_within,
                size=draw_count,
            )
            probabilities = inverse_alr_batch(eta)
            values[:, variant_index] = probabilities[:, 0] + 0.5 * probabilities[:, 2]
        binding_count = int(np.ceil(0.75 * len(rubric.variants)))
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
    return {name: value / len(design.rubrics) for name, value in totals.items()}


def main(argv: Sequence[str] | None = None) -> int:
    """Run all deterministic calibration replicates with per-start checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=5.0)
    arguments = parser.parse_args(argv)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    design_payload = json.loads(PRODUCTION_DESIGN.read_text(encoding="utf-8"))
    calibration = cast(dict[str, object], contract["calibration"])
    actual_design_hash = hashlib.sha256(PRODUCTION_DESIGN.read_bytes()).hexdigest()
    if actual_design_hash != str(calibration["blocking_design_sha256"]):
        raise ValueError("Production-shaped calibration design hash changed")
    replicate_count = int(cast(int, calibration["replicate_count"]))
    template = str(calibration["child_key_template"])
    template_data, truth_hyperparameters = _production_template(
        design_payload,
        contract,
    )
    first_simulation = simulate_layer1_dataset_with_truth(
        template_data,
        truth_hyperparameters,
        rng=np.random.default_rng(
            derive_child_seed(
                "diagnostic_resampling",
                template.format(replicate=1),
            )
        ),
    )
    first_data = first_simulation.data
    first_moments = method_of_moments_start(first_data)
    explicit_workers = None if arguments.workers == "auto" else int(arguments.workers)
    execution_config = Layer1ExecutionConfig(
        worker_mode="auto" if explicit_workers is None else "process",
        worker_count=explicit_workers,
        progress_interval_seconds=arguments.progress_interval,
    )
    auto_tune = (
        _auto_tune_model_executor(first_data, first_moments, execution_config)
        if explicit_workers is None
        else None
    )
    workers = resolve_worker_count(
        execution_config,
        auto_tune_result=auto_tune,
    )
    provenance = _execution_provenance(execution_config, workers, auto_tune)
    progress = RateLimitedProgressWriter(
        jsonl_path=RESULT_ROOT / "synthetic-calibration-progress.jsonl",
        interval_seconds=arguments.progress_interval,
    )
    fits: list[Layer1MmlFit] = []
    latent_mean_rmses: list[float] = []
    event_probabilities: list[dict[str, float]] = []
    with SpawnProcessExecutor(workers) as executor:
        for replicate in range(1, replicate_count + 1):
            simulation = simulate_layer1_dataset_with_truth(
                template_data,
                truth_hyperparameters,
                rng=np.random.default_rng(
                    derive_child_seed(
                        "diagnostic_resampling",
                        template.format(replicate=replicate),
                    )
                ),
            )
            data = simulation.data
            moments = method_of_moments_start(data)
            session = Layer1CheckpointSession.open(
                CALIBRATION_ROOT / f"replicate-{replicate:02d}",
                _identity(CONTRACT).model_copy(
                    update={
                        "input_sha256s": {
                            "synthetic_contract": hashlib.sha256(
                                CONTRACT.read_bytes()
                            ).hexdigest(),
                            "production_design": actual_design_hash,
                        },
                        "seed_contract": {
                            "stream": template.format(replicate=replicate)
                        },
                    }
                ),
                provenance,
                resume=not arguments.fresh,
                fresh=arguments.fresh,
            )
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
                replicate_id: int = replicate,
                current_session: Layer1CheckpointSession = session,
            ) -> None:
                progress.write(
                    ProgressEvent(
                        phase="synthetic_calibration",
                        work_unit_id=(f"replicate:{replicate_id}:start:{multiplier}"),
                        iteration=iteration_index,
                        objective=objective,
                        gradient_maximum=gradient,
                        completed=replicate_id - 1,
                        total=replicate_count,
                        checkpoint_path=current_session.root.as_posix(),
                    )
                )

            def completed_start(
                start: Layer1StartFit,
                replicate_id: int = replicate,
                current_session: Layer1CheckpointSession = session,
            ) -> None:
                current_session.commit_json_state(
                    f"primary/start_{str(start.phi_multiplier).replace('.', '_')}.json",
                    _start_payload(start),
                    task_id=(
                        f"calibration:{replicate_id}:start:{start.phi_multiplier}"
                    ),
                )

            fit = fit_layer1_mml(
                data,
                moments,
                executor=executor,
                completed_starts=completed,
                start_callback=completed_start,
                iteration_callback=iteration,
            )
            session.commit_json_state(
                "primary/fit.json",
                dict(_fit_payload(fit)),
                task_id=f"calibration:{replicate}:fit",
            )
            session.mark_completed()
            fits.append(fit)
            squared_errors: list[float] = []
            for rubric in data.rubrics:
                mode = conditional_mode(rubric, fit.hyperparameters, moments)
                fitted_variants = mode.value[2:].reshape(len(rubric.variants), 2)
                for variant, fitted_variant in zip(
                    rubric.variants,
                    fitted_variants,
                    strict=True,
                ):
                    difference = (
                        fitted_variant - simulation.variant_latents[variant.variant_id]
                    )
                    squared_errors.extend(np.square(difference))
            latent_mean_rmses.append(float(np.sqrt(np.mean(squared_errors))))
            event_probabilities.append(
                _fitted_event_probabilities(
                    fit,
                    replicate=replicate,
                    design=template_data,
                )
            )
            progress.write(
                ProgressEvent(
                    phase="synthetic_calibration_replicate_complete",
                    work_unit_id=f"replicate:{replicate}",
                    completed=replicate,
                    total=replicate_count,
                    checkpoint_path=session.root.as_posix(),
                ),
                force=True,
            )
    aggregate = _aggregate_recovery(
        fits,
        contract,
        latent_mean_rmses,
        event_probabilities,
        cast(
            dict[str, float],
            design_payload["reference_event_probabilities"],
        ),
    )
    payload = {
        "schema_version": 1,
        "replicate_count": replicate_count,
        "worker_count": workers,
        "auto_tuning": (
            {
                "selected_worker_count": auto_tune.selected_worker_count,
                "candidates": list(auto_tune.candidates),
            }
            if auto_tune is not None
            else None
        ),
        **aggregate,
    }
    atomic_write_bytes(
        CALIBRATION_RESULT,
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    if aggregate["passed"] is not True:
        raise RuntimeError("Repeated Layer 1 synthetic calibration failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
