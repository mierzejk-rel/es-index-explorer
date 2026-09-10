"""Independently diagnose the frozen synthetic Layer 1 MML optimum."""

import json
from pathlib import Path
from typing import cast

import numpy as np
from scipy.optimize import minimize

from es_index_explorer.question_analysis.layer1_model import (
    method_of_moments_start,
    pack_hyperparameters,
)
from es_index_explorer.question_analysis.storage import atomic_write_bytes
from scripts.profile_layer1 import _synthetic_design
from tests.oracles.layer1.reference import (
    ReferenceRubric,
    reference_laplace_log_marginal,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "question_analysis"
    / "layer1-synthetic-contract.json"
)
FIT_RESULT = (
    PROJECT_ROOT / "TestResults" / "layer1-performance" / "synthetic-recovery.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "TestResults"
    / "layer1-performance"
    / "synthetic-optimum-diagnosis.json"
)


def main() -> int:
    """Compare the fitted vector with an independent dense objective and Powell search."""
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    fit_result = json.loads(FIT_RESULT.read_text(encoding="utf-8"))
    stream_name, child_name = str(contract["generation_child_stream"]).split(
        ":", maxsplit=1
    )
    if stream_name != "diagnostic_resampling":
        raise ValueError("Synthetic recovery stream contract is invalid")
    data, truth = _synthetic_design(contract, child_name=child_name)
    moments = method_of_moments_start(data)
    fitted_vector = np.asarray(fit_result["fit"]["vector"], dtype=float)
    reference_rubrics = tuple(
        ReferenceRubric(
            rubric_id=rubric.rubric_id,
            dataset=rubric.dataset,
            variant_counts=tuple(variant.counts for variant in rubric.variants),
        )
        for rubric in data.rubrics
    )

    def objective(vector: np.ndarray) -> float:
        return -reference_laplace_log_marginal(
            vector,
            data.dataset_levels,
            reference_rubrics,
            dict(moments.rubric_latent_starts),
        )

    fitted_objective = objective(fitted_vector)
    difference_step = 1e-5
    gradient = np.empty_like(fitted_vector)
    for index in range(len(fitted_vector)):
        step = difference_step * max(1.0, abs(float(fitted_vector[index])))
        direction = np.zeros_like(fitted_vector)
        direction[index] = step
        gradient[index] = (
            objective(fitted_vector + direction) - objective(fitted_vector - direction)
        ) / (2.0 * step)
    alternative = minimize(
        objective,
        fitted_vector,
        method="Powell",
        options={
            "maxiter": 20,
            "maxfev": 500,
            "xtol": 1e-6,
            "ftol": 1e-10,
        },
    )
    truth_objective = objective(pack_hyperparameters(truth, data.dataset_levels))
    improvement = fitted_objective - float(alternative.fun)
    payload = {
        "schema_version": 1,
        "independent_reference": "tests/oracles/layer1/reference.py",
        "fitted_objective": fitted_objective,
        "truth_objective": truth_objective,
        "reference_gradient_maximum": float(np.max(np.abs(gradient))),
        "alternative_optimizer": {
            "method": "Powell",
            "success": bool(alternative.success),
            "message": str(alternative.message),
            "evaluations": int(alternative.nfev),
            "iterations": int(alternative.nit),
            "objective": float(alternative.fun),
            "improvement_from_fitted": improvement,
            "maximum_parameter_change": float(
                np.max(np.abs(np.asarray(alternative.x) - fitted_vector))
            ),
        },
        "confirmed_same_optimum": bool(
            improvement <= 1e-6 and float(np.max(np.abs(gradient))) <= 1e-4
        ),
        "recovery_failures": cast(
            dict[str, float],
            fit_result["recovery"],
        ),
    }
    atomic_write_bytes(
        OUTPUT,
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
