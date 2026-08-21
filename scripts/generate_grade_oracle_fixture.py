"""Generate the immutable ordinal-grade fixture from an r1-evals checkout."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SOURCE_RELATIVE_PATH = Path("src/r1_evals/rubrics/ordinal_grading_v2.py")


def main() -> None:
    """Generate the fixture and its deterministic provenance sidecar."""
    args = _parser().parse_args()
    repository = args.r1_evals_root.resolve()
    source = repository / SOURCE_RELATIVE_PATH
    sys.path.insert(0, str(repository / "src"))

    from r1_evals.rubrics.models_v2 import (  # ty: ignore[unresolved-import]
        CriterionResultV2,
        RubricGradingResultV2,
    )
    from r1_evals.rubrics.ordinal_grading_v2 import (  # ty: ignore[unresolved-import]
        compute_ordinal_grade,
    )

    rows: list[dict[str, object]] = []
    for n in range(1, args.max_n + 1):
        for passes in range(n + 1):
            for failures in range(n - passes + 1):
                undetermined = n - passes - failures
                criteria = [
                    *(
                        CriterionResultV2(
                            name=f"pass_{index}",
                            rationale="fixture",
                            state="PASS",
                            material=True,
                        )
                        for index in range(passes)
                    ),
                    *(
                        CriterionResultV2(
                            name=f"fail_{index}",
                            rationale="fixture",
                            state="FAIL",
                            material=True,
                        )
                        for index in range(failures)
                    ),
                    *(
                        CriterionResultV2(
                            name=f"undetermined_{index}",
                            rationale="fixture",
                            state="UNDETERMINED",
                            material=True,
                        )
                        for index in range(undetermined)
                    ),
                ]
                result = RubricGradingResultV2(
                    criteria=criteria,
                    score=RubricGradingResultV2.calculate_score(criteria),
                )
                for error_override, feedbacks in (
                    (False, None),
                    (True, ["true"]),
                ):
                    rows.append(
                        {
                            "P": passes,
                            "F": failures,
                            "U": undetermined,
                            "error_override": error_override,
                            "grade": compute_ordinal_grade(
                                result, error_scorer_feedbacks=feedbacks
                            ),
                        }
                    )

    fixture = {"schema_version": 1, "max_n": args.max_n, "rows": rows}
    fixture_bytes = _canonical_json(fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(fixture_bytes)

    provenance = {
        "schema_version": 1,
        "generator": Path(__file__).name,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "r1_evals_repository_head": _git(repository, "rev-parse", "HEAD"),
        "source_path": SOURCE_RELATIVE_PATH.as_posix(),
        "source_last_commit": _git(
            repository,
            "log",
            "-1",
            "--format=%H",
            "--",
            SOURCE_RELATIVE_PATH.as_posix(),
        ),
        "source_git_blob_sha": _git(
            repository, "hash-object", SOURCE_RELATIVE_PATH.as_posix()
        ),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "function": "r1_evals.rubrics.ordinal_grading_v2.compute_ordinal_grade",
        "max_n": args.max_n,
        "key": ["P", "F", "U", "error_override"],
        "error_override": (
            "Any detected rubric error mode or truthy errors_* scorer judgement"
        ),
    }
    args.provenance.write_bytes(_canonical_json(provenance))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r1-evals-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--max-n", type=int, default=21)
    return parser


def _canonical_json(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    main()
