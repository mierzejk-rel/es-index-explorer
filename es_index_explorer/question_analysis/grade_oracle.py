"""Runtime lookup for the source-generated ordinal-grade oracle."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from es_index_explorer.mlflow_analysis.experiment_arms import GRADE_ORDER
from es_index_explorer.question_analysis.errors import MalformedInputError

DEFAULT_GRADE_ORACLE = (
    Path(__file__).parent / "resources" / "ordinal-grade-oracle-v1.json"
)
DEFAULT_GRADE_ORACLE_PROVENANCE = (
    Path(__file__).parent / "resources" / "ordinal-grade-oracle-v1.provenance.json"
)
type GradeKey = tuple[int, int, int, bool]


@dataclass(frozen=True, slots=True)
class GradeOracle:
    """Validated `(P, F, U, error_override)` grade lookup."""

    max_n: int
    grades: dict[GradeKey, str]
    fixture_path: Path
    provenance_path: Path

    @classmethod
    def load(
        cls,
        fixture_path: Path = DEFAULT_GRADE_ORACLE,
        provenance_path: Path = DEFAULT_GRADE_ORACLE_PROVENANCE,
    ) -> "GradeOracle":
        """Load and structurally verify the immutable fixture."""
        fixture = _read_json(fixture_path)
        provenance = _read_json(provenance_path)
        if fixture.get("schema_version") != 1 or provenance.get("schema_version") != 1:
            raise MalformedInputError("Unsupported grade-oracle schema version")
        if (
            provenance.get("fixture_sha256")
            != sha256(fixture_path.read_bytes()).hexdigest()
        ):
            raise MalformedInputError(
                "Grade-oracle fixture does not match its provenance"
            )
        max_n = fixture.get("max_n")
        rows = fixture.get("rows")
        if not isinstance(max_n, int) or max_n < 1 or not isinstance(rows, list):
            raise MalformedInputError("Invalid grade-oracle fixture structure")

        grades: dict[GradeKey, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise MalformedInputError("Invalid grade-oracle row")
            try:
                key = (
                    int(row["P"]),
                    int(row["F"]),
                    int(row["U"]),
                    bool(row["error_override"]),
                )
                grade = str(row["grade"])
            except (KeyError, TypeError, ValueError) as error:
                raise MalformedInputError(f"Invalid grade-oracle row: {row}") from error
            if grade not in GRADE_ORDER or key in grades:
                raise MalformedInputError(
                    f"Invalid or duplicate grade-oracle key: {key}"
                )
            grades[key] = grade

        expected_keys = {
            (passes, failures, n - passes - failures, error_override)
            for n in range(1, max_n + 1)
            for passes in range(n + 1)
            for failures in range(n - passes + 1)
            for error_override in (False, True)
        }
        if set(grades) != expected_keys:
            raise MalformedInputError(
                "Grade-oracle fixture does not cover every frozen P/F/U/error key"
            )
        return cls(
            max_n=max_n,
            grades=grades,
            fixture_path=fixture_path.resolve(),
            provenance_path=provenance_path.resolve(),
        )

    def lookup(
        self, passes: int, failures: int, undetermined: int, error_override: bool
    ) -> str:
        """Return the authoritative grade for one state-count tuple."""
        key = (passes, failures, undetermined, error_override)
        try:
            return self.grades[key]
        except KeyError as error:
            raise MalformedInputError(
                f"Grade-oracle key is outside the fixture: {key}"
            ) from error

    def expectations_to_next_grade(
        self, passes: int, failures: int, undetermined: int, error_override: bool
    ) -> int | None:
        """Return the minimum non-pass flips required for a higher grade."""
        if error_override or passes + failures == 0:
            return None
        current_grade = self.lookup(passes, failures, undetermined, False)
        current_rank = GRADE_ORDER[current_grade]
        for flip_count in range(1, failures + undetermined + 1):
            minimum_failure_flips = max(0, flip_count - undetermined)
            maximum_failure_flips = min(failures, flip_count)
            for failure_flips in range(
                minimum_failure_flips, maximum_failure_flips + 1
            ):
                undetermined_flips = flip_count - failure_flips
                candidate = self.lookup(
                    passes + flip_count,
                    failures - failure_flips,
                    undetermined - undetermined_flips,
                    False,
                )
                if GRADE_ORDER[candidate] > current_rank:
                    return flip_count
        return None


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise MalformedInputError(f"Required grade-oracle resource is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MalformedInputError(
            f"Invalid grade-oracle JSON {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise MalformedInputError(f"Grade-oracle JSON must contain an object: {path}")
    return payload
