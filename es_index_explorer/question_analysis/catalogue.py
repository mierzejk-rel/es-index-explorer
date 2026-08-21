"""Rubric, variant, and expectation catalogue construction."""

import tomllib
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd

from es_index_explorer.question_analysis.errors import MalformedInputError

DATASET_RUBRIC_PATHS = {
    "emc2_set1": Path("air_assist/EMC2/uat/set_1"),
    "emc2_set2": Path("air_assist/EMC2/uat/set_2"),
    "mallinckrodt": Path("air_assist/mallinckrodt/rubrics_for_ga"),
}


@dataclass(frozen=True, slots=True)
class Catalogue:
    """Hold the three normalized catalogue grains."""

    rubrics: pd.DataFrame
    variants: pd.DataFrame
    expectations: pd.DataFrame
    input_paths: tuple[Path, ...]


def stable_id(kind: str, *parts: object) -> str:
    """Return a namespaced deterministic identity."""
    encoded = "\0".join(("simplemode-v1", kind, *(str(part) for part in parts)))
    return f"{kind}_{sha256(encoded.encode('utf-8')).hexdigest()}"


def build_catalogue(rubric_root: Path, task_path: Path) -> Catalogue:
    """Build complete catalogues from all authoritative rubric inputs."""
    rubric_root = rubric_root.resolve()
    allowed_use_cases = _load_use_cases(task_path)
    rubric_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    expectation_rows: list[dict[str, object]] = []
    input_paths: list[Path] = [task_path.resolve()]

    rubric_order = 0
    for eval_dataset, relative_root in DATASET_RUBRIC_PATHS.items():
        dataset_root = rubric_root / relative_root
        paths = sorted(dataset_root.glob("*.rubric.toml"))
        if not paths:
            raise MalformedInputError(f"No rubric TOMLs found under {dataset_root}")
        for path in paths:
            payload = _read_toml(path)
            meta = _require_mapping(payload, "meta", path)
            inputs = _require_rows(payload, "input", path)
            expectations = _require_rows(payload, "expectations", path)
            use_cases = _use_cases(meta.get("use_case"), path)
            unknown_use_cases = sorted(set(use_cases) - allowed_use_cases)
            if unknown_use_cases:
                raise MalformedInputError(
                    f"Unknown use case(s) in {path}: {', '.join(unknown_use_cases)}"
                )

            source_path = path.relative_to(rubric_root).as_posix()
            rubric_id = stable_id("rubric", eval_dataset, source_path)
            questions = tuple(_question(input_row, path) for input_row in inputs)
            content_sha256 = sha256(path.read_bytes()).hexdigest()
            raw_names = tuple(_expectation_name(row, path) for row in expectations)
            canonical_names = tuple(name.strip() for name in raw_names)
            name_counts = Counter(canonical_names)
            error_modes = payload.get("error_modes", [])
            if not isinstance(error_modes, list):
                raise MalformedInputError(f"Invalid [[error_modes]] in {path}")

            rubric_rows.append(
                {
                    "rubric_id": rubric_id,
                    "rubric_order": rubric_order,
                    "eval_dataset": eval_dataset,
                    "source_path": source_path,
                    "rubric_file_name": path.name,
                    "content_sha256": content_sha256,
                    "author": _optional_string(meta.get("author")),
                    "date": _optional_string(meta.get("date")),
                    "dataset_id": _optional_string(meta.get("dataset_id")),
                    "task_name": _optional_string(meta.get("task_name")),
                    "version": _optional_string(meta.get("version")),
                    "schema_version": _optional_string(meta.get("schema_version")),
                    "use_cases": list(use_cases),
                    "variant_count": len(inputs),
                    "expectation_count": len(expectations),
                    "distinct_expectation_name_count": len(name_counts),
                    "duplicate_expectation_name_count": sum(
                        count - 1 for count in name_counts.values()
                    ),
                    "material_expectation_count": sum(
                        bool(row.get("material", True)) for row in expectations
                    ),
                    "error_mode_count": len(error_modes),
                }
            )

            for variant_index, question in enumerate(questions):
                variant_rows.append(
                    {
                        "variant_id": stable_id(
                            "variant", eval_dataset, source_path, variant_index
                        ),
                        "rubric_id": rubric_id,
                        "rubric_order": rubric_order,
                        "eval_dataset": eval_dataset,
                        "source_path": source_path,
                        "rubric_file_name": path.name,
                        "variant_index": variant_index,
                        "question": question,
                        "use_cases": list(use_cases),
                        "variant_count": len(inputs),
                        "expectation_count": len(expectations),
                    }
                )

            occurrence_counts: Counter[str] = Counter()
            for expectation_index, row in enumerate(expectations):
                raw_name = raw_names[expectation_index]
                expectation_name = canonical_names[expectation_index]
                occurrence_index = occurrence_counts[expectation_name]
                occurrence_counts[expectation_name] += 1
                document_ids = _document_ids(row.get("document_ids"), path)
                expectation_rows.append(
                    {
                        "expectation_id": stable_id(
                            "expectation",
                            eval_dataset,
                            source_path,
                            expectation_index,
                        ),
                        "rubric_id": rubric_id,
                        "rubric_order": rubric_order,
                        "eval_dataset": eval_dataset,
                        "source_path": source_path,
                        "rubric_file_name": path.name,
                        "expectation_index": expectation_index,
                        "expectation_name": expectation_name,
                        "source_expectation_name": raw_name,
                        "expectation_name_occurrence_index": occurrence_index,
                        "description": _required_string(row.get("description"), path),
                        "material": bool(row.get("material", True)),
                        "document_ids": list(document_ids),
                        "expectation_document_count": len(document_ids),
                        "use_cases": list(use_cases),
                    }
                )

            input_paths.append(path.resolve())
            rubric_order += 1

    return Catalogue(
        rubrics=pd.DataFrame(rubric_rows),
        variants=pd.DataFrame(variant_rows),
        expectations=pd.DataFrame(expectation_rows),
        input_paths=tuple(input_paths),
    )


def _load_use_cases(task_path: Path) -> set[str]:
    payload = _read_toml(task_path)
    specification = _require_mapping(payload, "specification", task_path)
    rows = specification.get("use_cases")
    if not isinstance(rows, list):
        raise MalformedInputError(f"Missing [[specification.use_cases]] in {task_path}")
    names = {
        _required_string(row.get("name"), task_path)
        for row in rows
        if isinstance(row, dict)
    }
    if len(names) != len(rows):
        raise MalformedInputError(f"Invalid use-case taxonomy in {task_path}")
    return names


def _read_toml(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise MalformedInputError(f"Required TOML does not exist: {path}")
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise MalformedInputError(f"Invalid TOML {path}: {error}") from error


def _require_mapping(
    payload: dict[str, object], key: str, path: Path
) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise MalformedInputError(f"Missing [{key}] in {path}")
    return value


def _require_rows(
    payload: dict[str, object], key: str, path: Path
) -> list[dict[str, object]]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(row, dict) for row in value
    ):
        raise MalformedInputError(f"Missing or invalid [[{key}]] in {path}")
    return value


def _question(input_row: dict[str, object], path: Path) -> str:
    messages = input_row.get("messages")
    if not isinstance(messages, list):
        raise MalformedInputError(f"Input without messages in {path}")
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return _required_string(message.get("content"), path)
    raise MalformedInputError(f"Input without a user question in {path}")


def _expectation_name(row: dict[str, object], path: Path) -> str:
    name = _required_string(row.get("name"), path)
    if not name.strip():
        raise MalformedInputError(f"Empty expectation name in {path}")
    return name


def _use_cases(value: object, path: Path) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(dict.fromkeys(value))
    raise MalformedInputError(f"Invalid meta.use_case in {path}")


def _document_ids(value: object, path: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise MalformedInputError(f"Invalid expectation document_ids in {path}")
    return tuple(dict.fromkeys(value))


def _required_string(value: object, path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise MalformedInputError(f"Required string is missing in {path}")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
