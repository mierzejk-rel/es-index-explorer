"""Stage-neutral Simple Mode arm identities."""

import re
from dataclasses import dataclass
from typing import Literal

from es_index_explorer.mlflow_analysis.experiment_arms import split_arm_and_dataset
from es_index_explorer.question_analysis.errors import MalformedInputError

EVAL_DATASETS = ("emc2_set1", "emc2_set2", "mallinckrodt")

_STAGE_A = re.compile(
    r"^S-A-(?P<mode>bm25|dense|hybrid)-c(?P<calls>\d+)-rr-"
    r"f(?P<fetch>\d+)-g(?P<context>\d+)-rnone$"
)
_STAGE_B = re.compile(
    r"^S-B-(?P<branches>(?:bm25|dense)(?:-(?:bm25|dense))*)-"
    r"c(?P<calls>\d+)-union-f(?P<fetch>\d+)-rnone$"
)
_STAGE_C_ROUND_ROBIN = re.compile(
    r"^S-C-(?P<mode>bm25|dense|hybrid)-c(?P<calls>\d+)-rr-"
    r"f(?P<fetch>\d+)-g(?P<context>\d+)-rlow$"
)
_STAGE_C_UNION = re.compile(
    r"^S-C-(?P<branches>(?:bm25|dense)(?:-(?:bm25|dense))*)-"
    r"c(?P<calls>\d+)-union-f(?P<fetch>\d+)-rlow$"
)


@dataclass(frozen=True, slots=True)
class ArmIdentity:
    """Normalized identity and design dimensions for one run."""

    arm_id: str
    eval_dataset: str
    stage: Literal["A", "B", "C"]
    retrieval_family: str
    retrieval_modes: tuple[str, ...]
    calls: int
    merge_policy: Literal["round_robin", "union"]
    fetch: int
    global_context: int | None
    reasoning_effort: Literal["none", "low"]
    homogeneous: bool


def parse_arm(experiment_name: str) -> ArmIdentity:
    """Parse a Stage A, B, or C experiment name without conflating datasets."""
    short_name = experiment_name.rstrip("/").rsplit("/", maxsplit=1)[-1]
    try:
        arm_id, eval_dataset = split_arm_and_dataset(short_name, EVAL_DATASETS)
    except ValueError as error:
        raise MalformedInputError(str(error)) from error

    if match := _STAGE_A.fullmatch(arm_id):
        mode = match.group("mode")
        modes = ("bm25", "dense") if mode == "hybrid" else (mode,)
        return ArmIdentity(
            arm_id=arm_id,
            eval_dataset=eval_dataset,
            stage="A",
            retrieval_family=mode,
            retrieval_modes=modes,
            calls=int(match.group("calls")),
            merge_policy="round_robin",
            fetch=int(match.group("fetch")),
            global_context=int(match.group("context")),
            reasoning_effort="none",
            homogeneous=len(set(modes)) == 1,
        )

    if match := _STAGE_B.fullmatch(arm_id):
        return _union_identity(match, arm_id, eval_dataset, stage="B")

    if match := _STAGE_C_ROUND_ROBIN.fullmatch(arm_id):
        mode = match.group("mode")
        modes = ("bm25", "dense") if mode == "hybrid" else (mode,)
        return ArmIdentity(
            arm_id=arm_id,
            eval_dataset=eval_dataset,
            stage="C",
            retrieval_family=mode,
            retrieval_modes=modes,
            calls=int(match.group("calls")),
            merge_policy="round_robin",
            fetch=int(match.group("fetch")),
            global_context=int(match.group("context")),
            reasoning_effort="low",
            homogeneous=len(set(modes)) == 1,
        )

    if match := _STAGE_C_UNION.fullmatch(arm_id):
        return _union_identity(match, arm_id, eval_dataset, stage="C")

    raise MalformedInputError(f"Cannot parse Simple Mode arm identity {arm_id!r}")


def _union_identity(
    match: re.Match[str],
    arm_id: str,
    eval_dataset: str,
    *,
    stage: Literal["B", "C"],
) -> ArmIdentity:
    modes = tuple(match.group("branches").split("-"))
    calls = int(match.group("calls"))
    if calls != len(modes):
        raise MalformedInputError(
            f"Arm {arm_id!r} declares {calls} calls but encodes {len(modes)} branches"
        )
    return ArmIdentity(
        arm_id=arm_id,
        eval_dataset=eval_dataset,
        stage=stage,
        retrieval_family="mixed" if len(set(modes)) > 1 else modes[0],
        retrieval_modes=modes,
        calls=calls,
        merge_policy="union",
        fetch=int(match.group("fetch")),
        global_context=None,
        reasoning_effort="none" if stage == "B" else "low",
        homogeneous=len(set(modes)) == 1,
    )
