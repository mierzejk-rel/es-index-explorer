"""Fluent condition helpers for Object Manager queries."""

from dataclasses import dataclass
from typing import Iterable


def _quote(value: str) -> str:
    return f"'{value}'"


def _format_scalar(value: int | str | bool) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    return _quote(value)


def _format_list(values: Iterable[int | str]) -> str:
    formatted = [str(v) if isinstance(v, int) else _quote(v) for v in values]
    inner = ", ".join(formatted)
    return f"[{inner}]"


@dataclass(frozen=True)
class Cond:
    expr: str

    def __and__(self, other: "Cond") -> "Cond":
        return Cond(f"({self.expr}) AND ({other.expr})")

    def __or__(self, other: "Cond") -> "Cond":
        return Cond(f"({self.expr}) OR ({other.expr})")

    def __str__(self) -> str:
        return self.expr


class Field:
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def _q(self) -> str:
        return _quote(self.name)

    def eq(self, value: int | str | bool) -> Cond:
        return Cond(f"{self._q} == {_format_scalar(value)}")

    def in_(self, values: Iterable[int | str]) -> Cond:
        return Cond(f"{self._q} IN {_format_list(values)}")

    def in_saved_search(self, saved_search_id: int) -> Cond:
        return Cond(f"{self._q} IN SAVEDSEARCH {saved_search_id}")


def field(name: str) -> Field:
    return Field(name)
