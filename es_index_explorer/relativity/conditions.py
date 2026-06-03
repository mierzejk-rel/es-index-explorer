"""Fluent condition helpers for Object Manager queries."""

from dataclasses import dataclass
from typing import Iterable


def _quote(value: str) -> str:
    return f"'{value}'"


def _format_scalar(value: int | str | bool) -> str:
    match value:
        # case bool():
        #     return "True" if value else "False"
        case int():  # bool is a subclass of int
            return str(value)
        case _:
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

    def ne(self, value: int | str | bool) -> Cond:
        return Cond(f"{self._q} <> {_format_scalar(value)}")

    def gt(self, value: int | str) -> Cond:
        return Cond(f"{self._q} > {_format_scalar(value)}")

    def gte(self, value: int | str) -> Cond:
        return Cond(f"{self._q} >= {_format_scalar(value)}")

    def lt(self, value: int | str) -> Cond:
        return Cond(f"{self._q} < {_format_scalar(value)}")

    def lte(self, value: int | str) -> Cond:
        return Cond(f"{self._q} <= {_format_scalar(value)}")

    def between(self, low: int, high: int) -> Cond:
        return Cond(f"{self._q} BETWEEN {_format_scalar(low)} AND {_format_scalar(high)}")

    def in_(self, values: Iterable[int | str]) -> Cond:
        return Cond(f"{self._q} IN {_format_list(values)}")

    def in_saved_search(self, saved_search_id: int) -> Cond:
        return Cond(f"{self._q} IN SAVEDSEARCH {saved_search_id}")


def field(name: str) -> Field:
    return Field(name)
