"""Workflow errors with locked CLI exit categories."""

from es_index_explorer.question_analysis.contracts import ExitCode, FailureKind


class AnalysisError(RuntimeError):
    """Base error carrying a stable CLI and workflow category."""

    exit_code: ExitCode
    failure_kind: FailureKind

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class MalformedInputError(AnalysisError):
    """Input or persisted-state validation failed."""

    exit_code = ExitCode.MALFORMED_INPUT
    failure_kind = FailureKind.MALFORMED_INPUT


class GateFailureError(AnalysisError):
    """A blocking scientific or structural gate failed."""

    exit_code = ExitCode.GATE_FAILURE
    failure_kind = FailureKind.GATE_FAILURE


class PrerequisiteError(AnalysisError):
    """A required release-sequence step is incomplete."""

    exit_code = ExitCode.PREREQUISITE
    failure_kind = FailureKind.PREREQUISITE


class NumericalError(AnalysisError):
    """A required numerical result is non-computable."""

    exit_code = ExitCode.NUMERICAL
    failure_kind = FailureKind.NUMERICAL


class SingularRestrictionCovarianceError(NumericalError):
    """A restricted covariance is singular after the frozen PSD projection."""


class NonFiniteBootstrapReplicateError(NumericalError):
    """A bootstrap replicate produced a non-finite coefficient or statistic."""
