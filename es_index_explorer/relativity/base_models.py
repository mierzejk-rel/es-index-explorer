"""Shared Pydantic base models for Relativity APIs."""

from pydantic import BaseModel, ConfigDict


class HiddenInputBaseModel(BaseModel):
    """Hide input values in validation errors."""

    model_config = ConfigDict(hide_input_in_errors=True)
