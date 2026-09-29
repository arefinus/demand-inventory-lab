"""Exception types. Every error message must tell the user what to do next."""

from __future__ import annotations


class DemandInventoryError(Exception):
    """Base class for all package errors."""


class EmptySeriesError(DemandInventoryError):
    """Raised when a series has no observations."""


class InsufficientHistoryError(DemandInventoryError):
    """Raised when a series is too short for the requested model or horizon."""


class SchemaError(DemandInventoryError):
    """Raised when a table or API payload does not match the expected schema."""


class MissingDataError(DemandInventoryError):
    """Raised when a required input file is absent. CLI maps this to exit code 2."""


class ConfigError(DemandInventoryError):
    """Raised when configuration values are inconsistent or out of range."""
