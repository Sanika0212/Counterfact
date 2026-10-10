"""Counterfact: find the step that broke your agent."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("counterfact")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
