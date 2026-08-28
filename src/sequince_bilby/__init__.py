"""sequince-bilby: A Bilby wrapper for sequince."""

from importlib.metadata import PackageNotFoundError, version

from .proposal import BilbyPriorProposal

__all__ = ["BilbyPriorProposal"]

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
