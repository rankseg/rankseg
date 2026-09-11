from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from . import distribution, functional
from ._rankseg import RankSEG
from ._rankseg_algo import rankdice_ba, rankseg_rma
from .distribution import RefinedNormal, RefinedNormalPB

try:
    __version__ = _distribution_version("rankseg")
except _PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = (
    "__version__",
    "RankSEG",
    "functional",
    "distribution",
    "RefinedNormalPB",
    "RefinedNormal",
    "rankdice_ba",
    "rankseg_rma",
)
