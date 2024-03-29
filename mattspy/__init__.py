# flake8: noqa
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("mattspy")
except PackageNotFoundError:
    # package is not installed
    pass

from . import stats
from . import plotting
from .yield_result import ParallelResult
from .condor_yield import BNLCondorParallel
from .lsf_yield import SLACLSFParallel
from .loky_yield import LokyParallel
