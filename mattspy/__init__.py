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
from .condor_exec import BNLCondorExecutor
from .lsf_exec import SLACLSFExecutor
from .lsf_yield import SLACLSFYield
from .loky_yield import LokyYield
