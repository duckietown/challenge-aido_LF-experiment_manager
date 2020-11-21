from zuper_commons.logs import ZLogger

__version__ = "6.0.51"

logger = ZLogger(__name__)
logger.debug(f"duckietown_experiment_manager version {__version__} path {__file__}")
from .code import *
from .experiment_manager import *
