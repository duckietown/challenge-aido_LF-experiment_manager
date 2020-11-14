from zuper_commons.logs import ZLogger

__version__ = "6.0.47"

logger = ZLogger(__name__)
logger.debug(f"duckietown_experiment_manager version {__version__}")
from .code import *
from .experiment_manager import *
