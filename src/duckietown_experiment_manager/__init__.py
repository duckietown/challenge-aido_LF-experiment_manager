from zuper_commons.logs import ZLogger

__version__ = "6.2.1"

logger = ZLogger(__name__)
import os

path = os.path.dirname(os.path.dirname(__file__))


logger.debug(f"duckietown_experiment_manager version {__version__} path {path}")
from .code import *
from .experiment_manager import *
