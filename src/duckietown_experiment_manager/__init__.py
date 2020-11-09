from zuper_commons.logs import ZLogger

__version__ = "6.0.42"

logger = ZLogger(__name__)
logger.info(f"{__version__}")
from .code import *
from .experiment_manager import *
