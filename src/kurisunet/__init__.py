from .register import get_module
from .utils.logger import set_logger

set_logger(level="INFO")

__all__ = ["get_module"]
