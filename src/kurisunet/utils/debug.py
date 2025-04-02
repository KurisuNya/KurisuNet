from functools import wraps
import random

import torch

from ..utils.logger import get_logger
from .module import get_module_name

logger = get_logger("Utils")


def log_shape_hook(module, input, output):
    """Forward hook to log the shape of the input and output tensors of a module."""

    def get_shape(x):
        if isinstance(x, torch.Tensor):
            return (
                str(x.shape)
                .replace("torch.Size", "")
                .replace("([", "(")
                .replace("])", ")")
            )
        if isinstance(x, (list, tuple)):
            return str([get_shape(i) for i in x]).replace("'", "")
        return str(x)

    input_shape = get_shape(input[0] if len(input) == 1 else input)
    output_shape = get_shape(output)
    module_name = get_module_name(module)
    info = "Module: {:<20} Shape: {} -> {}"
    logger.info(info.format(module_name, input_shape, output_shape))


def reproduce(seed: int = 0, deterministic: bool = True):
    """
    WARN: This decorator has side effects on the entire process.
    Decorator to set the random seed for reproducibility.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True)

            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            try:
                import numpy as np
            except ImportError:
                logger.info("NumPy is not installed. Skipping NumPy seed setting.")
            else:
                np.random.seed(seed)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def is_close(a, b, rtol=1e-5, atol=1e-8) -> bool:
    """compare two tensors or tensor containers with logging"""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            logger.info(f"Length mismatch: {len(a)} != {len(b)}")
            return False
        return all(is_close(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            logger.info(f"Key mismatch: {set(a.keys())} != {set(b.keys())}")
            return False
        return all(is_close(a[k], b[k]) for k in a.keys())
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            logger.info(f"Shape mismatch: {a.shape} != {b.shape}")
            return False
        if not a.isclose(b, rtol, atol).all():
            logger.info(f"Value mismatch of two tensors")
            logger.debug(f"Difference: {a - b}")
            logger.info(f"Max difference: {torch.max(torch.abs(a - b))}")
            return False
        return True
    logger.info(f"Type mismatch or unsupported comparison of {type(a)} and {type(b)}")
    return False
