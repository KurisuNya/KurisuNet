from copy import copy
from typing import Callable

from ..basic.types import Env
from ..config.types import CustomModule, Module
from ..constants import OUTPUT_MODULE_NAME
from ..net import OutputModule
from ..utils.logger import get_logger

logger = get_logger("Register")


def register_module(obj):
    """Decorator to register a module with __name__ as name."""
    ModuleRegister.register(obj.__name__, obj)
    return obj


def register_converter(obj):
    """Decorator to register a converter with __name__ as name."""
    ConverterRegister.register(obj.__name__, obj)
    return obj


class ConverterRegister:
    """Register for converters."""

    __converters = {}

    @staticmethod
    def register(name: str, converter: Callable):
        """Register a converter with a name."""
        if name in ConverterRegister.__converters:
            raise ValueError(f"Converter {name} is already registered")
        ConverterRegister.__converters[name] = converter
        logger.debug(f"Converter {name} registered successfully")

    @staticmethod
    def get(name: str) -> Callable:
        """Get a converter by name."""
        if name not in ConverterRegister.__converters:
            raise ValueError(f"Converter {name} is not registered")
        return ConverterRegister.__converters[name]

    @staticmethod
    def get_env() -> Env:
        """Get the environment of registered converters."""
        return copy(ConverterRegister.__converters)

    @staticmethod
    def has(name: str) -> bool:
        """Check if a converter is registered."""
        return name in ConverterRegister.__converters

    @staticmethod
    def clear():
        """Clear the registered converters."""
        ConverterRegister.__converters.clear()


RegisterModule = Module | CustomModule


class ModuleRegister:
    """Register for modules."""

    __builtins: dict[str, RegisterModule] = {OUTPUT_MODULE_NAME: OutputModule}
    __modules: dict[str, RegisterModule] = copy(__builtins)

    @staticmethod
    def register(name: str, module: RegisterModule):
        """Register a module with a name."""
        if name in ModuleRegister.__modules:
            raise ValueError(f"Module {name} is already registered")
        ModuleRegister.__modules[name] = module
        logger.debug(f"Module {name} registered successfully")

    @staticmethod
    def get(name: str) -> RegisterModule:
        """Get a module by name."""
        if name not in ModuleRegister.__modules:
            raise ValueError(f"Module {name} is not registered")
        return ModuleRegister.__modules[name]

    @staticmethod
    def get_env() -> Env:
        """Get the environment of registered modules."""
        return copy(ModuleRegister.__modules)

    @staticmethod
    def has(name: str) -> bool:
        """Check if a module is registered."""
        return name in ModuleRegister.__modules

    @staticmethod
    def clear():
        """Clear the registered modules."""
        ModuleRegister.__modules = copy(ModuleRegister.__builtins)
