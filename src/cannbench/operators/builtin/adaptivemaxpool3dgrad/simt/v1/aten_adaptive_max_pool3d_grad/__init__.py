from importlib import import_module

try:
    _C = import_module(f"{__name__}._C")
except ImportError:
    _C = None

from . import ops

__all__ = ["ops"]
