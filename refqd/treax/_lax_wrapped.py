from jax import lax as _lax

import logging

from . import functional as _functional
from .core import TreeWrapped as _TreeWrapped


_log = logging.getLogger(__name__)
del logging


def __getattr__(name: str):
    _log.debug('__getattr__(%s)', name)
    if hasattr(_functional, name):
        obj = getattr(_functional, name)
    else:
        obj = getattr(_lax, name)
    if not callable(obj):
        raise AttributeError(name)

    return _TreeWrapped(obj)


__all__ = []
