from jax.lax import (
    RoundingMethod, ScatterDimensionNumbers, GatherDimensionNumbers, DotDimensionNumbers,
)
import jax._src.lax.lax
from jax._src.lax.lax import PrecisionLike
from jax._src.typing import DTypeLike, Shape

import numpy as np

import collections.abc
from collections.abc import Sequence, Callable
import typing
from typing import Union, Optional, TypeAlias, TypeVar, Any


unused = (
    RoundingMethod, ScatterDimensionNumbers, GatherDimensionNumbers, DotDimensionNumbers,
    jax._src.lax.lax,
    PrecisionLike,
    DTypeLike, Shape,
    np,
    collections.abc,
    Sequence, Callable,
    typing,
    Union, Optional, TypeAlias, TypeVar, Any
)


Tree: TypeAlias = Any
TreeT = TypeVar('TreeT', bound=Tree, covariant=True)


del TypeAlias, TypeVar, unused
