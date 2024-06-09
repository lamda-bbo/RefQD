import jax.numpy as jnp
from jax import lax
from jax._src.typing import DimSize, DType, DTypeLike, Shape
from jax._src.numpy.lax_numpy import dtype, PadValueLike
from jax._src.lax.lax import PrecisionLike

import numpy as np

from collections.abc import Sequence, Callable
import typing
from typing import Union, Optional, TypeAlias, TypeVar, Any
from types import EllipsisType


unused = (
    jnp,
    lax,
    DimSize, DType, DTypeLike, Shape,
    dtype, PadValueLike,
    PrecisionLike,
    np,
    Sequence, Callable,
    typing,
    Union, Optional, Any,
    EllipsisType,
)


Tree: TypeAlias = Any
TreeT = TypeVar('TreeT', bound=Tree, covariant=True)


del TypeAlias, TypeVar, unused
