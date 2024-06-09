import jax
import flax.linen as nn

from collections.abc import Callable
from typing import Optional, Final, overload


_ACTIVATIONS: Final[dict[str, Callable[[jax.Array], jax.Array]]] = {
    'tanh': nn.tanh,
    'relu': nn.relu,
    'leaky_relu': nn.leaky_relu,
}


@overload
def activation(name: str) -> Callable[[jax.Array], jax.Array]:
    ...


@overload
def activation(name: None) -> None:
    ...


def activation(name: Optional[str]) -> Optional[Callable[[jax.Array], jax.Array]]:
    if name is None:
        return None
    else:
        return _ACTIVATIONS[name]
