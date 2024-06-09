import jax
import jax.numpy as jnp
import flax.linen as nn
from qdax.types import Observation, Action

from collections.abc import Sequence, Callable
from typing import Optional, Literal

from .mlp import MLP
from .cnn import CNN


class SingleQModule(nn.Module):

    conv_features: Sequence[int] = ()
    conv_kernel_sizes: Sequence[Sequence[int]] = ()

    mlp_layer_sizes: tuple[int, ...] = (256, 256,)

    cnn_input_shape: Sequence[int] = (-1,)

    conv_activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    conv_strides: Optional[Sequence[int | Sequence[int]]] = None
    conv_padding: Literal['SAME', 'VALID'] = 'VALID'

    mlp_activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    mlp_final_activation: Optional[Callable[[jax.Array], jax.Array]] = None

    @nn.compact
    def __call__(  # pyright: ignore [reportIncompatibleMethodOverride]
        self, obs: Observation, actions: Action
    ) -> jax.Array:
        hidden = CNN(
            conv_features=self.conv_features,
            conv_kernel_sizes=self.conv_kernel_sizes,
            conv_activation=self.conv_activation,
            conv_strides=self.conv_strides,
            conv_padding=self.conv_padding,
            mlp_layer_sizes=(),
            mlp_activation=self.mlp_activation,
            mlp_final_activation=self.mlp_final_activation,
            cnn_input_shape=self.cnn_input_shape,
        )(obs)
        hidden = jnp.concatenate([hidden, actions], axis=-1)
        q = MLP(
            layer_sizes=(*self.mlp_layer_sizes, 1),
            activation=self.mlp_activation,
            final_activation=self.mlp_final_activation,
        )(hidden)
        return q


class QModule(SingleQModule):

    conv_features: Sequence[int] = ()
    conv_kernel_sizes: Sequence[Sequence[int]] = ()

    mlp_layer_sizes: tuple[int, ...] = (256, 256,)

    cnn_input_shape: Sequence[int] = (-1,)

    conv_activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    conv_strides: Optional[Sequence[int | Sequence[int]]] = None
    conv_padding: Literal['SAME', 'VALID'] = 'VALID'

    mlp_activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    mlp_final_activation: Optional[Callable[[jax.Array], jax.Array]] = None

    n_critics: int = 2

    def setup(self) -> None:
        critics: list[SingleQModule] = []
        for _ in range(self.n_critics):
            critics.append(
                SingleQModule(
                    conv_features=self.conv_features,
                    conv_kernel_sizes=self.conv_kernel_sizes,
                    conv_strides=self.conv_strides,
                    mlp_layer_sizes=self.mlp_layer_sizes,
                    conv_activation=self.conv_activation,
                    conv_padding=self.conv_padding,
                    mlp_activation=self.mlp_activation,
                    mlp_final_activation=self.mlp_final_activation,
                    cnn_input_shape=self.cnn_input_shape,
                )
            )
        self.critics = critics

    def __call__(self, obs: Observation, actions: Action) -> jax.Array:
        res: list[jax.Array] = []
        for i in range(self.n_critics):
            res.append(self.critics[i](obs, actions))
        return jnp.concatenate(res, axis=-1)

    def q1(self, obs: Observation, actions: Action) -> jax.Array:
        return self.critics[0](obs, actions).squeeze(axis=-1)
