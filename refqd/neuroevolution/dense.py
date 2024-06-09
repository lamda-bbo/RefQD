import jax
import jax.numpy as jnp
import flax.linen as nn


class Dense(nn.Dense):

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:

        if self.features == 0:
            _ = self.param(
                'kernel',
                nn.initializers.zeros_init(),
                (jnp.shape(inputs)[-1], self.features),
                self.param_dtype,
            )
            if self.use_bias:
                _ = self.param(
                    'bias', nn.initializers.zeros_init(), (self.features,), self.param_dtype
                )
            return inputs

        return super().__call__(inputs)
