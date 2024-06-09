import jax
from jax import Array, ShapeDtypeStruct
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike

from collections.abc import Callable
from types import EllipsisType


def asis(a: 'Array') -> 'Array':
    return a


def shape(a: 'Array') -> tuple[int, ...]:
    return jnp.shape(a)


def getitem(
    a: 'Array',
    indices: 'None | int | slice | EllipsisType | Array | tuple[None | int | slice | EllipsisType | Array, ...]',  # noqa: E501
) -> 'Array':
    return a[indices]


def setitem(
    a: 'Array',
    b: 'Array',
    indices: 'None | int | slice | EllipsisType | Array | tuple[None | int | slice | EllipsisType | Array, ...]',  # noqa: E501
) -> 'Array':
    return a.at[indices].set(b)


def concatenate(
    *arrays: 'Array', axis: int | None = 0, dtype: 'DTypeLike | None' = None
) -> 'Array':
    return jnp.concatenate(arrays, axis=axis, dtype=dtype)


def column_stack(*tup: 'Array') -> 'Array':
    return jnp.column_stack(tup)


def dstack(*tup: 'Array', dtype: 'DTypeLike | None' = None) -> 'Array':
    return jnp.dstack(tup, dtype=dtype)


def hstack(*tup: 'Array', dtype: 'DTypeLike | None' = None) -> 'Array':
    return jnp.hstack(tup, dtype=dtype)


def stack(
    *arrays: 'Array', axis: int = 0, out: None = None, dtype: 'DTypeLike | None' = None
) -> 'Array':
    return jnp.stack(arrays, axis=axis, out=out, dtype=dtype)


def vstack(*tup: 'Array', dtype: 'DTypeLike | None' = None) -> 'Array':
    return jnp.vstack(tup, dtype=dtype)


def duplicate(a: 'Array', repeats: int, axis: int = 0) -> 'Array':
    return jnp.repeat(jnp.expand_dims(a, axis=axis), repeats=repeats, axis=axis)


def shape_dtype(a: 'Array') -> 'ShapeDtypeStruct':
    return ShapeDtypeStruct(a.shape, a.dtype)


def reversed_broadcast(a: 'Array', to: 'Array') -> 'Array':
    assert len(a.shape) <= len(to.shape)
    for i in range(len(a.shape)):
        assert a.shape[i] == to.shape[i] or a.shape[i] == 1 or to.shape[i] == 1
    return jnp.expand_dims(a, axis=tuple(range(-(len(to.shape) - len(a.shape)), 0)))


def reversed_broadcasted_where_x(
    condition: 'Array',
    x: 'Array',
    y: 'Array',
) -> 'Array':
    return jnp.where(reversed_broadcast(condition, x), x, y)


def reversed_broadcasted_where_y(
    condition: 'Array',
    x: 'Array',
    y: 'Array',
) -> 'Array':
    return jnp.where(reversed_broadcast(condition, y), x, y)


def distance(a: 'Array', b: 'Array') -> 'Array':
    assert a.shape[-1] == b.shape[-1]
    func = jnp.subtract
    for _ in range(len(b.shape) - 1):
        func = jax.vmap(func, in_axes=(None, 0))
    for _ in range(len(a.shape) - 1):
        func = jax.vmap(func, in_axes=(0, None))
    return jnp.sum(jnp.square(func(a, b)), axis=-1)


def pool(
    inputs: jax.Array,
    init: ArrayLike,
    reduce_fn: Callable[..., jax.Array],
    window_shape: tuple[int, ...],
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
) -> jax.Array:
    assert len(window_shape) == len(strides) == len(padding)
    batch_shape = inputs.shape[:-len(window_shape) - 1]
    single_shape = inputs.shape[-len(window_shape) - 1:-1]
    channels = inputs.shape[-1]

    # Add padding to the input array
    padded = jnp.pad(
        inputs, (*(((0, 0),) * len(batch_shape)), *padding, (0, 0)), constant_values=init
    )

    # Calculate output shape
    single_out_shape = tuple(
        (
            single_shape[window_i] - window_shape[window_i]
            + padding[window_i][0] + padding[window_i][1]
        )
        // strides[window_i] + 1
        for window_i in range(len(single_shape))
    )

    # Create a grid of indices for pooling windows
    ij = tuple(jnp.arange(single_out_size) for single_out_size in single_out_shape)
    ij = jnp.meshgrid(*ij, indexing='ij')

    # Calculate the indices for pooling windows
    ij = tuple(_ij.reshape(-1, *((1,) * len(ij))) for _ij in ij)
    kl = tuple(jnp.expand_dims(
        jnp.arange(window_shape[window_i]), axis=tuple(range(-len(batch_shape), 0))
    ).swapaxes(0, window_i) for window_i in range(len(single_shape)))

    # Calculate the indices for the input array
    rcs = tuple(
        ij[window_i] * strides[window_i] + kl[window_i]
        for window_i in range(len(single_shape))
    )

    # Extract pooling windows from the input array
    windows = padded[..., *rcs, :]

    # Perform maximum pooling
    pooled = reduce_fn(windows, axis=(-3, -2))

    # Reshape the output to the desired shape
    out = pooled.reshape((*batch_shape, *single_out_shape, channels))

    return out


def max_pool(
    inputs: jax.Array,
    window_shape: tuple[int, ...],
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
) -> jax.Array:
    '''
    Performs maximum pooling on the input array.

    Args:
        x (jax.Array): Input array of shape.
        window_shape (tuple): Shape of the pooling window.
        strides (tuple): Strides of the pooling operation.
        padding (tuple): Padding sizes.

    Returns:
        jax.Array: Output array after maximum pooling.
    '''
    return pool(inputs, -jnp.inf, jnp.max, window_shape, strides, padding)


def min_pool(
    inputs: jax.Array,
    window_shape: tuple[int, ...],
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
) -> jax.Array:
    '''
    Performs minimum pooling on the input array.

    Args:
        x (jax.Array): Input array of shape.
        window_shape (tuple): Shape of the pooling window.
        strides (tuple): Strides of the pooling operation.
        padding (tuple): Padding sizes.

    Returns:
        jax.Array: Output array after minimum pooling.
    '''
    return pool(inputs, jnp.inf, jnp.min, window_shape, strides, padding)
