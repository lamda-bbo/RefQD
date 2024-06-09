import jax
import jax.numpy as jnp
import flax.struct
from qdax.core.neuroevolution.buffers import buffer

import numpy as np

import logging
import time
from functools import partial
from collections.abc import Callable
from typing import Self, TYPE_CHECKING

from ..utils import RNGKey, jax_jit, onp_callback, jax_pure_callback

if TYPE_CHECKING:
    from ..tasks import RLTask


_log = logging.getLogger(__name__)


global_buffer_data: list[np.ndarray] = []
global_restore_fn: list[Callable[[np.ndarray], np.ndarray]] = []
global_time_restore: float = 0.0
global_time_insert: float = 0.0
global_time_sample: float = 0.0


@onp_callback
def _update_global_buffer_data(
    fake_data: np.ndarray,
    flattened_transitions: np.ndarray,
    roll: np.ndarray,
    new_position: np.ndarray,
):
    global global_buffer_data
    global global_time_restore
    global global_time_insert
    global_time_insert -= time.monotonic()
    global_idx = int(fake_data.flatten()[0])
    replace_all = flattened_transitions.shape[-2] == global_buffer_data[global_idx].shape[-2]
    if replace_all:
        global_buffer_data[global_idx] = None  # type: ignore
    global_time_restore -= time.monotonic()
    flattened_transitions = global_restore_fn[global_idx](flattened_transitions)
    global_time_restore += time.monotonic()
    if replace_all:
        global_buffer_data[global_idx] = flattened_transitions
    else:
        global_buffer_data[global_idx] = _do_update_global_buffer_data(
            global_buffer_data[global_idx],
            flattened_transitions,
            roll.flatten()[0],
            new_position.flatten()[0],
        )
    global_time_insert += time.monotonic()
    return fake_data


def _do_update_global_buffer_data(
    buffer_data: np.ndarray,
    flattened_transitions: np.ndarray,
    roll: np.ndarray,
    new_position: np.ndarray,
):
    if roll != 0:
        _log.warning('Rolling...')
        buffer_data = np.roll(buffer_data, roll, axis=-2)
    buffer_data[
        ..., new_position:new_position + flattened_transitions.shape[-2], :
    ] = flattened_transitions
    return buffer_data


@onp_callback
def _take_from_global_buffer(data: np.ndarray, idx: np.ndarray):
    global global_time_sample
    global_time_sample -= time.monotonic()
    global_idx = int(data.flatten()[0])
    taken = np.vectorize(np.take, excluded=('axis', 'mode'), signature='(i,j),(k)->(k,j)')(
        global_buffer_data[global_idx], idx, axis=-2, mode='clip'
    )
    global_time_sample += time.monotonic()
    return taken


class CPUReplayBuffer(buffer.ReplayBuffer):

    flatten_dim: int = flax.struct.field(pytree_node=False)

    @classmethod
    def init(  # pyright: ignore [reportIncompatibleMethodOverride]
        cls,
        buffer_size: int,
        transition: buffer.Transition,
        rand: jax.Array,
        task: 'RLTask',
    ) -> Self:
        data = jnp.zeros((1, 1), dtype=jnp.float32)

        @onp_callback
        def onp_fn(data: np.ndarray, rand: np.ndarray):
            global global_buffer_data
            for size in reversed(rand.shape):
                data = np.repeat(np.expand_dims(data, axis=0), repeats=size, axis=0)
            shape = data.shape
            global_idx = len(global_buffer_data)
            _log.info(f'init.onp_fn: {global_idx}')
            data = np.zeros(
                (*data.shape[:-2], buffer_size, transition.flatten_dim), dtype=np.float32
            )
            global_buffer_data.append(data)
            global_restore_fn.append(task.onp_restore_transitions)
            return np.full(shape, global_idx, dtype=np.float32)

        data = jax_pure_callback(onp_fn, data, data, rand, vectorized=True)
        current_size = jnp.array(0, dtype=int)
        current_position = jnp.array(0, dtype=int)
        return cls(
            data=data,
            current_size=current_size,
            current_position=current_position,
            buffer_size=buffer_size,
            flatten_dim=transition.flatten_dim,
            transition=transition,
        )

    @partial(jax_jit, static_argnames=('sample_size',))
    def sample(  # pyright: ignore [reportIncompatibleVariableOverride]
        self,
        random_key: RNGKey,
        sample_size: int,
    ) -> tuple[buffer.Transition, RNGKey]:
        random_key, subkey = jax.random.split(random_key)
        idx = jax.random.randint(
            subkey,
            shape=(sample_size,),
            minval=0,
            maxval=self.current_size,
        )

        samples = jax_pure_callback(
            _take_from_global_buffer,
            jax.ShapeDtypeStruct((sample_size, self.flatten_dim), dtype=jnp.float32),
            self.data,
            idx,
            vectorized=True,
        )
        assert isinstance(samples, jax.Array)
        transitions = self.transition.__class__.from_flatten(samples, self.transition)
        return transitions, random_key

    @jax_jit
    def insert(  # pyright: ignore [reportIncompatibleVariableOverride]
        self, transitions: buffer.Transition
    ) -> Self:
        flattened_transitions = transitions.flatten()
        flattened_transitions = flattened_transitions.reshape(
            (-1, flattened_transitions.shape[-1])
        )
        num_transitions = flattened_transitions.shape[0]
        max_replay_size = self.buffer_size

        # Make sure update is not larger than the maximum replay size.
        if num_transitions > max_replay_size:
            raise ValueError(
                'Trying to insert a batch of samples larger than the maximum replay '
                f'size. num_samples: {num_transitions}, '
                f'max replay size {max_replay_size}'
            )

        # get current position
        position = self.current_position

        # check if there is an overlap
        roll = jnp.minimum(0, max_replay_size - position - num_transitions)

        # update the position accordingly
        new_position = position + roll

        # replace old data by the new one
        new_data = jax_pure_callback(
            _update_global_buffer_data,
            self.data,
            self.data, flattened_transitions, roll, new_position,
            vectorized=True
        )

        # update the position and the size
        new_position = (new_position + num_transitions) % max_replay_size
        new_size = jnp.minimum(self.current_size + num_transitions, max_replay_size)

        # update the replay buffer
        replay_buffer = self.replace(
            current_position=new_position,
            current_size=new_size,
            data=new_data,
        )

        return replay_buffer
