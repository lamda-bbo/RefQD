import jax
import jax.numpy as jnp
from qdax.core.neuroevolution.buffers import buffer
from qdax.types import Action, Descriptor

import numpy as np
import gymnasium.spaces

from typing import Any
from overrides import override

from .gym_task import GymJnpState
from .envpool_task import EnvPoolTask
from ..config.task import AtariConfig


class AtariTask(EnvPoolTask[AtariConfig]):

    @override
    def __init__(self, cfg: AtariConfig, batch_shape: tuple[int, ...]) -> None:
        super().__init__(cfg, batch_shape)
        assert self._cfg.subtype == 'Atari'

    @property
    def _make_kwargs(self) -> dict[str, Any]:
        return {
            'img_height': self._cfg.obs_shape[0],
            'img_width': self._cfg.obs_shape[0],
            'stack_num': 1,
            'episodic_life': True,
        }

    @property
    @override
    def behavior_descriptor_length(self) -> int:
        match self._cfg.name:
            case 'Pong-v5':
                return 1
            case 'Boxing-v5':
                return 2
            case _:
                raise NotImplementedError(self._cfg.name)

    @property
    @override
    def behavior_descriptor_limits(self) -> tuple[list[float], list[float]]:
        match self._cfg.name:
            case 'Pong-v5':
                return ([0.0], [1.0])
            case 'Boxing-v5':
                return ([0.0, 0.0], [1.0, 1.0])
            case _:
                raise NotImplementedError(self._cfg.name)

    @property
    @override
    def qd_offset(self) -> float:
        match self._cfg.name:
            case 'Pong-v5':
                return 22.0
            case 'Boxing-v5':
                return 100.0
            case _:
                raise NotImplementedError(self._cfg.name)

    @property
    @override
    def obs_space(self) -> gymnasium.spaces.Box:
        obs_space = super().obs_space

        assert isinstance(obs_space, gymnasium.spaces.Box)
        return type(obs_space)(
            low=np.zeros(
                (obs_space.shape[-2], obs_space.shape[-1], obs_space.shape[-3]), dtype=np.float32
            ),
            high=np.ones(
                (obs_space.shape[-2], obs_space.shape[-1], obs_space.shape[-3]), dtype=np.float32
            ),
        )

    @property
    @override
    def state_descriptor_length(self) -> int:
        match self._cfg.name:
            case 'Pong-v5':
                return 1
            case 'Boxing-v5':
                return 2
            case _:
                raise NotImplementedError(self._cfg.name)

    @override
    def _build_onp_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.divide(obs, np.float32(255.0), dtype=np.float32)
        obs = np.moveaxis(obs, -3, -1)
        return obs

    @override
    def _build_jnp_state(self, state: GymJnpState, action: Action) -> GymJnpState:
        obs, reward, done, truncated, info = state

        match self._cfg.name:
            case 'Pong-v5':
                info['state_descriptor'] = jnp.expand_dims(
                    (action >= 2).astype(jnp.float32), axis=-1
                )
            case 'Boxing-v5':
                info['state_descriptor'] = jnp.stack((
                    jnp.logical_or(action == 1, action >= 10).astype(jnp.float32),
                    (action >= 2).astype(jnp.float32),
                ), axis=-1)

        return obs, reward, done, truncated, info

    @override
    def extract_behavior_descriptor(
        self, transition: buffer.QDTransition, mask: jax.Array
    ) -> Descriptor:
        match self._cfg.name:
            case 'Pong-v5' | 'Boxing-v5':
                # reshape mask for bd extraction
                mask = jnp.expand_dims(mask, axis=-1)
                # Get behavior descriptor
                descriptors = jnp.sum(transition.state_desc * (1.0 - mask), axis=1)
                descriptors = descriptors / jnp.sum(1.0 - mask, axis=1)
            case _:
                raise NotImplementedError(self._cfg.name)

        return descriptors
