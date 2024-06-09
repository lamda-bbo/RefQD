import jax
import jax.numpy as jnp
from qdax.core.neuroevolution.buffers import buffer
from qdax.types import (
    Genotype, Observation, Action, Fitness, Descriptor, StateDescriptor, ExtraScores
)
from chex import ArrayTree

import numpy as np
import gym
import gymnasium
import gym.vector
import gymnasium.vector
from chex import ArrayNumpyTree

import logging
from functools import partial
import time
from collections.abc import Callable
from typing import Optional, TypeVar, Generic, Any, cast
from overrides import override

from .rl_task import RLTask, scoring_function_onp_envs
from ..config.task import GymConfig
from ..treax import numpy as tjnp
from ..utils import RNGKey, onp_callback, jax_pure_callback


_log = logging.getLogger(__name__)


GymOnpState = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, ArrayNumpyTree]]
GymJnpState = tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, ArrayTree]]


global_time_reset: float = 0.0
global_time_step: float = 0.0


_GymConfigT = TypeVar('_GymConfigT', bound=GymConfig)
_GymEnvT = TypeVar('_GymEnvT', gym.vector.VectorEnv, gymnasium.vector.VectorEnv)


class GymTask(RLTask[_GymConfigT, _GymEnvT, GymJnpState], Generic[_GymConfigT, _GymEnvT]):

    def __init__(self, cfg: _GymConfigT, batch_shape: tuple[int, ...]) -> None:
        super().__init__(cfg, batch_shape)
        if self._cfg.gymnasium:
            make_fn = gymnasium.vector.make
        else:
            make_fn = gym.vector.make

        self._env = cast(_GymEnvT, make_fn(
            self.env_name,
            num_envs=int(np.prod(self._batch_shape)),
            max_episode_steps=self._cfg.episode_len,
            **self._make_kwargs,
        ))
        fake_seeds = np.zeros(self._batch_shape, dtype=np.int32)
        state = tjnp.asarray(self._onp_reset(fake_seeds, fake_seeds))
        self._state_shape_dtype = tjnp.shape_dtype(
            tjnp.getitem(state, indices=tuple([0] * len(self._batch_shape)))
        )

    @property
    def _make_kwargs(self) -> dict[str, Any]:
        return {}

    @override
    def close(self) -> None:
        self.env.close()

    @property
    @override
    def obs_space(self) -> gym.Space | gymnasium.Space:
        return self.env.single_observation_space

    @property
    @override
    def action_space(self) -> gym.Space | gymnasium.Space:
        return self.env.single_action_space

    def _build_onp_state(
        self,
        obs: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: dict[str, Any],
    ) -> GymOnpState:
        obs = obs.reshape(*self._batch_shape, *obs.shape[1:])
        reward = reward.reshape(self._batch_shape)
        terminated = terminated.reshape(self._batch_shape)
        truncated = truncated.reshape(self._batch_shape)

        done = np.logical_or(terminated, truncated)
        extracted_info = {}
        return obs, reward, done, truncated, extracted_info

    @onp_callback
    def _onp_reset(self, seeds: np.ndarray, _: np.ndarray) -> GymOnpState:
        global global_time_reset
        global_time_reset -= time.time()
        _log.info('_onp_reset')
        seeds = seeds.flatten()
        obs, info = self.env.reset(seed=seeds.tolist())
        batch_shape = self._batch_shape
        onp_state = self._build_onp_state(
            obs=obs,
            reward=np.zeros(batch_shape, dtype=np.float32),
            terminated=np.zeros(batch_shape, dtype=np.bool_),
            truncated=np.zeros(batch_shape, dtype=np.bool_),
            info=info,
        )
        global_time_reset += time.time()
        return onp_state

    def _build_jnp_state(self, state: GymJnpState, action: Action) -> GymJnpState:
        return state

    @override
    def obs(self, state: GymJnpState) -> Observation:
        return state[0]

    @override
    def reset(
        self, random_key: RNGKey, extra: Optional[jax.Array] = None
    ) -> GymJnpState:
        assert extra is not None
        seed = jax.random.randint(random_key, (), 0, 0x7FFFFFFF)
        state = jax_pure_callback(
            self._onp_reset, self._state_shape_dtype, seed, extra, vectorized=True
        )
        return self._build_jnp_state(state, jnp.zeros_like(self.action_space.sample()))

    @onp_callback
    def _onp_step(self, action: np.ndarray) -> GymOnpState:
        global global_time_step
        global_time_step -= time.time()
        action = action.reshape(np.prod(self._batch_shape), *action.shape[len(self._batch_shape):])
        results = self.env.step(action)
        assert results is not None
        obs, reward, terminated, truncated, info = results
        onp_state = self._build_onp_state(
            obs=obs,
            reward=np.asarray(reward, dtype=np.float32),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        global_time_step += time.time()
        return onp_state

    @override
    def step(self, state: GymJnpState, action: Action) -> GymJnpState:
        action = self._unflatten_action(action)
        next_state = jax_pure_callback(
            self._onp_step, self._state_shape_dtype, action, vectorized=True
        )
        return self._build_jnp_state(next_state, action)

    @override
    def get_constant(self, random_key: RNGKey) -> RNGKey:
        return random_key

    @override
    def get_scoring_fn(self) -> Callable[
        [Genotype, Genotype, RNGKey, Any],
        tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ]:

        def play_step_fn(
            state: GymJnpState,
            representation_params: Genotype,
            decision_params: Genotype,
            random_key: RNGKey,
            rand: jax.Array,
        ):
            assert self._select_action_fn is not None

            obs = state[0]
            _log.debug('obs.shape = %s', obs.shape)
            obs = self._flatten_obs(obs)
            _log.debug('obs.shape = %s', obs.shape)
            actions = self._select_action_fn(representation_params, decision_params, obs)
            _log.debug('actions.shape = %s', actions.shape)

            state_desc = state[-1]['state_descriptor']
            assert isinstance(state_desc, StateDescriptor)

            next_state = self.step(state, actions)
            next_obs, rewards, dones, truncations, info = next_state

            _log.debug('next_obs.shape = %s', next_obs.shape)
            next_obs = self._flatten_obs(next_obs)
            _log.debug('next_obs.shape = %s', next_obs.shape)
            next_state_desc = info['state_descriptor']
            assert isinstance(next_state_desc, StateDescriptor)

            transition = buffer.QDTransition(
                obs=obs,
                next_obs=next_obs,
                rewards=rewards,
                dones=dones,
                actions=actions,
                truncations=truncations,
                state_desc=state_desc,
                next_state_desc=next_state_desc,
            )

            transition, rand = self.reduce_transitions(transition, rand)

            return next_state, representation_params, decision_params, random_key, rand, transition

        scoring_fn = partial(
            scoring_function_onp_envs,
            episode_length=self._cfg.episode_len,
            play_reset_fn=self.reset,
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=self.extract_behavior_descriptor,
            task=self,
        )

        return scoring_fn
