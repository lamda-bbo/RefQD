import jax
import jax.numpy as jnp
from qdax.core.neuroevolution.buffers import buffer
from qdax.types import (
    Genotype, Observation, Action, Fitness, Descriptor, StateDescriptor, ExtraScores
)

import numpy as np
import envpool
import envpool.python.lax
import envpool.python.envpool
import envpool.python.gymnasium_envpool
import envpool.python.protocol
import gymnasium

import logging
from functools import partial
import time
from collections.abc import Callable
from typing import Optional, TypeVar, Generic, Any
from overrides import override

from .rl_task import RLTask, scoring_function_onp_envs
from .gym_task import GymOnpState, GymJnpState
from ..config.task import EnvPoolConfig
from ..treax import numpy as tjnp
from ..utils import RNGKey, assert_cast, onp_callback, jax_pure_callback


_log = logging.getLogger(__name__)


global_time_reset: float = 0.0
global_time_step: float = 0.0


class GymnasiumEnvPoolEnv(
    envpool.python.protocol.EnvPool,
    envpool.python.gymnasium_envpool.GymnasiumEnvPoolMixin,
    envpool.python.envpool.EnvPoolMixin,
    envpool.python.lax.XlaMixin,
    gymnasium.Env,
):
    _state_keys: list[str]
    _action_keys: list[str]


_EnvPoolConfigT = TypeVar('_EnvPoolConfigT', bound=EnvPoolConfig)


class EnvPoolTask(
    RLTask[_EnvPoolConfigT, tuple[GymnasiumEnvPoolEnv, ...], GymJnpState], Generic[_EnvPoolConfigT]
):

    def __init__(self, cfg: _EnvPoolConfigT, batch_shape: tuple[int, ...]) -> None:
        super().__init__(cfg, batch_shape)
        self._seeds = np.zeros((np.prod(self._batch_shape[:-1], dtype=np.int32),), dtype=np.int32)
        state = tjnp.asarray(self._onp_reset(np.empty(()), np.empty(())))
        self._state_shape_dtype = tjnp.shape_dtype(
            tjnp.getitem(state, indices=tuple([0] * len(self._batch_shape)))
        )
        observation_space = self.env[0].observation_space
        assert isinstance(observation_space, gymnasium.Space)
        self._observation_space = observation_space
        action_space = self.env[0].action_space
        assert isinstance(action_space, gymnasium.Space)
        self._action_space = action_space
        del self._env, self._seeds

    @property
    def _make_kwargs(self) -> dict[str, Any]:
        return {}

    @override
    def close(self) -> None:
        for env in self.env:
            env.close()

    @property
    @override
    def obs_space(self) -> gymnasium.Space:
        return self._observation_space

    @property
    @override
    def action_space(self) -> gymnasium.Space:
        return self._action_space

    def _build_onp_obs(self, obs: np.ndarray) -> np.ndarray:
        return obs

    def _build_onp_state(self, results: list[tuple]) -> GymOnpState:
        stacked_results: GymOnpState = jax.tree_map(
            lambda *x: np.stack(x),
            *results,
        )
        obs, reward, terminated, truncated, info = stacked_results
        reward = np.asarray(reward, dtype=np.float32)

        obs = self._build_onp_obs(obs)

        obs = obs.reshape(*self._batch_shape, -1)
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
        if hasattr(self, '_env'):
            del self._env
        self._env = tuple(envpool.make_gymnasium(
            self.env_name,
            num_envs=self._batch_shape[-1],
            seed=seed,
            max_episode_steps=self._cfg.episode_len,
            **self._make_kwargs,
        ) for seed in self._seeds)
        for env in self.env:
            env.async_reset()
        onp_state = self._build_onp_state([env.recv() for env in self.env])

        if self._cfg.reduce_obs:
            self._latest_onp_obs = onp_state[0]

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
        action = action.reshape(
            np.prod(self._batch_shape[:-1], dtype=np.int32),
            *action.shape[len(self._batch_shape) - 1:],
        )
        for i, env in enumerate(self.env):
            env.send(action[i])
        onp_state = self._build_onp_state([env.recv() for env in self.env])

        if self._cfg.reduce_obs:
            self._latest_onp_obs = onp_state[0]

        global_time_step += time.time()
        return onp_state

    @override
    def step(self, state: GymJnpState, action: Action) -> GymJnpState:
        action = self._unflatten_action(action)
        next_state = jax_pure_callback(
            self._onp_step, self._state_shape_dtype, action, vectorized=True
        )
        return self._build_jnp_state(next_state, action)

    @onp_callback
    @override
    def _onp_reduce_init_obs(
        self, obs: np.ndarray, rand: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(self._onp_obs) >= 2 * (self._cfg.episode_len + 1):
            _log.warning('Clearing self._onp_obs...')
            self._onp_obs = self._onp_obs[self._cfg.episode_len + 1:]
        self._onp_obs.append(self._latest_onp_obs)
        return obs, rand

    @override
    def reduce_init_state(
        self, state: GymJnpState, rand: jax.Array
    ) -> tuple[Observation, jax.Array]:
        obs = self.obs(state)
        if not self._cfg.reduce_obs:
            return obs, rand
        obs = obs[..., :1]
        obs, rand = jax_pure_callback(
            self._onp_reduce_init_obs,
            (obs, rand),
            obs, rand,
            vectorized=True,
        )
        return obs, rand

    @onp_callback
    @override
    def _onp_reduce_obs(self, obs: np.ndarray, rand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._onp_obs.append(self._latest_onp_obs)
        return obs, rand

    @override
    def reduce_obs(self, obs: Observation, rand: jax.Array) -> tuple[jax.Array, jax.Array]:
        if not self._cfg.reduce_obs:
            return obs, rand
        obs = obs[..., :1]
        obs, rand = jax_pure_callback(
            self._onp_reduce_obs,
            (obs, rand),
            obs, rand,
            vectorized=True,
        )
        return obs, rand

    @onp_callback
    def _onp_get_constant(self, random_key: RNGKey, seeds: np.ndarray) -> RNGKey:
        seeds = seeds.reshape(-1)
        assert seeds.shape[0] == np.prod(self._batch_shape[:-1], dtype=np.int32)
        _log.info('_onp_get_constant: %s', seeds)
        self._seeds = seeds
        return random_key

    @override
    def get_constant(self, random_key: RNGKey) -> RNGKey:
        random_key, subkey = jax.random.split(random_key)
        seeds = jax.random.randint(subkey, (), 0, 0x7FFFFFFF)
        random_key = jax.random.key_data(random_key)
        random_key = assert_cast(RNGKey, jax.random.wrap_key_data(jax_pure_callback(
            self._onp_get_constant, random_key, random_key, seeds, vectorized=True
        )))
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
            actions = self._select_action_fn(representation_params, decision_params, obs)
            _log.debug('actions.shape = %s', actions.shape)

            state_desc = state[-1]['state_descriptor']
            assert isinstance(state_desc, StateDescriptor)

            next_state = self.step(state, actions)
            next_obs, rewards, dones, truncations, info = next_state

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
