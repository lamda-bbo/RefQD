
import jax
import brax.v1.envs
from qdax.core.neuroevolution.buffers import buffer
import qdax.environments
import qdax.tasks.brax_envs
from qdax.types import (
    Genotype, Observation, Action, Fitness, Descriptor, StateDescriptor, ExtraScores
)

import numpy as np
import gymnasium

import logging
from functools import partial
from collections.abc import Callable
from typing import Optional, TypeAlias, Any, cast
from overrides import override

from .rl_task import RLTask, scoring_function_jnp_envs
from ..config.task import QDaxBraxConfig
from ..utils import RNGKey, assert_cast, jax_jit


_log = logging.getLogger(__name__)


QDaxBraxEnvState: TypeAlias = brax.v1.envs.State


class QDaxBraxTask(RLTask[QDaxBraxConfig, qdax.environments.QDEnv, QDaxBraxEnvState]):

    def __init__(self, cfg: QDaxBraxConfig, batch_shape: tuple[int, ...]) -> None:
        super().__init__(cfg, batch_shape)
        env = qdax.environments.create(
            self._cfg.name,
            episode_length=self._cfg.episode_len,
        )
        self._env = cast(qdax.environments.QDEnv, env)

    @property
    @override
    def behavior_descriptor_length(self) -> int:
        return self.env.behavior_descriptor_length

    @property
    @override
    def behavior_descriptor_limits(self) -> tuple[list[float], list[float]]:
        return self.env.behavior_descriptor_limits

    @property
    @override
    def obs_space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(-np.inf, np.inf, shape=(self.env.observation_size,))

    @property
    @override
    def action_space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(-np.inf, np.inf, shape=(self.env.action_size,))

    @property
    @override
    def state_descriptor_length(self) -> int:
        return self.env.state_descriptor_length

    @property
    @override
    def qd_offset(self) -> float:
        return qdax.environments.reward_offset[self._cfg.name] * self._cfg.episode_len

    @override
    def obs(self, state: QDaxBraxEnvState) -> Observation:
        return assert_cast(Observation, state.obs)

    @override
    def reset(
        self, random_key: RNGKey, extra: Optional[jax.Array] = None
    ) -> QDaxBraxEnvState:
        return self.env.reset(random_key)

    @override
    def step(self, state: QDaxBraxEnvState, action: Action) -> QDaxBraxEnvState:
        return self.env.step(state, action)

    @override
    def get_constant(self, random_key: RNGKey) -> QDaxBraxEnvState:
        init_state = jax_jit(self.reset)(random_key)
        return init_state

    @override
    def extract_behavior_descriptor(
        self, transition: buffer.QDTransition, mask: jax.Array
    ) -> Descriptor:
        return qdax.environments.behavior_descriptor_extractor[self._cfg.name](transition, mask)

    @override
    def get_scoring_fn(self) -> Callable[
        [Genotype, Genotype, RNGKey, Any],
        tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ]:
        def play_step_fn(
            state: QDaxBraxEnvState,
            representation_params: Genotype,
            decision_params: Genotype,
            random_key: RNGKey,
            rand: jax.Array,
        ):
            assert self._select_action_fn is not None

            _log.debug('obs.shape = %s', state.obs.shape)
            actions = self._select_action_fn(
                representation_params, decision_params, assert_cast(Observation, state.obs)
            )
            _log.debug('actions.shape = %s', actions.shape)

            state_desc: StateDescriptor = state.info['state_descriptor']
            next_state = self.step(state, actions)

            transition = buffer.QDTransition(
                obs=state.obs,
                next_obs=next_state.obs,
                rewards=next_state.reward,
                dones=next_state.done,
                actions=actions,
                truncations=next_state.info['truncation'],
                state_desc=state_desc,
                next_state_desc=next_state.info['state_descriptor'],
            )

            transition, rand = self.reduce_transitions(transition, rand)

            return next_state, representation_params, decision_params, random_key, rand, transition

        scoring_fn = partial(
            scoring_function_jnp_envs,
            episode_length=self._cfg.episode_len,
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=self.extract_behavior_descriptor,
            task=self,
            map_states=False,
        )

        return scoring_fn
