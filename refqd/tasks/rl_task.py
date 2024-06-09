import jax
import jax.numpy as jnp
from qdax.core.neuroevolution.buffers import buffer
import qdax.tasks.brax_envs
from qdax.types import Genotype, Observation, Action, Fitness, Descriptor, ExtraScores

import numpy as np
import gym
import gymnasium

import logging
from functools import partial
from abc import abstractmethod
from collections.abc import Callable
from typing import Optional, TypeVar, Generic, Any

from .task import Task
from ..config.task import RLTaskConfig
from ..treax import numpy as tjnp
from ..treax import functional as F
from ..utils import RNGKey, jax_jit, onp_callback, jax_pure_callback, lax_scan


_EnvStateT = TypeVar('_EnvStateT')
_TransitionT = TypeVar('_TransitionT', bound=buffer.Transition)


@partial(jax_jit, static_argnames=('play_step_fn', 'episode_length'))
def _generate_unroll(
    init_state: _EnvStateT,
    representation_params: Genotype,
    decision_params: Genotype,
    rand: jax.Array,
    random_key: RNGKey,
    episode_length: int,
    play_step_fn: Callable[
        [_EnvStateT, Genotype, Genotype, RNGKey, jax.Array],
        tuple[
            _EnvStateT,
            Genotype,
            Genotype,
            RNGKey,
            jax.Array,
            _TransitionT,
        ],
    ],
) -> tuple[_EnvStateT, _TransitionT]:

    def _scan_play_step_fn(
        carry: tuple[_EnvStateT, Genotype, Genotype, RNGKey, jax.Array], _: Any
    ) -> tuple[
        tuple[_EnvStateT, Genotype, Genotype, RNGKey, jax.Array], _TransitionT
    ]:
        (
            env_state, representation_params, decision_params, random_key, rand, transitions
        ) = play_step_fn(*carry)
        return (env_state, representation_params, decision_params, random_key, rand), transitions

    (state, _, _, _, _), transitions = lax_scan(
        _scan_play_step_fn,
        (init_state, representation_params, decision_params, random_key, rand),
        (),
        length=episode_length,
    )
    return state, transitions


@partial(
    jax_jit,
    static_argnames=(
        'episode_length',
        'play_step_fn',
        'behavior_descriptor_extractor',
        'task',
        'map_states',
    ),
)
def scoring_function_jnp_envs(
    representation_params: Genotype,
    decision_params: Genotype,
    random_key: RNGKey,
    init_states: _EnvStateT,
    episode_length: int,
    play_step_fn: Callable[
        [_EnvStateT, Genotype, Genotype, RNGKey, jax.Array],
        tuple[_EnvStateT, Genotype, Genotype, RNGKey, jax.Array, buffer.QDTransition],
    ],
    behavior_descriptor_extractor: Callable[[buffer.QDTransition, jax.Array], Descriptor],
    task: 'RLTask',
    map_states: bool = False,
) -> tuple[Fitness, Descriptor, ExtraScores, RNGKey]:

    # Perform rollouts with each policy
    random_key, subkey = jax.random.split(random_key)
    unroll_fn = partial(
        _generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        random_key=subkey,
    )

    batch_size = jax.tree_util.tree_leaves(decision_params)[0].shape[0]
    if not map_states:
        init_states = tjnp.duplicate(init_states, batch_size)

    random_key, subkey = jax.random.split(random_key)
    rand = jax.random.uniform(subkey, (batch_size,))
    _, rand = jax.vmap(task.reduce_init_state)(init_states, rand)

    _, data = jax.vmap(unroll_fn, in_axes=(0, None, 0, 0))(
        init_states, representation_params, decision_params, rand
    )

    # create a mask to extract data properly
    mask = qdax.tasks.brax_envs.get_mask_from_transitions(data)

    # scores
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
    descriptors = behavior_descriptor_extractor(data, mask)

    return (
        fitnesses,
        descriptors,
        {
            'transitions': data,
        },
        random_key,
    )


@partial(
    jax_jit,
    static_argnames=(
        'episode_length',
        'play_reset_fn',
        'play_step_fn',
        'behavior_descriptor_extractor',
        'task',
    ),
)
def scoring_function_onp_envs(
    representation_params: Genotype,
    decision_params: Genotype,
    random_key: RNGKey,
    init_key: RNGKey,
    episode_length: int,
    play_reset_fn: Callable[[RNGKey, Optional[jax.Array]], _EnvStateT],
    play_step_fn: Callable[
        [_EnvStateT, Genotype, Genotype, RNGKey, jax.Array],
        tuple[_EnvStateT, Genotype, Genotype, RNGKey, jax.Array, buffer.QDTransition],
    ],
    behavior_descriptor_extractor: Callable[[buffer.QDTransition, jax.Array], Descriptor],
    task: 'RLTask',
) -> tuple[Fitness, Descriptor, ExtraScores, RNGKey]:

    random_key, subkey = jax.random.split(random_key)
    batch_size = jax.tree_util.tree_leaves(decision_params)[0].shape[0]
    keys = jax.random.split(
        subkey, batch_size
    )
    reset_fn = jax.vmap(play_reset_fn)
    init_key = F.duplicate(init_key, repeats=batch_size)
    init_states = reset_fn(init_key, keys)

    fitnesses, descriptors, extra_scores, random_key = scoring_function_jnp_envs(
        representation_params=representation_params,
        decision_params=decision_params,
        random_key=random_key,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=behavior_descriptor_extractor,
        task=task,
        map_states=True,
    )

    return fitnesses, descriptors, extra_scores, random_key


_RLTaskConfigT = TypeVar('_RLTaskConfigT', bound=RLTaskConfig)
_EnvT = TypeVar('_EnvT')


class RLTask(Task[_RLTaskConfigT], Generic[_RLTaskConfigT, _EnvT, _EnvStateT]):

    def __init__(self, cfg: _RLTaskConfigT, batch_shape: tuple[int, ...]) -> None:
        super().__init__(cfg, batch_shape)
        self._env: _EnvT
        self._select_action_fn: Callable[[Genotype, Genotype, Observation], Action]
        self._onp_obs: list[np.ndarray] = []

    def set_select_action_fn(
        self, select_action_fn: Callable[[Genotype, Genotype, Observation], Action]
    ) -> None:
        self._select_action_fn = select_action_fn

    @property
    def env_name(self) -> str:
        return self._cfg.name.replace('--', '/').split('__')[0]

    @property
    def env(self) -> _EnvT:
        return self._env

    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def obs_space(self) -> gym.Space | gymnasium.Space:
        ...

    @property
    @abstractmethod
    def action_space(self) -> gym.Space | gymnasium.Space:
        ...

    @property
    def observation_size(self) -> int:
        shape = self.obs_space.shape
        assert shape is not None
        return int(np.prod(shape))

    @property
    def action_size(self) -> int:
        shape = self.action_space.shape
        assert shape is not None
        return int(np.prod(shape))

    @property
    @abstractmethod
    def state_descriptor_length(self) -> int:
        ...

    def _flatten_obs(self, obs: Observation) -> Observation:
        shape = self.obs_space.shape
        assert shape is not None
        return obs.reshape(*obs.shape[:-len(shape)], self.observation_size)

    def _unflatten_action(self, action: Action) -> Action:
        shape = self.action_space.shape
        assert shape is not None
        return action.reshape(*action.shape[:-1], *shape)

    @abstractmethod
    def obs(self, state: _EnvStateT) -> Observation:
        ...

    @abstractmethod
    def reset(self, random_key: RNGKey, extra: Optional[jax.Array] = None) -> _EnvStateT:
        ...

    @abstractmethod
    def step(self, state: _EnvStateT, action: Action) -> _EnvStateT:
        ...

    @onp_callback
    def _onp_reduce_init_obs(
        self, obs: np.ndarray, rand: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(self._onp_obs) >= 2 * (self._cfg.episode_len + 1):
            self._onp_obs = self._onp_obs[self._cfg.episode_len + 1:]
        self._onp_obs.append(obs)
        res = obs[..., :1], rand
        return res

    def reduce_init_state(
        self, state: _EnvStateT, rand: jax.Array
    ) -> tuple[Observation, jax.Array]:
        obs = self._flatten_obs(self.obs(state))
        if not self._cfg.reduce_obs:
            return obs, rand
        obs, rand = jax_pure_callback(
            self._onp_reduce_init_obs,
            (obs[..., :1], rand),
            obs, rand,
            vectorized=True,
        )
        return obs, rand

    @onp_callback
    def _onp_reduce_obs(self, obs: np.ndarray, rand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._onp_obs.append(obs)
        res = obs[..., :1], rand
        return res

    def reduce_obs(self, obs: Observation, rand: jax.Array) -> tuple[jax.Array, jax.Array]:
        if not self._cfg.reduce_obs:
            return obs, rand
        obs, rand = jax_pure_callback(
            self._onp_reduce_obs,
            (obs[..., :1], rand),
            obs, rand,
            vectorized=True,
        )
        return obs, rand

    def reduce_transitions(
        self, transitions: buffer.QDTransition, rand: jax.Array
    ) -> tuple[buffer.QDTransition, jax.Array]:
        next_obs, rand = self.reduce_obs(transitions.next_obs, rand)
        return transitions.replace(obs=next_obs, next_obs=next_obs), rand

    def onp_restore_transitions(self, flattened_transitions: np.ndarray) -> np.ndarray:
        if not self._cfg.reduce_obs:
            return flattened_transitions
        onp_obs = self._onp_obs[-self._cfg.episode_len - 1:]
        self._onp_obs = self._onp_obs[:-self._cfg.episode_len - 1]
        return np.asarray(_jit_restore_transitions(
            flattened_transitions, onp_obs, self._cfg.episode_len  # type: ignore
        ))

    @abstractmethod
    def extract_behavior_descriptor(
        self, transition: buffer.QDTransition, mask: jax.Array
    ) -> Descriptor:
        ...


@partial(jax_jit, static_argnames=('episode_len',), donate_argnums=(0, 1), backend='cpu')
def _jit_restore_transitions(
    flattened_transitions: jax.Array, onp_obs: list[jax.Array], episode_len: int
) -> jax.Array:
    obs = jnp.stack(onp_obs, axis=-2)
    obs = jnp.concatenate((obs[..., :-1, :], obs[..., 1:, :]), axis=-1)
    shape = obs.shape
    obs = obs.reshape(*shape[:-3], shape[-3] * shape[-2], shape[-1])
    flattened_transitions = jnp.concatenate((obs, flattened_transitions[..., 2:]), axis=-1)
    return flattened_transitions
