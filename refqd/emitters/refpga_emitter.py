import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import flax.core.scope
import qdax.core.containers.mapelites_repertoire
import qdax.core.emitters.emitter
import qdax.core.emitters.standard_emitters
from qdax.core.neuroevolution.buffers import buffer
from qdax.types import Params, Fitness, Descriptor, ExtraScores

import gym.spaces
import gymnasium.spaces

from dataclasses import dataclass, field
from functools import partial
from collections.abc import Callable
from typing import Optional, TypeVar, Any, TYPE_CHECKING, cast

from .multi_emitter import RefEmitterState, RefMultiEmitter
from ..neuroevolution import make_se_td3_loss_fn, GenotypePair, QModule, CPUReplayBuffer, activation
from ..config.critic_net import SharedCNNCriticNetConfig
from ..treax import numpy as tjnp
from ..utils import (
    RNGKey, fnchain, assert_cast, jax_jit, jax_value_and_grad,
    lax_cond, lax_scan, optax_apply_updates,
)

if TYPE_CHECKING:
    from ..tasks import RLTask


@dataclass
class RefQPGConfig:
    env_batch_size: int = 100
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100

    critic_net: SharedCNNCriticNetConfig = field(default_factory=lambda: SharedCNNCriticNetConfig())

    replay_buffer_size: int = 1000000
    critic_learning_rate: float = 3e-4
    representation_learning_rate: float = 3e-4
    representation_lr_decay_rate: float = 1.0
    greedy_learning_rate: float = 3e-4
    decision_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    save_emitted_representation_params: bool = True
    num_decision_updating_representation: int = 100
    decision_factor: float = 1.0
    soft_tau_update: float = 0.005
    policy_delay: int = 2


class RefQPGEmitterState(RefEmitterState):
    critic_params: flax.core.scope.VariableDict
    target_critic_params: flax.core.scope.VariableDict
    critic_optimizer_state: optax.OptState
    representation_params: flax.core.scope.VariableDict
    emitted_representation_params: Optional[flax.core.scope.VariableDict]
    target_representation_params: flax.core.scope.VariableDict
    representation_opt_state: optax.OptState
    greedy_decision_params: flax.core.scope.VariableDict
    target_greedy_decision_params: flax.core.scope.VariableDict
    greedy_decision_opt_state: optax.OptState
    replay_buffer: buffer.ReplayBuffer
    random_key: RNGKey
    steps: jax.Array


_RefQPGEmitterStateT = TypeVar('_RefQPGEmitterStateT', bound=RefQPGEmitterState)


class RefQPGEmitter(qdax.core.emitters.emitter.Emitter):

    def __init__(
        self,
        config: RefQPGConfig,
        representation_net: nn.Module,
        decision_net: nn.Module,
        task: 'RLTask',
    ) -> None:
        self._cfg = config
        self._representation_net = representation_net
        self._decision_net = decision_net
        self._task = task

        obs_space = self._task.obs_space
        match obs_space:
            case gym.spaces.Box() | gymnasium.spaces.Box():
                cnn_input_shape = obs_space.shape
            case _:
                raise NotImplementedError(type(obs_space))
        assert self._cfg.critic_net.conv_padding in ('SAME', 'VALID')
        self._critic_network = QModule(
            conv_features=self._cfg.critic_net.conv_features,
            conv_kernel_sizes=self._cfg.critic_net.conv_kernel_sizes,
            conv_activation=activation(self._cfg.critic_net.conv_activation),
            conv_strides=self._cfg.critic_net.conv_strides,
            conv_padding=self._cfg.critic_net.conv_padding,
            mlp_layer_sizes=self._cfg.critic_net.mlp_hidden_layer_sizes,
            mlp_activation=activation(self._cfg.critic_net.mlp_activation),
            mlp_final_activation=activation(self._cfg.critic_net.mlp_final_activation),
            cnn_input_shape=cnn_input_shape,
            n_critics=2,
        )

        (
            self._policy_loss_fn, self._mixed_policy_loss_fn, self._critic_loss_fn
        ) = make_se_td3_loss_fn(
            representation_fn=fnchain(self._representation_net.apply, assert_cast(jax.Array)),
            decision_fn=fnchain(self._decision_net.apply, assert_cast(jax.Array)),
            critic_fn=fnchain(self._critic_network.apply, assert_cast(jax.Array)),
            q1_fn=fnchain(
                partial(self._critic_network.apply, method=self._critic_network.q1),
                assert_cast(jax.Array),
            ),
            reward_scaling=self._cfg.reward_scaling,
            discount=self._cfg.discount,
            noise_clip=self._cfg.noise_clip,
            policy_noise=self._cfg.policy_noise,
            decision_factor=self._cfg.decision_factor,
        )

        schedule = optax.exponential_decay(
            init_value=self._cfg.representation_learning_rate,
            transition_steps=self._cfg.num_critic_training_steps // self._cfg.policy_delay,
            decay_rate=self._cfg.representation_lr_decay_rate,
        )

        self._representation_optimizer = optax.adam(learning_rate=schedule)
        self._greedy_optimizer = optax.adam(
            learning_rate=self._cfg.greedy_learning_rate
        )
        self._critic_optimizer = optax.adam(
            learning_rate=self._cfg.critic_learning_rate
        )
        self._decision_optimizer = optax.adam(
            learning_rate=self._cfg.decision_learning_rate
        )

    @property
    def batch_size(self) -> int:
        return self._cfg.env_batch_size

    @property
    def use_all_data(self) -> bool:
        '''Whether to use all data or not when used along other emitters.

        RefQPGEmitter uses the transitions from the genotypes that were generated
        by other emitters.
        '''
        return True

    def init(
        self,
        init_genotypes: GenotypePair[flax.core.scope.VariableDict, flax.core.scope.VariableDict],
        random_key: RNGKey,
    ) -> tuple[RefQPGEmitterState, RNGKey]:

        observation_size = self._task.observation_size
        action_size = self._task.action_size
        descriptor_size = self._task.state_descriptor_length

        representation_params, init_decision_params = init_genotypes
        del init_genotypes
        target_representation_params = tjnp.asis(representation_params)

        if self._cfg.save_emitted_representation_params:
            emitted_representation_params = tjnp.asis(representation_params)
        else:
            emitted_representation_params = None

        # Initialise critic, greedy actor and population
        random_key, subkey = jax.random.split(random_key)
        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_action = jnp.zeros(shape=(action_size,))
        critic_params = self._critic_network.init(
            subkey, obs=fake_obs, actions=fake_action
        )
        target_critic_params = tjnp.asis(critic_params)

        greedy_decision_params = tjnp.getitem(init_decision_params, 0)
        target_greedy_decision_params = tjnp.asis(greedy_decision_params)

        # Prepare init optimizer states
        representation_optimizer_state = self._representation_optimizer.init(representation_params)
        critic_optimizer_state = self._critic_optimizer.init(critic_params)
        greedy_optimizer_state = self._greedy_optimizer.init(greedy_decision_params)

        # Initialize replay buffer
        dummy_transition = buffer.QDTransition.init_dummy(
            observation_dim=observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )

        replay_buffer = CPUReplayBuffer.init(
            buffer_size=self._cfg.replay_buffer_size,
            transition=dummy_transition,
            rand=jax.random.uniform(random_key),
            task=self._task,
        )

        # Initial training state
        random_key, subkey = jax.random.split(random_key)
        emitter_state = RefQPGEmitterState(
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            critic_optimizer_state=critic_optimizer_state,
            representation_params=representation_params,
            emitted_representation_params=emitted_representation_params,
            target_representation_params=target_representation_params,
            representation_opt_state=representation_optimizer_state,
            greedy_decision_params=greedy_decision_params,
            target_greedy_decision_params=target_greedy_decision_params,
            greedy_decision_opt_state=greedy_optimizer_state,
            replay_buffer=replay_buffer,
            random_key=subkey,
            steps=jnp.array(0),
        )

        return emitter_state, random_key

    @partial(jax_jit, static_argnames=('self',))
    def emit(
        self,
        repertoire: qdax.core.containers.mapelites_repertoire.MapElitesRepertoire,
        emitter_state: RefQPGEmitterState,
        random_key: RNGKey,
    ) -> tuple[GenotypePair[flax.core.scope.VariableDict, flax.core.scope.VariableDict], RNGKey]:
        batch_size = self._cfg.env_batch_size - 1
        decision_params, random_key = repertoire.sample(random_key, batch_size)
        decision_params = cast(flax.core.scope.VariableDict, decision_params)

        random_key, subkey = jax.random.split(random_key)
        decision_params = jax.vmap(
            self._mutation_function_pg,
            in_axes=(0, None, None),
        )(decision_params, emitter_state, subkey)

        decision_params = tjnp.concatenate(
            decision_params, tjnp.getitem(emitter_state.greedy_decision_params, None)
        )

        params = GenotypePair(emitter_state.representation_params, decision_params)

        return params, random_key

    @partial(jax_jit, static_argnames=('self',))
    def state_update(
        self,
        emitter_state: _RefQPGEmitterStateT,
        repertoire: qdax.core.containers.mapelites_repertoire.MapElitesRepertoire,
        genotypes: Optional[
            GenotypePair[flax.core.scope.VariableDict, flax.core.scope.VariableDict]
        ],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> _RefQPGEmitterStateT:
        assert 'transitions' in extra_scores.keys(), 'Missing transitions or wrong key'
        transitions = extra_scores['transitions']
        assert isinstance(transitions, buffer.Transition)

        replay_buffer = emitter_state.replay_buffer.insert(transitions)

        if self._cfg.save_emitted_representation_params:
            emitted_representation_params = tjnp.asis(emitter_state.representation_params)
        else:
            emitted_representation_params = None

        emitter_state = emitter_state.replace(
            emitted_representation_params=emitted_representation_params,
            replay_buffer=replay_buffer,
            steps=jnp.array(0),
        )

        def scan_train(
            emitter_state: _RefQPGEmitterStateT, _: Any
        ) -> tuple[_RefQPGEmitterStateT, None]:
            emitter_state = self._train(emitter_state, repertoire)
            return emitter_state, None

        emitter_state, _ = lax_scan(
            scan_train,
            emitter_state,
            None,
            length=self._cfg.num_critic_training_steps,
        )

        return emitter_state

    def update_repertoire(
        self,
        emitter_state: _RefQPGEmitterStateT,
        repertoire: qdax.core.containers.mapelites_repertoire.MapElitesRepertoire,
    ) -> tuple[_RefQPGEmitterStateT, qdax.core.containers.mapelites_repertoire.MapElitesRepertoire]:
        raise NotImplementedError

    @partial(jax_jit, static_argnames=('self',))
    def _train(
        self,
        emitter_state: _RefQPGEmitterStateT,
        repertoire: qdax.core.containers.mapelites_repertoire.MapElitesRepertoire,
    ) -> _RefQPGEmitterStateT:
        random_key = emitter_state.random_key
        transitions, random_key = emitter_state.replay_buffer.sample(
            random_key, sample_size=self._cfg.batch_size
        )

        critic_params = emitter_state.critic_params
        target_critic_params = emitter_state.target_critic_params
        critic_optimizer_state = emitter_state.critic_optimizer_state

        random_key, subkey = jax.random.split(random_key)
        _critic_loss, critic_gradient = jax_value_and_grad(self._critic_loss_fn)(
            critic_params,
            emitter_state.target_representation_params,
            emitter_state.target_greedy_decision_params,
            target_critic_params,
            transitions,
            subkey,
        )
        updates, critic_optimizer_state = self._critic_optimizer.update(
            critic_gradient, critic_optimizer_state
        )
        del critic_gradient
        critic_params = optax_apply_updates(critic_params, updates)
        del updates

        target_critic_params = jax.tree_map(
            lambda x1, x2: (1.0 - self._cfg.soft_tau_update) * x1 + self._cfg.soft_tau_update * x2,
            target_critic_params,
            critic_params,
        )

        emitter_state = emitter_state.replace(
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            critic_optimizer_state=critic_optimizer_state,
            random_key=random_key,
        )

        (
            representation_params,
            target_representation_params,
            representation_opt_state,
            greedy_decision_params,
            target_greedy_decision_params,
            greedy_decision_opt_state,
            random_key,
        ) = lax_cond(
            emitter_state.steps % self._cfg.policy_delay == 0,
            self._update_actor,
            lambda *_: (
                emitter_state.representation_params,
                emitter_state.target_representation_params,
                emitter_state.representation_opt_state,
                emitter_state.greedy_decision_params,
                emitter_state.target_greedy_decision_params,
                emitter_state.greedy_decision_opt_state,
                emitter_state.random_key,
            ),
            emitter_state,
            repertoire,
            transitions,
        )

        emitter_state = emitter_state.replace(
            representation_params=representation_params,
            target_representation_params=target_representation_params,
            representation_opt_state=representation_opt_state,
            greedy_decision_params=greedy_decision_params,
            target_greedy_decision_params=target_greedy_decision_params,
            greedy_decision_opt_state=greedy_decision_opt_state,
            random_key=random_key,
            steps=emitter_state.steps + 1,
        )
        return emitter_state

    @partial(jax_jit, static_argnames=('self',))
    def _update_actor(
        self,
        emitter_state: RefQPGEmitterState,
        repertoire: qdax.core.containers.mapelites_repertoire.MapElitesRepertoire,
        transitions: buffer.QDTransition,
    ):
        representation_params = emitter_state.representation_params
        target_representation_params = emitter_state.target_representation_params
        representation_opt_state = emitter_state.representation_opt_state
        greedy_decision_params = emitter_state.greedy_decision_params
        target_greedy_decision_params = emitter_state.target_greedy_decision_params
        greedy_decision_opt_state = emitter_state.greedy_decision_opt_state
        random_key = emitter_state.random_key

        decision_params, random_key = repertoire.sample(
            random_key, self._cfg.num_decision_updating_representation
        )

        (
            _loss, (representation_gradient, greedy_decision_gradient)
        ) = jax_value_and_grad(self._mixed_policy_loss_fn, argnums=(0, 1))(
            representation_params,
            greedy_decision_params,
            decision_params,
            emitter_state.critic_params,
            transitions,
        )

        (
            greedy_decision_updates, greedy_decision_opt_state
        ) = self._greedy_optimizer.update(
            greedy_decision_gradient, greedy_decision_opt_state
        )
        del greedy_decision_gradient
        greedy_decision_params = optax_apply_updates(
            greedy_decision_params, greedy_decision_updates
        )
        del greedy_decision_updates

        (
            representation_updates,
            representation_opt_state,
        ) = self._representation_optimizer.update(
            representation_gradient, representation_opt_state
        )
        del representation_gradient
        representation_params = optax_apply_updates(representation_params, representation_updates)
        del representation_updates

        target_representation_params, target_greedy_decision_params = jax.tree_map(
            lambda x1, x2: (1.0 - self._cfg.soft_tau_update) * x1
            + self._cfg.soft_tau_update * x2,
            (target_representation_params, target_greedy_decision_params),
            (representation_params, greedy_decision_params),
        )

        return (
            representation_params,
            target_representation_params,
            representation_opt_state,
            greedy_decision_params,
            target_greedy_decision_params,
            greedy_decision_opt_state,
            random_key,
        )

    @partial(jax_jit, static_argnames=('self',))
    def _mutation_function_pg(
        self,
        policy_params: flax.core.scope.VariableDict,
        emitter_state: RefQPGEmitterState,
        random_key: RNGKey,
    ) -> flax.core.scope.VariableDict:
        # Define new policy optimizer state
        policy_optimizer_state = self._decision_optimizer.init(policy_params)

        def scan_train_policy(
            carry: tuple[_RefQPGEmitterStateT, flax.core.scope.VariableDict, optax.OptState],
            random_key: RNGKey,
        ) -> tuple[tuple[_RefQPGEmitterStateT, flax.core.scope.VariableDict, optax.OptState], None]:
            emitter_state, policy_params, policy_optimizer_state = carry
            (
                policy_params,
                policy_optimizer_state,
            ) = self._train_policy(
                emitter_state,
                policy_params,
                policy_optimizer_state,
                random_key,
            )
            return (
                emitter_state,
                policy_params,
                policy_optimizer_state,
            ), None

        keys = jax.random.split(random_key, self._cfg.num_pg_training_steps)
        (emitter_state, policy_params, policy_optimizer_state,), _ = lax_scan(
            scan_train_policy,
            (emitter_state, policy_params, policy_optimizer_state),
            keys,
            length=self._cfg.num_pg_training_steps,
        )

        return policy_params

    @partial(jax_jit, static_argnames=('self',))
    def _train_policy(
        self,
        emitter_state: RefQPGEmitterState,
        desicion_params: flax.core.scope.VariableDict,
        optimizer_state: optax.OptState,
        random_key: RNGKey,
    ) -> tuple[flax.core.scope.VariableDict, optax.OptState]:
        transitions, _ = emitter_state.replay_buffer.sample(
            random_key, sample_size=self._cfg.batch_size
        )

        _loss, gradient = jax_value_and_grad(self._policy_loss_fn, argnums=1)(
            emitter_state.representation_params,
            desicion_params,
            emitter_state.critic_params,
            transitions,
        )
        updates, optimizer_state = self._decision_optimizer.update(gradient, optimizer_state)
        del gradient
        desicion_params = optax_apply_updates(desicion_params, updates)
        del updates

        return desicion_params, optimizer_state


@dataclass
class RefPGAMEConfig(RefQPGConfig):
    proportion_mutation_ga: float = 0.5


class RefPGAMEEmitter(RefMultiEmitter):
    def __init__(
        self,
        config: RefPGAMEConfig,
        representation_net: nn.Module,
        decision_net: nn.Module,
        task: 'RLTask',
        variation_fn: Callable[[Params, Params, RNGKey], tuple[Params, RNGKey]],
    ):
        self._config = config
        self._representation_net = representation_net
        self._decision_net = decision_net
        self._task = task
        self._variation_fn = variation_fn

        ga_batch_size = int(self._config.proportion_mutation_ga * config.env_batch_size)
        dqn_batch_size = config.env_batch_size - ga_batch_size

        refdqn_config = RefQPGConfig(
            env_batch_size=dqn_batch_size,
            num_critic_training_steps=self._config.num_critic_training_steps,
            num_pg_training_steps=self._config.num_pg_training_steps,
            critic_net=self._config.critic_net,
            replay_buffer_size=self._config.replay_buffer_size,
            critic_learning_rate=self._config.critic_learning_rate,
            representation_learning_rate=self._config.representation_learning_rate,
            representation_lr_decay_rate=self._config.representation_lr_decay_rate,
            greedy_learning_rate=self._config.greedy_learning_rate,
            decision_learning_rate=self._config.decision_learning_rate,
            noise_clip=self._config.noise_clip,
            policy_noise=self._config.policy_noise,
            discount=self._config.discount,
            reward_scaling=self._config.reward_scaling,
            batch_size=self._config.batch_size,
            save_emitted_representation_params=self._config.save_emitted_representation_params,
            num_decision_updating_representation=self._config.num_decision_updating_representation,
            decision_factor=self._config.decision_factor,
            soft_tau_update=self._config.soft_tau_update,
            policy_delay=self._config.policy_delay,
        )

        refdqn_emitter = RefQPGEmitter(
            config=refdqn_config,
            representation_net=representation_net,
            decision_net=decision_net,
            task=task,
        )

        ga_emitter = qdax.core.emitters.standard_emitters.MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_batch_size,
        )

        super().__init__(emitters=(refdqn_emitter, ga_emitter))
