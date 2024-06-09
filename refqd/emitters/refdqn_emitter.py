import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import flax.core.scope
import qdax.core.containers.mapelites_repertoire
import qdax.core.emitters.emitter
import qdax.core.emitters.standard_emitters
from qdax.core.neuroevolution.buffers import buffer
from qdax.types import Params, Observation, Action, Fitness, Descriptor, ExtraScores

from dataclasses import dataclass
from functools import partial
from collections.abc import Callable
from typing import Optional, TypeVar, Any, TYPE_CHECKING, cast

from .multi_emitter import RefEmitterState, RefMultiEmitter
from ..neuroevolution import GenotypePair, make_se_ddqn_loss_fn, CPUReplayBuffer
from ..treax import numpy as tjnp
from ..utils import (
    RNGKey, fnchain, jax_jit, jax_value_and_grad, lax_cond, lax_scan, optax_apply_updates,
)

if TYPE_CHECKING:
    from ..tasks import RLTask


@dataclass
class RefDQNEmitterConfig:
    env_batch_size: int = 100
    num_dqn_training_steps: int = 300
    num_mutation_steps: int = 100
    replay_buffer_size: int = 200000
    representation_learning_rate: float = 3e-4
    representation_lr_decay_rate: float = 1.0
    greedy_learning_rate: float = 3e-4
    learning_rate: float = 1e-3
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 32
    save_emitted_representation_params: bool = True
    target_policy_update_interval: int = 10
    num_decision_updating_representation: int = 100
    decision_factor: float = 1.0
    using_greedy: bool = True


class RefDQNEmitterState(RefEmitterState):
    representation_params: flax.core.scope.VariableDict
    emitted_representation_params: Optional[flax.core.scope.VariableDict]
    target_representation_params: flax.core.scope.VariableDict
    greedy_decision_params: flax.core.scope.VariableDict
    target_greedy_decision_params: flax.core.scope.VariableDict

    representation_optimizer_state: optax.OptState
    greedy_decision_optimizer_state: optax.OptState

    replay_buffer: buffer.ReplayBuffer
    random_key: RNGKey
    step: jax.Array


_RefDQNEmitterStateT = TypeVar('_RefDQNEmitterStateT', bound=RefDQNEmitterState)


class RefDQNEmitter(qdax.core.emitters.emitter.Emitter):

    def __init__(
        self,
        config: RefDQNEmitterConfig,
        representation_net: nn.Module,
        decision_net: nn.Module,
        task: 'RLTask',
    ) -> None:
        self._config = config
        self._task = task
        self._representation_net = representation_net
        self._decision_net = decision_net

        def policy_fn(
            representation_params: flax.core.scope.VariableDict,
            decision_params: flax.core.scope.VariableDict,
            obs: Observation,
        ) -> Action:
            representation = self._representation_net.apply(representation_params, obs)
            assert isinstance(representation, jax.Array)
            action = self._decision_net.apply(decision_params, representation)
            assert isinstance(action, Action)
            return action

        self._loss_fn = make_se_ddqn_loss_fn(
            policy_fn,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
        )

        self._vmapped_loss_fn = fnchain(
            jax.vmap(self._loss_fn, in_axes=(None, 0, None, 0, None)),
            partial(jnp.sum, axis=0),
        )

        schedule = optax.exponential_decay(
            init_value=self._config.representation_learning_rate,
            transition_steps=self._config.num_dqn_training_steps,
            decay_rate=self._config.representation_lr_decay_rate,
        )

        self._representation_optimizer = optax.adam(learning_rate=schedule)
        self._greedy_decision_optimizer = optax.adam(
            learning_rate=self._config.greedy_learning_rate
        )
        self._optimizer = optax.adam(learning_rate=self._config.learning_rate)

    @property
    def batch_size(self) -> int:
        return self._config.env_batch_size

    @property
    def use_all_data(self) -> bool:
        '''Whether to use all data or not when used along other emitters.

        RefDQNEmitter uses the transitions from the genotypes that were generated
        by other emitters.
        '''
        return True

    def init(
        self,
        init_genotypes: GenotypePair[flax.core.scope.VariableDict, flax.core.scope.VariableDict],
        random_key: RNGKey,
    ) -> tuple[RefDQNEmitterState, RNGKey]:
        observation_size = self._task.observation_size
        action_size = self._task.action_size
        descriptor_size = self._task.behavior_descriptor_length

        representation_params, init_decision_params = init_genotypes
        target_representation_params = tjnp.asis(representation_params)

        if self._config.save_emitted_representation_params:
            emitted_representation_params = tjnp.asis(representation_params)
        else:
            emitted_representation_params = None

        greedy_decision_params = tjnp.getitem(init_decision_params, 0)
        target_greedy_decision_params = tjnp.asis(greedy_decision_params)

        representation_optimizer_state = self._representation_optimizer.init(
            representation_params
        )
        greedy_decision_optimizer_state = self._greedy_decision_optimizer.init(
            greedy_decision_params
        )

        dummy_transition = buffer.QDTransition.init_dummy(
            observation_dim=observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )

        replay_buffer = CPUReplayBuffer.init(
            buffer_size=self._config.replay_buffer_size,
            transition=dummy_transition,
            rand=jax.random.uniform(random_key),
            task=self._task,
        )

        random_key, subkey = jax.random.split(random_key)
        emitter_state = RefDQNEmitterState(
            representation_params=representation_params,
            emitted_representation_params=emitted_representation_params,
            target_representation_params=target_representation_params,
            greedy_decision_params=greedy_decision_params,
            target_greedy_decision_params=target_greedy_decision_params,
            representation_optimizer_state=representation_optimizer_state,
            greedy_decision_optimizer_state=greedy_decision_optimizer_state,
            replay_buffer=replay_buffer,
            random_key=subkey,
            step=jnp.zeros((), dtype=jnp.int32),
        )

        return emitter_state, random_key

    @partial(jax_jit, static_argnames=('self',))
    def emit(
        self,
        repertoire: qdax.core.containers.mapelites_repertoire.MapElitesRepertoire,
        emitter_state: RefDQNEmitterState,
        random_key: RNGKey,
    ) -> tuple[GenotypePair[flax.core.scope.VariableDict, flax.core.scope.VariableDict], RNGKey]:
        batch_size = self._config.env_batch_size - self._config.using_greedy
        decision_params, random_key = repertoire.sample(random_key, batch_size)
        decision_params = cast(flax.core.scope.VariableDict, decision_params)

        random_key, subkey = jax.random.split(random_key)
        decision_params = jax.vmap(
            self._mutation_function,
            in_axes=(0, None, None, None),
        )(decision_params, emitter_state.representation_params, emitter_state.replay_buffer, subkey)

        if self._config.using_greedy:
            decision_params = tjnp.concatenate(
                decision_params, tjnp.getitem(emitter_state.greedy_decision_params, None)
            )

        params = GenotypePair(emitter_state.representation_params, decision_params)

        return params, random_key

    @partial(jax_jit, static_argnames=('self',))
    def state_update(  # pyright: ignore [reportIncompatibleVariableOverride]
        self,
        emitter_state: _RefDQNEmitterStateT,
        repertoire: qdax.core.containers.mapelites_repertoire.MapElitesRepertoire,
        genotypes: Optional[
            GenotypePair[flax.core.scope.VariableDict, flax.core.scope.VariableDict]
        ],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> _RefDQNEmitterStateT:
        assert 'transitions' in extra_scores.keys(), 'Missing transitions or wrong key'
        transitions = extra_scores['transitions']
        assert isinstance(transitions, buffer.Transition)

        if self._config.save_emitted_representation_params:
            emitted_representation_params = tjnp.asis(emitter_state.representation_params)
        else:
            emitted_representation_params = None

        replay_buffer = emitter_state.replay_buffer.insert(transitions)
        emitter_state = emitter_state.replace(
            emitted_representation_params=emitted_representation_params,
            replay_buffer=replay_buffer,
            step=jnp.array(0),
        )

        assert self._config.using_greedy

        def scan_train(
            emitter_state: _RefDQNEmitterStateT, _: Any
        ) -> tuple[_RefDQNEmitterStateT, None]:
            emitter_state = self._train(emitter_state, repertoire)
            return emitter_state, None

        emitter_state, _ = lax_scan(
            scan_train,
            emitter_state,
            None,
            length=self._config.num_dqn_training_steps,
        )

        return emitter_state

    @partial(jax_jit, static_argnames=('self',))
    def _train(
        self,
        emitter_state: _RefDQNEmitterStateT,
        repertoire: qdax.core.containers.mapelites_repertoire.MapElitesRepertoire,
    ) -> _RefDQNEmitterStateT:
        random_key = emitter_state.random_key
        replay_buffer = emitter_state.replay_buffer
        transitions, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )
        representation_params = emitter_state.representation_params
        target_representation_params = emitter_state.target_representation_params
        representation_optimizer_state = emitter_state.representation_optimizer_state
        greedy_decision_params = emitter_state.greedy_decision_params
        target_greedy_decision_params = emitter_state.target_greedy_decision_params
        greedy_decision_optimizer_state = emitter_state.greedy_decision_optimizer_state
        step = emitter_state.step

        decision_params, random_key = repertoire.sample(
            random_key, self._config.num_decision_updating_representation
        )

        (
            _greedy_loss, (greedy_representation_gradient, greedy_decision_gradient)
        ) = jax_value_and_grad(self._loss_fn, argnums=(0, 1))(
            representation_params,
            greedy_decision_params,
            target_representation_params,
            target_greedy_decision_params,
            transitions,
        )

        (
            greedy_decision_updates, greedy_decision_optimizer_state
        ) = self._greedy_decision_optimizer.update(
            greedy_decision_gradient, greedy_decision_optimizer_state
        )
        del greedy_decision_gradient
        greedy_decision_params = optax_apply_updates(
            greedy_decision_params, greedy_decision_updates
        )
        del greedy_decision_updates

        (
            _representation_loss, representation_gradient
        ) = jax_value_and_grad(self._vmapped_loss_fn)(
            representation_params,
            decision_params,
            target_representation_params,
            decision_params,
            transitions,
        )
        representation_gradient = jax.tree_map(
            lambda x1, x2: self._config.decision_factor * x1 + x2,
            representation_gradient,
            greedy_representation_gradient,
        )
        del greedy_representation_gradient

        (
            representation_updates,
            representation_optimizer_state,
        ) = self._representation_optimizer.update(
            representation_gradient, representation_optimizer_state
        )
        del representation_gradient
        representation_params = optax_apply_updates(representation_params, representation_updates)
        del representation_updates

        target_representation_params, target_greedy_decision_params = lax_cond(
            step % self._config.target_policy_update_interval == 0,
            lambda: (representation_params, greedy_decision_params),
            lambda: (target_representation_params, target_greedy_decision_params),
        )

        emitter_state = emitter_state.replace(
            random_key=random_key,
            representation_params=representation_params,
            target_representation_params=target_representation_params,
            representation_optimizer_state=representation_optimizer_state,
            greedy_decision_params=greedy_decision_params,
            target_greedy_decision_params=target_greedy_decision_params,
            greedy_decision_optimizer_state=greedy_decision_optimizer_state,
            step=step + 1,
        )

        return emitter_state

    @partial(jax_jit, static_argnames=('self',))
    def _mutation_function(
        self,
        decision_params: flax.core.scope.VariableDict,
        representation_params: flax.core.scope.VariableDict,
        replay_buffer: buffer.ReplayBuffer,
        random_key: RNGKey,
    ) -> flax.core.scope.VariableDict:
        target_decision_params = tjnp.asis(decision_params)
        optimizer_state = self._optimizer.init(decision_params)

        def scan_train_policy(
            carry: tuple[
                buffer.ReplayBuffer,
                flax.core.scope.VariableDict,
                flax.core.scope.VariableDict,
                optax.OptState,
            ],
            x: tuple[RNGKey, jax.Array],
        ) -> tuple[
            tuple[
                buffer.ReplayBuffer,
                flax.core.scope.VariableDict,
                flax.core.scope.VariableDict,
                optax.OptState,
            ],
            None,
        ]:
            replay_buffer, policy_params, target_policy_params, optimizer_state = carry
            random_key, update_target_policy = x
            (
                policy_params, target_policy_params, optimizer_state
            ) = self._train_policy(
                replay_buffer,
                policy_params,
                target_policy_params,
                optimizer_state,
                representation_params,
                random_key,
                update_target_policy,
            )
            return (
                replay_buffer, policy_params, target_policy_params, optimizer_state
            ), None

        keys = jax.random.split(random_key, self._config.num_mutation_steps)
        (replay_buffer, decision_params, target_decision_params, optimizer_state,), _ = lax_scan(
            scan_train_policy,
            (replay_buffer, decision_params, target_decision_params, optimizer_state,),
            (
                keys,
                jnp.arange(
                    1, self._config.num_mutation_steps + 1
                ) % self._config.target_policy_update_interval == 0,
            ),
            length=self._config.num_mutation_steps,
        )

        return decision_params

    @partial(jax_jit, static_argnames=('self',))
    def _train_policy(
        self,
        replay_buffer: buffer.ReplayBuffer,
        desicion_params: flax.core.scope.VariableDict,
        target_decision_params: flax.core.scope.VariableDict,
        optimizer_state: optax.OptState,
        representation_params: flax.core.scope.VariableDict,
        random_key: RNGKey,
        update_target: jax.Array,
    ) -> tuple[
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        optax.OptState,
    ]:
        transitions, _ = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

        _loss, gradient = jax_value_and_grad(self._loss_fn, argnums=1)(
            representation_params,
            desicion_params,
            representation_params,
            target_decision_params,
            transitions,
        )
        decision_updates, optimizer_state = self._optimizer.update(gradient, optimizer_state)
        del gradient
        desicion_params = optax_apply_updates(desicion_params, decision_updates)
        del decision_updates

        target_decision_params = lax_cond(
            update_target,
            lambda: desicion_params,
            lambda: target_decision_params,
        )

        return desicion_params, target_decision_params, optimizer_state


@dataclass
class RefDQNMEConfig(RefDQNEmitterConfig):
    proportion_mutation_ga: float = 0.5


class RefDQNMEEmitter(RefMultiEmitter):
    def __init__(
        self,
        config: RefDQNMEConfig,
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

        refdqn_config = RefDQNEmitterConfig(
            env_batch_size=dqn_batch_size,
            num_dqn_training_steps=self._config.num_dqn_training_steps,
            num_mutation_steps=self._config.num_mutation_steps,
            replay_buffer_size=self._config.replay_buffer_size,
            representation_learning_rate=self._config.representation_learning_rate,
            representation_lr_decay_rate=self._config.representation_lr_decay_rate,
            greedy_learning_rate=self._config.greedy_learning_rate,
            learning_rate=self._config.learning_rate,
            discount=self._config.discount,
            reward_scaling=self._config.reward_scaling,
            batch_size=self._config.batch_size,
            save_emitted_representation_params=self._config.save_emitted_representation_params,
            target_policy_update_interval=self._config.target_policy_update_interval,
            num_decision_updating_representation=self._config.num_decision_updating_representation,
            decision_factor=self._config.decision_factor,
            using_greedy=self._config.using_greedy,
        )

        refdqn_emitter = RefDQNEmitter(
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
