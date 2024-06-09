import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import qdax.core.containers.mapelites_repertoire
import qdax.core.emitters.multi_emitter
import qdax.core.emitters.pga_me_emitter
import qdax.core.emitters.qpg_emitter
import qdax.core.emitters.standard_emitters
from qdax.core.neuroevolution.buffers import buffer
import qdax.environments.base_wrappers
from qdax.types import Genotype, Params

import gym.spaces
import gymnasium.spaces

from dataclasses import dataclass, field
from functools import partial
from collections.abc import Callable
from typing import TypeVar, Any, TYPE_CHECKING
from overrides import override

from .multi_emitter import MultiEmitter
from ..neuroevolution import make_td3_loss_fn, QModule, CPUReplayBuffer, activation
from ..config.critic_net import NormalCNNCriticNetConfig
from ..utils import RNGKey, fnchain, assert_cast, jax_jit, lax_scan

if TYPE_CHECKING:
    from ..tasks import RLTask


@dataclass
class QualityPGConfig:
    '''Configuration for QualityPG Emitter'''

    env_batch_size: int = 100
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100

    critic_net: NormalCNNCriticNetConfig = field(default_factory=lambda: NormalCNNCriticNetConfig())

    # TD3 params
    replay_buffer_size: int = 1000000
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    policy_delay: int = 2


_QualityPGEmitterStateT = TypeVar(
    '_QualityPGEmitterStateT', bound=qdax.core.emitters.qpg_emitter.QualityPGEmitterState
)


class QualityPGEmitter(qdax.core.emitters.qpg_emitter.QualityPGEmitter):

    def __init__(
        self,
        config: QualityPGConfig,
        policy_network: nn.Module,
        env: 'RLTask',
    ) -> None:
        self._config = config
        self._env = env
        self._policy_network = policy_network

        # Init Critics
        obs_space = self._env.obs_space
        match obs_space:
            case gym.spaces.Box() | gymnasium.spaces.Box():
                cnn_input_shape = obs_space.shape
            case _:
                raise NotImplementedError(type(obs_space))
        assert self._config.critic_net.conv_padding in ('SAME', 'VALID')
        critic_network = QModule(
            conv_features=self._config.critic_net.conv_features,
            conv_kernel_sizes=self._config.critic_net.conv_kernel_sizes,
            conv_activation=activation(self._config.critic_net.conv_activation),
            conv_strides=self._config.critic_net.conv_strides,
            conv_padding=self._config.critic_net.conv_padding,
            mlp_layer_sizes=self._config.critic_net.mlp_hidden_layer_sizes,
            mlp_activation=activation(self._config.critic_net.mlp_activation),
            mlp_final_activation=activation(self._config.critic_net.mlp_final_activation),
            cnn_input_shape=cnn_input_shape,
            n_critics=2,
        )
        self._critic_network = critic_network

        # Set up the losses and optimizers - return the opt states
        self._policy_loss_fn, self._critic_loss_fn = make_td3_loss_fn(
            policy_fn=fnchain(policy_network.apply, assert_cast(jax.Array)),
            critic_fn=fnchain(critic_network.apply, assert_cast(jax.Array)),
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            noise_clip=self._config.noise_clip,
            policy_noise=self._config.policy_noise,
        )

        # Init optimizers
        self._actor_optimizer = optax.adam(
            learning_rate=self._config.actor_learning_rate
        )
        self._critic_optimizer = optax.adam(
            learning_rate=self._config.critic_learning_rate
        )
        self._policies_optimizer = optax.adam(
            learning_rate=self._config.policy_learning_rate
        )

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> tuple[qdax.core.emitters.qpg_emitter.QualityPGEmitterState, RNGKey]:

        observation_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length

        # Initialise critic, greedy actor and population
        random_key, subkey = jax.random.split(random_key)
        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_action = jnp.zeros(shape=(action_size,))
        critic_params = self._critic_network.init(
            subkey, obs=fake_obs, actions=fake_action
        )
        target_critic_params = jax.tree_util.tree_map(lambda x: x, critic_params)

        actor_params = jax.tree_util.tree_map(lambda x: x[0], init_genotypes)
        target_actor_params = jax.tree_util.tree_map(lambda x: x[0], init_genotypes)

        # Prepare init optimizer states
        critic_optimizer_state = self._critic_optimizer.init(critic_params)
        actor_optimizer_state = self._actor_optimizer.init(actor_params)

        # Initialize replay buffer
        dummy_transition = buffer.QDTransition.init_dummy(
            observation_dim=observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )

        replay_buffer = CPUReplayBuffer.init(
            buffer_size=self._config.replay_buffer_size,
            transition=dummy_transition,
            rand=jax.random.uniform(random_key),
            task=self._env,
        )

        # Initial training state
        random_key, subkey = jax.random.split(random_key)
        emitter_state = qdax.core.emitters.qpg_emitter.QualityPGEmitterState(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            actor_params=actor_params,
            actor_opt_state=actor_optimizer_state,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
            random_key=subkey,
            steps=jnp.array(0),
            replay_buffer=replay_buffer,
        )

        return emitter_state, random_key

    @partial(
        jax_jit,
        static_argnames=('self',),
    )
    @override
    def emit(
        self,
        repertoire: qdax.core.containers.mapelites_repertoire.MapElitesRepertoire,
        emitter_state: qdax.core.emitters.qpg_emitter.QualityPGEmitterState,
        random_key: RNGKey,
    ) -> tuple[Genotype, RNGKey]:
        batch_size = self._config.env_batch_size

        # sample parents
        mutation_pg_batch_size = int(batch_size - 1)
        parents, random_key = repertoire.sample(random_key, mutation_pg_batch_size)

        # apply the pg mutation
        random_key, subkey = jax.random.split(random_key)
        offsprings_pg = self.emit_pg(emitter_state, parents, subkey)

        # get the actor (greedy actor)
        offspring_actor = self.emit_actor(emitter_state)

        # add dimension for concatenation
        offspring_actor = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), offspring_actor
        )

        # gather offspring
        genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            offsprings_pg,
            offspring_actor,
        )

        return genotypes, random_key

    @partial(
        jax_jit,
        static_argnames=('self',),
    )
    def emit_pg(
        self,
        emitter_state: qdax.core.emitters.qpg_emitter.QualityPGEmitterState,
        parents: Genotype,
        random_key: RNGKey,
    ) -> Genotype:
        mutation_fn = partial(
            self._mutation_function_pg,
            emitter_state=emitter_state,
            random_key=random_key,
        )
        offsprings = jax.vmap(mutation_fn)(parents)

        return offsprings

    state_update = jax_jit(  # pyright: ignore [reportAssignmentType]
        qdax.core.emitters.qpg_emitter.QualityPGEmitter.state_update._fun,  # pyright: ignore [reportAttributeAccessIssue]
        static_argnames=('self',)
    )

    @partial(jax_jit, static_argnames=('self',))
    def _mutation_function_pg(
        self,
        policy_params: Genotype,
        emitter_state: qdax.core.emitters.qpg_emitter.QualityPGEmitterState,
        random_key: RNGKey,
    ) -> Genotype:
        # Define new policy optimizer state
        policy_optimizer_state = self._policies_optimizer.init(policy_params)

        def scan_train_policy(
            carry: tuple[
                qdax.core.emitters.qpg_emitter.QualityPGEmitterState, Genotype, optax.OptState
            ],
            random_key: RNGKey,
        ) -> tuple[
            tuple[qdax.core.emitters.qpg_emitter.QualityPGEmitterState, Genotype, optax.OptState],
            Any,
        ]:
            emitter_state, policy_params, policy_optimizer_state = carry
            (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
            ) = self._train_policy(
                emitter_state,
                policy_params,
                policy_optimizer_state,
                random_key,
            )
            return (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
            ), ()

        keys = jax.random.split(random_key, self._config.num_pg_training_steps)
        (emitter_state, policy_params, policy_optimizer_state,), _ = lax_scan(
            scan_train_policy,
            (emitter_state, policy_params, policy_optimizer_state),
            keys,
            length=self._config.num_pg_training_steps,
        )

        return policy_params

    @partial(jax_jit, static_argnames=('self',))
    def _train_policy(
        self,
        emitter_state: _QualityPGEmitterStateT,
        policy_params: Params,
        policy_optimizer_state: optax.OptState,
        random_key: RNGKey,
    ) -> tuple[_QualityPGEmitterStateT, Params, optax.OptState]:
        # Sample a batch of transitions in the buffer
        replay_buffer = emitter_state.replay_buffer
        transitions, _ = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

        # update policy
        policy_optimizer_state, policy_params = self._update_policy(
            critic_params=emitter_state.critic_params,
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            transitions=transitions,
        )

        return emitter_state, policy_params, policy_optimizer_state


@dataclass
class PGAMEConfig:
    '''Configuration for PGAME Algorithm'''

    env_batch_size: int = 100
    proportion_mutation_ga: float = 0.5
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100

    critic_net: NormalCNNCriticNetConfig = field(default_factory=lambda: NormalCNNCriticNetConfig())

    # TD3 params
    replay_buffer_size: int = 1000000
    critic_learning_rate: float = 3e-4
    greedy_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    policy_delay: int = 2


class PGAMEEmitter(MultiEmitter):

    def __init__(
        self,
        config: PGAMEConfig,
        policy_network: nn.Module,
        env: 'RLTask',
        variation_fn: Callable[[Params, Params, RNGKey], tuple[Params, RNGKey]],
    ) -> None:
        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = int(self._config.proportion_mutation_ga * config.env_batch_size)
        qpg_batch_size = config.env_batch_size - ga_batch_size

        qpg_config = QualityPGConfig(
            env_batch_size=qpg_batch_size,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps,
            replay_buffer_size=config.replay_buffer_size,
            critic_net=config.critic_net,
            critic_learning_rate=config.critic_learning_rate,
            actor_learning_rate=config.greedy_learning_rate,
            policy_learning_rate=config.policy_learning_rate,
            noise_clip=config.noise_clip,
            policy_noise=config.policy_noise,
            discount=config.discount,
            reward_scaling=config.reward_scaling,
            batch_size=config.batch_size,
            soft_tau_update=config.soft_tau_update,
            policy_delay=config.policy_delay,
        )

        # define the quality emitter
        q_emitter = QualityPGEmitter(
            config=qpg_config, policy_network=policy_network, env=env
        )

        # define the GA emitter
        ga_emitter = qdax.core.emitters.standard_emitters.MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_batch_size,
        )

        super().__init__(emitters=(q_emitter, ga_emitter))
