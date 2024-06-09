import jax
import jax.numpy as jnp
import flax.core.scope
import flax.struct
import qdax.core.emitters.emitter
import qdax.core.emitters.multi_emitter
import qdax.core.emitters.mutation_operators
import qdax.environments
from qdax.types import Genotype, Observation, Action, Fitness, Descriptor, Metrics

import numpy as np
import gym.spaces
import gymnasium.spaces

import hydra.utils
from omegaconf import OmegaConf
import logging
import wandb
import math
import time
import os
import tqdm
import tqdm.contrib.logging
from functools import partial
from collections.abc import Sequence, Callable
from contextlib import AbstractContextManager
from typing import Optional, TypeVar, Any, cast, assert_never
from overrides import override

from .manager_base import ManagerStateBase, ManagerConstantBase, ManagerBase
from .extended_me import ExtendedMAPElites
from .containers import (
    compute_cvt_centroids, ExtendedMapElitesRepertoire, CPUMapElitesRepertoire,
    ExtendedDepthRepertoire, CPUDepthRepertoire, get_repertoire_type,
)
from .containers import extended_me_repertoire, extended_depth_repertoire
from .emitters import RefEmitterState, PGAMEEmitter, RefPGAMEEmitter, DQNMEEmitter, RefDQNMEEmitter
from .neuroevolution import FakeRepresentation, CNN, activation, buffers
from .tasks import Task, RLTask, GymTask, QDaxBraxTask, EnvPoolTask, AtariTask
from .tasks import gym_task, envpool_task
from .metrics import qd_metrics
from .config.root import RootConfig
from .config.task import RLTaskConfig, QDaxBraxConfig, AtariConfig
from .config.framework import MEConfig
from .config.emitter import (
    PGAMEGEmitterConfig, VRefPGAMEGEmitterConfig, DQNMEGEmitterConfig, VRefDQNMEGEmitterConfig
)
from .config.network import CNNPolicyNetConfig, SharedCNNPolicyNetConfig
from .treax import functional as F
from .treax import numpy as tjnp
from .treax import chain as tchain
from .utils import (
    rich_highlight, fake_wrap, jax_jit, jax_compiled, lax_scan,
    transpose_dict_of_list, uninterrupted, PeriodicEMA, CSVLogger,
)


_log = logging.getLogger(__name__)


class ManagerState(ManagerStateBase):
    extra_keys: jax.Array
    repertoire: ExtendedMapElitesRepertoire
    tmp_repertoire: Optional[ExtendedMapElitesRepertoire]
    emitter_state: Optional[qdax.core.emitters.emitter.EmitterState]
    loop_completed: jax.Array
    total_time: jax.Array


class ManagerConstant(ManagerConstantBase):
    task_constant: Any
    reeval_task_constant: Any


_CallableT = TypeVar('_CallableT', bound=Callable)


class Manager(ManagerBase[RootConfig, ManagerState, ManagerConstant]):

    def __init__(self, cfg: RootConfig) -> None:
        super().__init__(cfg)

        if self._cfg.n_profile != 0:
            _log.warning('The profiler is enabled. The runtime may be longer.')

        self._task = self._create_task(self._cfg.emitter.env_batch_size)
        self._scoring_fn = self._task.get_scoring_fn()
        if self._cfg.emitter.reeval_env_batch_size is None:
            self._reeval_task = self._task
        else:
            self._reeval_task = self._create_task(self._cfg.emitter.reeval_env_batch_size)
        reeval_env_batch_size = (
            self._cfg.emitter.reeval_env_batch_size or self._cfg.emitter.env_batch_size
        )

        match self._cfg.network.task_type:
            case 'RL':
                qdax_brax_net_config = cast(CNNPolicyNetConfig, self._cfg.network)
                assert isinstance(self._task, RLTask)
                assert isinstance(self._reeval_task, RLTask)

                obs_space = self._task.obs_space
                match obs_space:
                    case gym.spaces.Box() | gymnasium.spaces.Box():
                        cnn_input_shape = obs_space.shape
                    case _:
                        raise NotImplementedError(type(obs_space))

                action_space = self._task.action_space
                match action_space:
                    case gym.spaces.Box() | gymnasium.spaces.Box():
                        output_size = self._task.action_size
                    case gym.spaces.Discrete() | gymnasium.spaces.Discrete():
                        output_size = int(action_space.n)
                    case _:
                        raise NotImplementedError(type(action_space))

                assert qdax_brax_net_config.conv_padding in ('SAME', 'VALID')

                if self._cfg.network.emitter_type in ('Normal', 'Share'):
                    if self._cfg.network.emitter_type == 'Normal':
                        self._representation_net = FakeRepresentation()
                        decision_cnn_input_shape = cnn_input_shape
                    elif self._cfg.network.emitter_type == 'Share':
                        qdax_brax_net_config = cast(
                            SharedCNNPolicyNetConfig, qdax_brax_net_config
                        )
                        assert qdax_brax_net_config.conv_padding in ('SAME', 'VALID')
                        self._representation_net = CNN(
                            conv_features=qdax_brax_net_config.representation_conv_features,
                            conv_kernel_sizes=qdax_brax_net_config.representation_conv_kernel_sizes,
                            conv_activation=activation(qdax_brax_net_config.conv_activation),
                            conv_strides=qdax_brax_net_config.representation_conv_strides,
                            conv_padding=qdax_brax_net_config.conv_padding,
                            mlp_layer_sizes=qdax_brax_net_config.representation_mlp_hidden_layer_sizes,  # noqa: E501
                            mlp_activation=activation(qdax_brax_net_config.mlp_activation),
                            mlp_final_activation=activation(qdax_brax_net_config.mlp_final_activation),  # noqa: E501
                            cnn_input_shape=cnn_input_shape,
                        )
                        decision_cnn_input_shape = qdax_brax_net_config.decision_cnn_input_shape
                    else:
                        assert_never(self._cfg.network.emitter_type)

                    self._net = CNN(
                        conv_features=qdax_brax_net_config.conv_features,
                        conv_kernel_sizes=qdax_brax_net_config.conv_kernel_sizes,
                        conv_activation=activation(qdax_brax_net_config.conv_activation),
                        conv_strides=qdax_brax_net_config.conv_strides,
                        conv_padding=qdax_brax_net_config.conv_padding,
                        mlp_layer_sizes=(*qdax_brax_net_config.mlp_hidden_layer_sizes, output_size),
                        mlp_activation=activation(qdax_brax_net_config.mlp_activation),
                        mlp_final_activation=activation(qdax_brax_net_config.mlp_final_activation),
                        cnn_input_shape=decision_cnn_input_shape,
                    )

                    match (obs_space, action_space):
                        case (
                            gym.spaces.Box() | gymnasium.spaces.Box(),
                            gym.spaces.Box() | gymnasium.spaces.Box(),
                        ):
                            def select_action_fn(
                                representation_genotypes: Genotype,
                                genotypes: Genotype,
                                obs: Observation,
                            ) -> Action:
                                representation = self._representation_net.apply(
                                    cast(flax.core.scope.VariableDict, representation_genotypes),
                                    obs,
                                )
                                assert isinstance(representation, jax.Array)
                                action = self._net.apply(
                                    cast(flax.core.scope.VariableDict, genotypes),
                                    representation,
                                )
                                assert isinstance(action, Action)
                                return action
                        case (
                            gym.spaces.Box() | gymnasium.spaces.Box(),
                            gym.spaces.Discrete() | gymnasium.spaces.Discrete(),
                        ):
                            def select_action_fn(
                                representation_genotypes: Genotype,
                                genotypes: Genotype,
                                obs: Observation,
                            ) -> Action:
                                representation = self._representation_net.apply(
                                    cast(flax.core.scope.VariableDict, representation_genotypes),
                                    obs,
                                )
                                assert isinstance(representation, Action)
                                action = self._net.apply(
                                    cast(flax.core.scope.VariableDict, genotypes),
                                    representation,
                                )
                                assert isinstance(action, Action)
                                action = jnp.argmax(action, axis=-1, keepdims=True)
                                return action
                        case _:
                            raise NotImplementedError((type(obs_space), type(action_space)))

                else:
                    raise NotImplementedError(self._cfg.network.emitter_type)

                self._task.set_select_action_fn(select_action_fn)
                if self._reeval_task is not self._task:
                    self._reeval_task.set_select_action_fn(select_action_fn)

            case _:
                raise NotImplementedError(self._cfg.task.typ)

        match self._cfg.emitter.subtype:
            case 'PGA-ME':
                pga_me_config = cast(PGAMEGEmitterConfig, self._cfg.emitter)
                self._emitter = PGAMEEmitter(
                    config=pga_me_config,
                    policy_network=self._net,
                    env=self._task,
                    variation_fn=partial(
                        qdax.core.emitters.mutation_operators.isoline_variation,
                        iso_sigma=pga_me_config.iso_sigma,
                        line_sigma=pga_me_config.line_sigma,
                    ),
                )
                if self._cfg.task.typ == 'RL':
                    rl_task_config = cast(RLTaskConfig, self._cfg.task)
                    self._step_multiplier = rl_task_config.episode_len
                else:
                    self._step_multiplier = 1
            case 'Ref-PGA-ME':
                refpga_me_config = cast(VRefPGAMEGEmitterConfig, self._cfg.emitter)
                assert isinstance(self._task, RLTask)
                assert not isinstance(self._representation_net, FakeRepresentation)
                self._emitter = RefPGAMEEmitter(
                    config=refpga_me_config,
                    representation_net=self._representation_net,
                    decision_net=self._net,
                    task=self._task,
                    variation_fn=partial(
                        qdax.core.emitters.mutation_operators.isoline_variation,
                        iso_sigma=refpga_me_config.iso_sigma,
                        line_sigma=refpga_me_config.line_sigma,
                    ),
                )
                if self._cfg.task.typ == 'RL':
                    rl_task_config = cast(RLTaskConfig, self._cfg.task)
                    self._step_multiplier = rl_task_config.episode_len
                else:
                    self._step_multiplier = 1
            case 'DQN-ME':
                dqn_me_config = cast(DQNMEGEmitterConfig, self._cfg.emitter)
                assert isinstance(self._task, RLTask)
                self._emitter = DQNMEEmitter(
                    config=dqn_me_config,
                    policy_network=self._net,
                    task=self._task,
                    variation_fn=partial(
                        qdax.core.emitters.mutation_operators.isoline_variation,
                        iso_sigma=dqn_me_config.iso_sigma,
                        line_sigma=dqn_me_config.line_sigma,
                    ),
                )
                if self._cfg.task.typ == 'RL':
                    rl_task_config = cast(RLTaskConfig, self._cfg.task)
                    self._step_multiplier = rl_task_config.episode_len
                else:
                    self._step_multiplier = 1
            case 'Ref-DQN-ME':
                refdqn_me_config = cast(VRefDQNMEGEmitterConfig, self._cfg.emitter)
                assert isinstance(self._task, RLTask)
                assert not isinstance(self._representation_net, FakeRepresentation)
                self._emitter = RefDQNMEEmitter(
                    config=refdqn_me_config,
                    representation_net=self._representation_net,
                    decision_net=self._net,
                    task=self._task,
                    variation_fn=partial(
                        qdax.core.emitters.mutation_operators.isoline_variation,
                        iso_sigma=refdqn_me_config.iso_sigma,
                        line_sigma=refdqn_me_config.line_sigma,
                    ),
                )
                if self._cfg.task.typ == 'RL':
                    rl_task_config = cast(RLTaskConfig, self._cfg.task)
                    self._step_multiplier = rl_task_config.episode_len
                else:
                    self._step_multiplier = 1
            case _:
                raise NotImplementedError(self._cfg.emitter.name)

        match self._cfg.framework.name:
            case 'ME':
                self._metrics_fn = partial(
                    qd_metrics,
                    qd_offset=self._task.qd_offset,
                )
                self._qd = ExtendedMAPElites(
                    scoring_function=self._scoring_fn,
                    reeval_scoring_fn=self._reeval_task.get_scoring_fn(),
                    reeval_env_batch_size=reeval_env_batch_size,
                    emitter=self._emitter,
                    metrics_function=self._metrics_fn,
                    refresh_depth=self._cfg.emitter.refresh_depth,
                )
            case _:
                raise NotImplementedError(self._cfg.framework.name)

        self._seed_array = jnp.asarray(self._cfg.seed, dtype=jnp.int32)
        if self._cfg.multiseed is None:
            assert self._seed_array.shape[0] == 1
            self._seed_array = self._seed_array[0]

    def _create_task(self, batch_size: int) -> Task:
        batch_shape = (batch_size,)
        if self._cfg.multiseed:
            batch_shape = (len(self._cfg.seed), *batch_shape)
        match self._cfg.task.subtype:
            case 'QDaxBrax':
                qdax_task_config = cast(QDaxBraxConfig, self._cfg.task)
                task = QDaxBraxTask(qdax_task_config, batch_shape)
            case 'Atari':
                atari_task_config = cast(AtariConfig, self._cfg.task)
                task = AtariTask(atari_task_config, batch_shape)
            case _:
                raise NotImplementedError(self._cfg.task.typ)
        return task

    def _wandb_init(self) -> None:
        resume = self._wandb_resume_arg

        original_dir = os.path.realpath(hydra.utils.get_original_cwd())
        id_dict = {
            'task': self._cfg.task.name,
            'framework': self._cfg.framework.name,
            'emitter': self._cfg.emitter.name,
            'network': self._cfg.network.name,
            'code': self._cfg.code,
            'run': self._cfg.run,
            'seed': '-'.join(map(str, self._cfg.seed)),
        }
        cfg_dict = OmegaConf.to_container(self._cfg)
        assert isinstance(cfg_dict, dict)
        wandb.init(
            project='RefQD',
            id='{emitter}.{network}.{framework}.{task}.{code}.{run}.{seed}'.format(**id_dict),
            resume=resume,
            name='{emitter} {network} {framework} {task} {code} {run} {seed}'.format(**id_dict),
            config=cfg_dict,
            tags=[
                self._cfg.task.name,
                self._cfg.framework.name,
                self._cfg.emitter.name,
                self._cfg.network.name,
            ],
            job_type=self._cfg.typ,
            dir=original_dir,
        )
        assert wandb.run is not None
        with open('wandb-run-dir', 'a') as f:
            f.write(f'{wandb.run.dir}\n')

    @property
    def _saved_keys(self) -> Sequence[str]:
        match self._cfg.emitter.subtype:
            case 'DQN-ME':
                return (
                    *super()._saved_keys,
                    'history',
                )
            case _:
                return (
                    *super()._saved_keys,
                    '$extended_me_repertoire.global_genotypes',
                    '$buffers.global_buffer_data',
                    'history',
                )

    @property
    def globals(self) -> dict[str, Any]:
        assert type(self) is Manager
        return globals()

    def _reduce(self, saved_dict: dict[str, Any]) -> dict[str, Any]:
        saved_dict['state'] = saved_dict['state'].replace(
            repertoire=saved_dict['state'].repertoire.replace(genotypes=None)
        )
        if hasattr(saved_dict['state'].repertoire, 'genotypes_depth'):
            saved_dict['state'] = saved_dict['state'].replace(
                repertoire=saved_dict['state'].repertoire.replace(genotypes_depth=None)
                )
        try:
            del saved_dict['$extended_me_repertoire.global_genotypes']
            del saved_dict['$buffers.global_buffer_data']
        except KeyError:
            pass
        return saved_dict

    def _do_init(self, seed: jax.Array) -> tuple[ManagerState, ManagerConstant, Metrics]:
        random_key = jax.random.key(seed)

        _log.info('Initializing the task...')
        random_key, task_key = jax.random.split(random_key)
        task_key, subkey = jax.random.split(task_key)
        task_constant = self._task.get_constant(subkey)
        if self._reeval_task is self._task:
            reeval_task_constant = task_constant
        else:
            reeval_task_constant = self._reeval_task.get_constant(subkey)
        del subkey
        del task_key

        _log.info('Initializing the network...')
        random_key, network_key = jax.random.split(random_key)
        match self._cfg.network.task_type:
            case 'RL':
                qdax_brax_net_config = cast(CNNPolicyNetConfig, self._cfg.network)
                assert isinstance(self._task, RLTask)

                fake_obs = jnp.zeros((self._task.observation_size,))

                assert qdax_brax_net_config.conv_padding in ('SAME', 'VALID')

                match self._cfg.network.emitter_type:
                    case 'Normal' | 'Share':
                        if not self._dry_run:
                            _log.info(
                                'Representation network:\n%s', self._representation_net.tabulate(
                                    jax.random.key(0),
                                    fake_obs,
                                    compute_flops=True,
                                    compute_vjp_flops=True,
                                ),
                            )
                        network_key, subkey = jax.random.split(network_key)
                        (
                            representation, initial_representation_genotypes
                        ) = self._representation_net.init_with_output(subkey, fake_obs)
                        del subkey

                        if not self._dry_run:
                            _log.info('Decision network:\n%s', self._net.tabulate(
                                jax.random.key(0),
                                representation,
                                compute_flops=True,
                                compute_vjp_flops=True,
                            ))
                        network_key, subkey = jax.random.split(network_key)
                        keys = jax.random.split(subkey, self._cfg.emitter.env_batch_size)
                        del subkey
                        initial_genotypes = jax.vmap(self._net.init, in_axes=(0, None))(
                            keys, representation
                        )
                        del keys

                    case _:
                        raise NotImplementedError(self._cfg.network.emitter_type)

            case _:
                raise NotImplementedError(self._cfg.task.typ)
        del network_key

        _log.info('Initializing the emitter...')
        random_key, emitter_key = jax.random.split(random_key)
        match self._cfg.emitter.subtype:
            case 'PGA-ME' | 'Ref-PGA-ME' | 'DQN-ME' | 'Ref-DQN-ME':
                pass
            case _:
                raise NotImplementedError(self._cfg.emitter.name)
        del emitter_key

        _log.info('Initializing the QD framework...')
        random_key, framework_key = jax.random.split(random_key)
        match self._cfg.framework.name:
            case 'ME':
                me_config = cast(MEConfig, self._cfg.framework)
                minbd, maxbd = self._task.behavior_descriptor_limits
                centroids, framework_key = compute_cvt_centroids(
                    self._task.behavior_descriptor_length,
                    me_config.n_init_cvt_samples,
                    me_config.n_centroids,
                    minbd,
                    maxbd,
                    framework_key,
                )
                framework_key, subkey = jax.random.split(framework_key)
                (
                    repertoire,
                    tmp_repertoire,
                    emitter_state,
                    metrics,
                    _,
                ) = self._qd.init(
                    init_representation_genotypes=initial_representation_genotypes,
                    init_decision_genotypes=initial_genotypes,
                    centroids=centroids,
                    random_key=subkey,
                    task_constant=task_constant,
                    using_representation=self._cfg.emitter.typ == 'Share',
                    repertoire_type=get_repertoire_type(self._cfg.emitter.repertoire_type),
                    repertoire_kwargs=self._cfg.emitter.repertoire_kwargs,
                )
                del subkey
            case _:
                raise NotImplementedError(self._cfg.framework.name)
        del framework_key

        metrics |= {
            'seed': seed,
            'n_loops': jnp.asarray(0, dtype=jnp.int32),
            'n_iters': jnp.asarray(0, dtype=jnp.int32),
            'n_evals': jnp.asarray(0, dtype=jnp.int32),
            'n_steps': jnp.asarray(0, dtype=jnp.int32),
        }

        random_key, extra_key = jax.random.split(random_key)
        extra_keys = jax.random.split(extra_key, 4)

        return (
            ManagerState(
                random_key=random_key,
                extra_keys=extra_keys,
                repertoire=repertoire,
                tmp_repertoire=tmp_repertoire,
                emitter_state=emitter_state,
                loop_completed=jnp.asarray(0, dtype=jnp.int32),
                total_time=jnp.asarray(0.0, dtype=jnp.float32),
            ),
            ManagerConstant(task_constant=task_constant, reeval_task_constant=reeval_task_constant),
            metrics,
        )

    def _log(self, metrics: Metrics) -> None:
        if len(metrics['n_loops'].shape) - (self._cfg.multiseed is not None) == 0:
            metrics = tjnp.expand_dims(metrics, axis=0)

        csv_list, last_dict = self._get_log_metrics(metrics)
        if self._dry_run:
            return
        self._csv_logger.log(csv_list)

        log_strs: list[str] = []
        has_value = True
        for key in self._csv_header:
            if key in last_dict.keys():
                assert has_value
            else:
                has_value = False
                continue
            metric = last_dict[key].item()
            if metric.is_integer() or 1e4 <= metric < 1e9:
                metric = round(metric)
            prefix = ''.join(map(lambda s: s[0], key.split('_')))
            style = '{}'
            if 'qs' in prefix:
                style = '[bright_red]{}[/bright_red]'
            elif 't' in prefix:
                style = '[dim]{}[/dim]'
            if isinstance(metric, float) and metric < 1e4:
                log_strs.append(style.format(f'{prefix} {metric:.5}'))
            else:
                log_strs.append(style.format(f'{prefix} {metric}'))
        log_str = ' '.join(log_strs)
        _log.info('%s', log_str)

        loop_completed = self.state.loop_completed.flatten()[0]
        if wandb.run is not None and loop_completed % self._reeval_period == 0:
            try:
                wandb.run.log(last_dict)
                wandb.run.summary.update(last_dict)
            except Exception:
                pass

    def _get_time_metrics(
        self,
        time_reeval: float,
        shape: tuple[int, ...],
        prefix: str,
    ) -> Metrics:
        time_metrics: Metrics = {f'{prefix}_time': jnp.full(shape, time_reeval, dtype=jnp.float32)}

        match self.state.repertoire:
            case CPUMapElitesRepertoire():
                time_metrics[f'{prefix}_time_repertoire_empty'] = jnp.full(
                    shape, extended_me_repertoire.global_time_empty, dtype=jnp.float32
                )
                time_metrics[f'{prefix}_time_repertoire_add'] = jnp.full(
                    shape, extended_me_repertoire.global_time_add, dtype=jnp.float32
                )
                time_metrics[f'{prefix}_time_repertoire_sample'] = jnp.full(
                    shape, extended_me_repertoire.global_time_sample, dtype=jnp.float32
                )
                extended_me_repertoire.global_time_empty = 0.0
                extended_me_repertoire.global_time_add = 0.0
                extended_me_repertoire.global_time_sample = 0.0
            case CPUDepthRepertoire():
                time_metrics[f'{prefix}_time_repertoire_empty'] = jnp.full(
                    shape, extended_depth_repertoire.global_time_empty, dtype=jnp.float32
                )
                time_metrics[f'{prefix}_time_repertoire_add'] = jnp.full(
                    shape, extended_depth_repertoire.global_time_add, dtype=jnp.float32
                )
                time_metrics[f'{prefix}_time_repertoire_sample'] = jnp.full(
                    shape, extended_depth_repertoire.global_time_sample, dtype=jnp.float32
                )
                time_metrics[f'{prefix}_time_repertoire_copy_from'] = jnp.full(
                    shape, extended_depth_repertoire.global_time_copy_from, dtype=jnp.float32
                )
                extended_depth_repertoire.global_time_empty = 0.0
                extended_depth_repertoire.global_time_add = 0.0
                extended_depth_repertoire.global_time_sample = 0.0
                extended_depth_repertoire.global_time_copy_from = 0.0
            case ExtendedMapElitesRepertoire() | ExtendedDepthRepertoire():
                pass
            case _:
                raise NotImplementedError(type(self.state.repertoire))

        match self._cfg.emitter.subtype:
            case 'PGA-ME' | 'Ref-PGA-ME' | 'DQN-ME' | 'Ref-DQN-ME':
                time_metrics[f'{prefix}_time_buffer_restore'] = jnp.full(
                    shape, buffers.global_time_restore, dtype=jnp.float32
                )
                time_metrics[f'{prefix}_time_buffer_insert'] = jnp.full(
                    shape, buffers.global_time_insert, dtype=jnp.float32
                )
                time_metrics[f'{prefix}_time_buffer_sample'] = jnp.full(
                    shape, buffers.global_time_sample, dtype=jnp.float32
                )
                buffers.global_time_restore = 0.0
                buffers.global_time_insert = 0.0
                buffers.global_time_sample = 0.0
            case _:
                raise NotImplementedError(self._cfg.emitter.name)

        match self._task:
            case QDaxBraxTask():
                pass
            case GymTask():
                time_metrics[f'{prefix}_time_env_reset'] = jnp.full(
                    shape, gym_task.global_time_reset, dtype=jnp.float32
                )
                time_metrics[f'{prefix}_time_env_step'] = jnp.full(
                    shape, gym_task.global_time_step, dtype=jnp.float32
                )
                gym_task.global_time_reset = 0.0
                gym_task.global_time_step = 0.0
            case EnvPoolTask():
                time_metrics[f'{prefix}_time_env_reset'] = jnp.full(
                    shape, envpool_task.global_time_reset, dtype=jnp.float32
                )
                time_metrics[f'{prefix}_time_env_step'] = jnp.full(
                    shape, envpool_task.global_time_step, dtype=jnp.float32
                )
                envpool_task.global_time_reset = 0.0
                envpool_task.global_time_step = 0.0
            case _:
                raise NotImplementedError(self._cfg.task.subtype)

        return time_metrics

    def _get_total_time_metrics(
        self,
        time_per_loop: float | jax.Array,
        total_time: float | jax.Array,
        shape: tuple[int, ...],
    ) -> Metrics:
        time_metrics: Metrics = {
            'time_per_loop': jnp.full(shape, time_per_loop, dtype=jnp.float32),
            'total_time': jnp.full(shape, total_time, dtype=jnp.float32),
        }
        return time_metrics

    def _reeval(
        self, state: ManagerState
    ) -> tuple[ManagerState, tuple[Metrics, Fitness, Descriptor]]:
        match self._cfg.emitter.typ:
            case 'Normal':
                representation_params = {}
            case 'Share':
                emitter_state = state.emitter_state
                while isinstance(
                    emitter_state, qdax.core.emitters.multi_emitter.MultiEmitterState
                ):
                    emitter_state = emitter_state.emitter_states[0]
                assert isinstance(emitter_state, RefEmitterState)
                if emitter_state.emitted_representation_params is not None:
                    representation_params = emitter_state.emitted_representation_params
                else:
                    representation_params = emitter_state.representation_params
            case _:
                raise NotImplementedError(self._cfg.emitter.typ)

        _, emitter_state, reeval_key, metrics, fitness, descriptor = self._map(self._qd.reeval)(
            None,
            state.emitter_state,
            state.extra_keys[..., 0],
            representation_params=representation_params,
            repertoire=state.repertoire,
            task_constant=self.constant.task_constant,
        )

        state = state.replace(
            emitter_state=emitter_state, extra_keys=state.extra_keys.at[..., 0].set(reeval_key)
        )

        return state, (metrics, fitness, descriptor)

    def reeval(
        self, wrapper: Callable[[str], AbstractContextManager[Any]], shape: tuple[int, ...]
    ) -> tuple[Metrics, Fitness, Descriptor]:
        if hasattr(self, '_reeval_fn'):
            reeval_fn = self._reeval_fn
        else:
            if self._dry_run:
                reeval_fn = jax_jit(self._reeval, donate_argnums=0)
            else:
                _log.info('Compiling reeval_fn...')
                reeval_fn = jax_compiled(self._reeval, donate_argnums=0)(
                    self.state
                )
                cost_stats = reeval_fn.cost_analysis()
                if (
                    isinstance(cost_stats, list) and len(cost_stats) == 1
                    and isinstance(cost_stats[0], dict)
                ):
                    cost_stats = [{
                        key: value
                        for key, value in sorted(cost_stats[0].items())
                        if ' {' not in key
                    }]
                _log.info('Cost analysis of reeval_fn: %s', cost_stats)
                _log.info(
                    'Memory analysis of reeval_fn: %s',
                    reeval_fn.memory_analysis(),
                )
                self._reeval_fn = reeval_fn

        with wrapper('tensorboard'):
            time_reeval = time.monotonic()
            (
                self.state, (metrics, reeval_fitness, reeval_descriptor)
            ) = reeval_fn(self.state)
            time_reeval = time.monotonic() - time_reeval
        metrics = {key: jnp.full(shape, value) for key, value in metrics.items()}
        metrics |= self._get_time_metrics(
            time_reeval,
            shape,
            'reeval',
        )
        reeval_fitness = reeval_fitness[jnp.newaxis]
        reeval_descriptor = reeval_descriptor[jnp.newaxis]

        return metrics, reeval_fitness, reeval_descriptor

    @override
    def init(self) -> None:
        if self._cfg.n_profile < 0:
            wrapper = jax.profiler.trace
        else:
            wrapper = fake_wrap

        do_init_fn = jax_jit(self._map(self._do_init))

        with wrapper('tensorboard'):
            time_do_init = time.monotonic()
            self.state, self.constant, metrics = do_init_fn(self._seed_array)
            time_do_init = time.monotonic() - time_do_init

        time_metrics = self._get_time_metrics(time_do_init, metrics['n_loops'].shape, 'opt')
        fitnesses = self.state.repertoire.fitnesses[jnp.newaxis]
        descriptors = self.state.repertoire.descriptors[jnp.newaxis]

        _log.info('Reevaluating...')
        (
            reeval_metrics, reeval_fitness, reeval_descriptor
        ) = self.reeval(wrapper, metrics['n_loops'].shape)

        _log.info('Logging...')
        metrics |= self._get_total_time_metrics(
            time_do_init, self.state.total_time, metrics['n_loops'].shape,
        )
        metrics |= time_metrics | reeval_metrics

        time_metrics_keys = tuple(key[4:] for key in time_metrics.keys())
        self._csv_header = (
            'seed',
            'n_loops',
            'n_iters',
            'n_evals',
            'n_steps',
            'qd_score',
            'max_fitness',
            'coverage',
            'min_fitness',
            'mean_fitness',
            'time_per_loop',
            'total_time',
            *(f'opt_{key}' for key in time_metrics_keys),
            'reeval_qd_score',
            'reeval_max_fitness',
            'reeval_coverage',
            'reeval_min_fitness',
            'reeval_mean_fitness',
            *(f'reeval_{key}' for key in time_metrics_keys),
        )
        assert all(key in self._csv_header for key in metrics.keys())
        assert all(key in metrics.keys() for key in self._csv_header)

        def get_log_metrics(
            metrics: Metrics
        ) -> tuple[list[Metrics], Metrics]:
            sorted_metrics: Metrics = {}
            for key in self._csv_header:
                if key in metrics.keys():
                    sorted_metrics[key] = metrics[key]

            csv_list = transpose_dict_of_list(tjnp.reshape(sorted_metrics, newshape=(-1,)))
            last_dict = tchain(tjnp.getitem, tjnp.mean)(sorted_metrics, -1)
            return csv_list, last_dict

        self._get_log_metrics = jax_jit(get_log_metrics)

        if self._dry_run:
            self.history = {}
        else:
            self.history = {
                'fitnesses': [np.asarray(fitnesses)],
                'descriptors': [np.asarray(descriptors)],
                'reeval_fitnesses': [np.asarray(reeval_fitness)],
                'reeval_descriptors': [np.asarray(reeval_descriptor)],
            }

        self._reeval_period = self._cfg.task.reeval_period * self._cfg.emitter.reeval_factor
        match self._cfg.task.typ:
            case 'RL':
                rl_task_config = cast(RLTaskConfig, self._cfg.task)
                if self._cfg.emitter.refresh_depth != 0:
                    reeval_factor = self._qd.calc_evals_per_reeval(self.state.repertoire)
                else:
                    reeval_factor = 0
                evals_per_reeval = (
                    self._cfg.emitter.env_batch_size
                    * rl_task_config.log_period
                    * self._reeval_period
                    + reeval_factor
                )
                total_reevals = (
                    rl_task_config.total_steps / evals_per_reeval / rl_task_config.episode_len
                )
            case _:
                raise NotImplementedError(self._cfg.task.typ)
        if not total_reevals.is_integer():
            _log.warning(f'total_reevals ({total_reevals}) is not an integer.')
        total_reevals = math.ceil(total_reevals)
        self._total_loops = total_reevals * self._reeval_period
        self._total_evals = total_reevals * evals_per_reeval

        if self._dry_run:
            return

        self._csv_logger = CSVLogger(self._cfg.metrics_filename, header=self._csv_header)
        _log.info('%s', ' '.join(self._csv_header))
        with open(self._cfg.metrics_filename, 'r') as f:
            n_lines = len(f.readlines())
        writing_initial_log = n_lines <= 1
        if writing_initial_log:
            self._log(metrics)

        if self._resumed:
            self.load()

    def _map(self, func: _CallableT, in_axes: int | None | Sequence[Any] = 0) -> _CallableT:
        match self._cfg.multiseed:
            case None:
                return func
            case 'vmap':
                return jax.vmap(func, in_axes=in_axes)
            case _:
                raise NotImplementedError(self._cfg.multiseed)

    def _calc_n(self, state: ManagerState) -> tuple[jax.Array, jax.Array, jax.Array]:
        loop_completed = state.loop_completed.flatten()[0]
        if isinstance(loop_completed, jax.Array):
            xnp = jnp
        else:
            xnp = np
        n_loops = xnp.full((self._cfg.task.log_period,), loop_completed, dtype=xnp.int32)
        stop_n_iter = loop_completed * self._cfg.task.log_period + 1
        n_iter = xnp.arange(-self._cfg.task.log_period, 0) + stop_n_iter
        n_evals = n_iter * self._cfg.emitter.env_batch_size
        if self._cfg.emitter.refresh_depth != 0:
            reeval_factor = self._qd.calc_evals_per_reeval(state.repertoire)
            n_previous_reevals = (
                xnp.floor((loop_completed - 1) / self._reeval_period).astype(xnp.int32)
                * reeval_factor
            )
            n_reevals = (
                xnp.floor((loop_completed) / self._reeval_period).astype(xnp.int32)
                * reeval_factor
            )
            n_evals += n_previous_reevals
            if isinstance(n_evals, jax.Array):
                n_evals = n_evals.at[-1].add(n_reevals - n_previous_reevals)
            else:
                n_evals[-1] += n_reevals - n_previous_reevals
        return n_loops, n_iter, n_evals  # type: ignore

    def _optimize(
        self, state: ManagerState, refresh_repertoire: jax.Array
    ) -> tuple[ManagerState, tuple[Metrics, Metrics, Fitness, Descriptor, Fitness, Descriptor]]:
        update_fn = self._map(self._qd.scan_update, in_axes=(0, None))
        (
            (repertoire, tmp_repertoire, emitter_state, random_key, _),
            (metrics, reeval_metrics, fitnesses, descriptors, reeval_fitness, reeval_descriptor),
        ) = lax_scan(
            update_fn,
            (
                state.repertoire,
                state.tmp_repertoire,
                state.emitter_state,
                state.random_key,
                self.constant.task_constant,
            ),
            jnp.full(
                (self._cfg.task.log_period,), False, dtype=bool
            ).at[-1].set(refresh_repertoire),
            length=self._cfg.task.log_period,
        )
        state = state.replace(
            repertoire=repertoire,
            tmp_repertoire=tmp_repertoire,
            emitter_state=emitter_state,
            random_key=random_key,
            loop_completed=state.loop_completed + 1,
        )

        n_loops, n_iter, n_evals = self._calc_n(state)
        d = {
            'n_loops': n_loops,
            'n_iters': n_iter,
            'n_evals': n_evals,
            'n_steps': n_evals * self._step_multiplier,
        }
        if len(self._seed_array.shape) > 0:
            d = tjnp.duplicate(d, repeats=self._seed_array.shape[-1], axis=-1)
        metrics.update(d)

        metrics['seed'] = F.duplicate(
            self._seed_array, repeats=self._cfg.task.log_period, axis=0
        )

        reeval_fitness = reeval_fitness[:1]
        reeval_descriptor = reeval_descriptor[:1]

        return (
            state,
            (metrics, reeval_metrics, fitnesses, descriptors, reeval_fitness, reeval_descriptor),
        )

    @override
    def run(self) -> None:
        if self._dry_run:
            optimize_fn = jax_jit(self._optimize, donate_argnums=0)
            self.state = self.state.replace(loop_completed=np.full(
                self.state.loop_completed.shape,
                self._total_loops - 1,
                self.state.loop_completed.dtype,
            ))
        else:
            _log.info('Compiling optimize_fn...')
            optimize_fn = jax_compiled(self._optimize, donate_argnums=0)(
                self.state, jnp.asarray(False)
            )
            cost_stats = optimize_fn.cost_analysis()
            if (
                isinstance(cost_stats, list) and len(cost_stats) == 1
                and isinstance(cost_stats[0], dict)
            ):
                cost_stats = [
                    {key: value for key, value in sorted(cost_stats[0].items()) if ' {' not in key}
                ]
            _log.info('Cost analysis of optimize_fn: %s', cost_stats)
            _log.info('Memory analysis of optimize_fn: %s', optimize_fn.memory_analysis())

        desc_dict = {
            'task': self._cfg.task.name,
            'framework': self._cfg.framework.name,
            'emitter': self._cfg.emitter.name,
            'run': self._cfg.run,
        }
        desc = rich_highlight(
            '\\[[purple]{task}/{framework}/{emitter}[/purple]|[purple]{run}[/purple]]'
            .format(**desc_dict)
        )
        time_per_loop_ema = PeriodicEMA(period=self._reeval_period)
        setattr(tqdm.std, 'EMA', partial(PeriodicEMA, period=self._reeval_period))
        wrapper = jax.profiler.trace
        n_profile = int(self._cfg.n_profile - self.state.loop_completed.flatten()[0])
        with tqdm.tqdm(
            initial=int(self._calc_n(self.state)[2].flatten()[-1]),
            total=self._total_evals,
            dynamic_ncols=True,
            leave=not self._dry_run,
        ) as pbar:
            with tqdm.contrib.logging.logging_redirect_tqdm():
                while self.state.loop_completed.flatten()[0] < self._total_loops:
                    with uninterrupted():
                        if n_profile <= 0:
                            wrapper = fake_wrap
                        n_profile -= 1
                        self.saved = False

                        pbar.set_description_str(f'{desc} optimizing')
                        refresh_repertoire = jnp.logical_and(
                            self._cfg.emitter.refresh_depth != 0,
                            (self.state.loop_completed.flatten()[0] + 1) % self._reeval_period == 0,
                        )
                        with wrapper('tensorboard'):
                            time_scan = time.monotonic()
                            (
                                self.state,
                                (
                                    metrics, reeval_metrics,
                                    fitness, descriptor,
                                    reeval_fitness, reeval_descriptor
                                ),
                            ) = optimize_fn(self.state, refresh_repertoire)
                            time_scan = time.monotonic() - time_scan
                        time_per_loop = time_scan
                        metrics |= self._get_time_metrics(
                            time_scan, metrics['n_loops'].shape, 'opt',
                        )
                        if self._dry_run or refresh_repertoire:
                            metrics |= reeval_metrics

                        if self._dry_run:
                            self.state = self.state.replace(loop_completed=np.full(
                                self.state.loop_completed.shape,
                                self._total_loops,
                                self.state.loop_completed.dtype,
                            ))
                        loop_completed = int(self.state.loop_completed.flatten()[0])
                        if (
                            loop_completed % self._reeval_period == 0
                        ):
                            pbar.set_description_str(f'{desc} reevaluating')
                            (
                                reeval_metrics, reeval_fitness, reeval_descriptor
                            ) = self.reeval(wrapper, metrics['n_loops'].shape)
                            metrics |= reeval_metrics

                        pbar.set_description_str(f'{desc} logging')
                        self.state = self.state.replace(
                            total_time=self.state.total_time + time_per_loop
                        )
                        metrics |= self._get_total_time_metrics(
                            time_per_loop, self.state.total_time, metrics['n_loops'].shape,
                        )
                        self._log(metrics)
                        if self._dry_run:
                            return
                        self.history['fitnesses'].append(np.asarray(fitness))
                        self.history['descriptors'].append(np.asarray(descriptor))
                        self.history['reeval_fitnesses'].append(np.asarray(reeval_fitness))
                        self.history['reeval_descriptors'].append(np.asarray(reeval_descriptor))

                        finish = loop_completed >= self._total_loops
                        if finish or loop_completed % self._cfg.checkpoint_saving_interval == 0:
                            _log.info('Saving...')
                            self.save()
                        if finish or loop_completed % self._cfg.metrics_uploading_interval == 0:
                            self.wandb_upload_metrics()

                        smoothed_time_per_loop = time_per_loop_ema(time_per_loop)
                        pbar.set_postfix({
                            't/loop': smoothed_time_per_loop,
                            't': '{}<{}'.format(
                                pbar.format_interval(round(
                                    float(self.state.total_time.flatten()[0])
                                )),
                                pbar.format_interval(round(
                                    (self._total_loops - loop_completed)
                                    * smoothed_time_per_loop
                                )),
                            )
                        }, refresh=False)
                        pbar.update(int(metrics['n_evals'].flatten()[-1]) - pbar.n)

                        if (
                            self._cfg.typ == 'time'
                            and loop_completed % (self._reeval_period * 2) == 0
                        ):
                            original_dir = os.path.realpath(hydra.utils.get_original_cwd())
                            path = 'logs/{task}/{framework}/{emitter}/{network}/{code}/{run}/tm.csv'
                            path = path.format(
                                task=self._cfg.task.name,
                                framework=self._cfg.framework.name,
                                emitter=self._cfg.emitter.name,
                                network=self._cfg.network.name,
                                code=self._cfg.code,
                                run=self._cfg.run,
                            )
                            with open(os.path.join(original_dir, path), 'w') as f:
                                f.write(f'{loop_completed},{pbar.n},{smoothed_time_per_loop}\n')
                            return

                    pbar.unpause()

    def clear(self) -> None:
        super().clear()
        del self.state, self.constant, self.history
        del self._task, self._reeval_task, self._scoring_fn, self._seed_array
