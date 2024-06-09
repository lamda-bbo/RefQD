from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from dataclasses import dataclass, field
from typing import Optional, TypeVar, Any

from .critic_net import CriticNetConfig, register_configs as register_critic_net_configs
from ..emitters import PGAMEConfig, RefPGAMEConfig, DQNMEConfig, RefDQNMEConfig


pg_defaults = [
    {'critic_net': MISSING},
]


@dataclass
class EmitterConfig:
    defaults: list[dict[str, Any]] = field(default_factory=lambda: [])

    name: str = MISSING
    typ: str = 'Normal'
    subtype: str = MISSING

    env_batch_size: int = 100
    reeval_env_batch_size: Optional[int] = None
    reeval_factor: int = 5

    refresh_depth: int = 0

    repertoire_type: str = 'GPU'
    repertoire_kwargs: dict[str, Any] = field(default_factory=lambda: {})


_configs: dict[str, type[EmitterConfig]] = {}
_property: dict[str, dict[str, str]] = {}


_EmitterConfigT = TypeVar('_EmitterConfigT', bound=EmitterConfig)


def _register(cls: type[_EmitterConfigT]) -> type[_EmitterConfigT]:
    assert cls.name != MISSING
    assert cls.name not in _configs
    _configs[cls.name] = cls
    _property[cls.name] = {'typ': cls.typ}
    return cls


@_register
@dataclass
class PGAMEGEmitterConfig(EmitterConfig, PGAMEConfig):
    defaults: list[dict[str, Any]] = field(default_factory=lambda: pg_defaults)

    name: str = 'PGA-ME-G'
    subtype: str = 'PGA-ME'

    proportion_mutation_ga: float = 0.5
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100

    critic_net: CriticNetConfig = MISSING

    replay_buffer_size: int = 1000000
    critic_learning_rate: float = 3e-4
    greedy_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 32
    soft_tau_update: float = 0.005
    policy_delay: int = 2

    iso_sigma: float = 0.005
    line_sigma: float = 0.05


@_register
@dataclass
class PGAMEEmitterConfig(PGAMEGEmitterConfig):
    name: str = 'PGA-ME'
    repertoire_type: str = 'CPU'


@_register
@dataclass
class PGAMESEmitterConfig(PGAMEEmitterConfig):
    name: str = 'PGA-ME-S'
    env_batch_size: int = 8
    reeval_env_batch_size: Optional[int] = 100
    reeval_factor: int = 60
    num_critic_training_steps: int = 24


@_register
@dataclass
class VRefPGAMEGEmitterConfig(EmitterConfig, RefPGAMEConfig):
    defaults: list[dict[str, Any]] = field(default_factory=lambda: pg_defaults)

    name: str = 'V-Ref-PGA-ME-G'
    typ: str = 'Share'
    subtype: str = 'Ref-PGA-ME'

    proportion_mutation_ga: float = 0.5
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100

    critic_net: CriticNetConfig = MISSING

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
    batch_size: int = 32
    save_emitted_representation_params: bool = True
    num_decision_updating_representation: int = 50
    decision_factor: float = 1.0
    soft_tau_update: float = 0.005
    policy_delay: int = 2

    iso_sigma: float = 0.005
    line_sigma: float = 0.05


@_register
@dataclass
class VRefPGAMEEmitterConfig(VRefPGAMEGEmitterConfig):
    name: str = 'V-Ref-PGA-ME'
    repertoire_type: str = 'CPU'


@_register
@dataclass
class RefPGAMEEmitterConfig(VRefPGAMEEmitterConfig):
    name: str = 'Ref-PGA-ME'
    representation_lr_decay_rate: float = 0.98
    repertoire_type: str = 'CPU-Depth'
    repertoire_kwargs: dict[str, Any] = field(default_factory=lambda: {'depth': 4})
    reeval_factor: int = 5
    refresh_depth: int = 1


@_register
@dataclass
class DQNMEGEmitterConfig(EmitterConfig, DQNMEConfig):
    name: str = 'DQN-ME-G'
    subtype: str = 'DQN-ME'

    proportion_mutation_ga: float = 0.5

    num_dqn_training_steps: int = 300
    num_mutation_steps: int = 100
    replay_buffer_size: int = 200000
    greedy_learning_rate: float = 3e-4
    learning_rate: float = 1e-3
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 32
    target_policy_update_interval: int = 10
    using_greedy: bool = True

    iso_sigma: float = 0.005
    line_sigma: float = 0.05


@_register
@dataclass
class DQNMEEmitterConfig(DQNMEGEmitterConfig):
    name: str = 'DQN-ME'
    repertoire_type: str = 'CPU'


@_register
@dataclass
class DQNMESEmitterConfig(DQNMEEmitterConfig):
    name: str = 'DQN-ME-S'
    env_batch_size: int = 8
    reeval_factor: int = 60
    num_dqn_training_steps: int = 24


@_register
@dataclass
class VRefDQNMEGEmitterConfig(EmitterConfig, RefDQNMEConfig):
    name: str = 'V-Ref-DQN-ME-G'
    typ: str = 'Share'
    subtype: str = 'Ref-DQN-ME'

    proportion_mutation_ga: float = 0.5

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
    num_decision_updating_representation: int = 50
    decision_factor: float = 1.0
    using_greedy: bool = True

    iso_sigma: float = 0.005
    line_sigma: float = 0.05


@_register
@dataclass
class VRefDQNMEEmitterConfig(VRefDQNMEGEmitterConfig):
    name: str = 'V-Ref-DQN-ME'
    repertoire_type: str = 'CPU'


@_register
@dataclass
class RefDQNMEEmitterConfig(VRefDQNMEEmitterConfig):
    name: str = 'Ref-DQN-ME'
    representation_lr_decay_rate: float = 0.98
    repertoire_type: str = 'CPU-Depth'
    repertoire_kwargs: dict[str, Any] = field(default_factory=lambda: {'depth': 2})
    reeval_factor: int = 5
    refresh_depth: int = 1


def register_configs(group: str) -> None:
    register_critic_net_configs(f'{group}/critic_net')

    cs = ConfigStore.instance()
    for name, cls in _configs.items():
        cs.store(
            group=group,
            name=name,
            node=cls,
        )


def get_properties(name: str) -> dict[str, str]:
    return _property[name]


def get_emitter_net_overrides(task_properties: dict[str, str], emitter_name: str) -> dict[str, str]:
    if hasattr(_configs[emitter_name], 'critic_net'):
        return {'critic_net': f'{task_properties["subtype"]}-{_property[emitter_name]["typ"]}'}
    else:
        return {}
