from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from dataclasses import dataclass
from typing import TypeVar


@dataclass
class TaskConfig:
    name: str = MISSING
    typ: str = MISSING
    subtype: str = MISSING

    log_period: int = MISSING
    reeval_period: int = MISSING


_configs: dict[str, type[TaskConfig]] = {}
_property: dict[str, dict[str, str]] = {}


_TaskConfigT = TypeVar('_TaskConfigT', bound=TaskConfig)


def _register(cls: type[_TaskConfigT]) -> type[_TaskConfigT]:
    assert cls.name != MISSING
    assert cls.name not in _configs
    _configs[cls.name] = cls
    _property[cls.name] = {'typ': cls.typ, 'subtype': cls.subtype}
    return cls


@dataclass
class RLTaskConfig(TaskConfig):
    typ: str = 'RL'
    episode_len: int = 250
    total_steps: int = 150_000_000
    reduce_obs: bool = True


@dataclass
class QDaxBraxConfig(RLTaskConfig):
    subtype: str = 'QDaxBrax'
    log_period: int = 10
    reeval_period: int = 1


@_register
@dataclass
class HopperUniEnvConfig(QDaxBraxConfig):
    name: str = 'hopper_uni'


@_register
@dataclass
class Walker2DUniEnvConfig(QDaxBraxConfig):
    name: str = 'walker2d_uni'


@_register
@dataclass
class HalfCheetahUniEnvConfig(QDaxBraxConfig):
    name: str = 'halfcheetah_uni'


@_register
@dataclass
class AntUniEnvConfig(QDaxBraxConfig):
    name: str = 'ant_uni'


@_register
@dataclass
class HumanoidUniEnvConfig(QDaxBraxConfig):
    name: str = 'humanoid_uni'


@_register
@dataclass
class AntOmniEnvConfig(QDaxBraxConfig):
    name: str = 'ant_omni'


@_register
@dataclass
class HumanoidOmniEnvConfig(QDaxBraxConfig):
    name: str = 'humanoid_omni'


@_register
@dataclass
class PointMazeEnvConfig(QDaxBraxConfig):
    name: str = 'pointmaze'


@_register
@dataclass
class AntMazeEnvConfig(QDaxBraxConfig):
    name: str = 'antmaze'


@_register
@dataclass
class AntTrapEnvConfig(QDaxBraxConfig):
    name: str = 'anttrap'


@_register
@dataclass
class HumanoidTrapEnvConfig(QDaxBraxConfig):
    name: str = 'humanoidtrap'


@_register
@dataclass
class AntNoTrapEnvConfig(QDaxBraxConfig):
    name: str = 'antnotrap'


@dataclass
class GymConfig(RLTaskConfig):
    gymnasium: bool = True


@dataclass
class EnvPoolConfig(RLTaskConfig):
    pass


@dataclass
class AtariConfig(EnvPoolConfig):
    subtype: str = 'Atari'
    log_period: int = 2
    reeval_period: int = 5
    obs_shape: tuple[int, int] = (84, 84)


@_register
@dataclass
class PongEnvConfig(AtariConfig):
    name: str = 'Pong-v5'
    episode_len: int = 2000


@_register
@dataclass
class BoxingEnvConfig(AtariConfig):
    name: str = 'Boxing-v5'
    episode_len: int = 1800


def register_configs(group: str) -> None:
    cs = ConfigStore.instance()
    for name, cls in _configs.items():
        cs.store(
            group=group,
            name=name,
            node=cls,
        )


def get_properties(name: str) -> dict[str, str]:
    return _property[name]
