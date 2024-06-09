from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from dataclasses import dataclass
from typing import TypeVar


@dataclass
class FrameworkConfig:
    name: str = MISSING


_configs: dict[str, type[FrameworkConfig]] = {}


_FrameworkConfigT = TypeVar('_FrameworkConfigT', bound=FrameworkConfig)


def _register(cls: type[_FrameworkConfigT]) -> type[_FrameworkConfigT]:
    assert cls.name != MISSING
    assert cls.name not in _configs
    _configs[cls.name] = cls
    return cls


@_register
@dataclass
class MEConfig(FrameworkConfig):
    name: str = 'ME'
    n_init_cvt_samples: int = 50000
    n_centroids: int = 1000


def register_configs(group: str) -> None:
    cs = ConfigStore.instance()
    for name, cls in _configs.items():
        cs.store(
            group=group,
            name=name,
            node=cls,
        )
