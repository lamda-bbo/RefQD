from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from dataclasses import dataclass
from typing import Optional, TypeVar, Any


@dataclass
class CriticNetConfig:
    task_subtype: str = MISSING
    emitter_type: str = MISSING
    name: str = MISSING


_configs: dict[str, type[CriticNetConfig]] = {}


_CriticNetConfigT = TypeVar('_CriticNetConfigT', bound=CriticNetConfig)


def _register(cls: type[_CriticNetConfigT]) -> type[_CriticNetConfigT]:
    assert cls.name != MISSING
    assert cls.name not in _configs
    _configs[cls.name] = cls
    return cls


@dataclass
class CNNCriticNetConfig(CriticNetConfig):
    conv_features: tuple[int, ...] = ()
    conv_kernel_sizes: tuple[Any, ...] = ()
    conv_strides: tuple[int, ...] = ()
    mlp_hidden_layer_sizes: tuple[int, ...] = (256, 256,)

    conv_activation: str = 'leaky_relu'
    conv_padding: str = 'SAME'

    mlp_activation: str = 'leaky_relu'
    mlp_final_activation: Optional[str] = None


@dataclass
class NormalCNNCriticNetConfig(CNNCriticNetConfig):
    emitter_type: str = 'Normal'


@dataclass
class SharedCNNCriticNetConfig(CNNCriticNetConfig):
    emitter_type: str = 'Share'


@_register
@dataclass
class NormalQDaxBraxCriticNetConfig(NormalCNNCriticNetConfig):
    task_subtype: str = 'QDaxBrax'
    name: str = f'{task_subtype}-{NormalCNNCriticNetConfig.emitter_type}'


@_register
@dataclass
class SharecQDaxBraxCriticNetConfig(SharedCNNCriticNetConfig):
    task_subtype: str = 'QDaxBrax'
    name: str = f'{task_subtype}-{SharedCNNCriticNetConfig.emitter_type}'


def register_configs(group: str) -> None:
    cs = ConfigStore.instance()
    for name, cls in _configs.items():
        cs.store(
            group=group,
            name=name,
            node=cls,
        )
