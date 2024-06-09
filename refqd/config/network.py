from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from dataclasses import dataclass
from typing import Optional, TypeVar, Any


@dataclass
class NetworkConfig:
    task_type: str = MISSING
    task_subtype: str = MISSING
    emitter_type: str = MISSING
    name: str = MISSING


_configs: dict[str, type[NetworkConfig]] = {}


_NetworkConfigT = TypeVar('_NetworkConfigT', bound=NetworkConfig)


def _register(cls: type[_NetworkConfigT]) -> type[_NetworkConfigT]:
    assert cls.name != MISSING
    assert cls.name not in _configs
    _configs[cls.name] = cls
    return cls


@dataclass
class CNNPolicyNetConfig(NetworkConfig):
    task_type: str = 'RL'

    conv_features: tuple[int, ...] = ()
    conv_kernel_sizes: tuple[Any, ...] = ()
    conv_strides: tuple[int, ...] = ()
    mlp_hidden_layer_sizes: tuple[int, ...] = (256, 256,)

    conv_activation: str = 'leaky_relu'
    conv_padding: str = 'SAME'

    mlp_activation: str = 'tanh'
    mlp_final_activation: Optional[str] = 'tanh'


@dataclass
class NormalCNNPolicyNetConfig(CNNPolicyNetConfig):
    emitter_type: str = 'Normal'


@dataclass
class SharedCNNPolicyNetConfig(CNNPolicyNetConfig):
    emitter_type: str = 'Share'

    mlp_hidden_layer_sizes: tuple[int, ...] = ()

    decision_cnn_input_shape: tuple[int, ...] = (-1,)
    representation_conv_features: tuple[int, ...] = ()
    representation_conv_kernel_sizes: tuple[Any, ...] = ()
    representation_conv_strides: tuple[int, ...] = ()
    representation_mlp_hidden_layer_sizes: tuple[int, ...] = (256, 256,)


@_register
@dataclass
class NormalQDaxBraxNetConfig(NormalCNNPolicyNetConfig):
    task_subtype: str = 'QDaxBrax'
    name: str = f'{task_subtype}-{NormalCNNPolicyNetConfig.emitter_type}'


@_register
@dataclass
class SharedQDaxBraxNetConfig(SharedCNNPolicyNetConfig):
    task_subtype: str = 'QDaxBrax'
    name: str = f'{task_subtype}-{SharedCNNPolicyNetConfig.emitter_type}'


@_register
@dataclass
class NormalAtariNetConfig(NormalCNNPolicyNetConfig):
    task_subtype: str = 'Atari'
    name: str = f'{task_subtype}-{NormalCNNPolicyNetConfig.emitter_type}'

    conv_features: tuple[int, ...] = (32, 64, 64)
    conv_kernel_sizes: tuple[Any, ...] = ((8, 8), (4, 4), (3, 3))
    conv_strides: tuple[int, ...] = (4, 2, 1)
    mlp_hidden_layer_sizes: tuple[int, ...] = (512,)

    conv_activation: str = 'leaky_relu'
    conv_padding: str = 'SAME'

    mlp_activation: str = 'leaky_relu'
    mlp_final_activation: Optional[str] = None


@_register
@dataclass
class SharedAtariNetConfig(SharedCNNPolicyNetConfig):
    task_subtype: str = 'Atari'
    name: str = f'{task_subtype}-{SharedCNNPolicyNetConfig.emitter_type}'

    conv_features: tuple[int, ...] = ()
    conv_kernel_sizes: tuple[Any, ...] = ()
    conv_strides: tuple[int, ...] = ()
    mlp_hidden_layer_sizes: tuple[int, ...] = ()

    decision_cnn_input_shape: tuple[int, ...] = (-1,)
    representation_conv_features: tuple[int, ...] = (32, 64, 64)
    representation_conv_kernel_sizes: tuple[Any, ...] = ((8, 8), (4, 4), (3, 3))
    representation_conv_strides: tuple[int, ...] = (4, 2, 1)
    representation_mlp_hidden_layer_sizes: tuple[int, ...] = (512,)


def register_configs(group: str) -> None:
    cs = ConfigStore.instance()
    for name, cls in _configs.items():
        cs.store(
            group=group,
            name=name,
            node=cls,
        )
