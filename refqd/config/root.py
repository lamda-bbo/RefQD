from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from dataclasses import dataclass, field
import json
import time
from typing import Optional, Any

from . import task
from . import framework
from . import emitter
from . import network
from .task import TaskConfig
from .framework import FrameworkConfig
from .emitter import EmitterConfig
from .network import NetworkConfig


defaults_base = [
]


@dataclass
class RootConfigBase:
    defaults: list[dict[str, Any]] = field(default_factory=lambda: defaults_base)

    multiseed: Optional[str] = None
    seed: tuple[int, ...] = MISSING
    seedstr: str = MISSING
    code: str = 'wang-refqd-icml24'
    run: str = 'normal'

    wandb: bool = True
    check_config: bool = True
    config_filename: str = 'cfg.yaml'
    runtime_env_filename: str = 'runtime_env.json'
    metrics_filename: str = 'metrics.csv'
    checkpoint_filename: str = 'checkpoint.pkl.lz4'
    compressed_checkpoint_filename: str = 'checkpoint.pkl.xz'
    reduced_checkpoint_filename: str = 'reduced.pkl.xz'
    tmpfile_postfix: str = '~'
    typ: str = 'main'
    pdb: bool = True
    debug_nans: bool = True
    fork_final: bool = True


defaults = [
    *defaults_base,
    {'task': MISSING},
    {'framework': 'ME'},
    {'emitter': MISSING},
    {'network': MISSING},
]


@dataclass
class RootConfig(RootConfigBase):
    defaults: list[dict[str, Any]] = field(default_factory=lambda: defaults)

    task: TaskConfig = MISSING
    framework: FrameworkConfig = MISSING
    emitter: EmitterConfig = MISSING
    network: NetworkConfig = MISSING

    checkpoint_saving_interval: int = 100
    metrics_uploading_interval: int = 20
    n_profile: int = 0


def register_all_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name='root_config', node=RootConfig)
    task.register_configs('task')
    framework.register_configs('framework')
    emitter.register_configs('emitter')
    network.register_configs('network')


def get_extra_overrides(overrides: dict[str, str]) -> dict[str, str]:
    assert 'task' in overrides.keys() and 'emitter' in overrides.keys()
    assert 'seed' in overrides.keys()
    assert 'seedstr' not in overrides.keys()

    extra_overrides: dict[str, str] = {}

    properties = {
        'task': task.get_properties(overrides['task']),
        'emitter': emitter.get_properties(overrides['emitter']),
    }
    if 'network' not in overrides:
        extra_overrides['network'] = '{task[subtype]}-{emitter[typ]}'.format(**properties)

    emitter_overrides = emitter.get_emitter_net_overrides(
        properties['task'], overrides['emitter']
    )
    for key, value in emitter_overrides.items():
        key = f'emitter/{key}'
        if key not in overrides:
            extra_overrides[key] = value

    seed = json.loads(overrides['seed'])
    if isinstance(seed, int):
        extra_overrides['seed'] = f'[{seed}]'
        extra_overrides['seedstr'] = str(seed)
    else:
        extra_overrides['seedstr'] = '-'.join(map(str, seed))
        if 'multiseed' not in overrides:
            extra_overrides['multiseed'] = 'vmap'

    if 'typ' in overrides.keys() and overrides['typ'] == 'dry':
        if 'wandb' in overrides.keys():
            assert overrides['wandb'] == 'False'
        else:
            extra_overrides['wandb'] = 'False'
        extra_overrides['hydra.run.dir'] = f'/tmp/refqd-dry-logs/{time.strftime("%m%dT%H%M%S")}'
        extra_overrides['hydra.sweep.dir'] = '/tmp/refqd-dry-logs'
        extra_overrides['hydra.sweep.subdir'] = time.strftime("%m%dT%H%M%S")

    return extra_overrides
