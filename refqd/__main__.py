import hydra
import os
import sys
from typing import TYPE_CHECKING


if __name__ == '__main__':
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if xla_flags != '':
        xla_flags = ' ' + xla_flags
    os.environ['XLA_FLAGS'] = f'--xla_gpu_deterministic_ops=true{xla_flags}'

    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'


from .main import main
from .config.root import register_all_configs, get_extra_overrides

if TYPE_CHECKING:
    from .config.root import RootConfig


def preprocess_argv() -> None:
    overrides: dict[str, str] = {}
    override_idx: dict[str, int] = {}
    for i, arg in enumerate(sys.argv[1:]):
        try:
            key, value = arg.split('=')
        except Exception:
            continue
        override_idx[key] = i + 1
        overrides[key] = value
    extra_overrides = get_extra_overrides(overrides)
    for key, value in extra_overrides.items():
        if key in override_idx.keys():
            sys.argv[override_idx[key]] = f'{key}={value}'
        else:
            sys.argv.append(f'{key}={value}')


if __name__ == '__main__':
    register_all_configs()
    preprocess_argv()

    @hydra.main(config_path='../config', config_name='config', version_base=None)
    def wrapped_main(cfg: 'RootConfig') -> None:
        main(cfg)

    wrapped_main()
