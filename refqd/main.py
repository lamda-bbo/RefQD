import logging
import wandb
import os
import sys
from typing import TYPE_CHECKING

from .manager import Manager
from .utils import wandb_alert, pdb_wrap, fork_detached

if TYPE_CHECKING:
    from .config.root import RootConfig


_log = logging.getLogger(__name__)


def set_time_format() -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        formatter = handler.formatter
        if formatter is not None:
            formatter.default_time_format = '%m-%d %H:%M:%S'


def main(cfg: 'RootConfig') -> None:
    set_time_format()

    with pdb_wrap('Manager.__init__', post_mortem=cfg.pdb):
        manager = Manager(cfg)

    with pdb_wrap('manager.dry_run', post_mortem=cfg.pdb):
        manager.dry_run()
    if cfg.typ == 'dry':
        return

    with pdb_wrap('manager.init', post_mortem=cfg.pdb):
        manager.init()

    try:
        with pdb_wrap('manager.run', post_mortem=cfg.pdb):
            manager.run()
    except KeyboardInterrupt:
        _log.info('Saving...')
        manager.save()
        raise
    finally:
        manager.wandb_upload_metrics()
        reduced = manager.pickle_reduced()
        manager.clear()

        if wandb.run is not None:
            match sys.exc_info()[1]:
                case None:
                    title = 'Finished'
                    level = wandb.AlertLevel.INFO
                    exit_code = None
                case KeyboardInterrupt():
                    title = 'Killed'
                    level = wandb.AlertLevel.WARN
                    exit_code = 255
                case _:
                    title = 'Crashed'
                    level = wandb.AlertLevel.ERROR
                    exit_code = 1
            wandb_alert(title, wandb.run.id, level)
            wandb.run.finish(exit_code)
        _log.info('Command: %s', ' '.join(sys.orig_argv).replace('[', '"[').replace(']', ']"'))
        _log.info('Logdir: %s', os.getcwd())

        if cfg.fork_final:
            fork_detached(manager.final, reduced)
        else:
            manager.final(reduced)
