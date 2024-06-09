import jax
import flax.struct

import hydra.utils
from omegaconf import OmegaConf
import logging
import wandb
import pickle
import cloudpickle
import json
import git
import os
import threading
import socket
import requests_unixsocket
import lz4.frame
from abc import abstractmethod
from collections.abc import Sequence
from typing import TypeVar, Generic, Any, assert_never

from .treax import numpy as tjnp
from .config.root import RootConfigBase
from .utils import RNGKey, ignore_exception, jax_eval_shape, xz_write, xz_read, ExcThread


_log = logging.getLogger(__name__)


class ManagerStateBase(flax.struct.PyTreeNode):
    random_key: RNGKey


class ManagerConstantBase(flax.struct.PyTreeNode):
    pass


_LOG_AGENT_PATH = '/tmp/refqd-log-agent.sock'
_LOG_AGENT_URL = f"http+unix://{_LOG_AGENT_PATH.replace('/', '%2F')}"


_RootConfigBaseT = TypeVar('_RootConfigBaseT', bound=RootConfigBase)
_ManagerStateBaseT = TypeVar('_ManagerStateBaseT', bound=ManagerStateBase)
_ManagerConstantBaseT = TypeVar('_ManagerConstantBaseT', bound=ManagerConstantBase)


class ManagerBase(Generic[_RootConfigBaseT, _ManagerStateBaseT, _ManagerConstantBaseT]):

    def __init__(self, cfg: '_RootConfigBaseT') -> None:
        self._cfg = cfg

        _log.info('Config:\n\n%s', OmegaConf.to_yaml(self._cfg))
        _log.info('Logdir: %s', os.getcwd())

        if self._cfg.debug_nans:
            jax.config.update('jax_debug_nans', True)

        self._dry_run = False

        RESUME_KEY = 'RESUME'
        resume_flag = os.environ.get(RESUME_KEY, 'never')
        assert resume_flag in (
            'never', 'allow', 'must', 'force-allow', 'force-must', 'overwrite'
        )
        self._RESUME_FLAG = resume_flag

        assert not os.path.exists(self._cfg.checkpoint_filename)

        if self._cfg.wandb:
            self._wandb_init()

        if wandb.run is not None:
            self._resumed: bool = wandb.run.resumed
        else:
            self._resumed = os.path.exists(self._cfg.config_filename)

        match self._RESUME_FLAG:
            case 'never' | 'overwrite':
                assert not self._resumed
            case 'must' | 'force-must':
                assert self._resumed
            case 'allow' | 'force-allow':
                pass
            case void:
                assert_never(void)

        if self._resumed:
            _log.info('Resuming from the previous run...')
            self.check_config()
        if self._resumed:
            self.restore_metrics()
            self.restore_checkpoint()
        else:
            _log.info('Setting up a fresh run...')
            assert not os.path.exists(self._cfg.config_filename)
            assert not os.path.exists(self._cfg.metrics_filename)
            assert not os.path.exists(self._cfg.compressed_checkpoint_filename)
        self.save_config()

        self.saved = False
        self.compressed = False

        self.state: _ManagerStateBaseT
        self.constant: _ManagerConstantBaseT

    @property
    def _wandb_resume_arg(self):
        match self._RESUME_FLAG:
            case 'never' | 'allow' | 'must':
                resume = self._RESUME_FLAG
            case 'force-allow' | 'force-must':
                resume = self._RESUME_FLAG[6:]
            case 'overwrite':
                _log.warning('RESUME is set to overwrite. The previous run may be overwritten.')
                resume = None
            case void:
                assert_never(void)
        assert resume in ('never', 'allow', 'must', None)
        return resume

    @abstractmethod
    def _wandb_init(self) -> None:
        pass

    def _wandb_upload_now(self, filename: str) -> None:
        if wandb.run is not None:
            threading.Thread(
                target=self._do_upload_logs,
                args=(os.path.abspath(filename),)
            ).start()
            wandb.run.save(filename, policy='now')

    def wandb_upload_metrics(self) -> None:
        if wandb.run is not None:
            _log.info('Uploading metrics...')
            self._wandb_upload_now(self._cfg.metrics_filename)

    @staticmethod
    def restore_from_wandb(filename: str, warning: str) -> None:
        try:
            f = wandb.restore(filename, replace=True, root='.')
            assert f is not None
        except Exception as e:
            _log.warning(warning, e)
        else:
            f.close()

    def save_config(self) -> None:
        with open(self._cfg.config_filename, 'w') as f:
            f.write(OmegaConf.to_yaml(self._cfg))
        self._wandb_upload_now(self._cfg.config_filename)

        original_dir = os.path.realpath(hydra.utils.get_original_cwd())
        Repo = git.Repo  # pyright: ignore [reportPrivateImportUsage]
        repo = Repo(original_dir)
        runtime_env = {
            'hostname': socket.gethostname(),
            'commit_hash': repo.git.log('HEAD', n=1, pretty='format:%H'),
            'diff': repo.git.diff('HEAD'),
        }
        with open(self._cfg.runtime_env_filename, 'w') as f:
            json.dump(runtime_env, f, indent=4)
        self._wandb_upload_now(self._cfg.runtime_env_filename)

    def _check_config_yaml(self, old_config_yaml, new_config_yaml) -> bool:
        return old_config_yaml == new_config_yaml

    def check_config_locally(self) -> None:
        with open(self._cfg.config_filename, 'r') as f:
            old_config_yaml = f.read()
        new_config_yaml = OmegaConf.to_yaml(self._cfg)
        if not self._check_config_yaml(old_config_yaml, new_config_yaml):
            msg = (
                'The new config is different from the old config.\n'
                'The old config:\n\n{old}\nThe new config: \n\n{new}\n'
                .format(old=old_config_yaml, new=new_config_yaml)
            )
            match self._RESUME_FLAG:
                case 'force-allow' | 'force-must':
                    _log.warning(msg)
                case 'allow' | 'must':
                    raise ValueError(
                        msg +
                        'Delete the local files and set RESUME=overwrite '
                        'in environment variables to overwrite the previous run.\n'
                        'Set RESUME=force in environment variables to resume from the previous run.'
                    )
                case 'never' | 'overwrite':
                    raise AssertionError(self._RESUME_FLAG)
                case void:
                    assert_never(void)

    def check_config(self) -> None:
        if not self._cfg.check_config:
            _log.warning('Skipped from config checking')
            return
        _log.info('Checking the config...')

        try:
            self.check_config_locally()
        except FileNotFoundError:
            pass

        if wandb.run is not None:
            self.restore_from_wandb(
                self._cfg.config_filename,
                '%s Using the local config instead.',
            )
        else:
            _log.warning('Wandb is disabled. Using the local config instead.')

        try:
            self.check_config_locally()
        except FileNotFoundError as e:
            match self._RESUME_FLAG:
                case 'must' | 'force-must':
                    e.add_note('Failed to resume from the previous run.')
                    raise
                case 'allow' | 'force-allow':
                    _log.warning('%s. Restarting the run instead.', e)
                    self._resumed = False
                case 'never' | 'overwrite':
                    raise AssertionError(self._RESUME_FLAG)
                case void:
                    assert_never(void)

    @property
    def _saved_keys(self) -> Sequence[str]:
        return ('state', 'constant=')

    @property
    def globals(self) -> dict[str, Any]:
        assert type(self) is ManagerBase
        return globals()

    def _get_saved_value(self, key: str) -> Any:
        if key[-1] == '=':
            key = key[:-1]
        keys = key.split('.')
        del key
        if keys[0][0] != '$':
            value = getattr(self, keys[0])
        else:
            value = self.globals[keys[0][1:]]
        for key in keys[1:]:
            value = getattr(value, key)
        return value

    def _set_saved_value(self, key: str, value: Any, checkeq: bool = True) -> None:
        checkeq = checkeq and key[-1] == '='
        if checkeq:
            key = key[:-1]
        keys = key.split('.')
        del key
        if keys[0][0] != '$':
            obj = self
        else:
            if len(keys) == 1:
                if checkeq:
                    return
                    assert jax.tree_util.tree_all(tjnp.allclose(self.globals[keys[0][1:]], value))
                else:
                    self.globals[keys[0][1:]] = value
                return
            else:
                obj = self.globals[keys[0][1:]]
                keys = keys[1:]
        for key in keys[:-1]:
            obj = getattr(obj, key)
        if checkeq:
            return
            assert jax.tree_util.tree_all(tjnp.allclose(getattr(obj, keys[-1]), value))
        else:
            setattr(obj, keys[-1], value)

    @property
    def saved_dict(self) -> dict[str, Any]:
        return {key: self._get_saved_value(key) for key in self._saved_keys}

    def save(self) -> None:
        if self.saved:
            return
        with lz4.frame.open(self._cfg.checkpoint_filename + self._cfg.tmpfile_postfix, 'wb') as f:
            cloudpickle.dump(self.saved_dict, f)
        os.replace(
            self._cfg.checkpoint_filename + self._cfg.tmpfile_postfix,
            self._cfg.checkpoint_filename,
        )
        self.compressed = False
        self.saved = True
        if os.path.isfile(self._cfg.compressed_checkpoint_filename):
            os.remove(self._cfg.compressed_checkpoint_filename)

    @abstractmethod
    def _reduce(self, saved_dict: dict[str, Any]) -> dict[str, Any]:
        ...

    def pickle_reduced(self) -> bytes:
        saved_dict = self.saved_dict
        saved_dict = self._reduce(saved_dict)
        reduced = cloudpickle.dumps(saved_dict)
        return reduced

    def compress(self) -> None:
        if self.compressed:
            return
        with lz4.frame.open(self._cfg.checkpoint_filename, 'rb') as f:
            data = f.read()
            assert isinstance(data, bytes)
        xz_write(data, self._cfg.compressed_checkpoint_filename, verbose=not self._cfg.fork_final)
        os.remove(self._cfg.checkpoint_filename)
        self.compressed = True

    def save_reduced(self, reduced: bytes) -> None:
        xz_write(reduced, self._cfg.reduced_checkpoint_filename, verbose=not self._cfg.fork_final)

    def _do_upload_logs(self, path: str) -> None:
        try:
            res = requests_unixsocket.post(f'{_LOG_AGENT_URL}/upload', params={'path': path})
            res.raise_for_status()
            assert res.status_code == 200
        except Exception:
            pass

    def final(self, reduced: bytes) -> None:
        def log_info(msg: str) -> None:
            with ignore_exception(IOError):
                _log.info(msg)

        log_info('Compressing...')
        try:
            self.compress()
        except FileNotFoundError as e:
            with ignore_exception(IOError):
                _log.warning('%s. Failed to compress the checkpoint.', e)

        if self.saved:
            log_info('Saving reduced checkpoints...')
            self.save_reduced(reduced)

        if self._cfg.wandb:
            log_info('Uploading logs...')
            self._do_upload_logs(os.path.abspath('.'))

        log_info('Done')

    def restore_metrics(self) -> None:
        _log.info('Restoring the metrics...')
        if wandb.run is not None:
            self.restore_from_wandb(
                self._cfg.metrics_filename,
                '%s Using the local metrics file instead.',
            )
        else:
            _log.warning('Wandb is disabled. Using the local metrics instead.')

    def _do_restore_checkpoint(self) -> None:
        path = os.path.abspath(self._cfg.compressed_checkpoint_filename)
        res = requests_unixsocket.post(f'{_LOG_AGENT_URL}/restore', params={'path': path})
        res.raise_for_status()
        assert res.status_code == 200

        path = os.path.abspath(self._cfg.checkpoint_filename)
        try:
            res = requests_unixsocket.post(f'{_LOG_AGENT_URL}/restore', params={'path': path})
        except Exception:
            pass

    def restore_checkpoint(self) -> None:
        _log.info('Restoring the checkpoint...')
        if wandb.run is not None:
            self._thread_restore = ExcThread(
                target=self._do_restore_checkpoint,
            )
            self._thread_restore.start()
        else:
            _log.warning('Wandb is disabled. Using the local checkpoint instead.')

    def load(self) -> None:
        if wandb.run is not None:
            _log.info('Waiting for the checkpoint...')
            assert hasattr(self, '_thread_restore')
            try:
                self._thread_restore.join()
            except Exception as e:
                _log.warning('%s Using the local checkpoint instead.', e)
            del self._thread_restore

        if os.path.exists(self._cfg.metrics_filename):
            with open(self._cfg.metrics_filename, 'r') as f:
                n_lines = len(f.readlines())
            if n_lines > 2:
                assert os.path.exists(self._cfg.compressed_checkpoint_filename)
        else:
            assert not os.path.exists(self._cfg.compressed_checkpoint_filename)

        if os.path.exists(self._cfg.compressed_checkpoint_filename):
            _log.info('Loading the checkpoint...')
            assert not os.path.exists(self._cfg.checkpoint_filename)
            for key in self._saved_keys:
                self._set_saved_value(key, None)
            saved_dict: dict[str, Any] = pickle.loads(
                xz_read(self._cfg.compressed_checkpoint_filename, verbose=True)
            )
            assert tuple(saved_dict.keys()) == tuple(self._saved_keys)
            for key, value in saved_dict.items():
                self._set_saved_value(key, value)
            self.saved = True
            self.compressed = True
        else:
            match self._RESUME_FLAG:
                case 'must' | 'force-must':
                    e = FileNotFoundError(self._cfg.compressed_checkpoint_filename)
                    e.add_note('Failed to resume from the previous run.')
                    raise e
                case 'allow' | 'force-allow':
                    _log.warning('The checkpoint is not found. Restarting the run instead.')
                    self._resumed = False
                case 'never' | 'overwrite':
                    raise AssertionError(self._RESUME_FLAG)
                case void:
                    assert_never(void)

    @abstractmethod
    def init(self) -> None:
        ...

    @abstractmethod
    def run(self) -> None:
        ...

    def _do_dry_run(self) -> None:
        _log.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Dry Run Started >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        previous_dry_run = self._dry_run
        self._dry_run = True
        try:
            self.init()
            self.run()
            self.saved_dict
            _log.info(
                '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Dry Run Ended <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
            )
        finally:
            self._dry_run = previous_dry_run

    def dry_run(self) -> None:
        return jax_eval_shape(self._do_dry_run)

    def clear(self) -> None:
        for key in self._saved_keys:
            self._set_saved_value(key, None, checkeq=False)
