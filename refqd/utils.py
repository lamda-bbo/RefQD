import jax
import jax.numpy as jnp
import jax.core
from jax._src.api import AxisName
import jax._src.sharding_impls
import jaxlib.xla_client
import jaxlib.xla_extension
import optax

import numpy as np

import logging
import colorlog
from functools import partial
import wandb
import csv
import rich.console
import rich.traceback
import rich.text
import io
import time
import os
import subprocess
import threading
import signal
import pdb
from contextlib import contextmanager
from collections.abc import Sequence, Generator, Iterable, Mapping, Callable
from typing import (
    Literal, Optional, TypeAlias, TypeVar, TypeVarTuple, ParamSpec, Concatenate, Generic, Any,
    cast, assert_never, overload,
)
from types import TracebackType


_log = logging.getLogger(__name__)


RNGKey: TypeAlias = jax.Array


_AnyT = TypeVar('_AnyT')


class _Unspecified:
    pass


_unspecified = _Unspecified()


def caster(typ: type[_AnyT]) -> Callable[[Any], _AnyT]:
    def func(obj: Any) -> _AnyT:
        return obj

    return func


@overload
def assert_cast(typ: type[_AnyT], obj: Any) -> _AnyT:
    ...


@overload
def assert_cast(typ: type[_AnyT]) -> Callable[[Any], _AnyT]:
    ...


def assert_cast(typ: type[_AnyT], obj: Any = _unspecified) -> _AnyT | Callable[[Any], _AnyT]:
    def func(obj: Any) -> _AnyT:
        assert isinstance(obj, typ)
        return obj

    if obj == _unspecified:
        return func
    else:
        return func(obj)


_P = ParamSpec('_P')
_IntermT = TypeVar('_IntermT')
_ResultT = TypeVar('_ResultT')


def fnchain(
    func1: Callable[_P, _IntermT], func2: Callable[[_IntermT], _ResultT]
) -> Callable[_P, _ResultT]:

    def func(*args: _P.args, **kwargs: _P.kwargs):
        return func2(func1(*args, **kwargs))

    return func


deoptimized = False


@contextmanager
def deoptimize() -> Generator[None, None, None]:
    global deoptimized
    original_deoptimize_scan = deoptimized
    try:
        deoptimized = True
        yield
    finally:
        deoptimized = original_deoptimize_scan


def _get_func_path(func: Callable) -> str:
    try:
        module = func.__module__
    except AttributeError:
        module = '<unnamed>'
    try:
        name = func.__name__
    except AttributeError:
        name = '<unnamed>'
    return f'{module}.{name}'


def rich_unhighlight(s: str) -> str:
    return str(rich.text.Text.from_markup(str(rich.text.Text.from_ansi(s))))


def rich_highlight(obj: Any) -> str:
    f = io.StringIO()
    console = rich.console.Console(
        file=f, force_terminal=True, force_jupyter=False, width=32768,
    )
    console.print(obj)
    return f.getvalue()[:-1]


def rich_highlight_exception(
    exc_type: type[BaseException], exc_value: BaseException, traceback: TracebackType | None
) -> str:
    f = io.StringIO()
    console = rich.console.Console(
        file=f, force_terminal=True, force_jupyter=False,
    )
    rich_traceback = rich.traceback.Traceback.from_exception(exc_type, exc_value, traceback)
    console.print(rich_traceback)
    return f.getvalue()


def wandb_alert(title: str, text: str, level: wandb.AlertLevel, email: bool = False) -> None:
    if wandb.run is not None:
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] [{level.value}] {title}'
        if text:
            line += f': {text}'
        with open(os.path.expanduser('~/wandbtop_alert.log'), 'a') as f:
            f.write(f'{line}\n')
        if email:
            wandb.run.alert(
                title=title,
                text=text,
                level=level,
            )


def log_exception(
    e: Exception, where: str, level: Literal['info', 'warning', 'error', 'critical'] = 'error'
) -> None:
    match level:
        case 'info':
            log_fn = _log.info
            wandb_level = wandb.AlertLevel.INFO
        case 'warning':
            log_fn = _log.warning
            wandb_level = wandb.AlertLevel.WARN
        case 'error':
            log_fn = _log.error
            wandb_level = wandb.AlertLevel.ERROR
        case 'critical':
            log_fn = _log.critical
            wandb_level = wandb.AlertLevel.ERROR
        case void:
            assert_never(void)
    log_fn(f'{type(e).__name__} in {where}', exc_info=True, stacklevel=2)
    wandb_alert(
        title=f'{type(e).__name__} in {where}',
        text=f'{repr(e)}',
        level=wandb_level,
    )


@contextmanager
def fake_wrap(*args: Any, **kwargs: Any) -> Generator[None, None, None]:
    yield


@contextmanager
def pdb_wrap(
    name: str,
    level: Literal['info', 'warning', 'error', 'critical'] = 'critical',
    post_mortem: bool = True,
) -> Generator[None, None, None]:
    try:
        yield
    except Exception as e:
        log_exception(e, name, level=level)
        if post_mortem:
            pdb.post_mortem(e.__traceback__)
        raise


@contextmanager
def ignore_exception(e: type[Exception]) -> Generator[None, None, None]:
    try:
        yield
    except e:
        pass


def jax_jit(
    fun: Callable[_P, _ResultT],
    in_shardings=jax._src.sharding_impls.UNSPECIFIED,
    out_shardings=jax._src.sharding_impls.UNSPECIFIED,
    static_argnums: Optional[int | Sequence[int]] = None,
    static_argnames: Optional[str | Iterable[str]] = None,
    donate_argnums: int | Sequence[int] = (),
    keep_unused: bool = False,
    device: Optional[jaxlib.xla_client.Device] = None,
    backend: Optional[str] = None,
    inline: bool = False,
    abstracted_axes: Optional[Any] = None,
) -> Callable[_P, _ResultT]:
    new_fun = jax.jit(
        fun,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        keep_unused=keep_unused,
        device=device,
        backend=backend,
        inline=inline,
        abstracted_axes=abstracted_axes,
    )

    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _ResultT:
        if deoptimized:
            return fun(*args, **kwargs)
        else:
            try:
                return new_fun(*args, **kwargs)
            except (TypeError, FloatingPointError, jaxlib.xla_extension.XlaRuntimeError) as e:
                log_exception(e, f'jax_jit of {_get_func_path(fun)}')
                return fun(*args, **kwargs)

    return wrapped


class JITCompiled(jax.stages.Compiled, Generic[_P, _ResultT]):
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _ResultT:
        ...

    def cost_analysis(self) -> list[dict[str, float]] | Any | None:
        return super().cost_analysis()


def jax_compiled(
    fun: Callable[_P, _ResultT],
    in_shardings=jax._src.sharding_impls.UNSPECIFIED,
    out_shardings=jax._src.sharding_impls.UNSPECIFIED,
    static_argnums: Optional[int | Sequence[int]] = None,
    static_argnames: Optional[str | Iterable[str]] = None,
    donate_argnums: int | Sequence[int] = (),
    keep_unused: bool = False,
    device: Optional[jaxlib.xla_client.Device] = None,
    backend: Optional[str] = None,
    inline: bool = False,
    abstracted_axes: Optional[Any] = None,
) -> Callable[_P, JITCompiled[_P, _ResultT]]:
    jit_wrapped = jax.jit(
        fun,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        keep_unused=keep_unused,
        device=device,
        backend=backend,
        inline=inline,
        abstracted_axes=abstracted_axes,
    )

    def compile(*args: _P.args, **kwargs: _P.kwargs) -> JITCompiled[_P, _ResultT]:

        compiled = jit_wrapped.lower(*args, **kwargs).compile()

        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _ResultT:
            if deoptimized:
                return fun(*args, **kwargs)
            else:
                try:
                    return compiled(*args, **kwargs)
                except (TypeError, FloatingPointError, jaxlib.xla_extension.XlaRuntimeError) as e:
                    log_exception(e, f'jax_compiled of {_get_func_path(fun)}')
                    return fun(*args, **kwargs)

        wrapped.cost_analysis = compiled.cost_analysis  # pyright: ignore [reportFunctionMemberAccess]
        wrapped.memory_analysis = compiled.memory_analysis  # pyright: ignore [reportFunctionMemberAccess]

        return cast(JITCompiled[_P, _ResultT], wrapped)

    return compile


_P0 = TypeVar('_P0')
_P1 = TypeVar('_P1')
_PTuple = TypeVarTuple('_PTuple')
_AuxResultT = TypeVar('_AuxResultT')


@overload
def jax_value_and_grad(
    fun: Callable[Concatenate[_P0, _P], jax.Array],
    argnums: Literal[0] = 0,
    has_aux: Literal[False] = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[Concatenate[_P0, _P], tuple[jax.Array, _P0]]:
    ...


@overload
def jax_value_and_grad(
    fun: Callable[Concatenate[_P0, _P1, _P], jax.Array],
    argnums: Literal[1],
    has_aux: Literal[False] = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[Concatenate[_P0, _P1, _P], tuple[jax.Array, _P1]]:
    ...


@overload
def jax_value_and_grad(
    fun: Callable[Concatenate[_P0, _P1, _P], jax.Array],
    argnums: tuple[Literal[0], Literal[1]],
    has_aux: Literal[False] = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[Concatenate[_P0, _P1, _P], tuple[jax.Array, tuple[_P0, _P1]]]:
    ...


@overload
def jax_value_and_grad(
    fun: Callable[_P, jax.Array],
    argnums: int | Sequence[int],
    has_aux: Literal[False] = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[_P, tuple[jax.Array, Any]]:
    ...


@overload
def jax_value_and_grad(
    fun: Callable[Concatenate[_P0, _P], tuple[jax.Array, _AuxResultT]],
    argnums: Literal[0],
    has_aux: Literal[True],
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[Concatenate[_P0, _P], tuple[tuple[jax.Array, _AuxResultT], _P0]]:
    ...


@overload
def jax_value_and_grad(
    fun: Callable[Concatenate[_P0, _P], tuple[jax.Array, _AuxResultT]],
    argnums: Literal[0] = 0,
    *,
    has_aux: Literal[True],
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[Concatenate[_P0, _P], tuple[tuple[jax.Array, _AuxResultT], _P0]]:
    ...


@overload
def jax_value_and_grad(
    fun: Callable[Concatenate[_P0, _P1, _P], tuple[jax.Array, _AuxResultT]],
    argnums: Literal[1],
    has_aux: Literal[True],
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[Concatenate[_P0, _P1, _P], tuple[tuple[jax.Array, _AuxResultT], _P1]]:
    ...


@overload
def jax_value_and_grad(
    fun: Callable[Concatenate[_P0, _P1, _P], tuple[jax.Array, _AuxResultT]],
    argnums: tuple[Literal[0], Literal[1]],
    has_aux: Literal[True],
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[Concatenate[_P0, _P1, _P], tuple[tuple[jax.Array, _AuxResultT], tuple[_P0, _P1]]]:
    ...


@overload
def jax_value_and_grad(
    fun: Callable[_P, tuple[jax.Array, _AuxResultT]],
    argnums: int | Sequence[int],
    has_aux: Literal[True],
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[_P, tuple[tuple[jax.Array, _AuxResultT], Any]]:
    ...


def jax_value_and_grad(
    fun: Callable[_P, _ResultT],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[_P, tuple[_ResultT, Any]]:
    new_fun = jax.value_and_grad(
        fun,
        argnums,
        has_aux,
        holomorphic,
        allow_int,
        reduce_axes,
    )
    return new_fun  # type: ignore


_CallableT = TypeVar('_CallableT', bound=Callable)


_jax_pure_callback = jax.pure_callback  # pyright: ignore [reportPrivateImportUsage]


def onp_callback(callback: _CallableT, /) -> _CallableT:
    callback.__onp_callback__ = True  # pyright: ignore [reportFunctionMemberAccess]
    return callback


def jax_pure_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: _ResultT,
    *args: Any,
    vectorized: bool = False,
    **kwargs: Any,
) -> _ResultT:
    original_callback = callback
    while isinstance(original_callback, partial):
        original_callback = original_callback.func
    assert hasattr(original_callback, '__onp_callback__')
    del original_callback
    try:
        return _jax_pure_callback(
            callback, result_shape_dtypes, *args, vectorized=vectorized, **kwargs
        )
    except jaxlib.xla_extension.XlaRuntimeError as e:
        log_exception(e, f'jax_pure_callback of {_get_func_path(callback)}')

        def new_callback(*args: Any, **kwargs: Any) -> Any:
            try:
                return callback(*args, **kwargs)
            except Exception as new_e:
                log_exception(
                    new_e, f'jax_pure_callback of {_get_func_path(callback)}', level='critical'
                )
                pdb.post_mortem(new_e.__traceback__)
                raise

        _jax_pure_callback(
            new_callback, result_shape_dtypes, *args, vectorized=vectorized, **kwargs
        )
        raise e


def jax_eval_shape(fun: Callable[_P, _ResultT], *args: _P.args, **kwargs: _P.kwargs) -> _ResultT:
    return jax.eval_shape(fun, *args, **kwargs)


_CarryT = TypeVar('_CarryT')
_X = TypeVar('_X')
_Y = TypeVar('_Y')


def _deoptimized_lax_cond(
    pred: jax.Array,
    true_fun: Callable[[*_PTuple], _ResultT],
    false_fun: Callable[[*_PTuple], _ResultT],
    *operands: *_PTuple,
) -> _ResultT:
    if pred:
        return true_fun(*operands)
    else:
        return false_fun(*operands)


def lax_cond(
    pred: jax.Array,
    true_fun: Callable[[*_PTuple], _ResultT],
    false_fun: Callable[[*_PTuple], _ResultT],
    *operands: *_PTuple,
) -> _ResultT:
    if isinstance(jax.core.get_aval(pred), jax.core.ConcreteArray):
        return _deoptimized_lax_cond(pred, true_fun, false_fun, *operands)
    else:
        return jax.lax.cond(pred, true_fun, false_fun, *operands)


def _deoptimized_lax_scan(
    f: Callable[[_CarryT, _X], tuple[_CarryT, _Y]],
    init: _CarryT,
    xs: _X,
    length: Optional[int] = None,
    reverse: bool = False,
    unroll: int = 1,
) -> tuple[_CarryT, _Y]:
    xs_flat, xs_tree = jax.tree_util.tree_flatten(xs)
    carry = init
    del init
    ys = []
    if length is None:
        length = jax.tree_util.tree_leaves(xs)[0].shape[0]
        assert length is not None
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(length)):
        xs_slice = [x[i] for x in xs_flat]
        carry, y = f(carry, jax.tree_util.tree_unflatten(xs_tree, xs_slice))
        ys.append(y)
    mry = maybe_reversed(ys)
    stacked_y = jax.tree_util.tree_map(lambda *y: jnp.stack(y), *mry)
    return carry, stacked_y


def lax_scan(
    f: Callable[[_CarryT, _X], tuple[_CarryT, _Y]],
    init: _CarryT,
    xs: _X,
    length: Optional[int] = None,
    reverse: bool = False,
    unroll: int = 1,
) -> tuple[_CarryT, _Y]:
    if deoptimized:
        return _deoptimized_lax_scan(f, init, xs, length=length, reverse=reverse, unroll=unroll)
    else:
        try:
            return jax.lax.scan(f, init, xs, length=length, reverse=reverse, unroll=unroll)
        except (TypeError, FloatingPointError, jaxlib.xla_extension.XlaRuntimeError) as e:
            log_exception(e, f'lax_scan of {_get_func_path(f)}')
            return _deoptimized_lax_scan(f, init, xs, length=length, reverse=reverse, unroll=unroll)


_ParamsT = TypeVar('_ParamsT', bound=optax.Params)


def optax_apply_updates(params: _ParamsT, updates: optax.Updates) -> _ParamsT:
    return cast(_ParamsT, optax.apply_updates(params, updates))


def transpose_dict_of_list(d: Mapping[str, Iterable[_AnyT]]) -> list[dict[str, _AnyT]]:
    return [dict(zip(d, t)) for t in zip(*d.values(), strict=True)]


@contextmanager
def uninterrupted(
    raise_signal_after_exception: bool = False
) -> Generator[dict[str, bool], None, None]:
    state = {
        'sigtstp_received': False,
        'sigint_received': False,
    }

    def sigtstp_handler(sig, frame) -> None:
        _log.info('SIGTSTP received')
        state['sigtstp_received'] = True
        state['sigint_received'] = False

    def sigint_handler(sig, frame) -> None:
        _log.info('SIGINT received')
        state['sigint_received'] = True
        state['sigtstp_received'] = False

    def sigquit_handler(sig, frame) -> None:
        _log.info('SIGQUIT received')
        state['sigint_received'] = True
        state['sigtstp_received'] = False
        signal.signal(signal.SIGTSTP, original_sigtstp_handler)
        signal.raise_signal(signal.SIGTSTP)
        signal.signal(signal.SIGTSTP, sigtstp_handler)

    original_sigtstp_handler = signal.signal(signal.SIGTSTP, sigtstp_handler)
    original_sigint_handler = signal.signal(signal.SIGINT, sigint_handler)
    original_sigquit_handler = signal.signal(signal.SIGQUIT, sigquit_handler)

    try:
        yield state
    except BaseException:
        if not raise_signal_after_exception:
            state['sigtstp_received'] = False
            state['sigint_received'] = False
        raise
    finally:
        signal.signal(signal.SIGTSTP, original_sigtstp_handler)
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGQUIT, original_sigquit_handler)
        if state['sigtstp_received']:
            _log.info('Raising SIGTSTP')
            signal.raise_signal(signal.SIGTSTP)
        elif state['sigint_received']:
            _log.info('Raising SIGINT')
            signal.raise_signal(signal.SIGINT)


def fork_detached(func: Callable[_P, Any], *args: _P.args, **kwargs: _P.kwargs) -> None:
    pid = os.fork()
    if pid > 0:
        os.waitid(os.P_PID, pid, os.WEXITED)
        return

    os.setsid()

    pid = os.fork()
    if pid > 0:
        exit(0)

    func(*args, **kwargs)

    exit(os.EX_OK)


_XZ_CPU_LIMIT_KEY = 'XZ_CPU_LIMIT'


def xz_write(data: bytes, filename: str, verbose: bool = False) -> None:
    v_arg = ('-v',) if verbose else ()
    n_threads = 0
    if _XZ_CPU_LIMIT_KEY in os.environ.keys():
        n_threads = os.environ[_XZ_CPU_LIMIT_KEY]
    with open(filename, 'wb') as f:
        subprocess.run(['xz', *v_arg, '-T', str(n_threads), '-6'], input=data, stdout=f, check=True)


def xz_read(filename: str, verbose: bool = False) -> bytes:
    v_arg = ('-v',) if verbose else ()
    n_threads = 0
    if _XZ_CPU_LIMIT_KEY in os.environ.keys():
        n_threads = os.environ[_XZ_CPU_LIMIT_KEY]
    proc = subprocess.run(
        ['xz', '-dc', *v_arg, '-T', str(n_threads), filename], stdout=subprocess.PIPE, check=True
    )
    return proc.stdout


class PeriodicEMA:
    def __init__(self, lmbda: float = 0.3, period: int = 1) -> None:
        self._period = period
        self._lmbda = lmbda
        self._last = 0.0
        self._calls = 0
        self._history = np.zeros((self._period,), dtype=np.float32)

    def __call__(self, x: Optional[float] = None):
        if x is not None:
            self._history[self._calls % self._period] = x
            self._calls += 1
        if self._calls < self._period:
            if self._calls == 0:
                return 0.0
            return float(np.mean(self._history[:self._calls]))
        if x is not None:
            x = float(np.mean(self._history))
            self._last = self._lmbda * x + (1 - self._lmbda) * self._last
        return self._last / (1 - (1 - self._lmbda) ** (self._calls - (self._period - 1)))


class CSVLogger:

    def __init__(self, filename: str, header: Sequence[str]) -> None:
        self._filename = filename
        self._header = header
        if not os.path.exists(self._filename):
            with open(self._filename, 'w') as file:
                writer = csv.DictWriter(file, fieldnames=self._header)
                # write the header
                writer.writeheader()

    def log(self, metrics: Iterable[dict[str, Any]]) -> None:
        with open(self._filename, 'a') as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write new metrics in raws
            writer.writerows(metrics)


class UnRichFormatter(logging.Formatter):

    def formatMessage(self, record: logging.LogRecord):
        record.message = rich_unhighlight(record.message)
        return super().formatMessage(record)


class RichFormatter(colorlog.ColoredFormatter):

    def formatMessage(self, record: logging.LogRecord):
        if '\x1b' not in record.message:
            with ignore_exception(Exception):
                record.message = rich_highlight(record.message)
        return super().formatMessage(record)

    def formatException(
        self, ei: (
            tuple[type[BaseException], BaseException, TracebackType | None]
            | tuple[None, None, None]
        )
    ) -> str:
        exc_type, exc_value, traceback = ei
        assert exc_type is not None and exc_value is not None
        return rich_highlight_exception(exc_type, exc_value, traceback)


class ExcThread(threading.Thread):
    def run(self) -> None:
        self._exc = None
        try:
            return super().run()
        except BaseException as e:
            self._exc = e

    def join(self, timeout: float | None = None) -> None:
        super().join(timeout)
        if self._exc is not None:
            raise self._exc
