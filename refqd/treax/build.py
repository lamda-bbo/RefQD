import jax.numpy as jnp
from jax import lax

import os.path
import sys
import inspect
import builtins
from collections.abc import Sequence, Iterable
from typing import Union, cast
from types import ModuleType

from . import functional
from . import _numpy_head
from . import _lax_head
from ..utils import assert_cast


class Str:
    def __init__(self, s: str):
        self.s = s

    def __repr__(self) -> str:
        return self.s


def transform_ann(ann: str) -> Str:
    ann = ann.replace('T]', 'TreeT]').replace('_TreeT', 'TreeT')
    if ann == 'T':
        ann = 'TreeT'
    ann = (
        ann.replace('jax.Array', 'TreeT').replace('jax.typing.ArrayLike', 'TreeT')
        .replace('ArrayLike', 'TreeT').replace('Array', 'TreeT')
        .replace('ShapeDtypeStruct', 'TreeT').replace('TreeT | DuckTypedTreeT', 'TreeT')
        .replace('DuckTypedTreeT', 'TreeT')
        .replace('core.Shape', 'Shape')
        .replace('lax.PrecisionType', 'PrecisionType')
        .replace(
            'convolution.ConvGeneralDilatedDimensionNumbers', 'ConvGeneralDilatedDimensionNumbers'
        )
    )
    if ann == 'ufunc':
        ann = 'Any'
    return Str(ann)


def transform_ann_obj(ann: object) -> Str | type[inspect._empty]:
    if hasattr(ann, '__origin__') and ann.__origin__ is Union:  # pyright: ignore [reportAttributeAccessIssue]
        ann = str(ann)
    elif isinstance(ann, type(tuple[int])):
        ann = str(ann)
    else:
        try:
            ann = assert_cast(str, ann.__name__)  # pyright: ignore [reportAttributeAccessIssue]
        except AttributeError:
            ann = str(ann)
    ann = ann.replace('T]', 'TreeT]').replace('_TreeT', 'TreeT')
    if ann == 'T':
        ann = 'TreeT'
    if ann[:12] == 'typing.Union':
        ann = 'TreeT'
    ann = (
        ann.replace('jax.Array', 'TreeT').replace('jax.typing.ArrayLike', 'TreeT')
        .replace('ArrayLike', 'TreeT').replace('Array', 'TreeT')
        .replace('ShapeDtypeStruct', 'TreeT').replace('TreeT | DuckTypedTreeT', 'TreeT')
        .replace('DuckTypedTreeT', 'TreeT')
    )
    if ann == 'ufunc' or ann == '_empty':
        return inspect._empty
    return Str(ann)


def transform_default(dflt):
    match dflt:
        case builtins.float:
            return Str('float')
        case lax.dot_general:
            return Str('lax.dot_general')
        case jnp.int32:
            return Str('jnp.int32')
        case _:
            if 'RoundingMethod.AWAY_FROM_ZERO' in repr(dflt):
                return Str('RoundingMethod.AWAY_FROM_ZERO')
            return dflt


def build_api(names: Iterable, module: ModuleType) -> tuple[Sequence[Str], Sequence[str]]:
    interface_names: list[Str] = []
    interfaces: list[str] = []

    for name in names:
        if not name[0].islower():
            continue
        if name in ('cond', 'while_loop', 'fori_loop', 'scan', 'switch'):
            continue
        if hasattr(functional, name):
            obj = getattr(functional, name)
        else:
            obj = getattr(module, name)
        if not callable(obj):
            continue

        try:
            sig = inspect.signature(obj)
        except ValueError as e:
            print(f'{name}:', repr(e), file=sys.stderr)
            continue

        ret = sig.return_annotation
        if isinstance(ret, str):
            ret = transform_ann(ret)
        elif ret == sig.empty:
            if obj.__module__ == 'jax._src.numpy.ufuncs':
                ret = transform_ann('Array')
        else:
            ret = transform_ann_obj(ret)

        params = sig.parameters.copy()
        treetcnt = 0
        for key in params.keys():
            if isinstance(params[key].annotation, str):
                params[key] = params[key].replace(
                    annotation=transform_ann(params[key].annotation),
                )
            elif params[key].annotation == params[key].empty:
                if obj.__module__ == 'jax._src.numpy.ufuncs':
                    params[key] = params[key].replace(
                        annotation=(
                            (
                                transform_ann('None')
                                if key == 'out'
                                else transform_ann('Array | None')
                            )
                            if params[key].default is None else
                            transform_ann('Array')
                        ),
                    )
            else:
                params[key] = params[key].replace(
                    annotation=transform_ann_obj(params[key].annotation),
                )
            if isinstance(params[key].annotation, Str):
                if key == 'condition':
                    params[key].annotation.s = params[key].annotation.s.replace('TreeT', 'Tree')
                if 'TreeT' in params[key].annotation.s:
                    treetcnt += 1
                if (
                    treetcnt > 1 and obj.__module__ != 'jax._src.numpy.ufuncs'
                    and 'condition' not in params.keys()
                ):
                    params[key].annotation.s = params[key].annotation.s.replace('TreeT', 'Tree')
            if params[key].default != params[key].empty:
                params[key] = params[key].replace(
                    default=transform_default(params[key].default),
                )

        sig = sig.replace(
            parameters=cast(Sequence[inspect.Parameter], params.values()),
            return_annotation=ret,
        )
        strsig = str(sig)
        if strsig.count('TreeT') == 0:
            continue
        if strsig.count('TreeT') == 1:
            strsig = strsig.replace('TreeT', 'Tree')
        interface_names.append(Str(name))
        interfaces.append(f'\n\ndef {name}{strsig}:  # noqa: E501\n    ...\n')

    interfaces.append('\n\n__all__ = [\n')
    for name in interface_names:
        interfaces.append(f"    '{name}',\n")
    interfaces.append(']\n')
    return interface_names, interfaces


def build_numpy_api() -> None:
    head_names = tuple(Str(name) for name in dir(_numpy_head) if name[:2] != '__')
    interface_names, interfaces = build_api(sorted(set(dir(jnp)) | set(dir(functional))), jnp)
    with open(os.path.join(os.path.dirname(__file__), '_numpy_api.py'), 'w') as f:
        f.write(f'from ._numpy_head import {str(head_names)[1:-1]}  # noqa: E501\n')
        f.writelines(interfaces)
    with open(os.path.join(os.path.dirname(__file__), 'numpy.py'), 'w') as f:
        f.write(
            f'''from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._numpy_api import {str(interface_names)[1:-1]}  # noqa: E501, F401
else:
    from ._numpy_wrapped import __getattr__  # noqa: F401
del TYPE_CHECKING
'''
        )


def build_lax_api() -> None:
    head_names = tuple(Str(name) for name in dir(_lax_head) if name[:2] != '__')
    interface_names, interfaces = build_api(sorted(set(dir(lax))), lax)
    with open(os.path.join(os.path.dirname(__file__), '_lax_api.py'), 'w') as f:
        f.write(f'from ._lax_head import {str(head_names)[1:-1]}  # noqa: E501\n')
        f.writelines(interfaces)
    with open(os.path.join(os.path.dirname(__file__), 'lax.py'), 'w') as f:
        f.write(
            f'''from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._lax_api import {str(interface_names)[1:-1]}  # noqa: E501, F401
else:
    from ._lax_wrapped import __getattr__  # noqa: F401
del TYPE_CHECKING
'''
        )


def main() -> None:
    build_numpy_api()
    build_lax_api()


if __name__ == '__main__':
    main()
