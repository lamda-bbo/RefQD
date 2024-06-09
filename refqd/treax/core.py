import jax

from collections.abc import Callable
from typing import TypeVar, ParamSpec, Generic, cast


_CallableT = TypeVar('_CallableT', bound=Callable)
_P = ParamSpec('_P')
_ResultT = TypeVar('_ResultT')


class TreeWrapped(Generic[_CallableT, _P, _ResultT]):

    def __init__(self, func: _CallableT) -> None:
        self._wrapped = func

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _ResultT:
        mapped: list[bool] = []
        mapped_args = []
        for arg in args:
            if jax.tree_util.all_leaves((arg,)) or arg is None:
                mapped.append(False)
            else:
                mapped.append(True)
                mapped_args.append(arg)

        if mapped_args == []:
            return self._wrapped(*args, **kwargs)

        def func(*mapped_args):
            mapped_idx = 0
            final_args = []
            for i, is_mapped in enumerate(mapped):
                if is_mapped:
                    final_args.append(mapped_args[mapped_idx])
                    mapped_idx += 1
                else:
                    final_args.append(args[i])

            return self._wrapped(*final_args, **kwargs)

        return jax.tree_map(func, *mapped_args)


def chain(
    func1: Callable[_P, _ResultT], *funcs: Callable[[_ResultT], _ResultT]
) -> TreeWrapped[Callable[_P, _ResultT], _P, _ResultT]:

    assert isinstance(func1, TreeWrapped)
    func1_ = cast(TreeWrapped[Callable[_P, _ResultT], _P, _ResultT], func1)
    assert all(isinstance(func, TreeWrapped) for func in funcs)
    funcs = cast(tuple[TreeWrapped[Callable[[_ResultT], _ResultT], [_ResultT], _ResultT]], funcs)

    def func(*args: _P.args, **kwargs: _P.kwargs) -> _ResultT:
        res = func1_._wrapped(*args, **kwargs)
        for func in funcs:
            res = func._wrapped(res)
        return res

    return TreeWrapped(func)
