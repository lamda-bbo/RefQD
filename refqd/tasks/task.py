from qdax.types import Genotype, Fitness, Descriptor, ExtraScores

from abc import abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar, Any

from ..config.task import TaskConfig
from ..utils import RNGKey


_TaskConfigT = TypeVar('_TaskConfigT', bound=TaskConfig)


class Task(Generic[_TaskConfigT]):

    def __init__(self, cfg: _TaskConfigT, batch_shape: tuple[int, ...]):
        self._cfg = cfg
        self._batch_shape = batch_shape

    @property
    @abstractmethod
    def behavior_descriptor_length(self) -> int:
        ...

    @property
    @abstractmethod
    def behavior_descriptor_limits(self) -> tuple[list[float], list[float]]:
        ...

    @property
    @abstractmethod
    def qd_offset(self) -> float:
        ...

    @abstractmethod
    def get_constant(self, random_key: RNGKey) -> Any:
        ...

    @abstractmethod
    def get_scoring_fn(self) -> Callable[
        [Genotype, Genotype, RNGKey, Any],
        tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ]:
        ...
