import jax
from qdax.types import Genotype

from typing import NamedTuple, TypeVar, Generic

from ..utils import RNGKey


_GenotypeT1 = TypeVar('_GenotypeT1', bound=Genotype)
_GenotypeT2 = TypeVar('_GenotypeT2', bound=Genotype)


class GenotypePair(NamedTuple, Generic[_GenotypeT1, _GenotypeT2]):
    representation: _GenotypeT1
    decision: _GenotypeT2


class FakeRepresentation:

    def tabulate(self, random_key: RNGKey, data: jax.Array, **kwargs) -> str:
        return ''

    def init_with_output(self, random_key: RNGKey, data: jax.Array) -> tuple[jax.Array, Genotype]:
        return data, {}

    def init(self, random_key: RNGKey, data: jax.Array) -> Genotype:
        return {}

    def apply(self, params: Genotype, data: jax.Array) -> jax.Array:
        return data
