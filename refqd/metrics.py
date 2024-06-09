import jax.numpy as jnp
from qdax.types import Metrics

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .containers import ExtendedRepertoire


class MetricsFn(Protocol):
    def __call__(self, repertoire: 'ExtendedRepertoire') -> Metrics:
        ...


def qd_metrics(
    repertoire: 'ExtendedRepertoire',
    qd_offset: float,
) -> Metrics:

    # get metrics
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
    qd_score += qd_offset * jnp.sum(1.0 - repertoire_empty)
    coverage = 100 * jnp.mean(1.0 - repertoire_empty)
    max_fitness = jnp.max(repertoire.fitnesses)
    min_fitness = jnp.min(repertoire.fitnesses, initial=max_fitness, where=~repertoire_empty)
    mean_fitness = jnp.mean(repertoire.fitnesses, where=~repertoire_empty)

    metrics = {
        'qd_score': qd_score,
        'max_fitness': max_fitness,
        'coverage': coverage,
        'min_fitness': min_fitness,
        'mean_fitness': mean_fitness,
    }

    return metrics
