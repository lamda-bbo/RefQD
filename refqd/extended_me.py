import jax
import jax.numpy as jnp
import qdax.core.map_elites
import qdax.core.emitters.emitter
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, Metrics

from functools import partial
from collections.abc import Callable
from typing import Optional, TypeVar, Any, cast, overload

from .containers import ExtendedRepertoire, ExtendedMapElitesRepertoire, ExtendedDepthRepertoire
from .neuroevolution import GenotypePair
from .metrics import MetricsFn
from .treax import numpy as tjnp
from .utils import RNGKey, jax_jit, jax_eval_shape, lax_cond, lax_scan


_ExtendedRepertoireT = TypeVar(
    '_ExtendedRepertoireT',
    bound=ExtendedRepertoire,
)


_EmitterStateT = TypeVar(
    '_EmitterStateT',
    bound=Optional[qdax.core.emitters.emitter.EmitterState],
)


class ExtendedMAPElites:

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, Genotype, RNGKey, Any],
            tuple[Fitness, Descriptor, ExtraScores, RNGKey],
        ],
        reeval_scoring_fn: Callable[
            [Genotype, Genotype, RNGKey, Any],
            tuple[Fitness, Descriptor, ExtraScores, RNGKey],
        ],
        reeval_env_batch_size: int,
        emitter: qdax.core.emitters.emitter.Emitter,
        metrics_function: MetricsFn,
        refresh_depth: int,
    ):
        self._scoring_function = scoring_function
        self._reeval_scoring_fn = reeval_scoring_fn
        self._reeval_env_batch_size = reeval_env_batch_size
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._refresh_depth = refresh_depth

    @partial(jax_jit, static_argnames=(
        'self', 'using_representation', 'repertoire_type', 'repertoire_kwargs'
    ))
    def init(
        self,
        init_representation_genotypes: Genotype,
        init_decision_genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
        task_constant: Any,
        using_representation: bool,
        repertoire_type: type[ExtendedRepertoire],
        repertoire_kwargs: dict[str, Any],
    ) -> tuple[
        ExtendedRepertoire,
        Optional[ExtendedRepertoire],
        Optional[qdax.core.emitters.emitter.EmitterState],
        Metrics,
        RNGKey,
    ]:
        # score initial genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_representation_genotypes, init_decision_genotypes, random_key, task_constant
        )

        # init the repertoire
        repertoire = repertoire_type.init(
            genotypes=init_decision_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            **repertoire_kwargs,
        )

        metrics = self._metrics_function(repertoire)

        if using_representation:
            init_genotypes = GenotypePair(init_representation_genotypes, init_decision_genotypes)
        else:
            init_genotypes = init_decision_genotypes

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key,
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        if self._refresh_depth != 0:
            tmp_repertoire = repertoire_type.init(
                genotypes=init_decision_genotypes,
                fitnesses=-fitnesses,
                descriptors=descriptors,
                centroids=centroids,
                **repertoire_kwargs,
            )
            tmp_repertoire = tmp_repertoire.empty()
        else:
            tmp_repertoire = None

        return repertoire, tmp_repertoire, emitter_state, metrics, random_key

    def calc_evals_per_reeval(self, repertoire: ExtendedRepertoire):
        match repertoire:
            case ExtendedDepthRepertoire():
                if self._refresh_depth != -1:
                    n_genotypes = repertoire.fitnesses.shape[-1] * self._refresh_depth
                else:
                    n_genotypes = repertoire.fitnesses_depth.shape[-1]
            case ExtendedMapElitesRepertoire():
                n_genotypes = repertoire.fitnesses.shape[-1]
            case _:
                raise NotImplementedError(type(repertoire))
        return n_genotypes

    @overload
    def reeval(
        self,
        new_repertoire: None,
        emitter_state: _EmitterStateT,
        reeval_key: RNGKey,
        representation_params: Genotype,
        repertoire: ExtendedRepertoire,
        task_constant: Any,
    ) -> tuple[
        None,
        _EmitterStateT,
        RNGKey,
        Metrics,
        Fitness,
        Descriptor,
    ]:
        ...

    @overload
    def reeval(
        self,
        new_repertoire: _ExtendedRepertoireT,
        emitter_state: _EmitterStateT,
        reeval_key: RNGKey,
        representation_params: Genotype,
        repertoire: _ExtendedRepertoireT,
        task_constant: Any,
    ) -> tuple[
        _ExtendedRepertoireT,
        _EmitterStateT,
        RNGKey,
        Metrics,
        Fitness,
        Descriptor,
    ]:
        ...

    def reeval(
        self,
        new_repertoire: Optional[_ExtendedRepertoireT],
        emitter_state: _EmitterStateT,
        reeval_key: RNGKey,
        representation_params: Genotype,
        repertoire: _ExtendedRepertoireT,
        task_constant: Any,
    ) -> tuple[
        Optional[_ExtendedRepertoireT],
        _EmitterStateT,
        RNGKey,
        Metrics,
        Fitness,
        Descriptor,
    ]:
        update_repertoire = new_repertoire is not None
        if not update_repertoire:
            new_repertoire = ExtendedMapElitesRepertoire.init(
                {},
                repertoire.fitnesses[:1],
                repertoire.descriptors[:1],
                repertoire.centroids,
            )
            new_repertoire = new_repertoire.empty()
        if update_repertoire:
            n_genotypes = self.calc_evals_per_reeval(repertoire)
        else:
            match repertoire:
                case ExtendedDepthRepertoire():
                    n_genotypes = repertoire.fitnesses_depth.shape[-1]
                case ExtendedMapElitesRepertoire():
                    n_genotypes = repertoire.fitnesses.shape[-1]
                case _:
                    raise NotImplementedError(type(repertoire))
        assert n_genotypes % self._reeval_env_batch_size == 0

        def _scan_reeval(
            carry: tuple[
                _ExtendedRepertoireT, _EmitterStateT
            ],
            x: tuple[RNGKey, jax.Array],
        ) -> tuple[
            tuple[_ExtendedRepertoireT, _EmitterStateT],
            tuple[Fitness, Descriptor],
        ]:
            new_repertoire, emitter_state = carry
            reeval_key, idx = x
            decision_params, fitness, descriptor = repertoire.get_from_idx(idx)
            representation_params_ = representation_params
            new_fitness, new_descriptor, extra_scores, _ = self._reeval_scoring_fn(
                representation_params_, decision_params, reeval_key, task_constant
            )
            new_fitness = jnp.where(fitness == -jnp.inf, -jnp.inf, new_fitness)
            new_repertoire = new_repertoire.add(
                decision_params if update_repertoire else {}, new_descriptor, new_fitness,
            )
            return (new_repertoire, emitter_state), (new_fitness, new_descriptor)

        reeval_key, subkey = jax.random.split(reeval_key)
        keys = jax.random.split(subkey, n_genotypes // self._reeval_env_batch_size)
        idx = repertoire.get_sorted_idx().T.reshape(-1)[:n_genotypes].reshape(
            -1, self._reeval_env_batch_size
        )
        (new_repertoire, emitter_state), (fitness, descriptor) = lax_scan(
            _scan_reeval,
            (new_repertoire, emitter_state),
            (keys, idx),
        )

        fitness = fitness.reshape(-1)
        descriptor = descriptor.reshape(-1, descriptor.shape[-1])

        metrics = self._metrics_function(new_repertoire)
        metrics = {f'reeval_{key}': value for key, value in metrics.items()}

        if not update_repertoire:
            new_repertoire = None

        return new_repertoire, emitter_state, reeval_key, metrics, fitness, descriptor

    @overload
    def update(
        self,
        repertoire: _ExtendedRepertoireT,
        empty_repertoire: None,
        emitter_state: _EmitterStateT,
        random_key: RNGKey,
        task_constant: Any,
        refresh_repertoire: jax.Array,
    ) -> tuple[
        _ExtendedRepertoireT,
        None,
        _EmitterStateT,
        RNGKey,
        Metrics,
        Metrics,
        Fitness,
        Descriptor,
    ]:
        ...

    @overload
    def update(
        self,
        repertoire: _ExtendedRepertoireT,
        empty_repertoire: _ExtendedRepertoireT,
        emitter_state: _EmitterStateT,
        random_key: RNGKey,
        task_constant: Any,
        refresh_repertoire: jax.Array,
    ) -> tuple[
        _ExtendedRepertoireT,
        _ExtendedRepertoireT,
        _EmitterStateT,
        RNGKey,
        Metrics,
        Metrics,
        Fitness,
        Descriptor,
    ]:
        ...

    @partial(jax_jit, static_argnames=('self',))
    def update(
        self,
        repertoire: _ExtendedRepertoireT,
        empty_repertoire: Optional[_ExtendedRepertoireT],
        emitter_state: _EmitterStateT,
        random_key: RNGKey,
        task_constant: Any,
        refresh_repertoire: jax.Array,
    ) -> tuple[
        _ExtendedRepertoireT,
        Optional[_ExtendedRepertoireT],
        _EmitterStateT,
        RNGKey,
        Metrics,
        Metrics,
        Fitness,
        Descriptor,
    ]:
        # generate offsprings with the emitter
        genotypes, random_key = self._emitter.emit(
            cast(ExtendedMapElitesRepertoire, repertoire), emitter_state, random_key
        )
        # scores the offsprings
        if isinstance(genotypes, GenotypePair):
            representation_params = genotypes[0]
            decision_genotypes = genotypes[1]
        else:
            representation_params = {}
            decision_genotypes = genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            representation_params, decision_genotypes, random_key, task_constant
        )

        def fake_reeval_metrics() -> tuple[Metrics, Fitness, Descriptor]:
            _, _, _, reeval_metrics, reeval_fitness, reeval_descriptor = tjnp.zeros_like(
                jax_eval_shape(
                    self.reeval,
                    cast(_ExtendedRepertoireT, empty_repertoire),  # ignore the error of Pylance
                    emitter_state,
                    random_key,
                    representation_params,
                    repertoire,
                    task_constant,
                )
            )
            return reeval_metrics, reeval_fitness, reeval_descriptor

        def unrefresh(
            repertoire: _ExtendedRepertoireT,
            new_repertoire: Optional[_ExtendedRepertoireT],
            emitter_state: _EmitterStateT,
            random_key: RNGKey,
        ):
            repertoire = repertoire.add(decision_genotypes, descriptors, fitnesses)
            reeval_metrics, reeval_fitness, reeval_descriptor = fake_reeval_metrics()
            return (
                repertoire, new_repertoire, emitter_state, random_key,
                reeval_metrics, reeval_fitness, reeval_descriptor,
            )

        def refresh(
            repertoire: _ExtendedRepertoireT,
            new_repertoire: Optional[_ExtendedRepertoireT],
            emitter_state: _EmitterStateT,
            random_key: RNGKey,
        ):
            if new_repertoire is None:  # ERROR
                reeval_metrics, reeval_fitness, reeval_descriptor = fake_reeval_metrics()
                return (
                    repertoire.empty(), new_repertoire, emitter_state, random_key,
                    reeval_metrics, reeval_fitness, reeval_descriptor,
                )
            new_repertoire = cast(_ExtendedRepertoireT, new_repertoire)
            if isinstance(new_repertoire, ExtendedDepthRepertoire) and self._refresh_depth != -1:
                new_repertoire = new_repertoire.copy_from(repertoire, layers=self._refresh_depth)
            else:
                new_repertoire = new_repertoire.empty()
            new_repertoire = new_repertoire.add(decision_genotypes, descriptors, fitnesses)
            (
                new_repertoire, emitter_state, random_key,
                reeval_metrics, reeval_fitness, reeval_descriptor,
            ) = self.reeval(
                new_repertoire,
                emitter_state,
                random_key,
                representation_params,
                repertoire,
                task_constant,
            )
            return (
                new_repertoire, repertoire, emitter_state, random_key,
                reeval_metrics, reeval_fitness, reeval_descriptor,
            )

        (
            repertoire, empty_repertoire, emitter_state, random_key,
            reeval_metrics, reeval_fitness, reeval_descriptor,
        ) = lax_cond(
            refresh_repertoire, refresh, unrefresh,
            repertoire, empty_repertoire, emitter_state, random_key,
        )

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return (
            repertoire, empty_repertoire, emitter_state, random_key,
            metrics, {}, reeval_fitness, reeval_descriptor,
        )

    @overload
    def scan_update(
        self,
        carry: tuple[
            _ExtendedRepertoireT,
            None,
            _EmitterStateT,
            RNGKey,
            Any,
        ],
        refresh_repertoire: jax.Array,
    ) -> tuple[
        tuple[
            _ExtendedRepertoireT,
            None,
            _EmitterStateT,
            RNGKey,
            Any,
        ],
        tuple[Metrics, Metrics, Fitness, Descriptor, Fitness, Descriptor]
    ]:
        ...

    @overload
    def scan_update(
        self,
        carry: tuple[
            _ExtendedRepertoireT,
            _ExtendedRepertoireT,
            _EmitterStateT,
            RNGKey,
            Any,
        ],
        refresh_repertoire: jax.Array,
    ) -> tuple[
        tuple[
            _ExtendedRepertoireT,
            _ExtendedRepertoireT,
            _EmitterStateT,
            RNGKey,
            Any,
        ],
        tuple[Metrics, Metrics, Fitness, Descriptor, Fitness, Descriptor]
    ]:
        ...

    @partial(jax_jit, static_argnames=('self',))
    def scan_update(
        self,
        carry: tuple[
            _ExtendedRepertoireT,
            Optional[_ExtendedRepertoireT],
            _EmitterStateT,
            RNGKey,
            Any,
        ],
        refresh_repertoire: jax.Array,
    ) -> tuple[
        tuple[
            _ExtendedRepertoireT,
            Optional[_ExtendedRepertoireT],
            _EmitterStateT,
            RNGKey,
            Any,
        ],
        tuple[Metrics, Metrics, Fitness, Descriptor, Fitness, Descriptor]
    ]:
        repertoire, empty_repertoire, emitter_state, random_key, task_constant = carry
        (
            repertoire, empty_repertoire, emitter_state, random_key,
            metrics, reeval_metrics, reeval_fitness, reeval_descriptor,
        ) = self.update(
            repertoire,
            empty_repertoire,
            emitter_state,
            random_key,
            task_constant,
            refresh_repertoire,
        )
        return (
            (repertoire, empty_repertoire, emitter_state, random_key, task_constant),
            (
                metrics, reeval_metrics,
                repertoire.fitnesses, repertoire.descriptors,
                reeval_fitness, reeval_descriptor,
            ),
        )
