import jax
import jax.numpy as jnp
import flax.struct
import qdax.core.containers.mapelites_repertoire
from qdax.types import Centroid, Descriptor, Fitness, Genotype

import numpy as np

import logging
import time
from functools import partial
from typing import Optional, Self, TYPE_CHECKING
from overrides import override

from ..cluster import KMeans
from ..treax import chain as tchain, numpy as tjnp
from ..utils import RNGKey, assert_cast, jax_jit, onp_callback, jax_pure_callback, jax_eval_shape


_log = logging.getLogger(__name__)


def compute_cvt_centroids(
    num_descriptors: int,
    num_init_cvt_samples: int,
    num_centroids: int,
    minval: float | list[float],
    maxval: float | list[float],
    random_key: RNGKey,
) -> tuple[jax.Array, RNGKey]:
    minval_ = jnp.array(minval)
    maxval_ = jnp.array(maxval)

    # assume here all values are in [0, 1] and rescale later
    random_key, subkey = jax.random.split(random_key)
    x = jax.random.uniform(key=subkey, shape=(num_init_cvt_samples, num_descriptors))

    # compute k means
    random_key, subkey = jax.random.split(random_key)
    k_means = KMeans(
        k=num_centroids,
        init='k-means++',
        n_init=1,
    )
    k_means_state = k_means.fit(subkey, x)
    centroids = k_means_state.cluster_centers_
    # rescale now
    return jnp.asarray(centroids) * (maxval_ - minval_) + minval_, random_key


class ExtendedMapElitesRepertoire(
    qdax.core.containers.mapelites_repertoire.MapElitesRepertoire
):

    def get_sorted_idx(self) -> jax.Array:
        return jnp.arange(self.fitnesses.shape[0])

    def get_from_idx(self, idx: jax.Array) -> tuple[Genotype, Fitness, Descriptor]:
        return tjnp.getitem((self.genotypes, self.fitnesses, self.descriptors), idx)

    @jax_jit
    @override
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
    ) -> Self:

        batch_of_indices = qdax.core.containers.mapelites_repertoire.get_cells_indices(
            batch_of_descriptors, self.centroids
        )
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
        batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)

        num_centroids = self.centroids.shape[0]

        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(axis=-1),
            num_segments=num_centroids,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, batch_of_fitnesses, -jnp.inf
        )

        # get addition condition
        repertoire_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1)
        current_fitnesses = jnp.take_along_axis(
            repertoire_fitnesses, batch_of_indices, 0
        )
        addition_condition = batch_of_fitnesses > current_fitnesses

        # assign fake position when relevant : num_centroids is out of bound
        batch_of_indices = jnp.where(
            addition_condition, batch_of_indices, num_centroids
        )

        # create new repertoire
        new_repertoire_genotypes = jax.tree_util.tree_map(
            lambda repertoire_genotypes, new_genotypes: repertoire_genotypes.at[
                assert_cast(jax.Array, batch_of_indices).squeeze(axis=-1)
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_fitnesses.squeeze(axis=-1)
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_descriptors
        )

        return self.replace(
            genotypes=new_repertoire_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
        )

    @classmethod
    @override
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
    ) -> Self:

        # retrieve one genotype from the population
        first_genotype = tjnp.getitem(genotypes, 0)

        # create a repertoire with default values
        repertoire = cls.init_default(
            genotype=first_genotype,
            centroids=centroids,
        )

        # add initial population to the repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses)

        return repertoire

    @classmethod
    def init_default(
        cls,
        genotype: Genotype,
        centroids: Centroid,
    ) -> Self:

        # get number of centroids
        num_centroids = centroids.shape[0]

        # default fitness is -inf
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

        # default genotypes is all 0
        default_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
            genotype,
        )

        # default descriptor is all zeros
        default_descriptors = jnp.zeros_like(centroids)

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
        )

    def empty(self) -> Self:
        return self.replace(
            genotypes=tjnp.zeros_like(self.genotypes),
            fitnesses=jnp.full_like(self.fitnesses, -jnp.inf),
        )

    if TYPE_CHECKING:
        def replace(self, **overrides) -> Self:
            return super().replace(**overrides)


global_genotypes: list[Genotype] = []

global_time_empty: float = 0.0
global_time_add: float = 0.0
global_time_sample: float = 0.0


@onp_callback
def _empty_genotypes(
    fake_genotypes: np.ndarray
) -> np.ndarray:
    global global_time_empty
    global_time_empty -= time.monotonic()
    fake_genotypes_ = fake_genotypes[..., 0]
    idx = int(fake_genotypes_.flatten()[0])

    def fn(genotypes: np.ndarray):
        genotypes[:] = np.float32(0)
        return genotypes

    global_genotypes[idx] = jax.tree_map(fn, global_genotypes[idx])

    global_time_empty += time.monotonic()
    return fake_genotypes


@onp_callback
def _update_genotypes(
    fake_genotypes: np.ndarray, new_genotypes: Genotype, batch_of_indices: np.ndarray
) -> np.ndarray:
    global global_time_add
    global_time_add -= time.monotonic()
    batch_of_indices = batch_of_indices.squeeze(axis=-1)
    fake_genotypes_ = fake_genotypes[..., 0]
    idx = int(fake_genotypes_.flatten()[0])
    batch_idxs = np.argwhere(fake_genotypes_ >= 0)

    def fn(genotypes: np.ndarray, new_genotypes: np.ndarray):
        n_centroids = genotypes.shape[batch_idxs.shape[-1]]
        for batch_idx in batch_idxs:
            batch_of_indices_ = batch_of_indices[*batch_idx]
            mask = batch_of_indices_ < n_centroids
            genotypes[*batch_idx, batch_of_indices_[mask]] = new_genotypes[*batch_idx, mask]
        return genotypes

    global_genotypes[idx] = jax.tree_map(fn, global_genotypes[idx], new_genotypes)
    global_time_add += time.monotonic()
    return fake_genotypes


@onp_callback
def _sample_genotypes(
    fake_genotypes: np.ndarray, indices: np.ndarray, fitnesses: Optional[np.ndarray] = None,
) -> Genotype:
    global global_time_sample
    global_time_sample -= time.monotonic()
    fake_genotypes = fake_genotypes[..., 0]
    idx = int(fake_genotypes.flatten()[0])
    batch_idxs = np.argwhere(fake_genotypes >= 0)
    indices = indices.astype(np.int32)
    if fitnesses is not None:
        for i in range(fitnesses.ndim - indices.ndim):
            indices = np.repeat(np.expand_dims(indices, axis=i), repeats=fitnesses.shape[i], axis=i)

    def fn(genotypes: np.ndarray):
        res = np.empty((*indices.shape, *genotypes.shape[indices.ndim:]), dtype=genotypes.dtype)
        for batch_idx in batch_idxs:
            res[*batch_idx] = genotypes[*batch_idx, indices[*batch_idx]]
        return res

    samples = jax.tree_map(fn, global_genotypes[idx])
    global_time_sample += time.monotonic()
    return samples


class CPUMapElitesRepertoire(ExtendedMapElitesRepertoire):

    genotype_shape_dtype: Genotype = flax.struct.field(pytree_node=False)

    def get_from_idx(self, idx: jax.Array) -> tuple[Genotype, Fitness, Descriptor]:
        assert idx.ndim == 1
        shape_dtype = jax_eval_shape(
            partial(tjnp.duplicate, repeats=idx.shape[0]), self.genotype_shape_dtype
        )
        samples = jax_pure_callback(
            _sample_genotypes, shape_dtype, self.genotypes, idx, self.fitnesses, vectorized=True
        )
        fitnesses, descriptors = tjnp.getitem((self.fitnesses, self.descriptors), idx)
        return samples, fitnesses, descriptors

    @partial(jax_jit, static_argnames=('num_samples',))
    def sample(self, random_key: RNGKey, num_samples: int) -> tuple[Genotype, RNGKey]:
        repertoire_empty = self.fitnesses == -jnp.inf
        p = (1.0 - repertoire_empty) / jnp.sum(1.0 - repertoire_empty)

        random_key, subkey = jax.random.split(random_key)
        indices = jax.random.choice(subkey, self.fitnesses.shape[0], shape=(num_samples,), p=p)

        shape_dtype = jax_eval_shape(
            partial(tjnp.duplicate, repeats=num_samples), self.genotype_shape_dtype
        )
        samples = jax_pure_callback(
            _sample_genotypes, shape_dtype, self.genotypes, indices, vectorized=True
        )

        return samples, random_key

    @jax_jit
    @override
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
    ) -> Self:

        batch_of_indices = qdax.core.containers.mapelites_repertoire.get_cells_indices(
            batch_of_descriptors, self.centroids
        )
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
        batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)

        num_centroids = self.centroids.shape[0]

        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(axis=-1),
            num_segments=num_centroids,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, batch_of_fitnesses, -jnp.inf
        )

        # get addition condition
        repertoire_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1)
        current_fitnesses = jnp.take_along_axis(
            repertoire_fitnesses, batch_of_indices, 0
        )
        addition_condition = batch_of_fitnesses > current_fitnesses

        # assign fake position when relevant : num_centroids is out of bound
        batch_of_indices = jnp.where(
            addition_condition, batch_of_indices, num_centroids
        )

        repertoire_genotypes = jax_pure_callback(
            _update_genotypes,
            self.genotypes,
            self.genotypes, batch_of_genotypes, batch_of_indices,
            vectorized=True,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_fitnesses.squeeze(axis=-1)
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_descriptors
        )

        return self.replace(
            genotypes=repertoire_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
        )

    @classmethod
    @override
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
    ) -> Self:

        @onp_callback
        def fake_transform(genotype: Genotype, fitnesses: Fitness) -> Genotype:
            global_idx = len(global_genotypes)
            _log.info(f'init.fake_transform: {global_idx}')

            genotype = jax.tree_util.tree_map(
                lambda x, y: np.full(fitnesses.shape[:-1] + x.shape, y, dtype=x.dtype),
                genotypes,
                genotype,
            )

            def fn(genotype: np.ndarray) -> np.ndarray:
                return np.zeros(
                    (*fitnesses.shape[:-1], centroids.shape[-2], *genotype.shape[fitnesses.ndim:]),
                    dtype=genotype.dtype,
                )

            initial_genotypes = jax.tree_map(fn, genotype)
            global_genotypes.append(initial_genotypes)

            fake_genotype = np.full_like(fitnesses, global_idx, dtype=np.int32)

            return genotype, fake_genotype

        genotypes, fake_genotype = jax_pure_callback(
            fake_transform,
            (genotypes, fitnesses.astype(jnp.int32)),
            genotypes, fitnesses,
            vectorized=True,
        )

        first_genotype_shape_dtype = tchain(tjnp.getitem, tjnp.shape_dtype)(genotypes, 0)

        # retrieve one genotype from the population
        first_genotype = tjnp.getitem(fake_genotype, 0)

        # create a repertoire with default values
        repertoire = cls.init_default(
            genotype=first_genotype,
            genotype_shape_dtype=first_genotype_shape_dtype,
            centroids=centroids,
        )

        # add initial population to the repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses)

        return repertoire

    @classmethod
    def init_default(
        cls,
        genotype: Genotype,
        genotype_shape_dtype: Genotype,
        centroids: Centroid,
    ) -> Self:

        # get number of centroids
        num_centroids = centroids.shape[0]

        # default fitness is -inf
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

        default_genotypes = tjnp.duplicate(genotype, num_centroids)

        # default descriptor is all zeros
        default_descriptors = jnp.zeros_like(centroids)

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
            genotype_shape_dtype=genotype_shape_dtype,
        )

    @override
    def empty(self) -> Self:
        return self.replace(
            genotypes=jax_pure_callback(
                _empty_genotypes, self.genotypes, self.genotypes, vectorized=True
            ),
            fitnesses=jnp.full_like(self.fitnesses, -jnp.inf),
        )
