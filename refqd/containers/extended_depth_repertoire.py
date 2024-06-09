import jax
import jax.numpy as jnp
import flax.struct
import qdax.core.containers.mapelites_repertoire
from qdax.types import Centroid, Descriptor, Fitness, Genotype

import numpy as np

import logging
import time
from functools import partial
from collections.abc import Callable
from typing import Optional, TypeVar, Self, TYPE_CHECKING

from .depth_repertoire import DeepMapElitesRepertoire
from ..treax import chain as tchain, numpy as tjnp
from ..utils import RNGKey, jax_jit, onp_callback, jax_pure_callback, jax_eval_shape


_log = logging.getLogger(__name__)


_ArrayT = TypeVar('_ArrayT', bound=jax.Array | np.ndarray)


class ExtendedDepthRepertoire(DeepMapElitesRepertoire):

    def _reshape(self, x: _ArrayT) -> _ArrayT:
        return x.reshape(x.shape[0] // self.dims.shape[0], self.dims.shape[0], *x.shape[1:])

    def get_sorted_idx(self) -> jax.Array:
        fitnesses_depth = self._reshape(self.fitnesses_depth)
        idx = jnp.argsort(fitnesses_depth, axis=1)[:, ::-1]
        idx += jnp.expand_dims(jnp.arange(idx.shape[0]) * idx.shape[1], axis=-1)
        return idx

    def get_from_idx(self, idx: jax.Array) -> tuple[Genotype, Fitness, Descriptor]:
        return tjnp.getitem(
            (self.genotypes_depth, self.fitnesses_depth, self.descriptors_depth),
            idx,
        )

    def copy_from(self, other: Self, layers: int) -> Self:
        assert self.dims.shape == other.dims.shape

        self = self.empty()

        idx = other.get_sorted_idx()

        genotypes, fitnesses, descriptors = tjnp.getitem(
            (other.genotypes_depth, other.fitnesses_depth, other.descriptors_depth),
            indices=idx[:, layers],
        )
        genotypes_depth, fitnesses_depth, descriptors_depth = jax.tree_map(
            lambda s, o: self._reshape(s).at[:, layers:].set(o[idx[:, layers:]]).reshape(*s.shape),
            (self.genotypes_depth, self.fitnesses_depth, self.descriptors_depth),
            (other.genotypes_depth, other.fitnesses_depth, other.descriptors_depth),
        )
        assert self.fitnesses_depth.shape == fitnesses_depth.shape
        assert self.descriptors_depth.shape == descriptors_depth.shape
        return self.replace(
            genotypes=genotypes,
            genotypes_depth=genotypes_depth,
            fitnesses=fitnesses,
            fitnesses_depth=fitnesses_depth,
            descriptors=descriptors,
            descriptors_depth=descriptors_depth,
        )

    @jax_jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
    ) -> Self:

        out_of_bound = (
            self.dims.shape[0] * self.centroids.shape[0]
        )  # Index of non-added individuals

        # Get indices for given descriptors
        batch_of_indices = qdax.core.containers.mapelites_repertoire.get_cells_indices(
            batch_of_descriptors, self.centroids
        )

        # Filter dead individuals
        batch_of_indices = jnp.where(
            batch_of_fitnesses > -jnp.inf,
            batch_of_indices,
            out_of_bound,
        )

        # Get final indices of individuals added to top layer of the grid
        # (i.e. best indivs added in: genotypes, fitnesses, descriptors)
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices,
            num_segments=self.centroids.shape[0],
        )
        filter_fitnesses = jnp.where(
            best_fitnesses[batch_of_indices] == batch_of_fitnesses,
            batch_of_fitnesses,
            -jnp.inf,
        )
        current_fitnesses = jnp.take_along_axis(self.fitnesses, batch_of_indices, 0)
        addition_condition = filter_fitnesses > current_fitnesses
        final_batch_of_max_indices = jnp.where(
            addition_condition,
            batch_of_indices,
            out_of_bound,
        )

        # Get final indices of individuals added to the depth of the grid
        # (i.e. indivs in: genotypes_depth, fitnesses_depth, descriptors_depth)
        final_batch_of_indices = self._place_indivs(
            batch_of_indices, batch_of_fitnesses
        )

        # Create new grid
        new_grid_genotypes_depth = jax.tree_map(
            lambda grid_genotypes, new_genotypes: grid_genotypes.at[
                final_batch_of_indices
            ].set(new_genotypes),
            self.genotypes_depth,
            batch_of_genotypes,
        )
        new_grid_genotypes = jax.tree_map(
            lambda grid_genotypes, new_genotypes: grid_genotypes.at[
                final_batch_of_max_indices
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # Compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[final_batch_of_max_indices].set(
            batch_of_fitnesses
        )
        new_fitnesses_depth = self.fitnesses_depth.at[final_batch_of_indices].set(
            batch_of_fitnesses
        )
        new_descriptors = self.descriptors.at[final_batch_of_max_indices].set(
            batch_of_descriptors
        )
        new_descriptors_depth = self.descriptors_depth.at[final_batch_of_indices].set(
            batch_of_descriptors
        )

        return self.replace(
            genotypes=new_grid_genotypes,
            genotypes_depth=new_grid_genotypes_depth,
            fitnesses=new_fitnesses,
            fitnesses_depth=new_fitnesses_depth,
            descriptors=new_descriptors,
            descriptors_depth=new_descriptors_depth,
            centroids=self.centroids,
            dims=self.dims,
        )

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        depth: int,
    ) -> Self:

        # Initialize grid with default values
        num_centroids = centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)
        default_fitnesses_depth = -jnp.inf * jnp.ones(shape=(num_centroids * depth))
        default_genotypes = jax.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape[1:]),
            genotypes,
        )
        default_genotypes_depth = jax.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids * depth,) + x.shape[1:]),
            genotypes,
        )
        default_descriptors = jnp.zeros(shape=(num_centroids, centroids.shape[-1]))
        default_descriptors_depth = jnp.zeros(
            shape=(num_centroids * depth, centroids.shape[-1])
        )
        dims = jnp.zeros(shape=(depth))

        repertoire = cls(
            genotypes=default_genotypes,
            genotypes_depth=default_genotypes_depth,
            fitnesses=default_fitnesses,
            fitnesses_depth=default_fitnesses_depth,
            descriptors=default_descriptors,
            descriptors_depth=default_descriptors_depth,
            centroids=centroids,
            dims=dims,
        )

        # Add initial values to the grid
        new_repertoire = repertoire.add(genotypes, descriptors, fitnesses)

        return new_repertoire

    def empty(self) -> Self:
        return self.replace(
            genotypes=tjnp.zeros_like(self.genotypes),
            genotypes_depth=tjnp.zeros_like(self.genotypes_depth),
            fitnesses=jnp.full_like(self.fitnesses, -jnp.inf),
            fitnesses_depth=jnp.full_like(self.fitnesses_depth, -jnp.inf),
        )

    if TYPE_CHECKING:
        def replace(self, **overrides) -> Self:
            return super().replace(**overrides)


global_genotypes: list[Genotype] = []
global_genotypes_depth: list[Genotype] = []

global_time_empty: float = 0.0
global_time_add: float = 0.0
global_time_sample: float = 0.0
global_time_copy_from: float = 0.0


@onp_callback
def _empty_genotypes(
    fake_genotypes: np.ndarray, fake_genotypes_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    global global_time_empty
    global_time_empty -= time.monotonic()
    fake_genotypes_ = fake_genotypes[..., 0]
    idx = int(fake_genotypes_.flatten()[0])

    def fn(genotypes: np.ndarray):
        genotypes[:] = np.float32(0)
        return genotypes

    global_genotypes[idx] = jax.tree_map(fn, global_genotypes[idx])
    global_genotypes_depth[idx] = jax.tree_map(fn, global_genotypes_depth[idx])

    global_time_empty += time.monotonic()
    return (fake_genotypes, fake_genotypes_depth)


@onp_callback
def _update_genotypes(
    fake_genotypes: np.ndarray, new_genotypes: Genotype, batch_of_indices: np.ndarray
) -> np.ndarray:
    global global_time_add
    global_time_add -= time.monotonic()
    batch_of_indices = batch_of_indices
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
def _update_genotypes_depth(
    fake_genotypes_depth: np.ndarray, new_genotypes: Genotype, batch_of_indices: np.ndarray
) -> np.ndarray:
    global global_time_add
    global_time_add -= time.monotonic()
    batch_of_indices = batch_of_indices
    fake_genotypes_depth_ = fake_genotypes_depth[..., 0]
    idx = int(fake_genotypes_depth_.flatten()[0])
    batch_idxs = np.argwhere(fake_genotypes_depth_ >= 0)

    def fn(genotypes_depth: np.ndarray, new_genotypes: np.ndarray):
        n_centroids = genotypes_depth.shape[batch_idxs.shape[-1]]
        for batch_idx in batch_idxs:
            batch_of_indices_ = batch_of_indices[*batch_idx]
            mask = batch_of_indices_ < n_centroids
            genotypes_depth[*batch_idx, batch_of_indices_[mask]] = new_genotypes[*batch_idx, mask]
        return genotypes_depth

    global_genotypes_depth[idx] = jax.tree_map(fn, global_genotypes_depth[idx], new_genotypes)
    global_time_add += time.monotonic()
    return fake_genotypes_depth


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


@onp_callback
def _sample_genotypes_depth(
    fake_genotypes_depth: np.ndarray,
    indices: np.ndarray,
    fitnesses_depth: Optional[np.ndarray] = None,
) -> Genotype:
    global global_time_sample
    global_time_sample -= time.monotonic()
    fake_genotypes_depth = fake_genotypes_depth[..., 0]
    idx = int(fake_genotypes_depth.flatten()[0])
    batch_idxs = np.argwhere(fake_genotypes_depth >= 0)
    indices = indices.astype(np.int32)
    if fitnesses_depth is not None:
        for i in range(fitnesses_depth.ndim - indices.ndim):
            indices = np.repeat(
                np.expand_dims(indices, axis=i), repeats=fitnesses_depth.shape[i], axis=i
            )

    def fn(genotypes: np.ndarray):
        res = np.empty((*indices.shape, *genotypes.shape[indices.ndim:]), dtype=genotypes.dtype)
        for batch_idx in batch_idxs:
            res[*batch_idx] = genotypes[*batch_idx, indices[*batch_idx]]
        return res

    samples = jax.tree_map(fn, global_genotypes_depth[idx])
    global_time_sample += time.monotonic()
    return samples


@onp_callback
def _copy_from(
    fake_genotypes: np.ndarray,
    fake_genotypes_depth: np.ndarray,
    fake_other_genotypes_depth: np.ndarray,
    remaining_idx: np.ndarray,
    layers: int,
    reshape_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    global global_time_copy_from
    global_time_copy_from -= time.monotonic()
    fake_genotypes_ = fake_genotypes[..., 0]
    idx = int(fake_genotypes_.flatten()[0])
    fake_other_genotypes_depth_ = fake_other_genotypes_depth[..., 0]
    other_idx = int(fake_other_genotypes_depth_.flatten()[0])
    batch_idxs = np.argwhere(fake_genotypes_ >= 0)

    def fn(genotypes: np.ndarray, genotypes_depth: np.ndarray, other_genotypes_depth: np.ndarray):
        for batch_idx in batch_idxs:
            remaining_idx_ = remaining_idx[*batch_idx]
            genotypes[*batch_idx] = other_genotypes_depth[*batch_idx, remaining_idx_[:, 0]]
            reshape_fn(genotypes_depth)[*batch_idx, :, layers:] = (
                other_genotypes_depth[*batch_idx, remaining_idx_]
            )

    jax.tree_map(
        fn, global_genotypes[idx], global_genotypes_depth[idx], global_genotypes_depth[other_idx]
    )

    global_time_copy_from += time.monotonic()
    return fake_genotypes, fake_genotypes_depth


class CPUDepthRepertoire(ExtendedDepthRepertoire):

    genotype_shape_dtype: Genotype = flax.struct.field(pytree_node=False)

    def get_from_idx(self, idx: jax.Array) -> tuple[Genotype, Fitness, Descriptor]:
        assert idx.ndim == 1
        shape_dtype = jax_eval_shape(
            partial(tjnp.duplicate, repeats=idx.shape[0]), self.genotype_shape_dtype
        )
        samples = jax_pure_callback(
            _sample_genotypes_depth,
            shape_dtype,
            self.genotypes_depth, idx, self.fitnesses_depth,
            vectorized=True,
        )
        fitnesses, descriptors = tjnp.getitem((self.fitnesses_depth, self.descriptors_depth), idx)
        return samples, fitnesses, descriptors

    def copy_from(self, other: Self, layers: int) -> Self:
        assert self.dims.shape == other.dims.shape

        self = self.empty()

        idx = other.get_sorted_idx()

        fitnesses, descriptors = tjnp.getitem(
            (other.fitnesses_depth, other.descriptors_depth),
            indices=idx[:, layers],
        )
        fitnesses_depth, descriptors_depth = jax.tree_map(
            lambda s, o: self._reshape(s).at[:, layers:].set(o[idx[:, layers:]]).reshape(*s.shape),
            (self.fitnesses_depth, self.descriptors_depth),
            (other.fitnesses_depth, other.descriptors_depth),
        )
        assert self.fitnesses_depth.shape == fitnesses_depth.shape
        assert self.descriptors_depth.shape == descriptors_depth.shape
        genotypes, genotypes_depth = jax_pure_callback(
            partial(_copy_from, layers=layers, reshape_fn=self._reshape),
            (self.genotypes, self.genotypes_depth),
            self.genotypes, self.genotypes_depth, other.genotypes_depth, idx[:, layers:],
            vectorized=True,
        )
        return self.replace(
            genotypes=genotypes,
            genotypes_depth=genotypes_depth,
            fitnesses=fitnesses,
            fitnesses_depth=fitnesses_depth,
            descriptors=descriptors,
            descriptors_depth=descriptors_depth,
        )

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
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
    ) -> Self:
        out_of_bound = (
            self.dims.shape[0] * self.centroids.shape[0]
        )  # Index of non-added individuals

        # Get indices for given descriptors
        batch_of_indices = qdax.core.containers.mapelites_repertoire.get_cells_indices(
            batch_of_descriptors, self.centroids
        )

        # Filter dead individuals
        batch_of_indices = jnp.where(
            batch_of_fitnesses > -jnp.inf,
            batch_of_indices,
            out_of_bound,
        )

        # Get final indices of individuals added to top layer of the grid
        # (i.e. best indivs added in: genotypes, fitnesses, descriptors)
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices,
            num_segments=self.centroids.shape[0],
        )
        filter_fitnesses = jnp.where(
            best_fitnesses[batch_of_indices] == batch_of_fitnesses,
            batch_of_fitnesses,
            -jnp.inf,
        )
        current_fitnesses = jnp.take_along_axis(self.fitnesses, batch_of_indices, 0)
        addition_condition = filter_fitnesses > current_fitnesses
        final_batch_of_max_indices = jnp.where(
            addition_condition,
            batch_of_indices,
            out_of_bound,
        )

        # Get final indices of individuals added to the depth of the grid
        # (i.e. indivs in: genotypes_depth, fitnesses_depth, descriptors_depth)
        final_batch_of_indices = self._place_indivs(
            batch_of_indices, batch_of_fitnesses
        )

        # Create new grid
        new_grid_genotypes_depth = jax_pure_callback(
            _update_genotypes_depth,
            self.genotypes_depth,
            self.genotypes_depth, batch_of_genotypes, final_batch_of_indices,
            vectorized=True,
        )
        new_grid_genotypes = jax_pure_callback(
            _update_genotypes,
            self.genotypes,
            self.genotypes, batch_of_genotypes, final_batch_of_max_indices,
            vectorized=True,
        )

        # Compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[final_batch_of_max_indices].set(
            batch_of_fitnesses
        )
        new_fitnesses_depth = self.fitnesses_depth.at[final_batch_of_indices].set(
            batch_of_fitnesses
        )
        new_descriptors = self.descriptors.at[final_batch_of_max_indices].set(
            batch_of_descriptors
        )
        new_descriptors_depth = self.descriptors_depth.at[final_batch_of_indices].set(
            batch_of_descriptors
        )

        return self.replace(
            genotypes=new_grid_genotypes,
            genotypes_depth=new_grid_genotypes_depth,
            fitnesses=new_fitnesses,
            fitnesses_depth=new_fitnesses_depth,
            descriptors=new_descriptors,
            descriptors_depth=new_descriptors_depth,
            centroids=self.centroids,
            dims=self.dims,
        )

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        depth: int,
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

            def fn_depth(genotype: np.ndarray) -> np.ndarray:
                return np.zeros(
                    (
                        *fitnesses.shape[:-1],
                        centroids.shape[-2] * depth,
                        *genotype.shape[fitnesses.ndim:],
                    ),
                    dtype=genotype.dtype,
                )

            initial_genotypes_depth = jax.tree_map(fn_depth, genotype)
            global_genotypes_depth.append(initial_genotypes_depth)

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

        # Initialize grid with default values
        num_centroids = centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)
        default_fitnesses_depth = -jnp.inf * jnp.ones(shape=(num_centroids * depth))
        default_genotypes = tjnp.duplicate(first_genotype, num_centroids)
        default_genotypes_depth = tjnp.duplicate(first_genotype, num_centroids * depth)
        default_descriptors = jnp.zeros(shape=(num_centroids, centroids.shape[-1]))
        default_descriptors_depth = jnp.zeros(
            shape=(num_centroids * depth, centroids.shape[-1])
        )
        dims = jnp.zeros(shape=(depth))

        repertoire = cls(
            genotypes=default_genotypes,
            genotypes_depth=default_genotypes_depth,
            fitnesses=default_fitnesses,
            fitnesses_depth=default_fitnesses_depth,
            descriptors=default_descriptors,
            descriptors_depth=default_descriptors_depth,
            centroids=centroids,
            dims=dims,
            genotype_shape_dtype=first_genotype_shape_dtype,
        )

        # Add initial values to the grid
        new_repertoire = repertoire.add(genotypes, descriptors, fitnesses)

        return new_repertoire

    def empty(self) -> Self:
        genotypes, genotypes_depth = jax_pure_callback(
            _empty_genotypes,
            (self.genotypes, self.genotypes_depth),
            self.genotypes, self.genotypes_depth,
            vectorized=True,
        )
        return self.replace(
            genotypes=genotypes,
            genotypes_depth=genotypes_depth,
            fitnesses=jnp.full_like(self.fitnesses, -jnp.inf),
            fitnesses_depth=jnp.full_like(self.fitnesses_depth, -jnp.inf),
        )
