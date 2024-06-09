import jax
import flax.core.scope
import qdax.core.containers.mapelites_repertoire
import qdax.core.emitters.emitter
import qdax.core.emitters.multi_emitter
from qdax.types import Genotype, Fitness, Descriptor, ExtraScores

from functools import partial
from typing import Optional

from ..neuroevolution import GenotypePair
from ..treax import numpy as tjnp
from ..utils import RNGKey, jax_jit


class MultiEmitter(qdax.core.emitters.multi_emitter.MultiEmitter):

    emit = jax_jit(  # pyright: ignore [reportAssignmentType]
        qdax.core.emitters.multi_emitter.MultiEmitter.emit._fun,  # pyright: ignore [reportAttributeAccessIssue]
        static_argnames=('self',)
    )

    state_update = jax_jit(  # pyright: ignore [reportAssignmentType]
        qdax.core.emitters.multi_emitter.MultiEmitter.state_update._fun,  # pyright: ignore [reportAttributeAccessIssue]
        static_argnames=('self',)
    )


class RefEmitterState(qdax.core.emitters.emitter.EmitterState):
    representation_params: flax.core.scope.VariableDict
    emitted_representation_params: Optional[flax.core.scope.VariableDict]


class RefMultiEmitter(MultiEmitter):

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> tuple[qdax.core.emitters.multi_emitter.MultiEmitterState, RNGKey]:

        # prepare keys for each emitter
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, len(self.emitters))

        # init all emitter states - gather them
        emitter_states = []
        for i, (emitter, subkey_emitter) in enumerate(zip(self.emitters, subkeys)):
            emitter_state, _ = emitter.init(init_genotypes, subkey_emitter)
            emitter_states.append(emitter_state)
            if i == 0:
                assert isinstance(init_genotypes, GenotypePair)
                init_genotypes = init_genotypes[1]

        return qdax.core.emitters.multi_emitter.MultiEmitterState(tuple(emitter_states)), random_key

    @partial(jax_jit, static_argnames=('self',))
    def emit(
        self,
        repertoire: Optional[qdax.core.containers.mapelites_repertoire.MapElitesRepertoire],
        emitter_state: qdax.core.emitters.multi_emitter.MultiEmitterState,
        random_key: RNGKey,
    ) -> tuple[GenotypePair[Genotype, Genotype], RNGKey]:
        assert emitter_state is not None
        assert len(emitter_state.emitter_states) == len(self.emitters)

        # prepare subkeys for each sub emitter
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, len(self.emitters))

        # emit from all emitters and gather offsprings
        representation_genotype: Genotype = {}
        all_offsprings: list[Genotype] = []
        for emitter, sub_emitter_state, subkey_emitter in zip(
            self.emitters,
            emitter_state.emitter_states,
            subkeys,
        ):
            genotype, _ = emitter.emit(repertoire, sub_emitter_state, subkey_emitter)
            if isinstance(genotype, GenotypePair):
                representation_genotype = genotype[0]
                genotype = genotype[1]
            batch_size = jax.tree_util.tree_leaves(genotype)[0].shape[0]
            assert batch_size == emitter.batch_size
            all_offsprings.append(genotype)

        # concatenate offsprings together
        offsprings = tjnp.concatenate(*all_offsprings, axis=0)

        return GenotypePair(representation_genotype, offsprings), random_key

    @partial(jax_jit, static_argnames=('self',))
    def state_update(
        self,
        emitter_state: qdax.core.emitters.multi_emitter.MultiEmitterState,
        repertoire: Optional[qdax.core.containers.mapelites_repertoire.MapElitesRepertoire] = None,
        genotypes: Optional[Genotype] = None,
        fitnesses: Optional[Fitness] = None,
        descriptors: Optional[Descriptor] = None,
        extra_scores: Optional[ExtraScores] = None,
    ) -> qdax.core.emitters.multi_emitter.MultiEmitterState:
        # update all the sub emitter states
        emitter_states = []

        for i, (emitter, sub_emitter_state, index_start, index_end) in enumerate(zip(
            self.emitters,
            emitter_state.emitter_states,
            self.indexes_start_batches,
            self.indexes_end_batches,
        )):
            # update with all genotypes, fitnesses, etc...
            if emitter.use_all_data:
                new_sub_emitter_state = emitter.state_update(
                    sub_emitter_state,
                    repertoire,
                    genotypes,
                    fitnesses,
                    descriptors,
                    extra_scores,
                )
                emitter_states.append(new_sub_emitter_state)
            # update only with the data of the emitted genotypes
            else:
                # extract relevant data
                sub_gen, sub_fit, sub_desc, sub_extra_scores = tjnp.getitem((
                    genotypes,
                    fitnesses,
                    descriptors,
                    extra_scores,
                ), indices=slice(index_start, index_end))
                # update only with the relevant data
                new_sub_emitter_state = emitter.state_update(
                    sub_emitter_state,
                    repertoire,
                    sub_gen,
                    sub_fit,
                    sub_desc,
                    sub_extra_scores,
                )
                emitter_states.append(new_sub_emitter_state)

            if i == 0 and genotypes is not None:
                assert isinstance(genotypes, GenotypePair)
                genotypes = genotypes[1]

        # return the update global emitter state
        return qdax.core.emitters.multi_emitter.MultiEmitterState(tuple(emitter_states))
