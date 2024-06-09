import jax
import jax.numpy as jnp
from qdax.core.neuroevolution.buffers import buffer
from qdax.types import Params, Observation, Action

from collections.abc import Callable
from typing import TypeVar

from ..treax import numpy as tjnp
from ..utils import RNGKey, jax_jit


_ParamsT = TypeVar('_ParamsT', bound=Params)


def make_td3_loss_fn(
    policy_fn: Callable[[_ParamsT, Observation], jax.Array],
    critic_fn: Callable[[_ParamsT, Observation, Action], jax.Array],
    reward_scaling: float,
    discount: float,
    noise_clip: float,
    policy_noise: float,
):

    @jax_jit
    def _policy_loss_fn(
        policy_params: _ParamsT,
        critic_params: _ParamsT,
        transitions: buffer.Transition,
    ) -> jax.Array:

        action = policy_fn(policy_params, transitions.obs)
        q_value = critic_fn(
            critic_params, transitions.obs, action
        )
        q1_action = q_value[..., 0]
        policy_loss = -jnp.mean(q1_action, axis=-1)
        return policy_loss

    @jax_jit
    def _critic_loss_fn(
        critic_params: _ParamsT,
        target_policy_params: _ParamsT,
        target_critic_params: _ParamsT,
        transitions: buffer.Transition,
        random_key: RNGKey,
    ) -> jax.Array:
        noise = (
            jax.random.normal(random_key, shape=transitions.actions.shape)
            * policy_noise
        ).clip(-noise_clip, noise_clip)

        next_action = (
            policy_fn(target_policy_params, transitions.next_obs) + noise
        ).clip(-1.0, 1.0)
        next_q = critic_fn(
            target_critic_params, transitions.next_obs, next_action
        )
        next_v = jnp.min(next_q, axis=-1)
        target_q = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_v
        )
        q_old_action = critic_fn(
            critic_params,
            transitions.obs,
            transitions.actions,
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)

        # Better bootstrapping for truncated episodes.
        q_error = q_error * jnp.expand_dims(1.0 - transitions.truncations, -1)

        # compute the loss
        q_losses = jnp.mean(jnp.square(q_error), axis=-2)
        q_loss = jnp.sum(q_losses, axis=-1)

        return q_loss

    return _policy_loss_fn, _critic_loss_fn


def make_se_td3_loss_fn(
    representation_fn: Callable[[_ParamsT, Observation], jax.Array],
    decision_fn: Callable[[_ParamsT, jax.Array], jax.Array],
    critic_fn: Callable[[_ParamsT, Observation, Action], jax.Array],
    q1_fn: Callable[[_ParamsT, Observation, Action], jax.Array],
    reward_scaling: float,
    discount: float,
    noise_clip: float,
    policy_noise: float,
    decision_factor: float,
):

    @jax_jit
    def _policy_loss_fn(
        representation_params: _ParamsT,
        decision_params: _ParamsT,
        critic_params: _ParamsT,
        transitions: buffer.Transition,
    ) -> jax.Array:

        action = decision_fn(
            decision_params, representation_fn(representation_params, transitions.obs)
        )
        q1_action = q1_fn(
            critic_params, transitions.obs, action
        )
        policy_loss = -jnp.mean(q1_action, axis=-1)
        return policy_loss

    @jax_jit
    def _mixed_policy_loss_fn(
        representation_params: _ParamsT,
        greedy_decision_params: _ParamsT,
        decision_params: _ParamsT,
        critic_params: _ParamsT,
        transitions: buffer.Transition,
    ) -> jax.Array:

        decision_params = tjnp.concatenate(
            decision_params, tjnp.getitem(greedy_decision_params, None)
        )
        representation = representation_fn(representation_params, transitions.obs)
        action = jax.vmap(decision_fn, in_axes=(0, None))(decision_params, representation)
        q1_action = jax.vmap(q1_fn, in_axes=(None, None, 0))(
            critic_params, transitions.obs, action
        )
        policy_loss = -jnp.mean(q1_action, axis=-1)
        if decision_factor == 0.0:
            policy_loss = policy_loss[0]
        elif decision_factor == 1.0:
            policy_loss = jnp.sum(policy_loss, axis=0)
        else:
            policy_loss = policy_loss[0] + jnp.sum(policy_loss[1:], axis=0) * decision_factor
        return policy_loss

    @jax_jit
    def _critic_loss_fn(
        critic_params: _ParamsT,
        target_representation_params: _ParamsT,
        target_decision_params: _ParamsT,
        target_critic_params: _ParamsT,
        transitions: buffer.Transition,
        random_key: RNGKey,
    ) -> jax.Array:
        noise = (
            jax.random.normal(random_key, shape=transitions.actions.shape)
            * policy_noise
        ).clip(-noise_clip, noise_clip)

        next_action = (
            decision_fn(
                target_decision_params,
                representation_fn(target_representation_params, transitions.next_obs),
            ) + noise
        ).clip(-1.0, 1.0)
        next_q = critic_fn(
            target_critic_params, transitions.next_obs, next_action
        )
        next_v = jnp.min(next_q, axis=-1)
        target_q = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_v
        )
        q_old_action = critic_fn(
            critic_params,
            transitions.obs,
            transitions.actions,
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)

        # Better bootstrapping for truncated episodes.
        q_error = q_error * jnp.expand_dims(1.0 - transitions.truncations, -1)

        # compute the loss
        q_losses = jnp.mean(jnp.square(q_error), axis=-2)
        q_loss = jnp.sum(q_losses, axis=-1)

        return q_loss

    return _policy_loss_fn, _mixed_policy_loss_fn, _critic_loss_fn


def make_ddqn_loss_fn(
    fn: Callable[[_ParamsT, Observation], Action],
    reward_scaling: float,
    discount: float,
):

    @jax_jit
    def _loss_fn(
        policy_params: _ParamsT,
        target_policy_params: _ParamsT,
        transitions: buffer.Transition,
    ) -> jax.Array:
        next_action = jnp.argmax(fn(policy_params, transitions.next_obs), axis=-1, keepdims=True)
        next_v = jnp.take_along_axis(
            fn(target_policy_params, transitions.next_obs), next_action, axis=-1
        ).squeeze(axis=-1)
        assert isinstance(next_v, jax.Array)
        target_q = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_v
        )
        q = jnp.take_along_axis(
            fn(policy_params, transitions.obs),
            (transitions.actions + 0.5 * jnp.sign(transitions.actions)).astype(jnp.int32),
            axis=-1,
        ).squeeze(axis=-1)
        assert isinstance(q, jax.Array)
        q_error = q - target_q

        # Better bootstrapping for truncated episodes.
        q_error = q_error * (1.0 - transitions.truncations)

        q_loss = jnp.mean(jnp.square(q_error), axis=0)

        return q_loss

    return _loss_fn


def make_se_ddqn_loss_fn(
    fn: Callable[[_ParamsT, _ParamsT, Observation], Action],
    reward_scaling: float,
    discount: float,
):

    @jax_jit
    def _loss_fn(
        representation_params: _ParamsT,
        decision_params: _ParamsT,
        target_representation_params: _ParamsT,
        target_decision_params: _ParamsT,
        transitions: buffer.Transition,
    ) -> jax.Array:
        next_action = jnp.argmax(
            fn(representation_params, decision_params, transitions.next_obs),
            axis=-1,
            keepdims=True,
        )
        next_v = jnp.take_along_axis(
            fn(target_representation_params, target_decision_params, transitions.next_obs),
            next_action,
            axis=-1,
        ).squeeze(axis=-1)
        assert isinstance(next_v, jax.Array)
        target_q = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_v
        )
        q = jnp.take_along_axis(
            fn(representation_params, decision_params, transitions.obs),
            (transitions.actions + 0.5 * jnp.sign(transitions.actions)).astype(jnp.int32),
            axis=-1,
        ).squeeze(axis=-1)
        assert isinstance(q, jax.Array)
        q_error = q - target_q

        # Better bootstrapping for truncated episodes.
        q_error = q_error * (1.0 - transitions.truncations)

        q_loss = jnp.mean(jnp.square(q_error), axis=0)

        return q_loss

    return _loss_fn
