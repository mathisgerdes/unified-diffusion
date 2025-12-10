"""Semantic representations for diffusion targets (WHITENED component space).

Representations:
- epsilon: additive noise in the forward SDE
- score: ∇_x log p_t(x)
- velocity (vector field): reverse-time ODE drift

Basis-transform behavior (linear orthonormal bases):
- score is a covector and transforms with the inverse-adjoint Jacobian; in our
  orthonormal spaces this equals applying the same unitary used for data → comp.
- epsilon and velocity are vectors; they transform like data (primal), i.e.,
  apply the forward transform to move into a new basis and the inverse to return.

All conversions here operate in WHITENED COMPONENT space where the schedule is
diagonal and the forward noise is injected. If we have unwhitened components,
whiten first (or switch basis via `Score`/`Data`) before converting between objects.
"""

from enum import StrEnum
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .schedules import NoisingState


def epsilon_to_score(eps: jax.Array, ns: NoisingState) -> jax.Array:
    """score = -ε / σ"""
    ns = ns.broadcast_dims(shape=eps.shape)
    return -eps / ns.sigma


def score_to_epsilon(score: jax.Array, ns: NoisingState) -> jax.Array:
    """ε = -σ · score"""
    ns = ns.broadcast_dims(shape=score.shape)
    return -ns.sigma * score


def score_to_velocity(score: jax.Array, x: jax.Array, ns: NoisingState) -> jax.Array:
    """v = β/2 · (x + score)"""
    ns = ns.broadcast_dims(shape=x.shape)
    return ns.beta / 2 * (x + score)


def velocity_to_score(v: jax.Array, x: jax.Array, ns: NoisingState) -> jax.Array:
    """score = 2v/β - x"""
    ns = ns.broadcast_dims(shape=x.shape)
    return 2 * v / ns.beta - x


def epsilon_to_velocity(eps: jax.Array, x: jax.Array, ns: NoisingState) -> jax.Array:
    """v = β/2 · (x - ε/σ)"""
    score = epsilon_to_score(eps, ns)
    return score_to_velocity(score, x, ns)


def velocity_to_epsilon(v: jax.Array, x: jax.Array, ns: NoisingState) -> jax.Array:
    """ε = σx - 2σv/β"""
    score = velocity_to_score(v, x, ns)
    return score_to_epsilon(score, ns)
