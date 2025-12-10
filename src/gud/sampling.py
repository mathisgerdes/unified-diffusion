"""Sampling utilities for forward and reverse diffusion SDEs.

All stochastic calculus is performed in whitened component space where the
noising schedule is diagonal. Network input and outputs should be Data objects.
"""

from __future__ import annotations

from typing import Callable

import bijx
import diffrax
import jax
import jax.numpy as jnp

from .conversions import epsilon_to_score
from .schedules import NoisingState, Schedule
from .spaces import Data, Space


def ou_process(rng: jax.Array, x: Data, ns: NoisingState) -> tuple[Data, jax.Array]:
    """Forward OU step in whitened component space.

    - Noise is drawn in component-white shape and run through `space.transform_noise`
      to respect possible unitary transforms (FFT).
    - Returns the noised sample and the raw injected noise `z`.
    """
    z = jax.random.normal(rng, x.comp_white.shape)
    z = x.space.transform_noise(z)
    ns = ns.broadcast_dims(shape=x.shape)
    noised = ns.alpha * x.comp_white + ns.sigma * z
    return Data(x.space, comp_white=noised), Data(x.space, comp_white=z)


def make_reverse_sde_terms(
    model_fn: Callable[[NoisingState, Data], Data],
    schedule: Schedule,
    space: Space,
) -> tuple[
    Callable[[float, jax.Array, tuple], jax.Array],
    Callable[[float, jax.Array, tuple], jax.Array],
]:
    """Construct drift and diffusion for the reverse OU SDE in whitened comps.

    Args:
        model_fn: Callable (ns, data) -> epsilon prediction as Data object.
        schedule: Noise schedule returning `NoisingState`.
        space:    Space describing transforms and noise handling.

    Returns:
        (drift, diffusion) callables suitable for diffrax SDE solvers.
    """

    def drift(t: float, x: jax.Array, _) -> jax.Array:
        ns = schedule(t, shape=x.shape)
        eps_pred = model_fn(ns, Data(space, comp_white=x)).comp_white
        score = epsilon_to_score(eps_pred, ns)
        return -ns.beta * (x / 2 + score)

    def diffusion(t: float, x: jax.Array, _) -> jax.Array:
        beta = schedule(t, shape=x.shape).beta
        return jnp.sqrt(beta)

    return drift, diffusion


def sample_reverse_sde(
    rng: jax.Array,
    count: int,
    model_fn: Callable[[NoisingState, Data], Data],
    schedule: Schedule,
    space: Space,
    data_shape: tuple[int, ...],
    *,
    solver: diffrax.AbstractSolver | None = None,
    dt: float = 0.01,
    t_start: float = 1.0,
    t_end: float = 0.0,
    saveat: diffrax.SaveAt | None = None,
) -> bijx.DiffraxSolution:
    """Run reverse SDE sampling in whitened component space and return solution.

    Notes:
    - Integration happens in whitened component space.
    - Network is expected to take x and return epsilon as Data objects.
    - Noise is transformed via `space.transform_noise` to respect unitary bases.

    """
    solver = solver or diffrax.Euler()
    if saveat is None:
        saveat = diffrax.SaveAt(t1=True)

    config = bijx.DiffraxConfig(
        solver=solver,
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        saveat=saveat,
    )

    drift, diffusion = make_reverse_sde_terms(
        model_fn, schedule, space
    )

    x0 = jax.random.normal(rng, shape=(count, *data_shape))
    x0 = space.transform_noise(x0)

    sol = config.solve_sde(
        drift,
        diffusion,
        x0,
        rng,
        noise_transform=space.transform_noise,
    )

    ys = sol.ys
    ys = ys.reshape(-1, *data_shape)
    ys = space.wrap(comp_white=ys).data
    return ys.reshape(sol.ys.shape)
