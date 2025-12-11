"""Component-wise noise schedules for diffusion processes.

Mathematical Framework
======================

The variance-preserving Ornstein-Uhlenbeck SDE is:

    dx_i = -β_i(t)/2 · x_i dt + √β_i(t) dW_i

with solution: x_i(t) = α_i(t)·x_i(0) + σ_i(t)·ε, where ε ~ N(0,1).

Key Quantities
--------------
All quantities can be scalar (uniform across components) or per-component arrays.
We use γ (negative log signal-to-noise ratio) as the primary parameterization:

    γ_i(t) = log(σ²_i/α²_i) = logit(σ²_i)

From which all other quantities derive:
    α²_i = sigmoid(-γ_i)     signal retention
    σ²_i = sigmoid(γ_i)      noise variance
    β_i  = σ²_i · dγ_i/dt    instantaneous noising rate

The signal-to-noise ratio (including data variance Σ_i) is:
    SNR_i(t) = Σ_i · exp(-γ_i(t))

Boundary Behavior
-----------------
As γ → -∞: α → 1, σ → 0 (pure signal, no noise)
As γ → +∞: α → 0, σ → 1 (no signal, pure noise)


Shape Conventions
-----------------
Data layout: (batch, data_dim, channels).
Schedules act on the data_dim axis and always broadcast over channels.

Canonical shapes:
- Scalar schedules: γ has shape () → broadcasts to (batch, data_dim, 1)
- Component-wise schedules: γ has shape (data_dim,) or (batch, data_dim)
- Time t is scalar () or batched (batch,)

Use NoisingState.broadcast_dims()/match_batch_dim() to obtain
(batch, data_dim, 1) along with t broadcast to (batch,).

Component-wise Schedules
------------------------
LinearSchedule covers both simultaneous and offset/clipped trajectories by
choosing per-component γ_min,i/γ_max,i and optional clipping. Factories
(`soft_conditioning_schedule`, `variance_adjusted_schedule`, `column_schedule`,
`row_schedule`) compute endpoints/offsets and return LinearSchedule instances.

Design parameters commonly used by the factories:
- labels: ordering variables l_i that set relative noising order.
- softness: overlap between component SNR trajectories
  (∞ = identical = standard diffusion, 0 = fully separated/autoregressive).
"""

from typing import Callable, Protocol

import jax
import jax.numpy as jnp
from flax import nnx

# =============================================================================
# Noising State Container
# =============================================================================

@nnx.dataclass
class NoisingState(nnx.Pytree):
    """Noising state at time t, parameterized by γ (negative log SNR).

    Note that here the SNR is defined purely in terms of the OU process as α²/σ²,
    i.e. agnostic of the data variance Σ.

    Shape conventions:
        - γ: () for scalar schedules, (data_dim,) for component-wise
        - t: () for single time, (batch,) for batched evaluation
        - β: same shape as γ, or None if not computed

    All derived quantities (α, σ, snr) maintain γ's shape.
    Use match_batch_dim() to broadcast to (batch, data_dim, 1) with
    t broadcast to (batch,).
    """
    gamma: jax.Array
    t: jax.Array
    beta: jax.Array | None = None

    @property
    def alpha_sq(self) -> jax.Array:
        """Squared signal retention: α² = sigmoid(-γ)."""
        return jax.nn.sigmoid(-self.gamma)

    @property
    def sigma_sq(self) -> jax.Array:
        """Squared noise level: σ² = sigmoid(γ)."""
        return jax.nn.sigmoid(self.gamma)

    @property
    def alpha(self) -> jax.Array:
        """Signal retention factor: α = √sigmoid(-γ)."""
        return jnp.sqrt(self.alpha_sq)

    @property
    def sigma(self) -> jax.Array:
        """Noise standard deviation: σ = √sigmoid(γ)."""
        return jnp.sqrt(self.sigma_sq)

    @property
    def snr(self) -> jax.Array:
        """Signal-to-noise ratio (schedule only): α²/σ² = exp(-γ)."""
        return jnp.exp(-self.gamma)

    @property
    def log_snr(self) -> jax.Array:
        """Log SNR: -γ."""
        return -self.gamma

    def broadcast_dims(
        self,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> "NoisingState":
        """Broadcast noising state to (batch, data_dim, 1).

        If shape is provided, shapes may be (batch, data_dim, channels) or
        (data_dim, channels) in which case batch is treated as 1. The final
        axis of γ/β is forced to be singleton so that schedules always
        broadcast over channels.
        """
        if shape is None:
            return self

        shape = tuple(shape)

        if len(shape) == 3:
            batch, data_dim, channels = shape
        elif len(shape) == 2:
            batch = 1
            data_dim, channels = shape
        else:
            raise ValueError(f"Expected shape (batch, data_dim, channels) or (data_dim, channels); got {shape}.")

        if channels < 1:
            raise ValueError(f"channels dimension must be >=1, got {channels}.")

        target_shape = (batch, data_dim, 1)
        target_core = (batch, data_dim)

        def _broadcast(value: jax.Array | None, name: str) -> jax.Array | None:
            if value is None:
                return None
            value = jnp.asarray(value)
            if value.shape == target_shape:
                pass
            elif value.shape == target_core:
                value = value[..., None]
            elif value.ndim == 0:
                value = jnp.broadcast_to(value, target_shape)
            elif value.shape == (data_dim,):
                value = jnp.broadcast_to(value[None, :, None], target_shape)
            elif value.shape == (batch,):
                value = jnp.broadcast_to(value[:, None, None], target_shape)
            elif value.shape == (data_dim, 1):
                value = jnp.broadcast_to(value[None, ...], target_shape)
            elif value.shape == (batch, data_dim, 1):
                value = jnp.broadcast_to(value, target_shape)
            else:
                raise ValueError(
                    f"Cannot broadcast {name} shape {value.shape} to {target_shape}"
                )

            if value.shape[-1] != 1:
                raise ValueError(f"{name} must have a singleton channel axis; got {value.shape}.")
            return value

        gamma = _broadcast(self.gamma, "gamma")
        beta = _broadcast(self.beta, "beta")

        t = jnp.asarray(self.t)
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (batch,))
        elif t.shape == (batch,):
            pass
        elif t.shape == (batch, 1):
            t = t.reshape((batch,))
        elif t.shape == (1,) and batch == 1:
            t = jnp.broadcast_to(t, (batch,))
        else:
            raise ValueError(f"Cannot broadcast t shape {t.shape} to batch {batch}.")

        return NoisingState(gamma=gamma, t=t, beta=beta)


# =============================================================================
# Schedule Protocol and Base Classes
# =============================================================================


class Schedule(Protocol):
    """Protocol for noise schedules.

    A schedule maps diffusion time t ∈ [0, 1] to noising state.
    The primary method is gamma(t), from which all else derives.
    """

    def gamma(self, t: jax.Array) -> jax.Array:
        """Compute γ(t), the negative log SNR.

        Args:
            t: Diffusion time in [0, 1]. Shape () or (batch,).

        Returns:
            γ with shape () for scalar schedules, (data_dim,) for component-wise,
            or (batch, ...) if t is batched.
        """
        ...

    def __call__(
        self,
        t: jax.Array,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> NoisingState:
        """Compute noising state at time t and optionally broadcast it.

        If shape is provided, the returned state is broadcast to
        (batch, data_dim, 1) with t broadcast to (batch,).
        """
        ...


class GammaSchedule(nnx.Module):
    """Schedule defined by an arbitrary differentiable γ(t) function.

    This is the most flexible schedule type. The noising rate β is
    computed via automatic differentiation: β = σ² · dγ/dt.
    """

    def __init__(self, gamma_fn: Callable[[jax.Array], jax.Array]):
        """Initialize from a gamma function.

        Args:
            gamma_fn: Function t → γ(t). Must be differentiable.
                Can return scalar or array depending on schedule type.
        """
        self.gamma_fn = gamma_fn

    def gamma(self, t: jax.Array) -> jax.Array:
        return self.gamma_fn(t)

    def __call__(
        self,
        t: jax.Array,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> NoisingState:
        gamma = self.gamma(t)
        sigma_sq = jax.nn.sigmoid(gamma)
        # β = σ² · dγ/dt via forward-mode autodiff
        _, dgamma_dt = jax.jvp(self.gamma, (t,), (jnp.ones_like(t),))
        beta = sigma_sq * dgamma_dt
        state = NoisingState(gamma=gamma, t=t, beta=beta)
        return state.broadcast_dims(shape=shape)


@nnx.dataclass
class LinearSchedule(nnx.Pytree):
    """Linear interpolation in γ-space.

    γ(t) = γ_min + t · (γ_max - γ_min)

    This is the simplest and most commonly used schedule.
    For component-wise schedules, γ_min and γ_max are arrays of shape (data_dim,).
    """
    # γ at t=0 (high SNR). Scalar or (data_dim,)
    gamma_min: jax.Array = nnx.data(default_factory=lambda: jnp.array(-13.3))
    # γ at t=1 (low SNR). Scalar or (data_dim,)
    gamma_max: jax.Array = nnx.data(default_factory=lambda: jnp.array(5.0))
    # Optional clipping range (low, high). If None, no clipping is applied.
    gamma_cap: tuple[jax.Array | float, jax.Array | float] | None = nnx.data(default=None)

    @property
    def is_componentwise(self) -> bool:
        """True if schedule varies per component."""
        return self.gamma_min.ndim > 0

    @property
    def delta_gamma(self) -> jax.Array:
        """Total γ change: γ_max - γ_min."""
        return self.gamma_max - self.gamma_min

    @property
    def clipping(self) -> tuple[jax.Array | float, jax.Array | float] | None:
        """Active clipping bounds or None if unclipped."""
        return self.gamma_cap

    def gamma(self, t: jax.Array) -> jax.Array:
        if self.is_componentwise:
            t = jnp.expand_dims(t, -1)
        gamma = self.gamma_min + t * self.delta_gamma
        if self.gamma_cap is not None:
            lo, hi = self.gamma_cap
            gamma = jnp.clip(gamma, lo, hi)
        return gamma

    def __call__(
        self,
        t: jax.Array,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> NoisingState:
        gamma = self.gamma(t)
        sigma_sq = jax.nn.sigmoid(gamma)
        # Use autodiff so clipping (if present) zeros out β where γ is flat.
        _, dgamma_dt = jax.jvp(self.gamma, (t,), (jnp.ones_like(t),))
        beta = sigma_sq * dgamma_dt
        state = NoisingState(gamma=gamma, t=t, beta=beta)
        return state.broadcast_dims(shape=shape)


# =============================================================================
# Factory Functions
# =============================================================================


def linear_schedule(
    gamma_min: float | jax.Array = -13.3,
    gamma_max: float | jax.Array = 5.0,
    *,
    gamma_cap: tuple[float | jax.Array, float | jax.Array] | None = None,
    clip: bool = False,
) -> LinearSchedule:
    """Create linear γ schedule.

    Standard bounds: γ_min ≈ -13.3 (SNR ≈ 10⁶), γ_max ≈ 5 (SNR ≈ 0.007).
    Set clip=True to clamp γ to [gamma_min, gamma_max], or pass gamma_cap
    explicitly for asymmetric clipping.
    """
    cap = gamma_cap
    if clip and cap is None:
        cap = (jnp.asarray(gamma_min), jnp.asarray(gamma_max))
    return LinearSchedule(
        gamma_min=jnp.asarray(gamma_min),
        gamma_max=jnp.asarray(gamma_max),
        gamma_cap=cap,
    )


def vp_schedule(beta_min: float = 0.1, beta_max: float = 20.0) -> GammaSchedule:
    """Create variance-preserving schedule (DDPM-style).

    The noising rate β(t) = β_min + t·(β_max - β_min) is integrated to get γ.
    """

    def gamma_fn(t: jax.Array) -> jax.Array:
        # ∫₀ᵗ β(s)ds = β_min·t + ½(β_max - β_min)·t²
        beta_int = beta_min * t + 0.5 * (beta_max - beta_min) * t**2
        # σ² = 1 - exp(-∫β) = -expm1(-∫β)
        sigma_sq = -jnp.expm1(-beta_int)
        return jax.scipy.special.logit(sigma_sq)

    return GammaSchedule(gamma_fn)


def cosine_schedule(s: float = 0.008) -> GammaSchedule:
    """Create cosine schedule (improved DDPM).

    α(t) = cos((t + s)/(1 + s) · π/2)²

    The offset s prevents singularity at t=1.
    """

    def gamma_fn(t: jax.Array) -> jax.Array:
        alpha_sq = jnp.cos((t + s) / (1 + s) * jnp.pi / 2) ** 2
        sigma_sq = 1 - alpha_sq
        return jnp.log(sigma_sq) - jnp.log(alpha_sq)

    return GammaSchedule(gamma_fn)


def componentwise_linear_endpoints(
    labels: jax.Array,
    gamma_min_target: float = -7.0,
    gamma_max_target: float = 3.0,
    softness: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Compute per-component γ endpoints for soft-conditioning schedules.

    Implements the GUD paper's linear schedule. All components
    share the same slope Δγ but have offset starting points based on labels.

    The relationship to paper notation (with a = 1/softness):

        γ_min,i = γ̃_denoise + a(l_i - l_max)     [paper includes +log Σ_i]
        Δγ = γ̃_noise - γ̃_denoise + a(l_max - l_min)

    Boundary guarantees:
        - Component with l_i = l_max: γ(0) = gamma_min_target
        - Component with l_i = l_min: γ(1) = gamma_max_target

    To include data variance (as in the paper), encode it in labels:
        labels = ordering_labels + softness * log(variances)

    Args:
        labels: Ordering variables l_i, shape (data_dim,).
            Higher label = higher γ throughout = lower SNR = more noised.
            In reverse (generation), lower-label components are "generated first"
            (contain signal earlier) because they have higher SNR.
        gamma_min_target: γ at t=0 for highest-label component.
            Corresponds to γ̃_denoise in paper.
        gamma_max_target: γ at t=1 for lowest-label component.
            Corresponds to γ̃_noise in paper.
        softness: Overlap between components. Equals 1/a in paper notation.
            softness → ∞: standard diffusion (all same schedule)
            softness → 0: fully autoregressive (no overlap)

    Returns:
        (gamma_min, gamma_max): Per-component arrays of shape (data_dim,).

    Example (standard diffusion with variance-based ordering):
        >>> variances = jnp.array([1.0, 0.1, 0.01])  # PCA variances
        >>> labels = -jnp.log(variances)  # Higher variance = lower label
        >>> gamma_min, gamma_max = componentwise_linear_endpoints(
        ...     labels, gamma_min_target=-7, gamma_max_target=3, softness=1.0)
        >>> # This gives same SNR trajectory as standard diffusion
    """
    labels = jnp.asarray(labels)
    l_min, l_max = jnp.min(labels), jnp.max(labels)
    l_range = l_max - l_min

    # Offset from highest label component
    offset = (labels - l_max) / softness

    gamma_min = gamma_min_target + offset
    # All components share the same slope (Δγ)
    delta_gamma = gamma_max_target - gamma_min_target + l_range / softness
    gamma_max = gamma_min + delta_gamma

    return gamma_min, gamma_max


def sequential_offsets(
    labels: jax.Array,
    softness: float = 2.0,
) -> jax.Array:
    """Compute time offsets for sequential schedules.

    Maps labels to time offsets τ_i ∈ [0, τ_max] where larger softness means
    smaller τ_max (more overlap), and smaller softness means larger τ_max
    (more sequential).

    Args:
        labels: Ordering variables l_i, shape (data_dim,).
                Higher label = later noising (larger time offset).
        softness: Controls overlap. softness=1 is fully sequential,
                  larger values increase overlap.

    Returns:
        offsets: Time offsets τ_i of shape (data_dim,).
    """
    labels = jnp.asarray(labels)
    l_min, l_max = jnp.min(labels), jnp.max(labels)

    # Normalize labels to [0, 1]
    normalized = jnp.where(
        l_max > l_min,
        (labels - l_min) / (l_max - l_min),
        jnp.zeros_like(labels),
    )

    # Scale to [0, τ_max] where τ_max determines sequentiality:
    # softness ↑ → tau_max ↓ (more overlap), softness ↓ → tau_max ↑ (more sequential).
    tau_max = jnp.clip(1.0 / softness, 0.0, 0.99)
    return normalized * tau_max


def offsets_to_linear_schedule(
    offsets: jax.Array,
    gamma_min: float | jax.Array,
    gamma_max: float | jax.Array,
    normalize: bool = True,
) -> LinearSchedule:
    """Fold sequential offsets into a clipped LinearSchedule.

    Produces the exact γ(t) and β(t) that SequentialSchedule implemented:
    - components wait at γ_min until their offset
    - then rise with slope Δγ / (1 - τ_range)
    - and clip at γ_max.
    """
    offsets = jnp.asarray(offsets)
    gamma_min = jnp.asarray(gamma_min)
    gamma_max = jnp.asarray(gamma_max)

    if normalize:
        offsets = offsets - jnp.min(offsets)

    offset_range = jnp.max(offsets)
    active_time = jnp.maximum(1.0 - offset_range, 1e-6)
    delta_gamma = gamma_max - gamma_min
    slope = delta_gamma / active_time

    gamma_min_i = gamma_min - offsets * slope
    gamma_max_i = gamma_min_i + slope
    cap = (gamma_min, gamma_max)
    return LinearSchedule(gamma_min=gamma_min_i, gamma_max=gamma_max_i, gamma_cap=cap)


def soft_conditioning_schedule(
    labels: jax.Array,
    gamma_min_target: float = -7.0,
    gamma_max_target: float = 3.0,
    softness: float = 1.0,
    *,
    gamma_cap: tuple[float | jax.Array, float | jax.Array] | None = None,
    clip: bool = False,
) -> LinearSchedule:
    """Create a soft-conditioning linear schedule from labels.

    Components with higher labels are noised later, creating a
    hierarchical generation process. The softness parameter controls
    how much overlap there is between components.

    Args:
        labels: Ordering variables, shape (data_dim,).
        gamma_min_target: Target log-SNR at t=0 for "first" components.
        gamma_max_target: Target log-SNR at t=1 for "last" components.
        softness: Overlap parameter (larger = more overlap = softer).
        gamma_cap: Optional asymmetric clipping bounds (low, high).
        clip: If True and gamma_cap is None, clip to [gamma_min, gamma_max].

    Returns:
        LinearSchedule with per-component endpoints.
    """
    gamma_min, gamma_max = componentwise_linear_endpoints(
        labels, gamma_min_target, gamma_max_target, softness
    )
    cap = gamma_cap
    if clip:
        # Clip to the intended target bounds, not the expanded endpoints.
        cap = (jnp.asarray(gamma_min_target), jnp.asarray(gamma_max_target)) if cap is None else cap
    return LinearSchedule(gamma_min, gamma_max, cap)


def variance_adjusted_schedule(
    variances: jax.Array,
    gamma_min_snr: float = -7.0,
    gamma_max_snr: float = 3.0,
    softness: float = 1.0,
    ordering: jax.Array | None = None,
    *,
    gamma_cap: tuple[float | jax.Array, float | jax.Array] | None = None,
    clip: bool = False,
) -> LinearSchedule:
    """Create schedule adjusted for per-component data variance.

    Implements GUD paper. Large-variance components maintain higher
    SNR throughout diffusion and are thus "generated first" in the reverse process.
    The softness parameter controls how separated the SNR trajectories are.

    The per-component γ endpoints are:
        γ_min,i = γ̃_denoise + log Σ_i + (l_i - l_max) / softness
        γ_max,i = γ_min,i + Δγ

    where Δγ is shared across all components.

    Args:
        variances: Per-component data variances Σ_i, shape (data_dim,).
        gamma_min_snr: Target log-SNR at t=0 (γ̃_denoise in paper).
        gamma_max_snr: Target log-SNR at t=1 (γ̃_noise in paper).
        softness: Controls SNR trajectory overlap. Equals 1/a in paper notation.
                  softness=1: Standard diffusion (same γ for all, SNR differs by Σ_i)
                  softness<1: More separated (larger Σ has even higher SNR)
                  softness>1: Less separated (SNR trajectories more similar)
        ordering: Ordering variables l_i. If None, uses -log(variances),
                  which at softness=1 gives standard diffusion behavior.
        gamma_cap: Optional clipping bounds (low, high) in γ space.
        clip: If True and gamma_cap is None, clip to [gamma_min, gamma_max].

    Returns:
        LinearSchedule with variance-adjusted per-component endpoints.

    Example:
        >>> variances = compute_pca_variances(data)
        >>> # Standard diffusion behavior
        >>> sched = variance_adjusted_schedule(variances, softness=1.0)
        >>> # More hierarchical (large-variance components generated earlier)
        >>> sched = variance_adjusted_schedule(variances, softness=0.5)
    """
    variances = jnp.asarray(variances)
    log_var = jnp.log(variances)

    if ordering is None:
        # Default: order by variance (larger variance = lower ordering = noised first)
        ordering = -log_var

    # Combine variance adjustment with ordering-based offsets
    # This matches paper eq. 13: γ_min,i = γ̃_denoise + log Σ_i + a(l_i - l_max)
    labels = ordering / softness + log_var

    gamma_min, gamma_max = componentwise_linear_endpoints(
        labels,
        gamma_min_target=gamma_min_snr,
        gamma_max_target=gamma_max_snr,
        softness=1.0,  # Already incorporated into labels
    )
    cap = gamma_cap
    if clip and cap is None:
        cap = (gamma_min, gamma_max)
    return LinearSchedule(gamma_min, gamma_max, cap)


def column_schedule(
    n_rows: int,
    n_cols: int,
    gamma_min: float = -13.0,
    gamma_max: float = 5.0,
    softness: float = 2.0,
) -> LinearSchedule:
    """Create column-wise sequential schedule for 2D spatial data.

    For generating images column-by-column (left to right).
    All pixels in the same column share the same offset.
    Column 0 is generated first, column n_cols-1 last.

    Assumes row-major (C-order) flattening: pixel (row, col) → row * n_cols + col.
    This matches the standard numpy/jax flatten order.

    Args:
        n_rows: Number of rows in the spatial grid.
        n_cols: Number of columns in the spatial grid.
        gamma_min: Lower γ bound.
        gamma_max: Upper γ bound.
        softness: Controls overlap between columns.

    Returns:
        LinearSchedule equivalent to the sequential-offset construction,
        with offsets of shape (n_rows * n_cols,) folded into γ endpoints.

    Example:
        >>> # For 32x32 images flattened to (batch, 1024, channels)
        >>> sched = column_schedule(n_rows=32, n_cols=32, softness=2.0)
        >>> assert sched.offsets.shape == (1024,)
    """
    # Create per-column offsets
    col_labels = jnp.arange(n_cols, dtype=jnp.float32)
    col_offsets = sequential_offsets(col_labels, softness)

    # Tile to full spatial size: each row repeats the column pattern
    # Row-major: [col_0, col_1, ..., col_{n-1}] repeated n_rows times
    offsets = jnp.tile(col_offsets, n_rows)

    return offsets_to_linear_schedule(
        offsets=offsets,
        gamma_min=jnp.asarray(gamma_min),
        gamma_max=jnp.asarray(gamma_max),
    )


def row_schedule(
    n_rows: int,
    n_cols: int,
    gamma_min: float = -13.0,
    gamma_max: float = 5.0,
    softness: float = 2.0,
) -> LinearSchedule:
    """Create row-wise sequential schedule for 2D spatial data.

    For generating images row-by-row (top to bottom).
    All pixels in the same row share the same offset.

    Args:
        n_rows: Number of rows in the spatial grid.
        n_cols: Number of columns in the spatial grid.
        gamma_min: Lower γ bound.
        gamma_max: Upper γ bound.
        softness: Controls overlap between rows.

    Returns:
        LinearSchedule equivalent to the sequential-offset construction,
        with offsets of shape (n_rows * n_cols,) folded into γ endpoints.
    """
    # Create per-row offsets
    row_labels = jnp.arange(n_rows, dtype=jnp.float32)
    row_offsets = sequential_offsets(row_labels, softness)

    # Repeat each row offset for all columns in that row
    # Row-major: [row_0]*n_cols, [row_1]*n_cols, ...
    offsets = jnp.repeat(row_offsets, n_cols)

    return offsets_to_linear_schedule(
        offsets=offsets,
        gamma_min=jnp.asarray(gamma_min),
        gamma_max=jnp.asarray(gamma_max),
    )
