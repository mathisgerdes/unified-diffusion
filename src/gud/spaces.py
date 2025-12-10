"""Spaces for diagonalizing diffusion processes.

A Space transforms data to a basis where the OU process is diagonal.
This is achieved via:
1. Orthogonal/unitary transform (e.g., PCA, FFT)
2. Diagonal rescaling (whitening by component std)

Usage
-----
Wrap tensors in `Data(space=..., data=...)` (or any single representation).
Access whichever attribute you need (`data`, `comp`, `comp_white`, `data_white`); lazy conversion happens automatically, so you rarely call `Space` methods directly.

Space hierarchy
---------------
Four spaces are involved:

    Data x (real)
        ↓ to_component_space (e.g., FFT)
    Components (may be complex)
        ↓ whiten (divide by std)
    Whitened components (may be complex) ← diffusion math operates here
        ↓ from_component_space (e.g., IFFT)
    Whitened data (real) ← neural networks operate here

Shape convention
----------------
All tensors use shape (batch, space, channel), each a single integer.
Flatten spatial structure before entering the space, e.g., images (b, 28, 28, 3) become (b, 28*28, 3).

Key methods (when needed)
- encode(x): data → whitened components (for diffusion math)
- decode(w): whitened components → data
- from_component_space(w): whitened components → whitened data (for networks)
- to_component_space(y): whitened data → whitened components

For unitary transforms (FFT), networks output in "whitened data" space which
is real-valued, avoiding complex-valued network outputs.
"""

from enum import StrEnum

import jax
import jax.numpy as jnp
from flax import nnx


class SpaceType(StrEnum):
    DATA = "data"
    COMP = "comp"
    COMP_WHITE = "comp_white"
    DATA_WHITE = "data_white"

    @classmethod
    def coerce(cls, space: "SpaceKey") -> "SpaceType":
        """Canonicalize user-provided space identifiers to SpaceType."""
        if isinstance(space, cls):
            return space
        try:
            return cls(space)
        except ValueError:
            try:
                return cls[space]
            except KeyError as err:
                raise ValueError(f"Unknown space identifier: {space}") from err


SpaceKey = SpaceType | str


class Space(nnx.Module):
    """Transformation to a space where diffusion is diagonal.

    For standard diffusion, both transforms are identity.
    Override for PCA, Fourier, wavelet, etc.
    """

    def to_component_space(self, x: jax.Array) -> jax.Array:
        """Orthogonal/unitary transform. Preserves noise statistics."""
        return x

    def from_component_space(self, x: jax.Array) -> jax.Array:
        """Inverse of to_component_space."""
        return x

    def whiten(self, x: jax.Array) -> jax.Array:
        """Rescale components (divide by std)."""
        return x

    def unwhiten(self, x: jax.Array) -> jax.Array:
        """Inverse of whiten (multiply by std)."""
        return x

    def encode(self, x: jax.Array) -> jax.Array:
        """Transform + whiten: x → whitened components."""
        return self.whiten(self.to_component_space(x))

    def decode(self, x: jax.Array) -> jax.Array:
        """Unwhiten + inverse transform: whitened components → x."""
        return self.from_component_space(self.unwhiten(x))

    def transform_noise(self, noise: jax.Array) -> jax.Array:
        """Transform noise for unitary (non-orthogonal) cases.

        Override with `to_component_space(noise)` for unitary transforms
        where real noise needs special handling (e.g., FFT with real data).
        """
        return noise

    def wrap(
        self,
        *,
        data: jax.Array | None = None,
        comp: jax.Array | None = None,
        comp_white: jax.Array | None = None,
        data_white: jax.Array | None = None,
        is_covector: bool = False,
    ) -> "Data | Score":
        """Wrap data in appropriate lazy container.

        Args:
            data: Data in original space.
            comp: Data in component space.
            comp_white: Data in whitened component space.
            data_white: Data in whitened original space.
            is_score: If True, return Score; else return Data.

        Returns:
            Data or Score instance with provided representations.
        """
        kwargs = {
            "space": self,
            "data": data,
            "comp": comp,
            "comp_white": comp_white,
            "data_white": data_white,
        }

        if is_covector:
            return Score(**kwargs)
        else:
            return Data(**kwargs)


class _LazySpaceData(nnx.Pytree):
    """Shared lazy conversion logic over Space representations."""

    _ORDERED_ATTRS = ("data", "comp", "comp_white", "data_white")

    def __getattribute__(self, name: str, /):
        if name in {"data", "comp", "comp_white", "data_white"}:
            value = object.__getattribute__(self, name)
            if value is not None:
                return value

            self._materialize(name)
            return object.__getattribute__(self, name)

        return super().__getattribute__(name)

    def _any_data(self):
        """Return first available (non-None) data representation."""
        for attr in self._ORDERED_ATTRS:
            val = object.__getattribute__(self, attr)
            if val is not None:
                return val
        raise ValueError("No data representations available")

    @property
    def shape(self):
        """Shape of the first available representation."""
        return self._any_data().shape

    def __len__(self) -> int:
        """Length of the first available representation."""
        return len(self._any_data())

    @property
    def data_available(self):
        """List of available (non-None) data representations."""
        return [attr for attr in self._ORDERED_ATTRS
                if object.__getattribute__(self, attr) is not None]

    def _conversion_edges(self) -> dict[SpaceType, dict[SpaceType, callable]]:
        raise NotImplementedError

    def _conversion_path(
        self, from_space: SpaceKey, to_space: SpaceKey
    ) -> list[callable]:
        """Shortest conversion path as a list of functions."""
        from_space = SpaceType.coerce(from_space)
        to_space = SpaceType.coerce(to_space)
        if from_space == to_space:
            return []

        edges = self._conversion_edges()
        frontier: list[SpaceType] = [from_space]
        parents: dict[SpaceType, tuple[SpaceType, callable]] = {}

        while frontier:
            current = frontier.pop(0)
            for nxt, fn in edges.get(current, {}).items():
                if nxt in parents or nxt == from_space:
                    continue
                parents[nxt] = (current, fn)
                if nxt == to_space:
                    break
                frontier.append(nxt)
            if to_space in parents:
                break

        if to_space not in parents:
            raise ValueError(f"Unsupported conversion from {from_space} to {to_space}")

        path: list[callable] = []
        node = to_space
        while node != from_space:
            parent, fn = parents[node]
            path.append(fn)
            node = parent
        path.reverse()
        return path

    def _convert(self, x: jax.Array, from_space: SpaceKey, to_space: SpaceKey) -> jax.Array:
        """Convert data between spaces using the shortest available path."""
        for fn in self._conversion_path(from_space, to_space):
            x = fn(x)
        return x

    def _available_sources(self) -> list[tuple[str, jax.Array]]:
        """List available (attr, value) pairs in a deterministic order."""
        pairs = []
        for attr in self._ORDERED_ATTRS:
            val = object.__getattribute__(self, attr)
            if val is not None:
                pairs.append((attr, val))
        return pairs

    def _materialize(self, target_attr: str) -> None:
        """Fill missing representation by converting from any available one."""
        if target_attr not in self._ORDERED_ATTRS:
            raise AttributeError(target_attr)

        target_space = SpaceType.coerce(target_attr)
        sources = self._available_sources()
        if not sources:
            raise ValueError("No data available to materialize any space.")

        last_error: Exception | None = None
        for source_attr, value in sources:
            source_space = SpaceType.coerce(source_attr)
            try:
                converted = self._convert(value, source_space, target_space)
            except ValueError as err:
                last_error = err
                continue
            object.__setattr__(self, target_attr, converted)
            return

        error_msg = (
            f"Unable to convert any available spaces to {target_attr}. "
            f"Last error: {last_error}"
        )
        raise ValueError(error_msg)


@nnx.dataclass
class Data(_LazySpaceData):
    """Lazy data wrapper following primal transformations."""

    space: Space

    data: jax.Array | None = nnx.data(default=None)
    comp: jax.Array | None = nnx.data(default=None)
    comp_white: jax.Array | None = nnx.data(default=None)
    data_white: jax.Array | None = nnx.data(default=None)

    def _conversion_edges(self) -> dict[SpaceType, dict[SpaceType, callable]]:
        """Primitive conversions as a directed adjacency map for data."""
        s = self.space
        return {
            SpaceType.DATA: {
                SpaceType.COMP: s.to_component_space,
                SpaceType.COMP_WHITE: s.encode,
            },
            SpaceType.COMP: {
                SpaceType.DATA: s.from_component_space,
                SpaceType.COMP_WHITE: s.whiten,
            },
            SpaceType.COMP_WHITE: {
                SpaceType.COMP: s.unwhiten,
                SpaceType.DATA: s.decode,
                SpaceType.DATA_WHITE: s.from_component_space,
            },
            SpaceType.DATA_WHITE: {
                SpaceType.COMP_WHITE: s.to_component_space,
            },
        }


@nnx.dataclass
class Score(_LazySpaceData):
    """Lazy wrapper for Stein scores (dual transformations)."""

    space: Space

    data: jax.Array | None = nnx.data(default=None)
    comp: jax.Array | None = nnx.data(default=None)
    comp_white: jax.Array | None = nnx.data(default=None)
    data_white: jax.Array | None = nnx.data(default=None)

    def _conversion_edges(self) -> dict[SpaceType, dict[SpaceType, callable]]:
        """Adjacency for score transformations (Jacobian inverse transpose)."""
        s = self.space
        return {
            # data → comp: inverse-transpose of from_component_space; orthonormal means same as to_component_space
            SpaceType.DATA: {
                SpaceType.COMP: s.to_component_space,
                SpaceType.COMP_WHITE: lambda x: s.unwhiten(s.to_component_space(x)),
            },
            # comp → data: inverse-transpose of to_component_space; orthonormal means from_component_space
            SpaceType.COMP: {
                SpaceType.DATA: s.from_component_space,
                SpaceType.COMP_WHITE: s.unwhiten,
            },
            # comp_white → comp: invert scale; whiten divides by scale
            SpaceType.COMP_WHITE: {
                SpaceType.COMP: s.whiten,
                SpaceType.DATA_WHITE: s.from_component_space,
            },
            # data_white → comp_white: inverse-transpose of from_component_space; orthonormal means to_component_space
            SpaceType.DATA_WHITE: {
                SpaceType.COMP_WHITE: s.to_component_space,
            },
        }


class LinearSpace(Space):
    """PCA-like space with orthogonal components and diagonal variance."""

    def __init__(
        self,
        components: jax.Array | None = None,
        scale: jax.Array | None = None,
    ):
        """Initialize linear space.

        Args:
            components: Orthogonal matrix U where x_component = U @ x.
                Shape (data_dim, data_dim). None means identity.
            scale: Standard deviation per component for whitening.
                Shape (data_dim,). None means no whitening.
        """
        self.components = nnx.Variable(components) if components is not None else None
        self.scale = nnx.Variable(scale) if scale is not None else None

    @classmethod
    def from_covariance(cls, cov: jax.Array) -> "LinearSpace":
        """Create from covariance matrix via eigendecomposition."""
        eigvals, eigvecs = jnp.linalg.eigh(cov)
        return cls(components=eigvecs.T, scale=jnp.sqrt(eigvals))

    def to_component_space(self, x: jax.Array) -> jax.Array:
        if self.components is None:
            return x
        return jnp.einsum("...ic,ji->...jc", x, self.components.value)

    def from_component_space(self, x: jax.Array) -> jax.Array:
        if self.components is None:
            return x
        return jnp.einsum("...ic,ij->...jc", x, self.components.value)

    def whiten(self, x: jax.Array) -> jax.Array:
        if self.scale is None:
            return x
        return x / jnp.expand_dims(self.scale.value, -1)

    def unwhiten(self, x: jax.Array) -> jax.Array:
        if self.scale is None:
            return x
        return x * jnp.expand_dims(self.scale.value, -1)


class FourierSpace(Space):
    """Fourier space for stationary data.

    Transforms spatial data to frequency domain where stationary
    processes have diagonal covariance.
    """

    def __init__(
        self,
        space_shape: tuple[int, ...],
        scale: jax.Array | None = None,
    ):
        """Initialize Fourier space.

        Args:
            space_shape: Spatial dimensions, e.g., (32, 32) for 2D.
            scale: Power spectrum sqrt for whitening. Shape (prod(space_shape),).
        """
        self.space_shape = space_shape
        self.scale = nnx.Variable(scale) if scale is not None else None

    @property
    def space_dim(self) -> int:
        return len(self.space_shape)

    @property
    def _space_axes(self) -> tuple[int, ...]:
        """Axes for FFT (excluding batch and channel)."""
        return tuple(range(-2, -2 - self.space_dim, -1))[::-1]

    def _reshape_to_spatial(self, x: jax.Array) -> jax.Array:
        """(..., flat_space, channels) → (..., *space_shape, channels)."""
        return x.reshape(*x.shape[:-2], *self.space_shape, x.shape[-1])

    def _reshape_to_flat(self, x: jax.Array) -> jax.Array:
        """(..., *space_shape, channels) → (..., flat_space, channels)."""
        return x.reshape(*x.shape[: -self.space_dim - 1], -1, x.shape[-1])

    def to_component_space(self, x: jax.Array) -> jax.Array:
        x = self._reshape_to_spatial(x)
        x = jnp.fft.fftn(x, self.space_shape, self._space_axes, norm="ortho")
        return self._reshape_to_flat(x)

    def from_component_space(self, x: jax.Array) -> jax.Array:
        x = self._reshape_to_spatial(x)
        x = jnp.fft.ifftn(x, self.space_shape, self._space_axes, norm="ortho")
        return self._reshape_to_flat(x).real

    def whiten(self, x: jax.Array) -> jax.Array:
        if self.scale is None:
            return x
        return x / jnp.expand_dims(self.scale.value, -1)

    def unwhiten(self, x: jax.Array) -> jax.Array:
        if self.scale is None:
            return x
        return x * jnp.expand_dims(self.scale.value, -1)

    def transform_noise(self, noise: jax.Array) -> jax.Array:
        # For FFT with real data, transform noise to get correct complex structure
        return self.to_component_space(noise)
