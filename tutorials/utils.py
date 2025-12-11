from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx
from torchvision import datasets, transforms


def mnist_numpy(
    *,
    dataset: str = "mnist",  # "mnist" or "fashion-mnist"
    split: str = "train",
    data_dir: str | Path = "./datasets",
    dtype: np.dtype = np.float32,
    flatten: bool = True,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Download MNIST or Fashion-MNIST and return NumPy arrays.

    Args:
        dataset: Either "mnist" or "fashion-mnist"
        split: Either "train" or "test"
        data_dir: Directory to store data. Defaults to "./{dataset}"
        dtype: Data type for arrays
        flatten: If True, flatten images to (N, 784, 1)
        download: Whether to download if not present
    """
    ds_class = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST

    train = split == "train"
    root = Path(data_dir).expanduser()
    transform = transforms.ToTensor()
    ds = ds_class(
        root=str(root),
        train=train,
        download=download,
        transform=transform,
    )
    images: list[np.ndarray] = []
    labels: list[int] = []
    for img, label in ds:
        images.append(np.array(img, dtype=np.float32) * 2 - 1)  # scale to [-1, 1]
        labels.append(int(label))
    x = np.stack(images, axis=0)
    if not flatten:
        x = np.moveaxis(x, 1, -1)  # (N, 28, 28, 1)
    else:
        x = x.reshape(x.shape[0], -1, 1)
    y = np.array(labels, dtype=np.int64)
    return x.astype(dtype), y


class MNIST:
    def __init__(
        self, batch_size: int = 64,
        split: str = "train",
        dataset: str = "mnist",
        data_dir: str | Path = "./datasets",
        dtype: np.dtype = np.float32,
        flatten: bool = True,
        download: bool = True,
        shuffle: bool = True,
        rng: int = 0,
    ):
        self.batch_size = batch_size
        self.split = split
        self.dataset = dataset
        self.data_dir = data_dir
        self.dtype = dtype
        self.flatten = flatten
        self.download = download
        self.shuffle = shuffle
        self.rng = np.random.RandomState(rng)

        images, labels = mnist_numpy(
            dataset=dataset,
            split=split,
            dtype=dtype,
            flatten=flatten,
            download=download,
        )
        self.images = images
        self.labels = labels
        self.index = 0
        if self.shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(len(self.labels))
        self.rng.shuffle(indices)
        self.images = self.images[indices]
        self.labels = self.labels[indices]

    def next(self):
        n = len(self.images)

        if self.batch_size > n:
            msg = f"batch_size {self.batch_size} exceeds dataset size {n}."
            raise ValueError(msg)

        if self.index + self.batch_size > n:
            if self.shuffle:
                self._shuffle()
            self.index = 0

        start = self.index
        end = start + self.batch_size
        batch = (self.images[start:end], self.labels[start:end])

        self.index = end

        return batch


class SinusoidalTimeEmbedding(nnx.Module):
    """Sinusoidal time embeddings for diffusion models."""

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, t: jax.Array) -> jax.Array:
        """Create sinusoidal embeddings for time t.

        Args:
            t: Time values, shape (batch,) or scalar

        Returns:
            Time embeddings, shape (batch, dim) or (dim,)
        """
        if t.ndim == 0:
            t = t[None]  # Handle scalar time

        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

        if self.dim % 2 == 1:
            emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)

        return emb.squeeze(0) if t.ndim == 0 else emb


class UNet(nnx.Module):
    """Simple U-Net architecture for epsilon prediction in diffusion models.

    A streamlined U-Net with 3 levels, residual blocks, and self-attention
    optimized for Fashion MNIST diffusion models.
    """

    def __init__(
        self,
        space_shape: tuple[int, ...],
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        channel_multipliers: list[int] = nnx.static(default_factory=lambda: [1, 2, 4]),
        dropout: float = 0.05,
        time_embed_dim: int = 64,
        attention_heads: int | None = 4,
        num_res_blocks: int = 1,
        rngs: nnx.Rngs,
    ):
        """Initialize U-Net.

        Args:
            space_shape: Spatial dimensions of input (H, W)
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_channels: Base number of channels
            channel_multipliers: List of channel multipliers for each level
            dropout: Dropout probability
            time_embed_dim: Dimension for time embeddings
            attention_heads: Number of attention heads (None to disable)
            num_res_blocks: Number of residual blocks per level
            rngs: Random number generators
        """
        self.space_shape = space_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_multipliers = channel_multipliers

        # Time embedding
        self.time_embed = nnx.Sequential(
            SinusoidalTimeEmbedding(dim=time_embed_dim),
            nnx.Linear(time_embed_dim, time_embed_dim, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(time_embed_dim, time_embed_dim, rngs=rngs),
        )

        # Input conv
        self.input_conv = nnx.Conv(
            in_features=in_channels, out_features=base_channels,
            kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)),
            rngs=rngs
        )

        # Encoder
        self.enc_conv1 = nnx.Conv(
            in_features=base_channels, out_features=base_channels,
            kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)),
            rngs=rngs
        )

        # Create encoder levels dynamically
        self.enc_res_blocks = nnx.List()
        self.enc_down_convs = nnx.List()

        for i, multiplier in enumerate(self.channel_multipliers):
            in_ch = base_channels * multiplier
            out_ch = base_channels * self.channel_multipliers[i + 1] if i < len(self.channel_multipliers) - 1 else in_ch

            # Residual blocks for this level
            self.enc_res_blocks.append(nnx.List([
                ResBlock(in_ch, in_ch, time_embed_dim, dropout, rngs=rngs)
                for _ in range(num_res_blocks)
            ]))

            # Downsampling conv (except for last level)
            if i < len(self.channel_multipliers) - 1:
                self.enc_down_convs.append(nnx.Conv(
                    in_features=in_ch, out_features=out_ch,
                    kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)),
                    rngs=rngs
                ))

        # Attention block for bottleneck level
        bottleneck_channels = base_channels * self.channel_multipliers[-1]
        if attention_heads is not None and attention_heads > 0:
            self.enc_attn = AttentionBlock(bottleneck_channels, attention_heads, rngs=rngs)
        else:
            self.enc_attn = None

        # Middle (bottleneck)
        bottleneck_channels = base_channels * self.channel_multipliers[-1]
        self.middle_res = ResBlock(bottleneck_channels, bottleneck_channels, time_embed_dim, dropout, rngs=rngs)

        # Decoder
        self.dec_up_convs = nnx.List()
        self.dec_res_blocks = nnx.List()

        # Create decoder levels in reverse order (from bottleneck back to base)
        reversed_multipliers = list(reversed(self.channel_multipliers))

        for i in range(len(reversed_multipliers) - 1):
            current_ch = base_channels * reversed_multipliers[i]
            next_ch = base_channels * reversed_multipliers[i + 1]

            # Upsampling conv reduces channels
            self.dec_up_convs.append(nnx.Conv(
                in_features=current_ch, out_features=next_ch,
                kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)),
                rngs=rngs
            ))

            # Residual blocks: first one takes concatenated input, subsequent ones preserve channels
            res_blocks = []
            for j in range(num_res_blocks):
                if j == 0:
                    # First ResBlock in level: takes concatenated input
                    res_blocks.append(ResBlock(2 * next_ch, next_ch, time_embed_dim, dropout, rngs=rngs))
                else:
                    # Subsequent ResBlocks: preserve channel count
                    res_blocks.append(ResBlock(next_ch, next_ch, time_embed_dim, dropout, rngs=rngs))
            self.dec_res_blocks.append(nnx.List(res_blocks))

        # Final output level (no upsampling, just residual blocks)
        final_channels = base_channels * reversed_multipliers[-1]
        self.dec_res_blocks.append(nnx.List([
            ResBlock(final_channels, final_channels, time_embed_dim, dropout, rngs=rngs)
            for _ in range(num_res_blocks)
        ]))

        # Output
        self.output_norm = nnx.LayerNorm(base_channels, rngs=rngs)
        self.output_conv = nnx.Conv(
            in_features=base_channels, out_features=out_channels,
            kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)),
            kernel_init=nnx.initializers.normal(stddev=1e-4),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs
        )

    def __call__(self, t, x):
        """Predict epsilon from noised data and time.

        Args:
            t: Time values, shape (batch,) or scalar
            x: Noised data with shape (batch, data_dim, channels)

        Returns:
            Predicted epsilon with same shape as x
        """
        batch_size, data_dim, channels = x.shape
        assert channels == self.in_channels, f"Expected {self.in_channels} channels, got {channels}"

        # Reshape to spatial layout: (batch, H, W, channels)
        h, w = self.space_shape
        x_img = rearrange(x, 'b (h w) c -> b h w c', h=h, w=w)

        # Time embedding
        t_array = jnp.asarray(t)
        t_embed = self.time_embed(t_array)

        # Encoder
        h = jax.nn.silu(self.input_conv(x_img))
        h = jax.nn.silu(self.enc_conv1(h))

        skip_connections = []
        for i, res_blocks in enumerate(self.enc_res_blocks):
            # Apply residual blocks
            for res_block in res_blocks:
                h = res_block(h, t_embed)

            # Store skip connection
            skip_connections.append(h)

            # Downsample (except for last level)
            if i < len(self.enc_down_convs):
                h = jax.nn.silu(self.enc_down_convs[i](h))

        # Apply attention at bottleneck if enabled
        if self.enc_attn is not None:
            h = self.enc_attn(h)

        # Middle
        h = self.middle_res(h, t_embed)

        # Decoder
        skip_connections = list(reversed(skip_connections))  # Use skips in reverse order

        for i in range(len(self.dec_up_convs)):
            # Calculate target spatial dimensions
            # Each upsampling doubles the spatial dimensions
            target_h = h.shape[1] * 2
            target_w = h.shape[2] * 2

            # Resize to target dimensions
            h = jax.image.resize(h, (batch_size, target_h, target_w, h.shape[3]), method='bilinear')
            h = jax.nn.silu(self.dec_up_convs[i](h))

            # Concatenate with corresponding skip connection (skip bottleneck level)
            h = jnp.concatenate([h, skip_connections[i + 1]], axis=-1)

            # Apply residual blocks
            for res_block in self.dec_res_blocks[i]:
                h = res_block(h, t_embed)

        # Final residual blocks (no upsampling)
        for res_block in self.dec_res_blocks[-1]:
            h = res_block(h, t_embed)

        # Output
        h = jax.nn.silu(self.output_norm(h))
        h = self.output_conv(h)

        # Flatten back to original shape
        return rearrange(h, 'b h w c -> b (h w) c')


class ResBlock(nnx.Module):
    """Residual block with time conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time conditioning
        self.time_proj = nnx.Linear(time_embed_dim, out_channels, rngs=rngs)

        # Main convolution path
        self.conv1 = nnx.Conv(
            in_channels, out_channels, kernel_size=(3, 3),
            strides=(1, 1), padding=((1, 1), (1, 1)), rngs=rngs
        )
        self.conv2 = nnx.Conv(
            out_channels, out_channels, kernel_size=(3, 3),
            strides=(1, 1), padding=((1, 1), (1, 1)), rngs=rngs
        )

        # Normalization
        self.norm1 = nnx.LayerNorm(out_channels, rngs=rngs)
        self.norm2 = nnx.LayerNorm(out_channels, rngs=rngs)

        # Dropout
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nnx.Conv(
                in_channels, out_channels, kernel_size=(1, 1),
                strides=(1, 1), padding=((0, 0), (0, 0)), rngs=rngs
            )
        else:
            self.skip_conv = None

    def __call__(self, x, t_embed):
        """Apply residual block with time conditioning."""
        h = x

        # First convolution
        h = self.conv1(h)
        h = self.norm1(h)
        h = jax.nn.silu(h + self.time_proj(t_embed)[:, None, None, :])

        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)

        # Skip connection
        if self.skip_conv is not None:
            x = self.skip_conv(x)

        return x + h


class AttentionBlock(nnx.Module):
    """Self-attention block."""

    def __init__(self, channels: int, num_heads: int = 4, rngs: nnx.Rngs = None):
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads

        self.norm = nnx.LayerNorm(channels, rngs=rngs)

        # QKV projection
        self.qkv_proj = nnx.Linear(channels, 3 * channels, rngs=rngs)
        self.out_proj = nnx.Linear(channels, channels, rngs=rngs)

    def __call__(self, x):
        """Apply self-attention."""
        b, h, w, c = x.shape
        assert c == self.channels

        # Normalization
        x_norm = self.norm(x)

        # Flatten spatial dimensions for attention
        x_flat = x_norm.reshape(b, h * w, c)

        # QKV
        qkv = self.qkv_proj(x_flat)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention
        q = q.reshape(b, h * w, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, h * w, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, h * w, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention
        scale = self.head_dim ** -0.5
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn = jax.nn.softmax(attn, axis=-1)

        # Apply attention to values
        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, h * w, c)

        # Output projection
        out = self.out_proj(out)
        out = out.reshape(b, h, w, c)

        return x + out
