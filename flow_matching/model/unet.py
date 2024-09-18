from functools import partial
from typing import Any, Collection, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from flow_matching.model.base import Model

Dtype = Any


class UpSample(nn.Module):
    dim: int
    kernel_size: int
    dtype: Dtype

    @nn.compact
    def __call__(self, x: Float[ArrayLike, "h w c"]) -> Float[Array, "{2*h} {2*w} c"]:
        # h, w, c = jnp.shape(x)
        # x_ = jax.image.resize(x, (2 * h, 2 * w, c), method="nearest")
        # x_ = nn.Conv(self.dim, self.kernel_size, dtype=self.dtype)(x_)
        x_ = nn.ConvTranspose(
            self.dim,
            (self.kernel_size, self.kernel_size),
            strides=(2, 2),
            dtype=self.dtype,
        )(x)
        return x_


class DownSample(nn.Module):
    dim: int
    kernel_size: int
    dtype: Dtype

    @nn.compact
    def __call__(self, x: Float[ArrayLike, "h w c"]) -> Float[Array, "{h//2} {w//2} c"]:
        # h, w, c = jnp.shape(x)
        # x_ = jax.image.resize(x, (h // 2, w // 2, c), method="linear")
        # x_ = nn.Conv(self.dim, self.kernel_size, dtype=self.dtype)(x_)
        x_ = nn.Conv(
            self.dim,
            (self.kernel_size, self.kernel_size),
            strides=(2, 2),
            dtype=self.dtype,
        )(x)
        return x_


class SinusoidalPosEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, t: Float[ArrayLike, ""]) -> Float[Array, "d"]:
        """Refer to https://arxiv.org/pdf/1706.03762.pdf#subsection.3.5"""
        assert self.dim % 2 == 0, f"Dim must be even, got {self.dim}"

        d_model = self.dim // 2
        freqs = t * jnp.exp(-(2 * jnp.arange(d_model) / d_model) * jnp.log(10000))
        emb = jnp.concatenate((jnp.sin(freqs), jnp.cos(freqs)))

        return emb


class TimeEmbedding(nn.Module):
    sinusoidal_dim: int
    time_dim: int
    dtype: Dtype

    @nn.compact
    def __call__(self, t: Float[ArrayLike, ""]) -> Float[Array, "{self.time_dim}"]:
        x = SinusoidalPosEmbedding(self.sinusoidal_dim)(t)
        x = nn.Dense(self.time_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.time_dim, dtype=self.dtype)(x)
        return x


class Block(nn.Module):
    dim: int
    kernel_size: int
    num_groups: int
    dropout: float
    dtype: Dtype

    @nn.compact
    def __call__(
        self,
        x: Float[ArrayLike, "h w c"],
        deterministic: bool,
        scale_shift: tuple[Float[ArrayLike, "c"], Float[ArrayLike, "c"]] | None = None,
    ) -> Float[Array, "h w c"]:
        x_ = nn.GroupNorm(self.num_groups, dtype=self.dtype)(x)
        x_ = nn.gelu(x_)
        x_ = nn.Dropout(rate=self.dropout)(x_, deterministic)
        x_ = nn.Conv(self.dim, self.kernel_size, dtype=self.dtype)(x_)

        if scale_shift is not None:
            scale, shift = scale_shift
            x_ = x_ * (1 + scale) + shift

        return x_


class ResnetBlock(nn.Module):
    dim: int
    kernel_size: int
    num_groups: int
    dropout: float
    dtype: Dtype

    @nn.compact
    def __call__(
        self,
        x: Float[ArrayLike, "h w c"],
        time_emb: Float[ArrayLike, "c_"],
        deterministic: bool,
    ) -> Float[Array, "h w {self.dim}"]:
        h = Block(self.dim, self.kernel_size, self.num_groups, 0.0, self.dtype)(
            x, deterministic=True
        )

        time_emb = nn.silu(time_emb)
        scale_shift = nn.Dense(self.dim * 2, dtype=self.dtype)(time_emb)
        scale, shift = jnp.split(scale_shift, 2, axis=-1)

        h = Block(
            self.dim, self.kernel_size, self.num_groups, self.dropout, self.dtype
        )(h, deterministic, scale_shift=(scale, shift))

        if jnp.shape(x)[-1] != self.dim:
            x = nn.Conv(self.dim, kernel_size=(1, 1))(x)

        return h + x


class ResidualAttentionBlock(nn.Module):
    dim: int
    num_heads: int
    num_groups: int
    dtype: Dtype

    @nn.compact
    def __call__(self, x: Float[ArrayLike, "h w c"]) -> Float[Array, "h w c"]:
        w, h, c = jnp.shape(x)
        res = nn.GroupNorm(self.num_groups, dtype=self.dtype)(x)
        res = jnp.reshape(res, (w * h, c))
        res = nn.MultiHeadDotProductAttention(
            self.num_heads, self.dtype, qkv_features=self.dim, out_features=c
        )(res)
        res = jnp.reshape(res, (w, h, c))
        return res + x


class UNet(Model):
    dim_init: int
    kernel_size: int
    dim_mults: Sequence[int]

    attention_resolutions: Collection[int]
    attention_num_heads: int
    num_res_blocks: int

    time_embed_dim: int

    num_groups: int
    dropout: float

    dtype: Dtype

    @nn.compact
    def forward(
        self, x: Float[ArrayLike, "h w c"], t: Float[ArrayLike, ""], train: bool = True
    ) -> Float[Array, "h w c"]:
        channels = jnp.shape(x)[-1]

        res = partial(
            ResnetBlock,
            kernel_size=self.kernel_size,
            num_groups=self.num_groups,
            dropout=self.dropout,
            dtype=self.dtype,
        )
        res_atten = partial(
            ResidualAttentionBlock,
            num_heads=self.attention_num_heads,
            num_groups=self.num_groups,
            dtype=self.dtype,
        )

        time_emb = TimeEmbedding(self.dim_init, self.time_embed_dim, self.dtype)(t)
        x = nn.Conv(self.dim_init, self.kernel_size, dtype=self.dtype)(x)

        hs = [x]
        # downsample
        for i, dim_mult in enumerate(self.dim_mults):
            is_last = i == len(self.dim_mults) - 1
            dim = self.dim_init * dim_mult

            for _ in range(self.num_res_blocks):
                x = res(dim)(x, time_emb, deterministic=not train)

                # apply attention at certain levels of resolutions
                if jnp.shape(x)[0] in self.attention_resolutions:
                    x = res_atten(dim)(x)

                hs.append(x)

            if not is_last:
                x = DownSample(dim, self.kernel_size, self.dtype)(x)
                hs.append(x)

        # middle
        dim_mid = self.dim_init * self.dim_mults[-1]
        x = res(dim_mid)(x, time_emb, deterministic=not train)
        x = res_atten(dim_mid)(x)
        x = res(dim_mid)(x, time_emb, deterministic=not train)

        # upsample
        for i, dim_mult in enumerate(reversed(self.dim_mults)):
            is_last = i == len(self.dim_mults) - 1
            dim = self.dim_init * dim_mult

            for _ in range(self.num_res_blocks + 1):
                # concatenate by last (channel) dimension
                x = jnp.concatenate((x, hs.pop()), axis=-1)
                x = res(dim)(x, time_emb, deterministic=not train)

                # apply attention at certain levels of resolutions
                if jnp.shape(x)[0] in self.attention_resolutions:
                    x = res_atten(dim)(x)

            if not is_last:
                x = UpSample(dim, self.kernel_size, self.dtype)(x)

        assert len(hs) == 0, "Not all hidden states are used"

        # final
        x = res(self.dim_init)(x, time_emb, deterministic=not train)
        return nn.Conv(channels, kernel_size=1, dtype=self.dtype)(x)
