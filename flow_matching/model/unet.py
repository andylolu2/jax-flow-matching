from typing import Collection

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, DTypeLike, Float, Key, PRNGKeyArray
from typing_extensions import Self

from flow_matching.model.base import Model, ModelConfig


class UNetConfig(ModelConfig):
    dim_init: int
    kernel_size: int
    dim_mults: list[int]

    attention_resolutions: Collection[int]
    attention_num_heads: int
    num_res_blocks: int

    time_embed_dim: int

    num_groups: int
    dropout: float

    dtype: DTypeLike


def _optional_split(
    rng: Key[ArrayLike, ""] | None, n: int = 2
) -> Key[Array, "{n}"] | tuple[None, ...]:
    return jax.random.split(rng, n) if rng is not None else (None,) * n


class UpSample(nn.Module):
    dim: int
    config: UNetConfig

    @nn.compact
    def __call__(self, x: Float[ArrayLike, "h w c"]) -> Float[Array, "{2*h} {2*w} c"]:
        return nn.ConvTranspose(
            features=self.dim,
            kernel_size=(self.config.kernel_size, self.config.kernel_size),
            strides=(2, 2),
            dtype=self.config.dtype,
        )(x)


class DownSample(nn.Module):
    dim: int
    config: UNetConfig

    @nn.compact
    def __call__(self, x: Float[ArrayLike, "h w c"]) -> Float[Array, "{h//2} {w//2} c"]:
        return nn.Conv(
            self.dim,
            (self.config.kernel_size, self.config.kernel_size),
            strides=(2, 2),
            dtype=self.config.dtype,
        )(x)


class SinusoidalPosEmbedding(nn.Module):
    config: UNetConfig

    @nn.compact
    def __call__(self, t: Float[ArrayLike, ""]) -> Float[Array, "d"]:
        """Refer to https://arxiv.org/pdf/1706.03762.pdf#subsection.3.5"""
        assert (
            self.config.dim_init % 2 == 0
        ), f"Dim must be even, got {self.config.dim_init}"

        d_model = self.config.dim_init // 2
        freqs = t * jnp.exp(-(2 * jnp.arange(d_model) / d_model) * jnp.log(10000))
        emb = jnp.concatenate((jnp.sin(freqs), jnp.cos(freqs)))

        return emb


class TimeEmbedding(nn.Module):
    config: UNetConfig

    @nn.compact
    def __call__(
        self, t: Float[ArrayLike, ""]
    ) -> Float[Array, "{self.config.time_embed_dim}"]:
        x = SinusoidalPosEmbedding(self.config)(t)
        x = nn.Dense(self.config.time_embed_dim, dtype=self.config.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.config.time_embed_dim, dtype=self.config.dtype)(x)
        return x


class ResnetBlock(nn.Module):
    dim: int
    config: UNetConfig

    @nn.compact
    def __call__(
        self,
        x: Float[ArrayLike, "h w c"],
        time_emb: Float[ArrayLike, "c_"],
        train: bool,
        rng: PRNGKeyArray | None,
    ) -> Float[Array, "h w {self.dim}"]:
        h = x
        h = nn.GroupNorm(self.config.num_groups)(h)
        h = nn.silu(h)
        h = nn.Conv(self.dim, self.config.kernel_size, dtype=self.config.dtype)(h)

        h += nn.Dense(self.dim, dtype=self.config.dtype)(nn.silu(time_emb))

        h = nn.GroupNorm(self.config.num_groups)(h)
        h = nn.silu(h)
        h = nn.Dropout(rate=self.config.dropout)(h, deterministic=not train, rng=rng)
        h = nn.Conv(
            self.dim,
            self.config.kernel_size,
            dtype=self.config.dtype,
            kernel_init=nn.zeros_init(),
        )(h)

        if jnp.shape(x)[-1] != self.dim:
            x = nn.Conv(self.dim, kernel_size=(1, 1), dtype=self.config.dtype)(x)

        return h + x


class ResidualAttentionBlock(nn.Module):
    dim: int
    config: UNetConfig

    @nn.compact
    def __call__(self, x: Float[ArrayLike, "h w c"]) -> Float[Array, "h w c"]:
        h, w, c = jnp.shape(x)
        res = nn.GroupNorm(self.config.num_groups)(x)
        res = jnp.reshape(res, (h * w, c))
        res = nn.MultiHeadDotProductAttention(
            self.config.attention_num_heads,
            self.config.dtype,
            qkv_features=self.dim,
            out_features=c,
            out_kernel_init=nn.zeros_init(),
        )(res)
        res = jnp.reshape(res, (h, w, c))
        return res + x


class UNet(Model):
    config: UNetConfig

    @classmethod
    def create(cls, config: UNetConfig) -> Self:
        return cls(config=config)

    @nn.compact
    def forward(
        self,
        x: Float[ArrayLike, "h w c"],
        t: Float[ArrayLike, ""],
        train: bool,
        rng: PRNGKeyArray | None = None,
    ) -> Float[Array, "h w c"]:
        channels = jnp.shape(x)[-1]

        time_emb = TimeEmbedding(self.config)(t)
        x = nn.Conv(
            self.config.dim_init, self.config.kernel_size, dtype=self.config.dtype
        )(x)

        hs = [x]
        # downsample
        for i, dim_mult in enumerate(self.config.dim_mults):
            is_last = i == len(self.config.dim_mults) - 1
            dim = self.config.dim_init * dim_mult

            for _ in range(self.config.num_res_blocks):
                rng, rng_ = _optional_split(rng)
                x = ResnetBlock(dim, self.config)(x, time_emb, train, rng_)

                # apply attention at certain levels of resolutions
                if jnp.shape(x)[0] in self.config.attention_resolutions:
                    x = ResidualAttentionBlock(dim, self.config)(x)

                hs.append(x)

            if not is_last:
                x = DownSample(dim, self.config)(x)
                hs.append(x)

        # middle
        dim_mid = self.config.dim_init * self.config.dim_mults[-1]
        rng, rng_ = _optional_split(rng)
        x = ResnetBlock(dim_mid, self.config)(x, time_emb, train, rng_)
        x = ResidualAttentionBlock(dim_mid, self.config)(x)
        rng, rng_ = _optional_split(rng)
        x = ResnetBlock(dim_mid, self.config)(x, time_emb, train, rng_)

        # upsample
        for i, dim_mult in enumerate(reversed(self.config.dim_mults)):
            is_last = i == len(self.config.dim_mults) - 1
            dim = self.config.dim_init * dim_mult

            for _ in range(self.config.num_res_blocks + 1):
                # concatenate by last (channel) dimension
                x = jnp.concatenate((x, hs.pop()), axis=-1)
                rng, rng_ = _optional_split(rng)
                x = ResnetBlock(dim, self.config)(x, time_emb, train, rng_)

                # apply attention at certain levels of resolutions
                if jnp.shape(x)[0] in self.config.attention_resolutions:
                    x = ResidualAttentionBlock(dim, self.config)(x)

            if not is_last:
                x = UpSample(dim, self.config)(x)

        assert len(hs) == 0, "Not all hidden states are used"

        # final
        x = nn.GroupNorm(self.config.num_groups)(x)
        x = nn.silu(x)
        return nn.Conv(
            channels,
            self.config.kernel_size,
            dtype=self.config.dtype,
            kernel_init=nn.zeros_init(),
        )(x)
