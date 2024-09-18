from datetime import datetime

import jax.numpy as jnp
from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    config.seed = 0
    config.num_steps = 50000
    config.log_steps = 500
    config.save_steps = 1000
    config.eval = dict(
        steps=1000,
        n_batches=100,
        batch_size=64,
    )

    config.generate = dict(
        steps=1000,
        samples=100,
    )

    config.checkpoint_dir = f"checkpoints/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config.batch_size = 64

    config.dataset = dict(
        name="mnist",
        seed=0,
    )
    config.model = dict(
        name="unet",
        dim_init=32,
        kernel_size=3,
        dim_mults=[1, 2, 4, 8],
        attention_resolutions=[16],
        attention_num_heads=4,
        num_res_blocks=1,
        time_embed_dim=32 * 4,
        num_groups=4,
        dropout=0,
        dtype=jnp.bfloat16,
    )
    config.optimizer = dict(
        name="adam",
        learning_rate=1e-4,
    )

    return config_dict.FrozenConfigDict(config)
