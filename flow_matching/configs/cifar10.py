from datetime import datetime

import jax.numpy as jnp
from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    config.seed = 0
    config.num_steps = 50000
    config.log = dict(
        steps=500,
    )
    config.save = dict(
        steps=10000,
    )
    config.eval = dict(
        steps=2000,
        n_batches=500,
        batch_size=128,
    )
    config.generate = dict(
        steps=2000,
        samples=100,
    )

    config.checkpoint_dir = f"checkpoints/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config.batch_size = 64

    config.dataset = dict(
        name="cifar10",
        seed=0,
    )
    config.model = dict(
        name="unet",
        dim_init=128,
        kernel_size=3,
        dim_mults=[1, 2, 2, 2],
        attention_resolutions=[16],
        attention_num_heads=4,
        num_res_blocks=2,
        time_embed_dim=128 * 4,
        num_groups=4,
        dropout=0.1,
        dtype=jnp.bfloat16,
    )
    config.optimizer = dict(
        name="adam",
        learning_rate=2e-4,
    )

    return config_dict.FrozenConfigDict(config)
