from datetime import datetime

from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    config.seed = 0
    config.num_steps = 5000
    config.log_steps = 1000
    config.save_steps = 1000
    config.eval = dict(
        steps=1000,
        n_batches=100,
        batch_size=64,
    )

    config.generate = dict(
        steps=1000,
        samples=1000,
    )

    config.checkpoint_dir = f"checkpoints/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config.batch_size = 64

    config.dataset = dict(
        name="toy",
        seed=0,
    )
    config.model = dict(
        name="mlp",
        dims=[16, 16],
    )
    config.optimizer = dict(
        name="adam",
        learning_rate=3e-3,
    )

    return config
