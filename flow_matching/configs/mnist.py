from datetime import datetime

from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    config.seed = 0
    config.num_steps = 5000
    config.log_steps = 500
    config.save_steps = 500
    config.eval_steps = 500

    config.generate = dict(
        steps=500,
        samples=1000,
    )

    config.checkpoint_dir = f"checkpoints/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config.batch_size = 64

    config.dataset = dict(
        name="mnist",
        seed=0,
    )
    config.model = dict(
        name="mlp",
        dims=[128, 128],
    )
    config.optimizer = dict(
        name="adam",
        learning_rate=1e-3,
    )

    return config_dict.FrozenConfigDict(config)
