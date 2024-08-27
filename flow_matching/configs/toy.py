from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    config.seed = 0
    config.num_steps = 1000
    config.log_steps = 100
    config.save_steps = 200
    config.eval_steps = 200
    config.checkpoint_dir = "checkpoints"
    config.batch_size = 64

    config.dataset = dict(
        name="toy",
        seed=0,
    )
    config.model = dict(
        name="mlp",
        dims=[8, 8],
    )
    config.optimizer = dict(
        name="adam",
        learning_rate=1e-3,
    )

    return config
