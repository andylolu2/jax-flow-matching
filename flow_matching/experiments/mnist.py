from datetime import datetime
from pathlib import Path

import jax.numpy as jnp

from flow_matching.dataset.builder import MnistConfig
from flow_matching.model.builder import UNetConfig
from flow_matching.optimizer.builder import AdamConfig
from flow_matching.train import (
    EvalConfig,
    GenerateConfig,
    LogConfig,
    SaveConfig,
    TrainConfig,
    main,
)

config = TrainConfig(
    model=UNetConfig(
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
    ),
    optimizer=AdamConfig(
        learning_rate=1e-4,
    ),
    train_dataset=MnistConfig(
        seed=0,
        split="train",
    ),
    val_dataset=MnistConfig(
        seed=0,
        split="val",
    ),
    log=LogConfig(
        steps=500,
    ),
    save=SaveConfig(
        steps=50000,
    ),
    eval=EvalConfig(
        steps=5000,
        n_batches=1000,
        batch_size=128,
    ),
    generate=GenerateConfig(
        steps=5000,
        samples=100,
    ),
    seed=0,
    num_steps=500000,
    exp_dir=Path("checkpoints") / datetime.now().strftime("%Y%m%d-%H%M%S"),
    batch_size=64,
)

main(config)
