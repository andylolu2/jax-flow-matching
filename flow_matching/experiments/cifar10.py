from datetime import datetime

import jax.numpy as jnp

from flow_matching.dataset.builder import Cifar10Config
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
        dim_init=128,
        kernel_size=3,
        dim_mults=(1, 2, 2, 2),
        attention_resolutions=(16,),
        attention_num_heads=4,
        num_res_blocks=2,
        time_embed_dim=128 * 4,
        num_groups=32,
        dropout=0.1,
        dtype=jnp.bfloat16,
    ),
    optimizer=AdamConfig(
        learning_rate=1e-4,
    ),
    train_dataset=Cifar10Config(
        seed=0,
        split="train",
    ),
    val_dataset=Cifar10Config(
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
    num_steps=500000,
    exp_dir=f"hf://andylolu24/flow-matching-cifar10/checkpoints/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    batch_size=64,
    num_compile_steps=10,
)

main(config)
