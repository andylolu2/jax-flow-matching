from datetime import datetime
from pathlib import Path

from flow_matching.dataset.builder import ToyConfig
from flow_matching.model.builder import MLPConfig
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
    model=MLPConfig(
        dims=(32, 32),
    ),
    optimizer=AdamConfig(
        learning_rate=3e-3,
    ),
    train_dataset=ToyConfig(
        seed=0,
    ),
    val_dataset=ToyConfig(
        seed=1,
    ),
    log=LogConfig(
        steps=1000,
    ),
    save=SaveConfig(
        steps=1000,
    ),
    eval=EvalConfig(
        steps=1000,
        n_batches=1000,
        batch_size=128,
    ),
    generate=GenerateConfig(
        steps=1000,
        samples=1000,
    ),
    num_steps=5000,
    exp_dir=(Path("checkpoints") / datetime.now().strftime("%Y%m%d-%H%M%S"))
    .resolve()
    .absolute()
    .as_uri(),
    batch_size=64,
)

main(config)
