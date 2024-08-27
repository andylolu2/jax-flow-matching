from os import PathLike
from pathlib import Path

import flax.struct
import jax
import optax
from absl import app, flags, logging
from clu.metric_writers import create_default_writer, write_values
from clu.metrics import Collection, LastValue
from flax.training.train_state import TrainState as _TrainState
from jaxtyping import PRNGKeyArray
from ml_collections import config_flags

from flow_matching.dataset.base import Dataset
from flow_matching.dataset.toy import ToyDataset
from flow_matching.model.base import Model, ModelMetrics
from flow_matching.model.mlp import MLP

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration file.", short_name="c"
)


@flax.struct.dataclass
class TrainMetrics(Collection):
    step: LastValue.from_output("step")  # type: ignore
    epoch: LastValue.from_output("epoch")  # type: ignore


class TrainState(_TrainState):
    rng: PRNGKeyArray
    train_metrics: TrainMetrics
    model_metrics: ModelMetrics
    train_dataset: Dataset
    val_dataset: Dataset


def build_dataset(name: str, **kwargs) -> tuple[Dataset, Dataset]:
    if name == "toy":
        return ToyDataset.create(**kwargs), ToyDataset.create(**kwargs)

    raise ValueError(f"Unknown dataset: {name}")


def build_model(name: str, **kwargs) -> Model:
    if name == "mlp":
        return MLP(**kwargs)

    raise ValueError(f"Unknown model: {name}")


def build_optimizer(name: str, **kwargs) -> optax.GradientTransformation:
    if name == "adam":
        return optax.adam(**kwargs)

    raise ValueError(f"Unknown optimizer: {name}")


def train_step(train_state: TrainState, config) -> TrainState:
    x, train_dataset = train_state.train_dataset.sample(config.batch_size)
    rng, fwd_rng = jax.random.split(train_state.rng)
    grads, new_model_metrics = jax.grad(train_state.apply_fn, has_aux=True)(
        train_state.params, x, fwd_rng
    )
    train_state = train_state.apply_gradients(grads=grads)
    model_metrics = train_state.model_metrics.merge(new_model_metrics)
    new_train_metrics = TrainMetrics.single_from_model_output(
        step=train_state.step,
        epoch=train_state.train_dataset.epoch,
    )
    train_metrics = train_state.train_metrics.merge(new_train_metrics)
    return train_state.replace(
        model_metrics=model_metrics,
        train_metrics=train_metrics,
        train_dataset=train_dataset,
        rng=rng,
    )


def log_metrics(train_state: TrainState) -> None:
    writer = create_default_writer()
    for metric in (train_state.train_metrics, train_state.model_metrics):
        values = metric.compute()
        write_values(writer, step=int(train_state.step), metrics=values)


def save_checkpoint(train_state: TrainState, path: PathLike) -> None:
    logging.info("Saving checkpoint to %s", path)


def evaluate(train_state: TrainState, dataset: Dataset) -> None:
    logging.info("Evaluating on validation dataset")


def main(_):
    config = FLAGS.config
    logging.info("Config: %s", config)

    rng = jax.random.PRNGKey(config.seed)
    model_rng, fwd_rng, state_rng = jax.random.split(rng, 3)
    train_dataset, val_dataset = build_dataset(**config.dataset)
    model = build_model(**config.model)
    optimizer = build_optimizer(**config.optimizer)

    # Training loop.
    params = model.init(model_rng, train_dataset.sample(1)[0], fwd_rng)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        train_metrics=TrainMetrics.empty(),
        model_metrics=ModelMetrics.empty(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        rng=state_rng,
    )

    while (step := train_state.step) < config.num_steps:
        train_state = train_step(train_state, config)

        if step % config.log_steps == 0:
            log_metrics(train_state)
            train_state = train_state.replace(
                train_metrics=TrainMetrics.empty(), model_metrics=ModelMetrics.empty()
            )
        if step % config.save_steps == 0:
            save_checkpoint(train_state, Path(config.checkpoint_dir) / f"step_{step}")
        if step % config.eval_steps == 0:
            evaluate(train_state, val_dataset)


if __name__ == "__main__":
    app.run(main)
