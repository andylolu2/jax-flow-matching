import math
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any, Callable

import flax.struct
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import orbax.checkpoint
from absl import app, flags, logging
from clu.metric_writers import create_default_writer, write_values
from clu.metrics import Average, Collection, LastValue
from diffrax import Dopri5, ODETerm, diffeqsolve
from flax.training.train_state import TrainState as _TrainState
from jaxtyping import Array, Float, PRNGKeyArray
from ml_collections import FrozenConfigDict, config_flags

from flow_matching.dataset.base import Dataset
from flow_matching.dataset.mnist import MnistDataset
from flow_matching.dataset.toy import ToyDataset
from flow_matching.model.base import Model, ModelMetrics
from flow_matching.model.cnn import CNN
from flow_matching.model.mlp import MLP
from flow_matching.model.unet import UNet

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration file.", short_name="c", lock_config=False
)


@flax.struct.dataclass
class TrainMetrics(Collection):
    step: LastValue.from_output("step")  # type: ignore
    epoch: LastValue.from_output("epoch")  # type: ignore


class TrainState(_TrainState):
    forward_fn: Callable = flax.struct.field(pytree_node=False)
    rng: PRNGKeyArray
    train_metrics: TrainMetrics
    model_metrics: ModelMetrics
    train_dataset: Dataset
    val_dataset: Dataset


def build_dataset(name: str, **kwargs) -> tuple[Dataset, Dataset]:
    if name == "toy":
        return (
            ToyDataset.create(**kwargs),
            ToyDataset.create(**kwargs),
        )
    elif name == "mnist":
        return (
            MnistDataset.create(**kwargs, split="train"),
            MnistDataset.create(**kwargs, split="val"),
        )

    raise ValueError(f"Unknown dataset: {name}")


def build_model(name: str, **kwargs) -> Model:
    if name == "mlp":
        return MLP(**kwargs)
    elif name == "cnn":
        return CNN(**kwargs)
    elif name == "unet":
        return UNet(**kwargs)

    raise ValueError(f"Unknown model: {name}")


def build_optimizer(name: str, **kwargs) -> optax.GradientTransformation:
    if name == "adam":
        return optax.adam(**kwargs)

    raise ValueError(f"Unknown optimizer: {name}")


@partial(jax.jit, static_argnames=("config",))
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
    checkpointer = orbax.checkpoint.StandardCheckpointer()
    checkpointer.save(Path(path).absolute().resolve(), train_state)
    checkpointer.wait_until_finished()


@partial(jax.jit, static_argnames=("batch_size",))
def _eval_step(
    train_state: TrainState,
    dataset: Dataset,
    batch_size: int,
    model_metrics: ModelMetrics,
):
    x, dataset = dataset.sample(batch_size)
    _, new_model_metrics = train_state.apply_fn(train_state.params, x, train_state.rng)
    model_metrics = model_metrics.merge(new_model_metrics)
    return model_metrics, dataset


def evaluate(
    train_state: TrainState, dataset: Dataset, n_batches: int, batch_size: int
) -> None:
    logging.info("Evaluating on validation dataset")

    model_metrics = ModelMetrics.empty()

    for _ in range(n_batches):
        model_metrics, dataset = _eval_step(
            train_state, dataset, batch_size, model_metrics
        )

    model_metrics = model_metrics.compute()
    logging.info("Validation metrics: %s", model_metrics)


@partial(jax.jit, static_argnames=("n",))
def _generate_samples(train_state: TrainState, n: int) -> Float[Array, "{n} ..."]:
    _x, _ = train_state.train_dataset.sample(1)
    size = math.prod(jnp.shape(_x)[1:])
    x0 = jax.random.multivariate_normal(
        train_state.rng, jnp.zeros(size), jnp.eye(size), shape=(n,)
    )
    x0 = jnp.reshape(x0, (n,) + jnp.shape(_x)[1:])

    forward_fn = lambda t, x, args: train_state.apply_fn(
        train_state.params, x, t, method=train_state.forward_fn
    )
    term = ODETerm(jax.vmap(forward_fn, in_axes=(None, 0, None)))
    solver = Dopri5()
    solution = diffeqsolve(term, solver, t0=0, t1=0.995, dt0=0.05, y0=x0)

    x1 = solution.ys[-1]
    return x1


def generate_samples(train_state: TrainState, n: int, save_path: PathLike) -> None:
    logging.info("Generating samples from model")

    x1 = _generate_samples(train_state, n)

    Path(save_path).mkdir(exist_ok=True, parents=True)

    if jnp.ndim(x1) == 2 and jnp.shape(x1)[-1] == 2:  # 2D samples
        x_real, _ = train_state.train_dataset.sample(n)
        fig, ax = plt.subplots()
        ax.scatter(x1[:, 0], x1[:, 1], s=5, alpha=0.5, label="Generated samples")
        ax.scatter(x_real[:, 0], x_real[:, 1], s=5, alpha=0.5, label="Real samples")
        ax.set_aspect("equal")
        ax.legend()
        fig.savefig(Path(save_path) / "samples.png")
        plt.close(fig)
    elif jnp.ndim(x1) == 3 or (
        jnp.ndim(x1) == 4 and jnp.shape(x1)[-1] in (1, 3)
    ):  # Images
        if jnp.ndim(x1) == 3:
            x1 = jnp.expand_dims(x1, -1)
        if jnp.shape(x1)[-1] == 1:
            x1 = jnp.repeat(x1, 3, axis=-1)

        n, h, w, c = jnp.shape(x1)

        grid_size = math.ceil(math.sqrt(n))
        canvas = jnp.zeros((grid_size * h, grid_size * w, c))

        for i in range(n):
            row = i // grid_size
            col = i % grid_size
            canvas = canvas.at[row * h : (row + 1) * h, col * w : (col + 1) * w, :].set(
                x1[i]
            )

        fig, ax = plt.subplots()
        ax.imshow(canvas)
        ax.axis("off")
        fig.savefig(Path(save_path) / "samples.png")
        plt.close(fig)

    else:
        logging.warning("Cannot plot samples with shape %s", jnp.shape(x1))


def main(_):
    config: Any = FrozenConfigDict(FLAGS.config)
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
        forward_fn=model.forward,
        params=params,
        tx=optimizer,
        train_metrics=TrainMetrics.empty(),
        model_metrics=ModelMetrics.empty(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        rng=state_rng,
    )

    while train_state.step < config.num_steps:
        train_state = train_step(train_state, config)

        step = train_state.step
        if config.log_steps > 0 and step % config.log_steps == 0:
            log_metrics(train_state)
            train_state = train_state.replace(
                train_metrics=TrainMetrics.empty(), model_metrics=ModelMetrics.empty()
            )
        if config.save_steps > 0 and step % config.save_steps == 0:
            save_checkpoint(train_state, Path(config.checkpoint_dir) / f"step_{step}")
        if config.eval.steps > 0 and step % config.eval.steps == 0:
            evaluate(
                train_state, val_dataset, config.eval.n_batches, config.eval.batch_size
            )
        if config.generate.steps > 0 and step % config.generate.steps == 0:
            generate_samples(
                train_state,
                config.generate.samples,
                Path(config.checkpoint_dir) / f"step_{step}",
            )


if __name__ == "__main__":
    app.run(main)
