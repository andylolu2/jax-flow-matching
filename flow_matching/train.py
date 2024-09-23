import math
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

import fsspec
import fsspec.generic
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint
import wandb
from absl import app, flags, logging
from clu import metric_writers
from diffrax import Dopri5, ODETerm, diffeqsolve
from jaxtyping import Array, Float
from matplotlib.backends.backend_agg import FigureCanvasAgg
from ml_collections import FrozenConfigDict, config_flags

from flow_matching.builder import (
    TrainMetrics,
    TrainState,
    build_dataset,
    build_model,
    build_optimizer,
    build_train_state,
)
from flow_matching.dataset.base import Dataset
from flow_matching.model.base import ModelMetrics
from flow_matching.third_party.clu.wandb_writer import WandbWriter

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration file.", short_name="c", lock_config=False
)


@partial(jax.jit, static_argnames=("config",))
def train_step(train_state: TrainState, config) -> TrainState:
    x, train_dataset = train_state.train_dataset.sample(config.batch_size)
    rng, fwd_rng = jax.random.split(train_state.rng)
    grads, new_model_metrics = jax.grad(
        jax.vmap(train_state.apply_fn, in_axes=(None, 0, 0, None)), has_aux=True
    )(train_state.params, x, jax.random.split(fwd_rng, config.batch_size), train=True)
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


def log_metrics(train_state: TrainState, writer: metric_writers.MetricWriter) -> None:
    for metric in (train_state.train_metrics, train_state.model_metrics):
        values = metric.compute()
        values = {f"train/{k}": v for k, v in values.items()}
        metric_writers.write_values(writer, step=int(train_state.step), metrics=values)


def save_checkpoint(train_state: TrainState, path: str) -> None:
    logging.info("Saving checkpoint to %s", path)

    # save to temporary directory, then copy to final destination
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        checkpointer = orbax.checkpoint.StandardCheckpointer()
        checkpointer.save(tmp_path, train_state)
        checkpointer.wait_until_finished()

        fs = fsspec.generic.GenericFileSystem()
        fs.put(tmp_path, path, recursive=True)
        fs.copy


@partial(jax.jit, static_argnames=("batch_size",))
def _eval_step(
    train_state: TrainState,
    dataset: Dataset,
    batch_size: int,
    model_metrics: ModelMetrics,
):
    x, dataset = dataset.sample(batch_size)
    _, new_model_metrics = train_state.apply_fn(
        train_state.params, x, train_state.rng, train=False
    )
    model_metrics = model_metrics.merge(new_model_metrics)
    return model_metrics, dataset


def evaluate(
    train_state: TrainState,
    dataset: Dataset,
    n_batches: int,
    batch_size: int,
    writer: metric_writers.MetricWriter,
) -> None:
    logging.info("Evaluating on validation dataset")

    model_metrics = ModelMetrics.empty()

    for _ in range(n_batches):
        model_metrics, dataset = _eval_step(
            train_state, dataset, batch_size, model_metrics
        )

    model_metrics = model_metrics.compute()
    model_metrics = {f"val/{k}": v for k, v in model_metrics.items()}
    metric_writers.write_values(
        writer, step=int(train_state.step), metrics=model_metrics
    )


def _generate_samples(train_state: TrainState, n: int) -> Float[Array, "{n} ..."]:
    _x, _ = train_state.train_dataset.sample(1)
    size = math.prod(jnp.shape(_x)[1:])
    x0 = jax.random.multivariate_normal(
        train_state.rng, jnp.zeros(size), jnp.eye(size), shape=(n,)
    )
    x0 = jnp.reshape(x0, (n,) + jnp.shape(_x)[1:])

    forward_fn = lambda t, x, args: train_state.apply_fn(
        train_state.params, x, t, train=False, method=train_state.forward_fn
    )
    term = ODETerm(jax.vmap(forward_fn, in_axes=(None, 0, None)))
    solver = Dopri5()
    solution = diffeqsolve(term, solver, t0=0, t1=0.995, dt0=0.05, y0=x0)

    x1 = solution.ys[-1]
    return x1


def generate_samples(
    train_state: TrainState, n: int, writer: metric_writers.MetricWriter
) -> None:
    logging.info("Generating samples from model")

    x1 = _generate_samples(train_state, n)

    if jnp.ndim(x1) == 2 and jnp.shape(x1)[-1] == 2:  # 2D samples
        x_real, _ = train_state.train_dataset.sample(n)
        fig, ax = plt.subplots()
        ax.scatter(x1[:, 0], x1[:, 1], s=5, alpha=0.5, label="Generated samples")
        ax.scatter(x_real[:, 0], x_real[:, 1], s=5, alpha=0.5, label="Real samples")
        ax.set_aspect("equal")
        ax.legend()

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        writer.write_images(int(train_state.step), {"val/samples": rgba})

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

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        writer.write_images(int(train_state.step), {"val/samples": rgba})

        plt.close(fig)

    else:
        logging.warning("Cannot plot samples with shape %s", jnp.shape(x1))


def main(_):
    config: Any = FrozenConfigDict(FLAGS.config)
    wandb.login()
    wandb.init(project="jax-flow-matching", dir="_wandb")
    writer = metric_writers.AsyncMultiWriter(
        [metric_writers.LoggingWriter(), WandbWriter()]
    )
    writer.write_hparams(config.to_dict())

    rng = jax.random.PRNGKey(config.seed)
    model_rng, fwd_rng, state_rng = jax.random.split(rng, 3)
    train_dataset, val_dataset = build_dataset(**config.dataset)
    model = build_model(**config.model)
    optimizer = build_optimizer(**config.optimizer)
    train_state = build_train_state(
        model, optimizer, train_dataset, val_dataset, state_rng
    )

    while train_state.step < config.num_steps:
        train_state = train_step(train_state, config)

        step = int(train_state.step)
        if config.log.steps > 0 and step % config.log.steps == 0:
            log_metrics(train_state, writer)
            train_state = train_state.replace(
                train_metrics=TrainMetrics.empty(), model_metrics=ModelMetrics.empty()
            )
        if config.save.steps > 0 and step % config.save.steps == 0:
            save_checkpoint(train_state, f"{config.checkpointj_dir}/step_{step}/")
        if config.eval.steps > 0 and step % config.eval.steps == 0:
            evaluate(
                train_state,
                val_dataset,
                config.eval.n_batches,
                config.eval.batch_size,
                writer,
            )
        if config.generate.steps > 0 and step % config.generate.steps == 0:
            generate_samples(train_state, config.generate.samples, writer)


if __name__ == "__main__":
    app.run(main)
