import gc
import math
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Callable

import flax.struct
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint
import wandb
from absl import logging
from clu import metric_writers
from clu.metrics import Collection, LastValue
from diffrax import Dopri5, ODETerm, diffeqsolve
from flax.training.train_state import TrainState as _TrainState
from jaxtyping import Array, Float, PRNGKeyArray
from matplotlib.backends.backend_agg import FigureCanvasAgg

from flow_matching.dataset.base import Dataset
from flow_matching.dataset.builder import DatasetConfig, build_dataset
from flow_matching.model.base import ModelMetrics
from flow_matching.model.builder import ModelConfig, build_model
from flow_matching.optimizer.builder import OptimizerConfig, build_optimizer
from flow_matching.third_party.clu import WandbWriter
from flow_matching.third_party.fsspec import copy
from flow_matching.third_party.pydantic import BaseModel


class LogConfig(BaseModel):
    steps: int


class SaveConfig(BaseModel):
    steps: int


class EvalConfig(BaseModel):
    steps: int
    n_batches: int
    batch_size: int


class GenerateConfig(BaseModel):
    steps: int
    samples: int


class TrainConfig(BaseModel):
    model: ModelConfig
    optimizer: OptimizerConfig
    train_dataset: DatasetConfig
    val_dataset: DatasetConfig
    log: LogConfig
    save: SaveConfig
    eval: EvalConfig
    generate: GenerateConfig

    num_steps: int
    exp_dir: str
    batch_size: int
    seed: int = 0
    dry_run: bool = False
    num_compile_steps: int = 5
    ema_step_size: float = 1 - 0.9999

    def __post_init__(self):
        assert self.log.steps % self.num_compile_steps == 0
        assert self.save.steps % self.num_compile_steps == 0
        assert self.eval.steps % self.num_compile_steps == 0


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
    ema_params: flax.core.FrozenDict[str, Any] = flax.struct.field(pytree_node=True)

    def apply_gradients(self, *, grads, ema_step_size: float, **kwargs):
        next_state = super().apply_gradients(grads=grads, **kwargs)
        new_ema_params = optax.incremental_update(
            new_tensors=next_state.params,
            old_tensors=self.ema_params,
            step_size=ema_step_size,
        )
        return next_state.replace(ema_params=new_ema_params)


def build_train_state(config: TrainConfig) -> TrainState:
    model = build_model(config.model)
    optimizer = build_optimizer(config.optimizer)
    train_dataset = build_dataset(config.train_dataset)
    val_dataset = build_dataset(config.val_dataset)

    rng = jax.random.PRNGKey(config.seed)
    model_rng, fwd_rng, state_rng = jax.random.split(rng, 3)
    x, _ = train_dataset.sample(1)
    params = model.init(model_rng, x[0], fwd_rng, train=True)
    return TrainState.create(
        apply_fn=model.apply,
        forward_fn=model.forward,
        params=params,
        ema_params=params,
        tx=optimizer,
        train_metrics=TrainMetrics.empty(),
        model_metrics=ModelMetrics.empty(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        rng=state_rng,
    )


@partial(jax.jit, static_argnames=("config",))
def train_step(train_state: TrainState, config: TrainConfig) -> TrainState:
    def single_step(train_state: TrainState) -> TrainState:
        x, train_dataset = train_state.train_dataset.sample(config.batch_size)
        rng, fwd_rng = jax.random.split(train_state.rng)

        def loss_fn(params):
            losses, metrics = jax.vmap(
                train_state.apply_fn, in_axes=(None, 0, 0, None)
            )(params, x, jax.random.split(fwd_rng, config.batch_size), True)
            return jnp.mean(losses), metrics

        grads, new_model_metrics = jax.grad(loss_fn, has_aux=True)(train_state.params)
        train_state = train_state.apply_gradients(
            grads=grads,
            ema_step_size=config.ema_step_size,
            train_dataset=train_dataset,
            rng=rng,
        )

        model_metrics = jax.lax.fori_loop(
            0,
            config.batch_size,
            lambda i, acc: acc.merge(jax.tree_map(lambda x: x[i], new_model_metrics)),
            train_state.model_metrics,
        )
        train_metrics = train_state.train_metrics.merge(
            TrainMetrics.single_from_model_output(
                step=jnp.array(train_state.step, dtype=jnp.float32),
                epoch=jnp.array(train_state.train_dataset.epoch, dtype=jnp.float32),
            )
        )
        return train_state.replace(
            model_metrics=model_metrics, train_metrics=train_metrics
        )

    train_state = jax.lax.fori_loop(
        0, config.num_compile_steps, lambda _, state: single_step(state), train_state
    )
    return train_state


def log_metrics(train_state: TrainState, writer: metric_writers.MetricWriter) -> None:
    for metric in (train_state.train_metrics, train_state.model_metrics):
        values = metric.compute()
        values = {f"train/{k}": v for k, v in values.items()}
        metric_writers.write_values(writer, step=int(train_state.step), metrics=values)


def save_checkpoint(train_state: TrainState, path: str) -> None:
    logging.info("Saving checkpoint to %s", path)

    # save to temporary directory, then copy to final destination
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "checkpoint"
        checkpointer = orbax.checkpoint.StandardCheckpointer()
        checkpointer.save(tmp_path, train_state)
        checkpointer.wait_until_finished()

        copy(str(tmp_path), path)


@partial(jax.jit, static_argnames=("batch_size",))
def _eval_step(
    train_state: TrainState,
    dataset: Dataset,
    batch_size: int,
    model_metrics: ModelMetrics,
):
    x, dataset = dataset.sample(batch_size)
    _, metrics = jax.vmap(train_state.apply_fn, in_axes=(None, 0, 0, None))(
        train_state.ema_params, x, jax.random.split(train_state.rng, batch_size), False
    )
    new_model_metrics = jax.tree_map(jnp.mean, metrics)
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


@partial(jax.jit, static_argnames=("n",))
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
    term: ODETerm = ODETerm(jax.vmap(forward_fn, in_axes=(None, 0, None)))  # type: ignore[call-arg]
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
        img = jnp.zeros((grid_size * h, grid_size * w, c))

        for i in range(n):
            row = i // grid_size
            col = i % grid_size
            img = img.at[row * h : (row + 1) * h, col * w : (col + 1) * w, :].set(x1[i])

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis("off")

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        writer.write_images(int(train_state.step), {"val/samples": rgba})

        plt.close(fig)

    else:
        logging.warning("Cannot plot samples with shape %s", jnp.shape(x1))


def main(config: TrainConfig) -> None:
    wandb.login()
    wandb.init(
        project="jax-flow-matching",
        dir="_wandb",
        mode="offline" if config.dry_run else "online",
        save_code=True,
    )
    logging.set_verbosity(logging.INFO)
    writer = metric_writers.AsyncMultiWriter(
        [metric_writers.LoggingWriter(), WandbWriter()]
    )
    writer.write_hparams(config.model_dump())

    train_state = build_train_state(config)

    while train_state.step < config.num_steps:
        train_state: TrainState = train_step(train_state, config)  # type: ignore[no-redef]

        step = int(train_state.step)
        if config.log.steps > 0 and step % config.log.steps == 0:
            log_metrics(train_state, writer)
            train_state = train_state.replace(
                train_metrics=TrainMetrics.empty(), model_metrics=ModelMetrics.empty()
            )
        if config.save.steps > 0 and step % config.save.steps == 0:
            save_checkpoint(train_state, f"{config.exp_dir}/step_{step}/")
        if config.eval.steps > 0 and step % config.eval.steps == 0:
            evaluate(
                train_state,
                train_state.val_dataset,
                config.eval.n_batches,
                config.eval.batch_size,
                writer,
            )
        if config.generate.steps > 0 and step % config.generate.steps == 0:
            generate_samples(train_state, config.generate.samples, writer)

        gc.collect()
