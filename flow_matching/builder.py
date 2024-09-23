from typing import Callable

import flax.struct
import jax
import optax
from clu.metrics import Collection, LastValue
from flax.training.train_state import TrainState as _TrainState
from jaxtyping import PRNGKeyArray

from flow_matching.dataset.base import Dataset
from flow_matching.dataset.cifar10 import Cifar10Dataset
from flow_matching.dataset.mnist import MnistDataset
from flow_matching.dataset.toy import ToyDataset
from flow_matching.model.base import Model, ModelMetrics
from flow_matching.model.cnn import CNN
from flow_matching.model.mlp import MLP
from flow_matching.model.unet import UNet


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
    elif name == "cifar10":
        return (
            Cifar10Dataset.create(**kwargs, split="train"),
            Cifar10Dataset.create(**kwargs, split="val"),
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


def build_train_state(
    model: Model,
    optimizer: optax.GradientTransformation,
    train_dataset: Dataset,
    val_dataset: Dataset,
    rng: PRNGKeyArray,
) -> TrainState:
    model_rng, fwd_rng, state_rng = jax.random.split(rng, 3)
    params = model.init(model_rng, train_dataset.sample(1)[0], fwd_rng, train=True)
    return TrainState.create(
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
