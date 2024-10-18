from typing import TYPE_CHECKING

from flow_matching.model.cnn import CNN, CNNConfig
from flow_matching.model.mlp import MLP, MLPConfig
from flow_matching.model.unet import UNet, UNetConfig

if TYPE_CHECKING:
    from flow_matching.model.base import Model, ModelConfig


def build_model(config: ModelConfig) -> Model:
    match config:
        case MLPConfig():
            return MLP.create(config)
        case CNNConfig():
            return CNN.create(config)
        case UNetConfig():
            return UNet.create(config)
        case _:
            raise ValueError(f"Unknown model config: {config}")
