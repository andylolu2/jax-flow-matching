import optax

from flow_matching.third_party.pydantic import BaseModel


class OptimizerConfig(BaseModel):
    pass


class AdamConfig(OptimizerConfig):
    learning_rate: float
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 1e-8


def build_optimizer(config: OptimizerConfig) -> optax.GradientTransformation:
    match config:
        case AdamConfig():
            return optax.adam(**config.model_dump())
        case _:
            raise ValueError(f"Unknown optimizer config: {config}")
