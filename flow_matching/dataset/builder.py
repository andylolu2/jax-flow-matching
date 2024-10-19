from flow_matching.dataset.base import Dataset, DatasetConfig
from flow_matching.dataset.cifar10 import Cifar10Config, Cifar10Dataset
from flow_matching.dataset.mnist import MnistConfig, MnistDataset
from flow_matching.dataset.toy import ToyConfig, ToyDataset


def build_dataset(config: DatasetConfig) -> Dataset:
    match config:
        case MnistConfig():
            return MnistDataset.create(config)
        case Cifar10Config():
            return Cifar10Dataset.create(config)
        case ToyConfig():
            return ToyDataset.create(config)
        case _:
            raise ValueError(f"Unknown dataset config: {config}")
