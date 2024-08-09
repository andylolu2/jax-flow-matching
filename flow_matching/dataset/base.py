import numpy as np


class Dataset:
    def dimensions(self) -> tuple[int, ...]:
        raise NotImplementedError

    def sample(self, batch_size: int) -> np.ndarray:
        raise NotImplementedError
