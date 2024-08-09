import numpy as np

from flow_matching.dataset.base import Dataset


class ToyDataset(Dataset):
    def __init__(self):
        self.means = np.array([[4.0, 0.0], [-4.0, 0.0]])
        self.covariances = np.array([
            [[1.0, 0.7], [0.7, 1.0]], 
            [[1.0, 0.7], [0.7, 1.0]], 
        ])

    def dimensions(self) -> tuple[int, ...]:
        return (2,)

    def sample(self, batch_size: int) -> np.ndarray:
        # return np.random.multivariate_normal(self.mean, self.covariance, batch_size)
        samples = np.array([
            np.random.multivariate_normal(mean, covariance, batch_size)
            for mean, covariance in zip(self.means, self.covariances)
        ])
        idx = np.random.randint(0, 2, batch_size)
        return samples[idx, np.arange(batch_size)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = ToyDataset()
    samples = dataset.sample(1000)
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.savefig("tmp/toy_dataset.png")
