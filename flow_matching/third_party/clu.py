from typing import Any, Mapping, Optional, TypeAlias

import numpy as np
import wandb
from clu.metric_writers import interface

Array: TypeAlias = interface.Array
Scalar: TypeAlias = interface.Scalar


class WandbWriter(interface.MetricWriter):
    """MetricWriter that writes Pytorch summary files."""

    def write_summaries(
        self,
        step: int,
        values: Mapping[str, Array],
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        assert (
            wandb.run is not None
        ), "wandb.init must be called before writing summaries"
        for k, v in values.items():
            wandb.run.summary[k] = v
        if metadata is not None:
            for k, v in metadata.items():
                wandb.run.summary[f"metadata/{k}"] = v

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        wandb.log({k: v for k, v in scalars.items()}, step=step)

    def write_images(self, step: int, images: Mapping[str, Array]):
        wandb.log({k: wandb.Image(v) for k, v in images.items()}, step=step)

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        wandb.log({k: wandb.Video(np.array(v)) for k, v in videos.items()}, step=step)

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        wandb.log(
            {k: wandb.Audio(v, sample_rate=sample_rate) for k, v in audios.items()},
            step=step,
        )

    def write_texts(self, step: int, texts: Mapping[str, str]):
        wandb.log({k: wandb.Html(v) for k, v in texts.items()}, step=step)

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Optional[Mapping[str, int]] = None,
    ):
        wandb.log(
            {
                tag: wandb.Histogram(
                    list(values),
                    num_bins=64 if num_buckets is None else num_buckets.get(tag, 64),
                )
                for tag, values in arrays.items()
            },
            step=step,
        )

    def write_hparams(self, hparams: Mapping[str, Any]):
        wandb.config.update(hparams)  # type: ignore

    def flush(self):
        wandb.log({}, commit=True)

    def close(self):
        self.flush()
