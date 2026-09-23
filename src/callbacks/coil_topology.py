import math
from pathlib import Path

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader

from src.data.coil100_datamodule import COIL100Dataset
from src.eval.topology import TopologyOptions, analyse, persistence, plot_persistence, report
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class COILTopologyCallback(Callback):
    """Embed every COIL-100 frame with the trained weights and score its topology."""

    def __init__(
        self,
        stride: int = 3,
        global_maxdim: int = 1,
        thresh_quantile: float = 0.4,
        h0_scale: float = 2.0,
        h1_cut: float = 0.3,
        holdout_objects: int = 20,
        batch_size: int = 256,
        num_workers: int = 8,
        save_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.options = TopologyOptions(
            stride=stride,
            global_maxdim=global_maxdim,
            thresh_quantile=thresh_quantile,
            h0_scale=h0_scale,
            h1_cut=h1_cut,
            holdout_objects=holdout_objects,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_embeddings = save_embeddings

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._measure(trainer, pl_module, self.num_workers)

    def _measure(self, trainer: Trainer, pl_module: LightningModule, workers: int) -> None:
        if not trainer.is_global_zero:
            return

        datamodule = trainer.datamodule  # type: ignore
        for attribute in ("data_dir", "eval_transform"):
            if getattr(datamodule, attribute, None) is None:
                raise RuntimeError(
                    f"{type(self).__name__} needs a datamodule exposing {attribute!r}; "
                    f"got {type(datamodule).__name__}. Attach it to a COIL-100 run."
                )

        dataset = COIL100Dataset(datamodule.data_dir, datamodule.eval_transform, split="all")
        log.info(f"Scoring topology on {len(dataset)} COIL-100 frames.")

        self._score(trainer, pl_module, dataset, "final", workers)

        best = getattr(trainer.checkpoint_callback, "best_model_path", "") or ""
        if not (best and Path(best).is_file()):
            log.info("No best checkpoint recorded; scored the final weights only.")
            return

        held = {key: value.detach().clone() for key, value in pl_module.state_dict().items()}
        pl_module.load_state_dict(torch.load(best, map_location="cpu", weights_only=False)["state_dict"])
        try:
            self._score(trainer, pl_module, dataset, "best", workers)
        finally:
            pl_module.load_state_dict(held)

    def _score(self, trainer: Trainer, pl_module: LightningModule, dataset: COIL100Dataset, tag: str, workers: int) -> None:
        embeddings, projections = self._embed(pl_module, dataset, workers)
        angles = dataset.angles.astype(np.float64)

        for name, tensor in (("embeddings", embeddings), ("projections", projections)):
            cloud = tensor.numpy().astype(np.float64)
            stats = analyse(cloud, angles, dataset.objects, self.options)
            report(f"COIL topology :: {tag} weights :: {name}", stats)
            if trainer.logger is not None:
                trainer.logger.log_metrics(
                    {f"topology/{tag}/{name}/{key}": value for key, value in stats.items() if math.isfinite(value)},
                    step=trainer.global_step,
                )

            figure = Path(trainer.default_root_dir) / f"persistence_{tag}_{name}.png"
            global_dgms, per_object, _ = persistence(cloud, dataset.objects, self.options)
            plot_persistence(global_dgms, per_object, f"{tag} weights, {name}", figure)
            if trainer.logger is not None and hasattr(trainer.logger, "log_image"):
                trainer.logger.log_image(key=f"topology/{tag}/{name}/persistence", images=[str(figure)])
            log.info(f"Wrote {figure}")

        if self.save_embeddings:
            destination = Path(trainer.default_root_dir) / f"coil_embeddings_{tag}.pt"
            torch.save(
                {
                    "embeddings": embeddings,
                    "projections": projections,
                    "angles": torch.from_numpy(dataset.angles),
                    "objects": torch.from_numpy(dataset.objects),
                },
                destination,
            )
            log.info(f"Wrote {destination}")

    def _embed(self, pl_module: LightningModule, dataset: COIL100Dataset, workers: int) -> tuple[torch.Tensor, torch.Tensor]:
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=workers, shuffle=False)
        was_training = pl_module.training
        pl_module.eval()

        embeddings, projections = [], []
        with torch.no_grad():
            for images, _ in loader:
                # The eval transform is multi-view with one view, as elsewhere in the repo.
                images = images.flatten(0, 1) if images.ndim == 5 else images
                features = pl_module.encoder(images.to(pl_module.device))  # type: ignore
                embeddings.append(features.float().cpu())
                projections.append(pl_module.projector(features).float().cpu())  # type: ignore

        if was_training:
            pl_module.train()
        return torch.cat(embeddings), torch.cat(projections)
