import pytorch_lightning as pl
import torch
import wandb
import numpy as np
import torchvision.utils as vutils
from hydra.utils import instantiate

class ImageReconstruction(pl.Callback):
    def __init__(self, dataloader, num_images=8, every_n_epochs=1, **kwargs):
        super().__init__()
        self.dataloader = dataloader
        self.num_images = num_images
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        with torch.no_grad():
            batch = next(iter(self.dataloader))
            if isinstance(batch, (tuple, list)):
                images, mask = batch[0][:self.num_images], batch[1][:self.num_images]
            else:
                images = batch[:self.num_images]
            images = images.to(pl_module.device)
            mask = images.to(pl_module.device)
            reconstructions = pl_module.reconstruct(images, mask)
        
        # Move to [0,1]
        images = (images + 1) / 2
        reconstructions = (reconstructions + 1) / 2
        reconstructions = torch.clamp(reconstructions, min=0, max=1)

        # Create a grid of originals and reconstructions
        grid = vutils.make_grid(
            torch.cat([images, reconstructions], dim=0),
            nrow=self.num_images,
            padding=2,
        )

        # Log to wandb
        image = wandb.Image(grid, caption=f"Epoch {trainer.current_epoch+1} (Top: Originals, Bottom: Reconstructions)")
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log({"Reconstructions": image, "epoch": trainer.current_epoch})



class ImageSamples(pl.Callback):
    def __init__(self, num_samples=8, every_n_epochs=1, **kwargs):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        with torch.no_grad():
            samples = pl_module.sample(self.num_samples)

        # Move to [0,1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, min=0, max=1)

        grid = vutils.make_grid(samples, nrow=int(np.sqrt(self.num_samples)), padding=2)
        model_name = pl_module.__class__.__name__
        image = wandb.Image(grid, caption=f"Epoch {trainer.current_epoch+1} ({model_name} Samples)")
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log({"MIWAE Samples": image, "epoch": trainer.current_epoch})



class ImageCorrectReconstruction(pl.Callback):
    def __init__(self, dataloader, num_images=8, every_n_epochs=1, **kwargs):
        super().__init__()
        self.dataloader = dataloader
        self.num_images = num_images
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        with torch.no_grad():
            batch = next(iter(self.dataloader))
            if isinstance(batch, (tuple, list)):
                images, mask = batch[0][:self.num_images], batch[1][:self.num_images]
            else:
                images = batch[:self.num_images]
            images = images.to(pl_module.device)
            mask = images.to(pl_module.device)
            reconstructions, init = pl_module.reconstruct(images, return_init=True)
        
        # Move to [0,1]
        images = (images + 1) / 2
        reconstructions = (reconstructions + 1) / 2
        reconstructions = torch.clamp(reconstructions, min=0, max=1)
        init = (init + 1) / 2
        init = torch.clamp(init, min=0, max=1)

        # Create a grid of originals and reconstructions
        grid = vutils.make_grid(
            torch.cat([images, init, reconstructions], dim=0),
            nrow=self.num_images,
            padding=2,
        )
        # Log to wandb
        image = wandb.Image(grid, caption=f"Epoch {trainer.current_epoch+1} (Top: Originals, Middel: MIWAE, Bottom: MIWAE+EBM)")
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log({"Reconstructions": image, "epoch": trainer.current_epoch})


class ImageCorrectedSamples(pl.Callback):
    def __init__(self, num_samples=8, every_n_epochs=1):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        with torch.no_grad():
            samples, init = pl_module.sample(self.num_samples, return_init=True)

        # Move to [0,1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, min=0, max=1)
        init = (init + 1) / 2
        init = torch.clamp(init, min=0, max=1)

        grid_samples = vutils.make_grid(samples, nrow=int(np.sqrt(self.num_samples)), padding=2)
        grid_init = vutils.make_grid(init, nrow=int(np.sqrt(self.num_samples)), padding=2)
        model_name = pl_module.__class__.__name__
        image_samples = wandb.Image(grid_samples, caption=f"Epoch {trainer.current_epoch+1} ({model_name} Corrected Samples)")
        image_init = wandb.Image(grid_init, caption=f"Epoch {trainer.current_epoch+1} ({model_name} Init Samples)")
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log({
            "Corrected Samples": image_samples,
            "Init Samples": image_init,
            "epoch": trainer.current_epoch
            })


def get_callbacks(config, loaders):
    splits = {
        'train': 0,
        'val': 1,
        'test': 2
    }
    callbacks = []
    if config.callbacks is None:
        return callbacks
    for _, callback_cfg in config.callbacks.items():
        split = callback_cfg.get('split', None)
        loader = loaders[splits[split]] if split in splits else None
        callbacks.append(instantiate(callback_cfg, dataloader=loader))
    return callbacks