import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import torch.nn as nn
import torch.optim as optim

import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import wandb

from src.models.miwae import MIWAE
from src.ebm import EBM
from src.data.utils import get_data_loaders
from src.callbacks import get_callbacks
def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--miwae_config', type=str, required=True, help='Path to MIWAE config file')
    args = parser.parse_args()
    
    # Load MIWAE config
    default_config = omegaconf.OmegaConf.load("configs/default.yaml")
    miwae_config = omegaconf.OmegaConf.load(args.miwae_config)
    miwae_config = omegaconf.OmegaConf.merge(default_config, miwae_config)

    # Load config
    config = omegaconf.OmegaConf.load(args.config)
    config = omegaconf.OmegaConf.merge(default_config, config)

    # Provide the required arguments for the model
    miwae = MIWAE.load_from_checkpoint(
        config.model.pretrained_ckpt,
        input_dim=miwae_config.model.input_dim,
        latent_dim=miwae_config.model.latent_dim,
        K=miwae_config.model.K,
        likelihood=miwae_config.model.likelihood,
        lr=miwae_config.train.lr,
    )
    miwae.eval()

    ebm_model = EBM(
        pretrained_model=miwae,
        input_dim=config.model.input_dim,
        hidden_dims=config.model.hidden_dims
    )

    train_loader, val_loader = get_data_loaders(config)

    # Wandb logger
    wandb_logger = pl.loggers.WandbLogger(
        project='EBM_Hackathon', 
        config=omegaconf.OmegaConf.to_container(config, resolve=True),
        save_dir=config.log_dir)

    # Callbacks
    callbacks = get_callbacks(config, val_loader)

    # Trainer
    trainer = Trainer(
        max_epochs=config.train.epochs,
        logger=wandb_logger,
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        default_root_dir=config.log_dir,
        callbacks=callbacks,
        
    )

    trainer.fit(ebm_model, train_loader, val_loader)

if __name__ == "__main__":
    main()