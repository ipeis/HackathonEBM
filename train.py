import argparse
import torch

import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from src.data.utils import get_data_loaders
from src.callbacks import get_callbacks

import os
from hydra.utils import instantiate
from src.utils import *

def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config
    default_config = omegaconf.OmegaConf.load("configs/default.yaml")
    config = omegaconf.OmegaConf.load(args.config)
    config = omegaconf.OmegaConf.merge(default_config, config)

    # Logs and data directories
    if 'LOGDIR' not in os.environ:
        log_dir = 'logs/'
        config.data.root = os.path.join('data/', config.data.root)
    else:
        log_dir = os.environ['LOGDIR']
        config.data.root = os.path.join(log_dir, 'data', config.data.root)
        config.log_dir = os.path.join(log_dir, 'logs', 'EBM_Hackathon')
    
    # Model
    model = instantiate(config)

    # Data loaders
    loaders = get_data_loaders(config)  #[train_loader, val_loader, ?test_loader]
    
    # Wandb logger
    wandb_logger = pl.loggers.WandbLogger(
        project='EBM_Hackathon', 
        config=omegaconf.OmegaConf.to_container(config, resolve=True),
        save_dir=config.log_dir)

    # Callbacks
    callbacks = get_callbacks(config, loaders)

    if torch.cuda.is_available() and config.train.precision == "16":
        torch.set_float32_matmul_precision('medium')
        
    # Trainer
    trainer = Trainer(
        max_epochs=config.train.epochs,
        logger=wandb_logger,
        accelerator=config.train.accelerator,
        devices=config.train.devices if torch.cuda.is_available() else None,
        strategy=config.train.strategy,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        default_root_dir=config.log_dir,
        callbacks=callbacks,   
        gradient_clip_val=config.train.gradient_clip_val
    )

    trainer.fit(model, *loaders)

if __name__ == "__main__":
    main()