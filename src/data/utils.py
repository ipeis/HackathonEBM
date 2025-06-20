
from hydra.utils import instantiate
from omegaconf import OmegaConf

def get_data_loaders(cfg):

    train_cfg = OmegaConf.merge(cfg.data, cfg.train_data)
    train_dataset = instantiate(train_cfg)
    train_loader = instantiate(cfg.dataloader, dataset=train_dataset, batch_size=cfg.train.batch_size, shuffle=True)

    if 'val_data' in cfg:
        val_cfg = OmegaConf.merge(cfg.data, cfg.val_data)
        val_dataset = instantiate(val_cfg)
        val_loader = instantiate(cfg.dataloader, dataset=val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    else:
        val_loader = None

    if 'test_data' in cfg:
        test_cfg = OmegaConf.merge(cfg.data, cfg.test_data)
        test_dataset = instantiate(test_cfg)
        test_loader = instantiate(cfg.dataloader, dataset=test_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
