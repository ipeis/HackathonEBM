import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.distributions as td
from src.likelihoods import get_likelihood
from src.sgld_sampler import SGLD

class EBM(pl.LightningModule):
    """

    """
    def __init__(self, pretrained_model, input_dim, hidden_dims):
        super(EBM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Create the energy model
        layers = [nn.Flatten(), nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))  # Final output layer (one dim)
        self.energy_model = nn.Sequential(*layers)

        self.pretrained_model = pretrained_model



    def forward(self, x):

        pos_energy = self.energy_model(x)
        neg_samples = self.pretrained_model.sample(x.shape[0]).detach()
        neg_energy = self.energy_model(neg_samples)

        # Contrastive term
        loss = (pos_energy - neg_energy).mean()

        # Energy norm regularization
        reg_loss = (pos_energy**2).mean() + (neg_energy**2).mean()

        # Gradient regularization
        grad_reg_loss = 0
        if self.training:
            interp = torch.rand(x.size(0), 1, 1, 1, device=x.device)
            x_interp = (interp * x + (1 - interp) * neg_samples)
            x_interp.requires_grad_(True)
            energy_interp = self.energy_model(x_interp).mean()
            grad_interp = torch.autograd.grad(energy_interp, x_interp, create_graph=True)[0]
            grad_interp_flat = grad_interp.view(grad_interp.size(0), -1)  # [B, C*H*W]
            grad_reg_loss = grad_interp_flat.norm(2, dim=1).mean()

        # Total loss
        total_loss = loss + 0.15 * reg_loss + 0.15 * grad_reg_loss

        loss_dict = {
            'energy_loss': loss,
            'pos_energy': pos_energy.mean(),
            'neg_energy': neg_energy.mean(),
            'reg_loss': reg_loss,
            'grad_reg_loss': grad_reg_loss
        }
        return total_loss, loss_dict
    
    def log_dict(self, metrics, prefix=""):
        """
        Logs a dictionary of metrics using Lightning's self.log.
        Args:
            metrics (dict): Dictionary of metric names and values.
            prefix (str): Optional prefix for metric names.
        """
        for k, v in metrics.items():
            name = f"{prefix}{k}" if prefix else k
            self.log(name, v, on_step=True, on_epoch=False, prog_bar=False)
    
    def sample(self, num_samples, n_iter=100, return_init=False):
        with torch.set_grad_enabled(True):
            init = self.pretrained_model.sample(num_samples)
            sampler = SGLD(
                n_iter=n_iter,
                step_size=1e-4,
                noise_std=0.01,
                clamp_min=-1.0,
                clamp_max=1.0,
                tau=0.1
            )
            samples = sampler(self.unnorm_log_prob, init)
        if return_init:
            return samples, init
        return samples
    
    def reconstruct(self, x, n_iter=100, return_init=False):
        with torch.set_grad_enabled(True):
            init = self.pretrained_model.reconstruct(x, torch.ones_like(x))
            sampler = SGLD(
                n_iter=n_iter,
                step_size=1e-4,
                noise_std=0.01,
                clamp_min=-1.0,
                clamp_max=1.0,
                tau=0.1
            )
            x_rec = sampler(self.unnorm_log_prob, init)
        if return_init:
            return x_rec, init
        return x_rec
    
    def unnorm_log_prob(self, x):
        energy = self.energy_model(x)[...,0]
        log_q_x = self.pretrained_model.log_prob(x, torch.ones_like(x), K=1000)
        unnorm_log_prob = -energy + log_q_x
        return unnorm_log_prob






    def training_step(self, batch, batch_idx):
        x, _ = batch
        bs = x.shape[0]
        loss, loss_dict = self.forward(x)
        self.log_dict(loss_dict, "train/")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss, loss_dict = self.forward(x)
        self.log_dict(loss_dict, 'val/')
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
