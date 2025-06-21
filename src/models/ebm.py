import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from hydra.utils import instantiate

class EBM(pl.LightningModule):
    def __init__(self, model, **kwargs):
        """
        Args:
            encoder: A neural network that encodes input x to a latent representation.
            energy_net: A network that outputs a scalar energy given the encoder's output.
        """
        super().__init__()
        self.energy_net = instantiate(model.energy_net)
        self.sampler = instantiate(model.sampler, log_prob=self.u_log_prob)

        kwargs['model'] = model
        self.save_hyperparameters(kwargs)

    def forward(self, x, *args, return_losses=False, **kwargs):
        x_sampled = self.sample(num_samples=x.shape[0])
        if return_losses:
            loss, loss_dict = self.loss(x, x_sampled, return_losses=return_losses)
            return loss, loss_dict
        else:
            return self.loss(x, x_sampled)
        
    def loss(self, x_real, x_sampled, return_losses=False):
        """
        Args:
            x_real: Samples from the true data distribution
            x_sampled: Negative samples (e.g., from a generator or Langevin dynamics)
        Returns:
            loss: A scalar loss value
        """
        if self.training:
            x_real.requires_grad_(True)

        energy_real = self.energy_net(x_real)[:,0]
        energy_sampled = self.energy_net(x_sampled)[:,0]
        loss = (energy_real - energy_sampled)

        # Gradient penalty
        if self.training:
            grad = torch.autograd.grad(energy_real.sum(), x_real, create_graph=True)[0]
            loss_grad = (grad ** 2).sum(dim=tuple(range(1, grad.ndim)))
            loss += self.hparams.model.lambda_grad * loss_grad

        # Energy regularization
        loss_energy = (energy_real ** 2 + energy_sampled ** 2)
        loss += self.hparams.model.lambda_energy * loss_energy
        
        if return_losses:
            loss_dict = {
                'energy_real': energy_real.mean(),
                'energy_sampled': energy_sampled.mean(),
                'loss_energy': loss_energy.mean(),
                }
            if self.training:
                loss_dict['grad_loss'] = loss_grad.mean()
            return loss, loss_dict
        else:        
            return loss
        
    def u_log_prob(self, x):
        """
        Unnormalized log probability of the data under the energy model.
        Args:

            x: Input data
        Returns:
            log_prob: Unnormalized log probability
        """
        energy = self.energy_net(x)
        return -energy  # EBM outputs energy, we return -energy for log probability
    
    def sample(self, num_samples=None):
        """
        Sample from the energy model using the provided sampler.
        Args:
            num_samples: Number of samples to generate
        Returns:
            samples: Generated samples from the energy model
        """
        if num_samples is None:
            num_samples = self.hparams.train.batch_size
        init = torch.randn(num_samples, self.hparams.model.input_dim, *self.hparams.model.input_size, device=self.device)
        samples = self.sampler(init=init)
        return samples            
    

    def training_step(self, batch, batch_idx):
        x, mask = batch
        loss, loss_dict = self.forward(x, mask, return_losses=True)
        self.log('train/loss', loss.mean(), on_step=True, on_epoch=False, prog_bar=True)
        loss_dict = {f'train/{k}': v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, on_step=True, on_epoch=False, prog_bar=False)
        return loss.mean()
        
    def validation_step(self, batch, batch_idx):
        x, mask = batch
        loss, loss_dict = self.forward(x, mask, return_losses=True)
        self.log('val/loss', loss.mean(), on_step=False, on_epoch=True, prog_bar=True)
        loss_dict = {f'val/{k}': v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False)
        return loss.mean()
    
    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.train.optimizer, params=self.parameters())
        return optimizer