import torch
import pytorch_lightning as pl
import torch.distributions as td
from hydra.utils import instantiate
from src.utils import reparam

class MIWAE(pl.LightningModule):
    """
    Missing data Importance Weighted Autoencoder (MIWAE)
    Reference: Mattei & Frellsen, 2019 (https://arxiv.org/abs/1902.10661)
    """
    def __init__(self, model, **kwargs):
        super(MIWAE, self).__init__()

        self.likelihood = instantiate(model.likelihood)
        self.encoder = instantiate(model.encoder)
        self.decoder = instantiate(model.decoder)

        kwargs['model'] = model
        
        self.save_hyperparameters(kwargs)

    def encode(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar

    def decode(self, z):
        # z: [batch, K, latent_dim]
        batch, K, latent_dim = z.size()
        input_size = self.hparams.model.input_size
        input_dim = self.hparams.model.input_dim

        z_flat = z.view(-1, latent_dim)
        x_recon = self.decoder(z_flat)
        return x_recon.view(batch, K, input_dim, *input_size)

    def forward(self, x, mask):
        return -self.elbo(x, mask).mean()

    def elbo(self, x, mask, K=None):
        """
        x: [batch, input_dim]
        mask: [batch, input_dim] binary mask (1=observed, 0=missing)
        """
        if K==None:
            K = self.hparams.model.K
        mean, logvar = self.encode(x)
        z = reparam(mean, logvar, K)  # [batch, K, latent_dim]

        # Compute log q(z|x_obs)
        qz_mean = mean.unsqueeze(1).expand(-1, K, -1)  # [batch, K, latent_dim]
        qz_logvar = logvar.unsqueeze(1).expand(-1, K, -1)
        qz = td.Normal(qz_mean, torch.exp(0.5 * qz_logvar))
        log_qz = qz.log_prob(z).sum(-1)  # [batch, K]

        # Compute log p(z)
        pz = td.Normal(torch.zeros_like(z), torch.ones_like(z))
        log_pz = pz.log_prob(z).sum(-1)  # [batch, K]

        # Decode
        x_recon = self.decode(z)  # [batch, K, input_dim]

        # Compute log p(x_obs|z)
        # Only observed values contribute to likelihood
        log_px = self.likelihood.log_prob_mask(x, x_recon, mask=mask, mode='sum')  # [batch, K]

        # Importance weights
        log_w = log_px + log_pz - log_qz  # [batch, K]
        log_w = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)

        # MIWAE ELBO
        elbo = torch.logsumexp(log_px + log_pz - log_qz, dim=1) - torch.log(torch.tensor(K, device=x.device, dtype=x.dtype))
        return elbo

    def log_prob(self, x, mask, K=None):
        return self.elbo(x, mask, K)
    
    def reconstruct(self, x, mask, K=None):
        """
        x: [batch, input_dim] 
        mask: [batch, input_dim] binary mask (1=observed, 0=missing)
        """
        if K is None:
            K = self.hparams.model.K
        with torch.no_grad():
            mean, logvar = self.encode(x)
            z = reparam(mean, logvar, K)  # [batch, K, latent_dim]
            
            # Decode
            x_recon = self.decode(z)  # [batch, K, input_dim]
            
            # Get weights
            log_px = self.likelihood.log_prob_mask(x, x_recon, mask=mask, mode='sum')  # [batch, K]
            log_pz = td.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)  # [batch, K]
            log_qz = td.Normal(mean.unsqueeze(1).expand(-1, K, -1), torch.exp(0.5 * logvar).unsqueeze(1)).log_prob(z).sum(-1)  # [batch, K]
            log_w = log_px + log_pz - log_qz  # [batch, K]
            log_w = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)  # Normalize weights

            # Reconstruct
            # Expand log_w to match x_recon's dimensions for broadcasting
            weights = torch.exp(log_w).view(log_w.shape[0], log_w.shape[1], *([1] * (x_recon.dim() - 2)))
            x_recon = x_recon * weights
            x_recon = x_recon.sum(dim=1)  # Weighted sum over K

        return x_recon

    def sample(self, num_samples, device=None):
        """
        Generate samples from the model's prior.
        Returns: [num_samples, *input_dim]
        """
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.hparams.model.latent_dim, device=device)
        x_samples = self.decoder(z)
        return x_samples        

    def impute(self, x, mask, K=None):
        """
        Impute missing values in x using the trained model.
        """
        with torch.no_grad():
            x_recon = self.reconstruct(x, mask, K=K)
            x_out = x.clone()
            x_out[mask == 0] = x_recon[mask == 0]
        return x_out
    
    def training_step(self, batch, batch_idx):
        x, mask = batch
        loss = self.forward(x, mask)
        self.log('train/loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, mask = batch
        loss = self.forward(x, mask)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.train.optimizer, params=self.parameters())
        return optimizer