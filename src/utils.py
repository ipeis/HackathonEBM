import torch
from omegaconf import OmegaConf

# This allows to reverse a list in the config
OmegaConf.register_new_resolver("reverse", lambda x: list(reversed(x)))

def reparam(mu, logvar, n_samples=1):
    """
    Reparameterization trick for VAE.
    Args:
        mu (torch.Tensor): Mean of the latent variable.
        logvar (torch.Tensor): Log variance of the latent variable.
    Returns:
        torch.Tensor: Sampled latent variable.
    """
    if n_samples > 1:
        mu = mu.unsqueeze(1).expand(-1, n_samples, -1)
        logvar = logvar.unsqueeze(1).expand(-1, n_samples, -1)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


