import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        input_size,  # tuple (H, W)
        hidden_channels,
        kernel_sizes,
        strides,
        paddings,
        latent_dim
    ):
        """
        CNN Encoder for VAE.

        Args:
            input_channels (int): # input channels.
            input_size (tuple): (H, W) of input image.
            hidden_channels (list[int]): Output channels of each Conv layer.
            kernel_sizes (list[int]): Kernel sizes for each Conv layer.
            strides (list[int]): Strides for each Conv layer.
            paddings (list[int]): Paddings for each Conv layer.
            latent_dim (int): Dimensionality of latent vector.
        """
        super().__init__()

        assert len(hidden_channels) == len(kernel_sizes) == len(strides) == len(paddings), \
            "All Conv layer configs must be same length"

        self.input_size = input_size
        self.latent_dim = latent_dim

        layers = []
        in_channels = input_channels
        h, w = input_size

        for out_channels, k, s, p in zip(hidden_channels, kernel_sizes, strides, paddings):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p))
            layers.append(nn.ReLU())
            in_channels = out_channels

            # Update feature map size
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1

        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        flattened_size = in_channels * h * w

        self.fc = nn.Linear(flattened_size, 2*latent_dim)

    def forward(self, x):
        h = self.flatten(self.conv(x))
        mu, logvar = self.fc(h).chunk(2, dim=-1)
        return mu, logvar
    

