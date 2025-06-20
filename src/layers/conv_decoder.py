import torch.nn as nn



# ------------------ ConvDecoder ------------------
class ConvDecoder(nn.Module):
    def __init__(
        self,
        output_channels,
        output_size,  # tuple (H, W)
        hidden_channels,
        kernel_sizes,
        strides,
        paddings,
        latent_dim
    ):
        """
        CNN Decoder for VAE.

        Args:
            output_channels (int): # of output image channels.
            output_size (tuple): (H, W) of the output image.
            hidden_channels (list[int]): Channels for each ConvTranspose layer.
            kernel_sizes (list[int]): Kernel sizes for ConvTranspose layers.
            strides (list[int]): Strides for ConvTranspose layers.
            paddings (list[int]): Paddings for ConvTranspose layers.
            latent_dim (int): Dimensionality of latent vector input.
        """
        super().__init__()
        assert len(hidden_channels) == len(kernel_sizes) == len(strides) == len(paddings), \
            "All ConvTranspose layer configs must be same length"

        self.output_size = output_size
        self.latent_dim = latent_dim

        # Compute initial feature map size using inverse of ConvTranspose2d formula
        h, w = output_size
        for k, s, p in zip(kernel_sizes, strides, paddings):
            h = (h - k + 2 * p) // s + 1
            w = (w - k + 2 * p) // s + 1

        self.init_h = h
        self.init_w = w
        self.init_channels = hidden_channels[0]

        self.fc = nn.Linear(latent_dim, self.init_channels * h * w)

        layers = []
        in_channels = hidden_channels[0]
        for out_channels, k, s, p in zip(hidden_channels[1:], kernel_sizes[1:], strides[1:], paddings[1:]):
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p))
            layers.append(nn.ReLU())
            in_channels = out_channels

        # Final output layer
        layers.append(nn.ConvTranspose2d(in_channels, output_channels, kernel_size=kernel_sizes[0], stride=strides[0], padding=paddings[0]))

        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), self.init_channels, self.init_h, self.init_w)
        x_recon = self.deconv(h)
        return x_recon


