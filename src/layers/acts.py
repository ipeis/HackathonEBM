import torch.nn as nn

activations = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'softmax': nn.Softmax,
    'softplus': nn.Softplus,
    'elu': nn.ELU,
    'selu': nn.SELU,
    'prelu': nn.PReLU,
    'gelu': nn.GELU,
    'swish': lambda: nn.SiLU(),  # Swish is often implemented as SiLU in PyTorch
}

