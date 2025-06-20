import torch

class SGLD:
    def __init__(
        self,
        n_iter,
        step_size=1e-3,
        noise_std=1.0,
        clamp_min=-1.0,
        clamp_max=1.0,
        tau=None  # Optional gradient clamp
    ):
        self.n_iter = n_iter
        self.step_size = step_size
        self.noise_std = noise_std
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.tau = tau

    def __call__(self, log_pdf, init):
        x = init.detach().clone().requires_grad_(True)
        for _ in range(self.n_iter):
            if x.grad is not None:
                x.grad.zero_()

            # Forward + backward
            energy = log_pdf(x).sum()
            energy.backward()

            # Gradient step
            grad = x.grad
            if self.tau is not None:
                grad = torch.clamp(grad, -self.tau, self.tau)

            noise = torch.randn_like(x) * self.noise_std
            x = x + 0.5 * self.step_size * grad + torch.sqrt(torch.tensor(self.step_size)) * noise

            # Clamp pixel values to [0, 1]
            x = x.detach().clamp(self.clamp_min, self.clamp_max).requires_grad_(True)

        return x.detach()