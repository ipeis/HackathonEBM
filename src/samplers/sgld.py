import torch
import math

class SGLD:
    def __init__(
        self,
        n_iter,
        log_prob,
        step_size=1e-3,
        noise_std=1.0,
        clamp_min=-1.0,
        clamp_max=1.0,
        tau=None  # Optional gradient clamp
    ):
        self.n_iter = n_iter
        self.log_prob = log_prob 
        self.step_size = step_size
        self.noise_std = noise_std
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.tau = tau

    def __call__(self, init):

        with torch.enable_grad():

            x = init.detach().clone().requires_grad_(True)
            sqrt_step_size = math.sqrt(self.step_size)

            for _ in range(self.n_iter):
                if x.grad is not None:
                    x.grad.zero_()

                energy = self.log_prob(x).sum()  # Typically: -E(x)
                energy.backward()

                grad = x.grad
                if self.tau is not None:
                    grad = torch.clamp(grad, -self.tau, self.tau)

                noise = torch.randn_like(x) * self.noise_std
                x = x + 0.5 * self.step_size * grad + sqrt_step_size * noise

                x = x.detach().clamp(self.clamp_min, self.clamp_max).requires_grad_(True)

        return x.detach()