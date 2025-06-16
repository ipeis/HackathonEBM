# HackathonEBM

In MIWAE_Pytorch_exercises_demo_ProbAI, a base model is implemented and trained on MNAR Data. We will modify this model with an Energy-Based Model (EBM) to handle the MNAR data more effectively.


Objective: 

1- Using a VAE in data space
    1.1- Train a VAE on MNAR data.
    The objective of this hackathon is to modify the VAE using EBM to handle MNAR data.
    VAE :
    $$p_{\phi}(x) = \int p_{\phi}(x|z)p(z)dz$$
    where we defined :
    $$p_{\phi}(x|z) = \mathcal{N}(x; \mu_{\theta}(z), \sigma_{\theta}(z))$$
    $$p(z) = \mathcal{N}(z; 0, I)$$
    where $\mu_{\theta}(z)$ and $\sigma_{\theta}(z)$.

As usual in the VAE setting, we will use the reparameterization trick to sample from $p_{\phi}(x|z)$:
    $$x = \mu_{\theta}(z) + \sigma_{\theta}(z) \odot \epsilon$$
    where $\epsilon \sim \mathcal{N}(0, I)$.
    
1.2- First idea is just to tilt the data output with an EBM. To that end, we consider an EBM of the form:
    $$p_{\theta}(x) = \frac{1}{Z_{\theta}} e^{-E_{\theta}(x)}p_{\phi(x)}$$
    where $E_{\theta}(x)$ is the energy function parameterised by a neural network, and $Z_{\theta}$ is the partition function that ensures normalisation.

We want to train this model to minimise the log-likelihood of the data under the EBM :
    $$\mathcal{L}_{EBM} = \sum_{i=1}^n  \frac{1}{n} (- E_{\theta}(x_i) + \log p_{\phi}(x_i)) +  \log \mathbb{E}_{p_{\phi}(\tilde{x})} \left[ e^{-E_{\theta}(\tilde{x})} \right]$$

Using Jensen's inequality, we can derive a lower bound for the log-likelihood:
    $$\mathcal{L}_{EBM} \geq \sum_{i=1}^n  \frac{1}{n} (- E_{\theta}(x_i) + \log p_{\phi}(x_i)) +   \mathbb{E}_{p_{\phi}(\tilde{x})} \left[ -E_{\theta}(\tilde{x}) \right]$$

One can obtain the gradient in $\theta$ (and make the $q_{\phi}$ disappear) as follows:

$$\nabla_{\theta} \mathcal{L}_{EBM} = \sum_{i=1}^n  \frac{1}{n} \left( -\nabla_{\theta} E_{\theta}(x_i)\right) +  \mathbb{E}_{p_{\phi}(\tilde{x})} \left[ -\nabla_{\theta} E_{\theta}(\tilde{x}) \right]$$.

The first term is the gradient of the energy function evaluated at the observed data, and the second term is the gradient of the energy function evaluated at samples from the VAE.

1.3 Sampling from the resulting model can be done by sampling from the VAE and then use a MCMC chain to update the samples according to the EBM. 
    The gradient of the full log-likelihood guides the MCMC chain:
    $$\nabla_{x} \log p_{\theta}(x) = -\nabla_{x} E_{\theta}(x) + \nabla_{x} \log p_{\phi}(x)$$
    wwhere $p_{\phi}(x)$ can be approximated by the ELBO of the VAE.

$$x_{t+1} = x_t - \eta \left( -\nabla_{x} E_{\theta}(x_t) + \nabla_{x} \log p_{\phi}(x_t) \right)$$
    where $\eta$ is the step size.







2 - 



