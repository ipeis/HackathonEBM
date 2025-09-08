# Hackathon EBM

This repository provides reference implementations of energy-based models (EBMs) and related baselines on the MNIST dataset using [PyTorch Lightning](https://www.pytorchlightning.ai/) and [Hydra](https://hydra.cc/).

## Requirements
- Python 3.10+
- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/index.html)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Hydra](https://hydra.cc/) and [OmegaConf](https://omegaconf.readthedocs.io/)
- [Weights & Biases](https://wandb.ai/) (optional for logging)

Install dependencies with:

```bash
pip install torch torchvision pytorch-lightning hydra-core omegaconf wandb
```

## Data Preparation
The experiments use MNIST. The dataset is downloaded automatically the first time you run a training command using `torchvision.datasets.MNIST`.

To place datasets and logs in a specific directory, set the `LOGDIR` environment variable before running:

```bash
export LOGDIR=/path/to/workdir  # optional
```

Data will then be stored in `mnist/` under this location. No manual preprocessing is required.

## Training
Training is handled by `train.py`, which loads configuration files from the `configs/` directory.

Run the default energy-based model on MNIST with:

```bash
python3 train.py --config-name mnist_ebm
```

Hydra allows overriding any configuration value from the command line, for example:

```bash
python3 train.py --config-name mnist_ebm train.batch_size=64 train.epochs=50
```

Logs are written to `logs/EBM_Hackathon` (or under `LOGDIR` if set) and tracked in Weights & Biases.

## Run Experiment
Several model configurations are provided in `configs/`.

Examples:

```bash
# Train a MIWAE model
python3 train.py --config-name mnist_miwae

# Train a VAE baseline
python3 train.py --config-name mnist_vae
```

Create new experiments by adding configuration files or overriding parameters at the command line:

```bash
python3 train.py --config-name <your_config> model.lambda_reg=0.1
```

All experiment results will be saved under the log directory specified above.

