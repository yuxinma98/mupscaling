This repository is the official implementation of **$\mu$pscaling Small Models: Principled Warm Starts and Hyperparameter Transfer**

# Requirements
To set up the environment using conda:
```bash
conda env create -f environment.yml
conda activate weight_transfer
```
An important requirement is the [mup package](https://github.com/microsoft/mup).

# Generic upscaling routine
The core logic for upscaling is located in `src/weight_transfer/transfer.py`. It provides two main functions:

- `transfer_weights(source_model, target_model)`: Upscale model's checkpoints including weights and buffers (like BatchNorm stats).
- `transfer_optimizer(source_optimizer, target_optimizer, source_model, target_model)`: Upscale optimizer states (e.g., momentum in SGD, `exp_avg` and `exp_avg_sq` in Adam)

Example usage:
```python
# create base model and load saved checkpoints
base_model = create_MLP_model(input_dim, hidden_dim, output_dim, num_layers, base_width=base_width)
base_model.load_state_dict(state_dict)

# create base optimizer and load saved checkpoints
base_optimizer = mup.MuSGD(base_model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
base_optimizer.load_state_dict(optimizer_state_dict)

# create widened model and transfer model's checkpoints
wide_model = create_MLP_model(input_dim, hidden_dim * multiplier, output_dim, num_layers, base_width=base_width, init_std=None, noise_std=noise_std) #initialized as injected noise
transfer_weights(base_model, wide_model)

# create optimizer for the widened model and transfer optimizer's states
wide_optimizer = mup.MuSGD(wide_model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
transfer_optimizer(base_optimizer, wide_optimizer, base_model, wide_model)

```
# Modification to mup and verification of equivalence

As described in Appendix C.4 of the paper, we implement slight modifications to the original mup package to ensure the widened model (without adding noise) is exactly equivalent to the base model.
This is implemented in `optim.py`, where the functions `MuAdam` and `MuAdamW` are modified from the corresponding functions in the mup package.

Equivalence is verified through pytests:
- `src/weight_transfer/MLP/test_weight_transfer.py`
- `src/weight_transfer/ResNet/test_weight_transfer.py`

# Verification of hyperparameter transfer

Hyperparameter transfer allows finding optimal hyperparameters (like learning rate) on a small model and applying them directly to a larger model.

For MLP on the ForestCoverType dataset, you can run:
```bash
python -m weight_transfer.MLP.check_hyperparam_transfer
```

# Upscaling experiments

We conducted upscaling experiments for three main architectures:
* For the MLP experiments using `[optimizer]` (chosen from SGD or AdamW), run
  ```bash
  python -m weight_transfer.MLP.main --dataset ForestCoverType --optimizer [optimizer]
  ```
* For the ResNet experiments using SGD, run
  ```bash
  python -m weight_transfer.ResNet.main --dataset CIFAR100 --optimizer SGD
  ```
* For the GPT-2 experiments using AdamW, run
  ```bash
  python src/weight_transfer/GPT2/train_gpt2.py --dataset fineweb_edu --learning-rate-exponent [lr_exp] --enable-mup
  ```
  Supports training and weight transfer for Transformer architectures using $\mu$P. For upscaling, use the `--up-scale-from` flag pointing to a saved checkpoint.