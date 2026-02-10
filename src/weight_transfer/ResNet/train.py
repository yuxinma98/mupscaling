from weight_transfer.transfer import transfer_weights, transfer_weights_sp, transfer_optimizer
from weight_transfer.ResNet.data import get_data_loaders_CNN
from weight_transfer.ResNet.models import create_ResNet
from weight_transfer.train import train
import weight_transfer.optim as replaced_optim
import torch
import mup
from torch.optim import SGD, AdamW
from weight_transfer import DATA_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
task_dict = {
    "CIFAR100": "classification",
}

def get_optimizer(model, optim, lr, wd=0, momentum=0, sp=True):
    """
    Initializes the optimizer based on the configuration and parameterization type.

    Args:
        model: The torch.nn.Module to optimize.
        config (dict): Configuration containing 'optimizer', 'lr', and 'weight_decay'.
        sp (bool): If True, use standard parameterization. If False, use MuP.
    """
    params = model.parameters()

    if sp:
        if optim == "SGD":
            return SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
        elif optim == "AdamW":
            return AdamW(params, lr=lr, weight_decay=wd)
    else:
        # MuP specific optimizers
        if optim == "SGD":
            return mup.MuSGD(params, lr=lr, weight_decay=wd, momentum=momentum)
        elif optim == "AdamW":
            return replaced_optim.MuAdamW(params, lr=lr, weight_decay=wd)

    raise ValueError(f"Optimizer {optim} not supported or 'sp' flag mismatch.")

def train_and_expand_CNN(config, log_weight_norms_flag=False):
    # prepare data
    train_loader, val_loader, output_dim = get_data_loaders_CNN(config["dataset"], DATA_DIR, config["batch_size"])
    # model with mup initialization
    model_small = create_ResNet(output_dim=output_dim, wm=config["hidden_dim"], init_std=1.0, base_width=1)
    model_large = create_ResNet(output_dim=output_dim, wm=config["multiplier"], init_std=config["noise_std"], base_width=1)

    # optimizers
    if config.get("optimizer") == "SGD":
        optimizer_small = mup.MuSGD(model_small.parameters(), lr=config["lr_bf"], weight_decay=config["weight_decay"], momentum=config["momentum"])
        optimizer_large = mup.MuSGD(model_large.parameters(), lr=config["lr_af"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    else:
        optimizer_small = replaced_optim.MuAdamW(model_small.parameters(), lr=config["lr_bf"], weight_decay=config["weight_decay"])
        optimizer_large = replaced_optim.MuAdamW(model_large.parameters(), lr=config["lr_af"], weight_decay=config["weight_decay"])

    # Train small model, expand, and continue training the large model
    model_small = train(
        model_small,
        train_loader,
        val_loader,
        optimizer_small,
        epochs=config["epoch_bf"],
        start_epoch=1,
        task=task_dict[config["dataset"]],
        device=device,
        log_weight_norms_flag=log_weight_norms_flag,
        lr_scheduler=config.get("lr_scheduler", False),
        lr_end_factor=config.get("lr_end_factor", 0.1),
        lr_total_iters=config.get("lr_total_iters", 500),
    )
    model_large = model_large.to(device)
    transfer_weights(model_small, model_large)
    transfer_optimizer(optimizer_small, optimizer_large, model_small, model_large)
    model_large = train(
        model_large,
        train_loader,
        val_loader,
        optimizer_large,
        epochs=config["epoch_af"],
        start_epoch=config["epoch_bf"] + 2,
        task=task_dict[config["dataset"]],
        device=device,
        log_weight_norms_flag=log_weight_norms_flag,
        lr_scheduler=config.get("lr_scheduler", False),
        lr_end_factor=config.get("lr_end_factor", 0.1),
        lr_total_iters=config.get("lr_total_iters", 500),
    )


def train_with_different_lr_CNN(config):
    train_loader, val_loader, output_dim = get_data_loaders_CNN(config["dataset"], DATA_DIR, config["batch_size"])
    model = create_ResNet(output_dim=output_dim, wm=config["hidden_dim"], init_std=1.0, base_width=1)
    
    if config.get("optimizer") == "SGD":
        optimizer_bf = mup.MuSGD(model.parameters(), lr=config["lr_bf"], weight_decay=config["weight_decay"], momentum=config["momentum"])
        optimizer_af = mup.MuSGD(model.parameters(), lr=config["lr_af"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    else:
        optimizer_bf = replaced_optim.MuAdamW(model.parameters(), lr=config["lr_bf"], weight_decay=config["weight_decay"])
        optimizer_af = replaced_optim.MuAdamW(model.parameters(), lr=config["lr_af"], weight_decay=config["weight_decay"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train(
        model,
        train_loader,
        val_loader,
        optimizer_bf,
        epochs=config["epoch_bf"],
        device=device,
        start_epoch=1,
        task=task_dict[config["dataset"]],
        lr_scheduler=config.get("lr_scheduler",False),
        lr_end_factor=config.get("lr_end_factor", 0.1),
        lr_total_iters=config.get("lr_total_iters",500),
    )
    transfer_optimizer(optimizer_bf, optimizer_af, model, model)
    model = train(
        model,
        train_loader,
        val_loader,
        optimizer_af,
        epochs=config["epoch_af"],
        device=device,
        start_epoch=config["epoch_bf"] + 2,
        task=task_dict[config["dataset"]],
        lr_scheduler=config.get("lr_scheduler",False),
        lr_end_factor=config.get("lr_end_factor", 0.1),
        lr_total_iters=config.get("lr_total_iters",500),
    )


def create_and_train_model(output_dim, config, train_loader, val_loader, stop_loss=None, path=None, log_weight_norms_flag=False, base_width=1):
    """
    Helper function to create, initialize and train a model from scratch.
    """
    model = create_ResNet(output_dim, wm=config["hidden_dim"], init_std=config["init_std"], base_width=base_width)
    model.to(torch.device(device))
    if base_width is None:
        sp = True
    else:
        sp = False
        
    optimizer = get_optimizer(model, config.get("optimizer"), config.get("lr"), config.get("weight_decay"), config["momentum"], sp)
    
    trained_model = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=config["epochs"],
        stop_loss=stop_loss,
        device=device,
        start_epoch=1,
        task=task_dict[config["dataset"]],
        path=path,
        log_weight_norms_flag=log_weight_norms_flag,
        lr_scheduler=config.get("lr_scheduler",False),
        lr_end_factor=config.get("lr_end_factor", 0.1),
        lr_total_iters=config.get("lr_total_iters",500),
    )

    return trained_model


def setup_transfer_models(output_dim, narrow_hidden_dim, wide_hidden_dim, state_dict, noise_std, base_width=1):
    """
    Helper function to create, initialize, and transfer weights from a narrow model to a wide model.
    """
    # Create and load narrow model
    narrow_model = create_ResNet(output_dim, wm=narrow_hidden_dim, init_std=1.0, base_width=base_width)
    narrow_model.load_state_dict(state_dict)
    narrow_model.to(torch.device(device))

    # Create and initialize wide model
    wide_model = create_ResNet(output_dim, wm=wide_hidden_dim, init_std=noise_std, base_width=base_width)
    wide_model.to(torch.device(device))

    # Transfer weights
    transfer_weights(narrow_model, wide_model)

    return narrow_model, wide_model


def setup_transfer_models_normalized_spectral(output_dim, narrow_hidden_dim, wide_hidden_dim, state_dict, t, base_width=1):
    """
    Helper function to create, initialize, and transfer weights from a narrow model to a wide model.
    """
    # Create and load narrow model
    narrow_model = create_ResNet(output_dim, wm=narrow_hidden_dim, init_std=1.0, base_width=base_width)
    narrow_model.load_state_dict(state_dict)
    narrow_model.to(torch.device(device))

    # Create and initialize wide model
    wide_model_noise = create_ResNet(output_dim, wm=wide_hidden_dim, init_std=1, base_width=base_width)
    wide_model_noise.to(torch.device(device))
    wide_model_transfer = create_ResNet(output_dim, wm=wide_hidden_dim, init_std=1, base_width=base_width)
    wide_model_transfer.to(torch.device(device))
    for param in wide_model_transfer.parameters():
        param.data.zero_()

    # Transfer weights
    if base_width is None:
        transfer_weights_sp(narrow_model, wide_model_transfer)
    else:
        transfer_weights(narrow_model, wide_model_transfer)
    wide_model = create_ResNet(output_dim, wm=wide_hidden_dim, init_std=1, base_width=base_width)
    wide_model.to(torch.device(device))
    for param, param_transfer, param_noise in zip(wide_model.parameters(), wide_model_transfer.parameters(), wide_model_noise.parameters()):
        if torch.allclose(param_noise.data, torch.zeros_like(param_noise.data)):
            param.data = param_transfer.data
        else:  # scale noise to match (spectral) norm of transferred weights
            if len(param.data.shape) == 1:
                noise = param_noise.data / torch.linalg.vector_norm(param_noise.data, ord=2) * torch.linalg.vector_norm(param_transfer.data, ord=2)
            else:
                noise = (
                    param_noise.data
                    / torch.linalg.norm(param_noise.data, ord=2, dim=(0, 1))
                    * torch.linalg.norm(param_transfer.data, ord=2, dim=(0, 1))
                )
            param.data = param_transfer.data + t * noise

    return narrow_model, wide_model


def setup_transfer_optimizers(optimizer, lr, weight_decay, optimizer_state_dict, narrow_model, wide_model, momentum=0.0, sp=False):
    """
    Helper function to transfer optimizer state from narrow model to wide model.
    """
    # Create optimizers
    narrow_optimizer = get_optimizer(narrow_model, optimizer, lr, weight_decay, momentum, sp)
    wide_optimizer = get_optimizer(wide_model, optimizer, lr, weight_decay, momentum, sp)

    # Load state dict into narrow optimizer
    if not sp:
        narrow_optimizer.load_state_dict(optimizer_state_dict)
        # Transfer optimizer state
        transfer_optimizer(narrow_optimizer, wide_optimizer, narrow_model, wide_model)

    return wide_optimizer


def load_upscale_and_train_model(output_dim, config, train_loader, val_loader, state_dict, optimizer_state_dict, stop_loss=None, path=None, base_width=1):
    """
    Helper function to load a narrow model from state dict, create a wide model by upscaling, transfer weights, and continue training.
    """
    narrow_model, wide_model = setup_transfer_models(
        output_dim, config["hidden_dim"], config["hidden_dim"] * config["multiplier"], state_dict, config["noise_std"], base_width
    )
    wide_optimizer = setup_transfer_optimizers(
        config["optimizer"], config["lr"], config["weight_decay"], optimizer_state_dict, narrow_model, wide_model, config["momentum"], 
        sp=True if base_width is None else False,
    )

    trained_model = train(
        wide_model,
        train_loader,
        val_loader,
        wide_optimizer,
        epochs=config["epochs"],
        stop_loss=stop_loss,
        device=device,
        start_epoch=1,
        task=task_dict[config["dataset"]],
        path=path,
        lr_scheduler=False,
    )

    return trained_model


def load_upscale_and_train_model_normalized_noise_spectral(
    output_dim, config, train_loader, val_loader, state_dict, optimizer_state_dict, stop_loss=None, path=None, base_width=1
):
    """
    Helper function to load a narrow model from state dict, create a wide model by upscaling, transfer weights, and continue training.
    Here the noise is normalized wrt the transferred weights. i.e. noise <- noise * \|W_transfer|\
    """
    narrow_model, wide_model = setup_transfer_models_normalized_spectral(
        output_dim, config["hidden_dim"], config["hidden_dim"] * config["multiplier"], state_dict, config["t"], base_width
    )
    wide_optimizer = setup_transfer_optimizers(
        config["optimizer"],
        config["lr"],
        config["weight_decay"],
        optimizer_state_dict,
        narrow_model,
        wide_model,
        config["momentum"],
        sp=True if base_width is None else False,
    )

    trained_model = train(
        wide_model,
        train_loader,
        val_loader,
        wide_optimizer,
        epochs=config["epochs"],
        stop_loss=stop_loss,
        device=device,
        start_epoch=1,
        task=task_dict[config["dataset"]],
        path=path,
        lr_scheduler=config.get("lr_scheduler",False),
        lr_end_factor=config.get("lr_end_factor", 0.1),
        lr_total_iters=config.get("lr_total_iters",500),
    )

    return trained_model
