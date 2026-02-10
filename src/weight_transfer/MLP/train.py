from weight_transfer.transfer import transfer_weights, transfer_weights_sp, transfer_optimizer
from weight_transfer.MLP.data import get_data_loaders_MLP
from weight_transfer.MLP.models import create_MLP_model, freeze_first_last_weights
from weight_transfer.train import train
import weight_transfer.optim as replaced_optim
from torch.optim import SGD, AdamW
from weight_transfer import DATA_DIR
import mup
import torch
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
task_dict = {
    "ForestCoverType": "classification",
}


def get_optimizer(model, optim, lr, momentum=0, wd=0, sp=True):
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


def setup_transfer_models(
    input_dim,
    output_dim,
    config,
    state_dict,
    base_width=1
):
    """
    Helper function to load state_dict into a narrow model, create a wide model by upscaling and add noise.
    """
    # Create and load narrow model
    narrow_model = create_MLP_model(
        input_dim,
        config["hidden_dim"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    narrow_model.load_state_dict(state_dict)
    narrow_model.to(torch.device(device))

    # Create and initialize wide model
    wide_model = create_MLP_model(
        input_dim,
        config["hidden_dim"] * config["multiplier"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        init_std=None,
        noise_std=config["noise_std"],
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    wide_model.to(torch.device(device))
    
    if config.get("freeze_weights", False): # if freeze first and last layer, do not add noise to those layers
        for param in wide_model.model[0].parameters():
            param.data.zero_()
        for param in wide_model.model[-1].parameters():
            param.data.zero_()

    # Transfer weights
    if base_width is None:
        transfer_weights_sp(narrow_model, wide_model)
    else:
        transfer_weights(narrow_model, wide_model)

    return narrow_model, wide_model


def setup_transfer_optimizers(optimizer, lr, momentum, weight_decay, optimizer_state_dict, narrow_model, wide_model, sp=False):
    """
    Helper function to transfer optimizer state from narrow model to wide model.
    """
    # Create optimizers
    narrow_optimizer = get_optimizer(narrow_model, optimizer, lr, momentum, weight_decay, sp)
    wide_optimizer = get_optimizer(wide_model, optimizer, lr, momentum, weight_decay, sp)

    if not sp:
        # Load state dict into narrow optimizer
        narrow_optimizer.load_state_dict(optimizer_state_dict)
        # Transfer optimizer state
        transfer_optimizer(narrow_optimizer, wide_optimizer, narrow_model, wide_model)

    return wide_optimizer


def setup_interpolation_models(input_dim, output_dim, config, state_dict, base_width=1):
    """
    Helper function to load state_dict into a narrow model, create a wide model by upscaling, and interpolate with noise.
    i.e. wide_model = sqrt(1-t) * wide_model_transfer + sqrt(t) * noise,
    wide_model_transfer is obtained by upscaling from narrow model to wide model,
    noise is scaled so that its norm matches that of wide_model_transfer.
    This ensure that wide_model has the same norm as wide_model_transfer.
    """
    # Create and load narrow model
    narrow_model = create_MLP_model(
        input_dim,
        config["hidden_dim"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    narrow_model.load_state_dict(state_dict)
    narrow_model.to(torch.device(device))

    # Create and initialize wide model
    wide_model_noise = create_MLP_model(
        input_dim,
        config["hidden_dim"] * config["multiplier"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        init_std=None,
        noise_std=1,
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    wide_model_noise.to(torch.device(device))
    
    if config.get("freeze_weights", False): # if freeze first and last layer, do not add noise to those layers
        for param in wide_model_noise.model[0].parameters():
            param.data.zero_()
        for param in wide_model_noise.model[-1].parameters():
            param.data.zero_()
            
    wide_model_transfer = create_MLP_model(
        input_dim,
        config["hidden_dim"] * config["multiplier"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        init_std=None,
        noise_std=1,
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    wide_model_transfer.to(torch.device(device))
    for param in wide_model_transfer.parameters():
        param.data.zero_()
    if base_width is None:
        transfer_weights_sp(narrow_model, wide_model_transfer)
    else:
        transfer_weights(narrow_model, wide_model_transfer)

    wide_model = create_MLP_model(
        input_dim,
        config["hidden_dim"] * config["multiplier"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        init_std=None,
        noise_std=1,
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    wide_model.to(torch.device(device))
    for param, param_transfer, param_noise in zip(wide_model.parameters(), wide_model_transfer.parameters(), wide_model_noise.parameters()):
        if torch.allclose(param_noise.data, torch.zeros_like(param_noise.data)):
            param.data = param_transfer.data
        else:
            noise = (
                param_noise.data / torch.linalg.norm(param_noise.data, ord=2) * torch.linalg.norm(param_transfer.data, ord=2)
            )  # scale noise to match (spectral) norm of transferred weights
            param.data = math.sqrt(1 - config["t"]) * param_transfer.data + math.sqrt(config["t"]) * noise

    return narrow_model, wide_model


def setup_transfer_models_normalized_hadamard(
    input_dim,
    output_dim,
    config,
    state_dict,
    base_width=1
):
    """
    Helper function to load state_dict into a narrow model, create a wide model by upscaling, and add normalized noise.
    i.e. wide_model = wide_model_transfer + t * (noise with sigma=1) \odot |W_transfer|
    """
    # Create and load narrow model
    narrow_model = create_MLP_model(
        input_dim,
        config["hidden_dim"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    narrow_model.load_state_dict(state_dict)
    narrow_model.to(torch.device(device))

    # Create and initialize wide model
    wide_model_noise = create_MLP_model(
        input_dim,
        config["hidden_dim"] * config["multiplier"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        init_std=None,
        noise_std=1,
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    wide_model_noise.to(torch.device(device))
    if config.get("freeze_weights", False): # if freeze first and last layer, do not add noise to those layers
        for param in wide_model_noise.model[0].parameters():
            param.data.zero_()
        for param in wide_model_noise.model[-1].parameters():
            param.data.zero_()
            
    wide_model_transfer = create_MLP_model(
        input_dim,
        config["hidden_dim"] * config["multiplier"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        init_std=None,
        noise_std=1,
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    wide_model_transfer.to(torch.device(device))
    for param in wide_model_transfer.parameters():
        param.data.zero_()
    if base_width is None:
        transfer_weights_sp(narrow_model, wide_model_transfer)
    else:
        transfer_weights(narrow_model, wide_model_transfer)

    wide_model = create_MLP_model(
        input_dim,
        config["hidden_dim"] * config["multiplier"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        init_std=None,
        noise_std=1,
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    wide_model.to(torch.device(device))
    for param, param_transfer, param_noise in zip(wide_model.parameters(), wide_model_transfer.parameters(), wide_model_noise.parameters()):
        if torch.allclose(param_noise.data, torch.zeros_like(param_noise.data)):  # not adding noise to some parameters
            param.data = param_transfer.data
        else:
            param.data = param_transfer.data + config["t"] * torch.normal(0, 1, param_transfer.data.shape).to(param_transfer.data.device) * torch.abs(param_transfer.data)

    return narrow_model, wide_model


def setup_transfer_models_normalized_spectral(
    input_dim, output_dim, config, state_dict, base_width=1, 
):
    """
    Helper function to load state_dict into a narrow model, create a wide model by upscaling, and add normalized noise.
    i.e. wide_model = wide_model_transfer + t * (noise with sigma=1) / norm of noise * norm of W_transfer
    """
    # Create and load narrow model
    narrow_model = create_MLP_model(
        input_dim,
        config["hidden_dim"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    narrow_model.load_state_dict(state_dict)
    narrow_model.to(torch.device(device))

    # Create and initialize wide model
    wide_model_noise = create_MLP_model(
        input_dim,
        config["hidden_dim"] * config["multiplier"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        init_std=None,
        noise_std=1,
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    wide_model_noise.to(torch.device(device))
    if config.get("freeze_weights", False): # if freeze first and last layer, do not add noise to those layers
        for param in wide_model_noise.model[0].parameters():
            param.data.zero_()
        for param in wide_model_noise.model[-1].parameters():
            param.data.zero_()
            
    wide_model_transfer = create_MLP_model(
        input_dim,
        config["hidden_dim"] * config["multiplier"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        init_std=None,
        noise_std=1,
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    wide_model_transfer.to(torch.device(device))
    for param in wide_model_transfer.parameters():
        param.data.zero_()
    if base_width is None:
        transfer_weights_sp(narrow_model, wide_model_transfer)
    else:
        transfer_weights(narrow_model, wide_model_transfer)

    wide_model = create_MLP_model(
        input_dim,
        config["hidden_dim"] * config["multiplier"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        init_std=None,
        noise_std=1,
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    wide_model.to(torch.device(device))
    for param, param_transfer, param_noise in zip(wide_model.parameters(), wide_model_transfer.parameters(), wide_model_noise.parameters()):
        if torch.allclose(param_noise.data, torch.zeros_like(param_noise.data)):
            param.data = param_transfer.data
        else:
            noise = (
                param_noise.data / torch.linalg.norm(param_noise.data, ord=2) * torch.linalg.norm(param_transfer.data, ord=2)
            )  # scale noise to match (spectral) norm of transferred weights
            param.data = param_transfer.data + config["t"] * noise

    return narrow_model, wide_model


def create_and_train_model(
    input_dim, output_dim, config, train_loader, val_loader, stop_loss=None, path=None, log_weight_norms_flag=False, base_width=1, **kwargs
):
    """
    Helper function to create, initialize and train a model from scratch.
    """
    model = create_MLP_model(
        input_dim,
        config["hidden_dim"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity=config.get("nonlinearity", "relu"),
        init_std=config["init_std"],
        noise_std=None,
        base_width=base_width,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    model.to(torch.device(device))
    if base_width is None:
        sp = True
    else:
        sp = False
    optimizer = get_optimizer(model, config.get("optimizer"), config.get("lr"), config.get("momentum", 0), config.get("weight_decay", 0), sp)
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
    )

    return trained_model


def load_and_train_model(input_dim, output_dim, config, train_loader, val_loader, state_dict, optimizer_state_dict, stop_loss=None, path=None):
    """
    Helper function to load a model from state dict and continue training.
    """
    model = create_MLP_model(
        input_dim,
        config["hidden_dim"],
        output_dim,
        config["num_layers"],
        bias=config["bias"],
        nonlinearity="relu",
        init_std=1.0,
        noise_std=None,
        base_width=1,
        freeze_weights=config.get("freeze_weights", False),
        readout_zero_init=config.get("readout_zero_init", True),
    )
    model.load_state_dict(state_dict)
    model.to(torch.device(device))
    optimizer = get_optimizer(model, config["optimizer"], config["lr"], config.get("momentum", 0), config.get("weight_decay",0), sp=False)
    optimizer.load_state_dict(optimizer_state_dict)

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
    )

    return trained_model


def load_upscale_and_train_model(
    input_dim, output_dim, config, train_loader, val_loader, state_dict, optimizer_state_dict, stop_loss=None, path=None, base_width=1
):
    """
    Helper function to load a narrow model from state dict, create a wide model by upscaling, transfer weights, and continue training.
    """
    narrow_model, wide_model = setup_transfer_models(
        input_dim,
        output_dim,
        config,
        state_dict,
        base_width=base_width
    )
    wide_optimizer = setup_transfer_optimizers(
        config.get("optimizer"), config["lr"], config.get("momentum",0), config.get("weight_decay",0), optimizer_state_dict, narrow_model, wide_model, sp=True if base_width is None else False,
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
    )

    return trained_model


def load_upscale_interpolation_and_train_model(
    input_dim, output_dim, config, train_loader, val_loader, state_dict, optimizer_state_dict, stop_loss=None, path=None, base_width=1
):
    """
    Helper function to load a narrow model from state dict, create a wide model by upscaling, transfer weights, and continue training.
    """
    narrow_model, wide_model = setup_interpolation_models(
        input_dim,
        output_dim,
        config,
        state_dict,
        base_width=base_width
    )
    wide_optimizer = setup_transfer_optimizers(
        config.get("optimizer"), config["lr"], config.get("momentum",0), config.get("weight_decay",0), optimizer_state_dict, narrow_model, wide_model, sp=True if base_width is None else False,
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
    )

    return trained_model


def load_upscale_and_train_model_normalized_noise_hadamard(
    input_dim, output_dim, config, train_loader, val_loader, state_dict, optimizer_state_dict, stop_loss=None, path=None, base_width=1
):
    """
    Helper function to load a narrow model from state dict, create a wide model by upscaling, transfer weights, and continue training.
    Here the noise is normalized wrt the transferred weights. i.e. noise <- noise \odot |W_transfer|
    """
    narrow_model, wide_model = setup_transfer_models_normalized_hadamard(
        input_dim,
        output_dim,
        config,
        state_dict,
        base_width=base_width
    )
    wide_optimizer = setup_transfer_optimizers(
        config.get("optimizer"), config["lr"], config.get("momentum",0), config.get("weight_decay",0), optimizer_state_dict, narrow_model, wide_model, sp=True if base_width is None else False,
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
    )

    return trained_model


def load_upscale_and_train_model_normalized_noise_spectral(
    input_dim, output_dim, config, train_loader, val_loader, state_dict, optimizer_state_dict, stop_loss=None, path=None, base_width=1
):
    """
    Helper function to load a narrow model from state dict, create a wide model by upscaling, transfer weights, and continue training.
    Here the noise is normalized wrt the transferred weights. i.e. noise <- noise * \|W_transfer|\
    """
    narrow_model, wide_model = setup_transfer_models_normalized_spectral(
        input_dim,
        output_dim,
        config,
        state_dict,
        base_width,
    )
    wide_optimizer = setup_transfer_optimizers(
        config["optimizer"],
        config["lr"],
        config.get("momentum",0),
        config.get("weight_decay",0),
        optimizer_state_dict,
        narrow_model,
        wide_model,
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
    )

    return trained_model
