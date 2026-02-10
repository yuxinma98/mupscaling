"""
Modifications in mup:
replace the final linear layer with MuReadout for μP

Further modifications in our implementation:
1. add init_std and noise_std to control 2 types of initialization schemes
    init_std: initialize with standard initialization scaled by init_std for small model.
    noise_std: initialize as added noise for upscaled model. The upscale_reset_parameters function is added for this.
2. add helper function create_MLP_model function to create and initialize a model with mup
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mup import MuReadout, set_base_shapes

nonlin_dict = {"linear": torch.nn.Identity(), "relu": torch.nn.ReLU(), "sigmoid": torch.nn.Sigmoid()}

class MLP_muP(nn.Module):
    """
    Directly use the muP package from https://github.com/microsoft/mup to create MLP in μP parametrization.
    If init_std is provided, the model is initialized for standard training.
    If noise_std is provided, the model is initialized for upscaling (noise added to break the symmetry).
    Note: zero-initialize the output weights can improve HP transfer.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        input_mult=1.0,
        output_mult=1.0,
        init_std=1.0,
        noise_std=None,
        bias=True,
        readout_zero_init=True,
        nonlinearity="relu",
    ):
        super().__init__()
        layers = nn.ModuleList()
        self.nonlinearity = nonlin_dict[nonlinearity]
        self.input_mult = input_mult
        self.output_mult = output_mult
        # Input layer
        input_layer = nn.Linear(input_dim, hidden_dim, bias=bias)
        layers.append(input_layer)
        layers.append(self.nonlinearity)
        # Hidden layers
        for _ in range(num_layers - 2):
            hidden_layer = nn.Linear(hidden_dim, hidden_dim, bias=bias)
            layers.append(hidden_layer)
            layers.append(self.nonlinearity)
        # Output layer
        ### This is the only μP related change ###
        self.readout_zero_init = readout_zero_init
        readout = MuReadout(hidden_dim, output_dim, bias=bias, readout_zero_init=readout_zero_init, output_mult=output_mult)
        ###########################################
        layers.append(readout)
        self.model = nn.Sequential(*layers)

        ########## Our modification for upscaling ##############
        assert (
            init_std is None and noise_std is not None or init_std is not None and noise_std is None
        ), "Exactly one of init_std or noise_std should be provided."
        self.init_std = init_std
        self.noise_std = noise_std
        if noise_std is None:
            self.reset_parameters()
        else:
            self.upscale_reset_parameters()
        ########################################################

    def reset_parameters(self):
        """
        Change the default uniform initialization to Gaussian initialization, with std = init_std / sqrt(fan_in) for weight and std = init_std / sqrt(fan_in) for bias.
            (Notice that the default initialization is uniform(-sqrt(1/fan_in), sqrt(1/fan_in)) for weight and uniform(-sqrt(1/fan_in), sqrt(1/fan_in)) for bias.
            The std is roughly 1/sqrt(3*fan_in) for weight and 1/sqrt(3*fan_in) for bias.)
        This is standard parametrization. After calling mup.set_base_shapes, the initialization std will be adjusted for different widths.
        Use zero initialization for output layer (MuReadout).
        """
        for layer in self.model:
            if isinstance(layer, nn.Linear) and not isinstance(layer, MuReadout):
                nn.init.kaiming_normal_(layer.weight, a=1, mode="fan_in")  # Gaussian initialization with std 1/sqrt(fan_in)
                layer.weight.data *= self.init_std  # scale by initialization std
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    std = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.normal_(layer.bias, 0, std)
                    layer.bias.data *= self.init_std
            if isinstance(layer, MuReadout) and not self.readout_zero_init:
                nn.init.kaiming_normal_(layer.weight, a=1, mode="fan_in")  # Gaussian initialization with std 1/sqrt(fan_in)
                layer.weight.data *= self.init_std  # scale by initialization std
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # always zero initialize output layer bias for stability
        self.model[0].weight.data /= self.input_mult**0.5  # adjust input layer weight for input_mult

    ########## Our modification for upscaling ##############
    def upscale_reset_parameters(self):
        """
        Initialize the weights for upscaling (noise added to break the symmetry).
        """
        for layer in self.model:
            if (isinstance(layer, nn.Linear) and not isinstance(layer, MuReadout)) or (isinstance(layer, MuReadout) and not self.readout_zero_init):
                nn.init.kaiming_normal_(layer.weight, a=1, mode="fan_in")  # Gaussian initialization with std 1/sqrt(fan_in)
                layer.weight.data *= self.noise_std  # scale by noise std
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # zero initialization for bias

    ########################################################

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.model):
            if i == 0:
                out = layer(out) * self.input_mult**0.5  # adjust input layer
            else:
                out = layer(out)
        return out


def freeze_first_last_weights(MLP):
    for param in MLP.model[0].parameters():
        param.requires_grad = False
    for param in MLP.model[-1].parameters():
        param.requires_grad = False


def create_MLP_model(input_dim, hidden_dim, output_dim, num_layers, base_width=1, freeze_weights=False, **kwargs):
    """Create and initialize a model.
    If base_width is None, this is Standard Parameterization initialization.
    If base_width is provided, this is μP initialization with MLP of the base_width as the base model. Default: base_width=1.
    """
    assert base_width is None or isinstance(base_width, int), "base_width must be None or an integer."
    model = MLP_muP(input_dim, hidden_dim, output_dim, num_layers, **kwargs)
    if base_width is not None:
        base_model = MLP_muP(input_dim, base_width, output_dim, num_layers, **kwargs)
        delta_model = MLP_muP(input_dim, base_width+1, output_dim, num_layers, **kwargs) # create a model that differ in all widths dimmensions from the base_model
    else:
        base_model = None
        delta_model = None
    set_base_shapes(model, base_model, delta=delta_model)
    if freeze_weights:
        freeze_first_last_weights(model)
    return model


class MLP_muP_flexible(nn.Module):
    """
    Allow flexible hidden dimensions.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        input_mult=1.0,
        output_mult=1.0,
        init_std=1.0,
        noise_std=None,
        bias=True,
        nonlinearity="relu",
        readout_zero_init=True,
    ):
        super().__init__()
        layers = nn.ModuleList()
        self.nonlinearity = nonlin_dict[nonlinearity]
        self.input_mult = input_mult
        self.output_mult = output_mult
        # Input layer
        input_layer = nn.Linear(input_dim, hidden_dims[0], bias=bias)
        layers.append(input_layer)
        layers.append(self.nonlinearity)
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            hidden_layer = nn.Linear(hidden_dims[i - 1], hidden_dims[i], bias=bias)
            layers.append(hidden_layer)
            layers.append(self.nonlinearity)
        # Output layer
        ### This is the only μP related change ###
        self.readout_zero_init = readout_zero_init
        readout = MuReadout(
            hidden_dims[-1], output_dim, bias=bias, readout_zero_init=readout_zero_init, output_mult=output_mult
        )  # zero initialize output layer, stabilize training
        ###########################################
        layers.append(readout)
        self.model = nn.Sequential(*layers)

        ########## Our modification for upscaling ##############
        assert (
            init_std is None and noise_std is not None or init_std is not None and noise_std is None
        ), "Exactly one of init_std or noise_std should be provided."
        self.init_std = init_std
        self.noise_std = noise_std
        if noise_std is None:
            self.reset_parameters()
        else:
            self.upscale_reset_parameters()
        ########################################################

    def reset_parameters(self):
        """
        Change the default uniform initialization to Gaussian initialization, with std = init_std / sqrt(fan_in) for weight and std = init_std / sqrt(fan_in) for bias.
            (Notice that the default initialization is uniform(-sqrt(1/fan_in), sqrt(1/fan_in)) for weight and uniform(-sqrt(1/fan_in), sqrt(1/fan_in)) for bias.
            The std is roughly 1/sqrt(3*fan_in) for weight and 1/sqrt(3*fan_in) for bias.)
        This is standard parametrization. After calling mup.set_base_shapes, the initialization std will be adjusted for different widths.
        Use zero initialization for output layer (MuReadout).
        """
        for layer in self.model:
            if isinstance(layer, nn.Linear) and not isinstance(layer, MuReadout):
                nn.init.kaiming_normal_(layer.weight, a=1, mode="fan_in")  # Gaussian initialization with std 1/sqrt(fan_in)
                layer.weight.data *= self.init_std  # scale by initialization std
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    std = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.normal_(layer.bias, 0, std)
                    layer.bias.data *= self.init_std
            if isinstance(layer, MuReadout) and not self.readout_zero_init:
                nn.init.kaiming_normal_(layer.weight, a=1, mode="fan_in")  # Gaussian initialization with std 1/sqrt(fan_in)
                layer.weight.data *= self.init_std  # scale by initialization std
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # always zero initialize output layer bias for stability
        self.model[0].weight.data /= self.input_mult**0.5  # adjust input layer weight for input_mult

    ########## Our modification for upscaling ##############
    def upscale_reset_parameters(self):
        """
        Initialize the weights for upscaling (noise added to break the symmetry).
        """
        for layer in self.model:
            if (isinstance(layer, nn.Linear) and not isinstance(layer, MuReadout)) or (isinstance(layer, MuReadout) and not self.readout_zero_init):
                nn.init.kaiming_normal_(layer.weight, a=1, mode="fan_in")  # Gaussian initialization with std 1/sqrt(fan_in)
                layer.weight.data *= self.noise_std  # scale by noise std
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # zero initialization for bias

    ########################################################

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.model):
            if i == 0:
                out = layer(out) * self.input_mult**0.5  # adjust input layer
            else:
                out = layer(out)
        return out


def create_MLP_model_flexible(input_dim, hidden_dims, output_dim, base_width=1, **kwargs):
    """Create and initialize a model with mup"""
    assert base_width is None or isinstance(base_width, int), "base_width must be None or an integer."
    model = MLP_muP_flexible(input_dim, hidden_dims, output_dim, **kwargs)
    if base_width is not None:
        base_model = MLP_muP_flexible(input_dim, [base_width] * len(hidden_dims), output_dim, **kwargs)
        delta_model = MLP_muP_flexible(input_dim, [base_width + 1] * len(hidden_dims), output_dim, **kwargs) # create a model that differ in all widths dimmensions from the base_model
    else:
        base_model = None
        delta_model = None
    set_base_shapes(model, base_model, delta=delta_model)
    return model
