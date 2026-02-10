"""
Plot coordinate check results for MLP models in both Standard Parameterization (SP) and μP.
Expect that in SP, as width increases, the training dynamics diverge, while in μP they remain stable.
"""

import numpy as np
from mup.coord_check import get_coord_data, plot_coord_data
from weight_transfer.MLP.models import create_MLP_model
from weight_transfer import DATA_DIR
from weight_transfer.MLP.data import get_data_loaders_MLP
import os

num_layers = 4
dataset = "ForestCoverType"
batch_size = 200
lr = 1
plot_dir = "coord_check_plots/"

os.makedirs(plot_dir, exist_ok=True)
train_loader, test_loader, input_dim, output_dim = get_data_loaders_MLP(dataset, batch_size=batch_size, data_dir=DATA_DIR)

# Standard Parameterization
models = {
    w: lambda w=w: create_MLP_model(
        input_dim, w, output_dim, num_layers, bias=True, nonlinearity="relu", base_width=None
    ).to("cuda")
    for w in 2 ** np.arange(7, 14)
}
df = get_coord_data(models, dataloader=train_loader, mup=False, lr=lr, optimizer="sgd", flatten_input=True, nsteps=3, nseeds=5, lossfn="xent")
plot_coord_data(df, legend=True, suptitle=f"MLP_SGD_SP", face_color="xkcd:light grey", save_to=f"{plot_dir}MLP_SGD_SP.png")

# μP Parameterization
models = {
    w: lambda w=w: create_MLP_model(
        input_dim, w, output_dim, num_layers, bias=True, nonlinearity="relu", base_width=1
    ).to("cuda")
    for w in 2 ** np.arange(7, 14)
}
df = get_coord_data(models, dataloader=train_loader, mup=True, lr=lr, optimizer="sgd", flatten_input=True, nsteps=3, nseeds=5, lossfn="xent")
plot_coord_data(df, legend=True, suptitle=f"MLP_SGD_MuP", face_color="xkcd:light grey", save_to=f"{plot_dir}MLP_SGD_MuP.png")
