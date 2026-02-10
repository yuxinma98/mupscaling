"""
Plot coordinate check results for ResNet models in both SP and μP.
Expect that in SP, as width increases, the training dynamics diverge, while in μP they remain stable.
"""

import os
import numpy as np
from weight_transfer.ResNet.data import get_data_loaders_CNN
from weight_transfer.ResNet.models import create_ResNet
from weight_transfer import DATA_DIR
from mup.coord_check import get_coord_data, plot_coord_data

num_layers = 4
dataset = "CIFAR100"
batch_size = 200
lr = 1
plot_dir = "coord_check_plots/"

os.makedirs(plot_dir, exist_ok=True)
train_loader, test_loader, output_dim = get_data_loaders_CNN(dataset, batch_size=batch_size, data_dir=DATA_DIR)

# Standard Parameterization
models = {w: lambda w=w: create_ResNet(output_dim, wm=w, base_width=None).to("cuda") for w in 2 ** np.arange(-2.0, 2)}
df = get_coord_data(models, dataloader=train_loader, mup=False, optimizer="sgd", nsteps=3, nseeds=5)
plot_coord_data(df, legend=False, suptitle=f"ResNet_SGD_SP", face_color="xkcd:light grey", save_to=f"{plot_dir}ResNet_SGD_SP.png")

# muP parametrization
models = {w: lambda w=w: create_ResNet(output_dim, wm=w, base_width=1).to("cuda") for w in 2 ** np.arange(-2.0, 2)}
df = get_coord_data(models, dataloader=train_loader, mup=True, optimizer="sgd", nsteps=3, nseeds=5)
plot_coord_data(df, legend=False, suptitle=f"ResNet_SGD_MuP", face_color="xkcd:light grey", save_to=f"{plot_dir}ResNet_SGD_MuP.png")
