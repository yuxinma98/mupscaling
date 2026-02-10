import wandb
from weight_transfer.train import fix_seed
from weight_transfer.MLP.data import get_data_loaders_MLP
from weight_transfer.MLP.train import create_and_train_model, load_upscale_and_train_model
import torch
import argparse
import numpy as np
from weight_transfer import DATA_DIR, LOG_DIR, WANDB_ENTITY
import os, sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, choices=["SGD", "AdamW"], default="SGD", help="Optimizer to use")
    parser.add_argument("--read_wandb", action="store_true", help="Whether to read from wandb for the sweep")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.optimizer == "AdamW":
        config = {
            "dataset": "ForestCoverType",
            "multiplier": 2,
            "num_layers": 4,
            "batch_size": 2000,
            "epochs_bf": 100,
            "epochs_af": 100,
            "bias": True,
            "init_std": 1,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "base_width": 512,
            "n_range":  [128, 180, 256, 390, 512, 768],
        }
        rel_config = {"lr_bf": 0.001, "lr_af": 0.0005, "noise_std": [2**i for i in range(-9, 2)]}
        best_config = {"lr_bf": 0.002}
        sweep_id = "e2xl5tnq"
    elif args.optimizer == "SGD":
        config = {
            "dataset": "ForestCoverType",
            "multiplier": 2,
            "num_layers": 4,
            "batch_size": 2000,
            "epochs_bf": 200,
            "epochs_af": 200,
            "bias": True,
            "init_std": 1,
            "optimizer": "SGD",
            "weight_decay": 1e-4,
            "base_width": 1024,
            "n_range": [128, 256, 512, 1024],
        }
        rel_config = {
            "lr_bf": 0.2,
            "lr_af": 0.2,
            "noise_std": np.arange(0, 2.1, 0.125).tolist()
        }
        best_config = {"lr_bf": 0.6}
        sweep_id = "tidnjloi"

    # Check hyperparam transfer
    config["epochs"] = config["epochs_bf"]
    for n in [128, 256, 512, 1024, 2048]:
        config["hidden_dim"] = n
        for lr_exp in np.arange(-5, 6, 0.5):
            for seed in range(5):
                config["seed"] = seed
                config["lr"] = rel_config["lr_bf"] * (2.0 ** float(lr_exp))
                # mup
                wandb.init(
                    project="weight_transfer2",
                    config=config,
                    reinit="finish_previous",
                    name=f"n={n}_lr_exp={lr_exp}_seed{seed}",
                    group=f"hyparam_transfer",
                )
                fix_seed(config["seed"])
                train_loader, val_loader, input_dim, output_dim = get_data_loaders_MLP(config["dataset"], DATA_DIR, config["batch_size"])
                base_model = create_and_train_model(
                    input_dim,
                    output_dim,
                    config,
                    train_loader,
                    val_loader,
                    base_width=config["base_width"],
                )
                wandb.finish()

    # sweep for best_config
    if not args.read_wandb:
        config["hidden_dim"] = 100
        config["epochs"] = config["epochs_bf"]
        config["seed"] = 0
        config["lr"] = best_config["lr_bf"]
        wandb.init(
            project="weight_transfer2",
            config=config,
            reinit="finish_previous",
            name=f"basemodel",
            group=f"hyparam_transfer_sweep(not_t)",
        )
        fix_seed(config["seed"])
        train_loader, val_loader, input_dim, output_dim = get_data_loaders_MLP(config["dataset"], DATA_DIR, config["batch_size"])
        base_model = create_and_train_model(input_dim, output_dim, config, train_loader, val_loader, path=LOG_DIR, base_width=config["base_width"])
        base_model_state_dict = torch.load(f"{LOG_DIR}/{wandb.run.id}.pt", map_location=torch.device(device))
        prefix = "_orig_mod."
        base_model_state_dict = {k.removeprefix(prefix): v for k, v in base_model_state_dict.items()}
        optimizer_state_dict = torch.load(f"{LOG_DIR}/{wandb.run.id}_optimizer.pt", map_location=torch.device(device))
        os.remove(f"{LOG_DIR}/{wandb.run.id}.pt")
        os.remove(f"{LOG_DIR}/{wandb.run.id}_optimizer.pt")
        wandb.finish()

        def sweep_train():
            with wandb.init(group="hyparam_transfer_sweep(not_t)") as run:
                # Get hyperparameters from the sweep agent
                lr_exp = run.config.lr_exp
                noise_std = run.config.noise_std

                current_config = config.copy()
                current_config["epochs"] = config["epochs_af"]
                current_config["lr"] = rel_config["lr_af"] * (2.0 ** float(lr_exp))
                current_config["noise_std"] = noise_std
                run.config.update(current_config)

                fix_seed(current_config["seed"])
                train_loader, val_loader, input_dim, output_dim = get_data_loaders_MLP(config["dataset"], DATA_DIR, config["batch_size"])
                upscaled_model = load_upscale_and_train_model(
                    input_dim, output_dim, current_config, train_loader, val_loader, base_model_state_dict, optimizer_state_dict, base_width=config["base_width"]
                )

        sweep_config = {
            "method": "grid",
            "metric": {"name": "last_train_loss", "goal": "minimize"},
            "parameters": {"lr_exp": {"values": np.arange(-3, 3, 1).tolist()}, "noise_std": {"values": rel_config["noise_std"][::2]}},
        }

        sweep_id = wandb.sweep(sweep_config, project="weight_transfer2")
        wandb.agent(sweep_id, function=sweep_train)
        wandb.teardown()

    api = wandb.Api()
    sweep = api.sweep(f"{WANDB_ENTITY}/weight_transfer2/{sweep_id}")
    best_run = sweep.best_run()
    best_config["lr_af"] = best_run.config["lr"]
    best_config["noise_std"] = best_run.config["noise_std"]
    if best_config["noise_std"] == 0:
        print("best noise is 0")
        sys.exit(0)

    # Check hyperparam transfer in upscaling (fixed noise, vary lr_af)
    config["noise_std"] = best_config["noise_std"]
    for seed in range(5):
        config["seed"] = seed
        for n in config["n_range"]:
            config["hidden_dim"] = n
            # train base model
            config["epochs"] = config["epochs_bf"]
            config["lr"] = best_config["lr_bf"]
            wandb.init(
                project="weight_transfer2",
                config=config,
                reinit="finish_previous",
                name=f"basemodel_n={n}_seed={seed}",
                group=f"hyparam_transfer_upscale_fix_noise(not_t)",
            )
            fix_seed(config["seed"])
            train_loader, val_loader, input_dim, output_dim = get_data_loaders_MLP(config["dataset"], DATA_DIR, config["batch_size"])
            base_model = create_and_train_model(
                input_dim, output_dim, config, train_loader, val_loader, path=LOG_DIR, base_width=config["base_width"]
            )
            base_model_state_dict = torch.load(f"{LOG_DIR}/{wandb.run.id}.pt", map_location=torch.device(device))
            prefix = "_orig_mod."
            base_model_state_dict = {k.removeprefix(prefix): v for k, v in base_model_state_dict.items()}
            optimizer_state_dict = torch.load(f"{LOG_DIR}/{wandb.run.id}_optimizer.pt", map_location=torch.device(device))
            os.remove(f"{LOG_DIR}/{wandb.run.id}.pt")
            os.remove(f"{LOG_DIR}/{wandb.run.id}_optimizer.pt")
            wandb.finish()

            for lr_exp in np.arange(-5, 6, 0.5):
                # train upscaled model
                config["lr"] = rel_config["lr_af"] * (2.0 ** float(lr_exp))
                config["epochs"] = config["epochs_af"]
                wandb.init(
                    project="weight_transfer2",
                    config=config,
                    reinit="finish_previous",
                    name=f"upscale_n={n}_lr_exp={lr_exp}_seed={seed}",
                    group=f"hyparam_transfer_upscale_fix_noise(not_t)",
                )
                fix_seed(config["seed"])
                train_loader, val_loader, input_dim, output_dim = get_data_loaders_MLP(config["dataset"], DATA_DIR, config["batch_size"])
                upscaled_model = load_upscale_and_train_model(
                    input_dim,
                    output_dim,
                    config,
                    train_loader,
                    val_loader,
                    base_model_state_dict,
                    optimizer_state_dict,
                    base_width=config["base_width"],
                )
                wandb.finish()

    # Check hyperparam transfer in upscaling (fixed lr_af, vary noise)
    for seed in range(5):
        config["seed"] = seed
        for n in config["n_range"]:
            config["hidden_dim"] = n
            # train base model
            config["epochs"] = config["epochs_bf"]
            config["lr"] = best_config["lr_bf"]
            wandb.init(
                project="weight_transfer2",
                config=config,
                reinit="finish_previous",
                name=f"basemodel_n={n}_seed={seed}",
                group=f"hyparam_transfer_upscale_fix_lr(not_t)",
            )
            fix_seed(config["seed"])
            train_loader, val_loader, input_dim, output_dim = get_data_loaders_MLP(config["dataset"], DATA_DIR, config["batch_size"])
            base_model = create_and_train_model(
                input_dim, output_dim, config, train_loader, val_loader, path=LOG_DIR, base_width=config["base_width"]
            )
            base_model_state_dict = torch.load(f"{LOG_DIR}/{wandb.run.id}.pt", map_location=torch.device(device))
            prefix = "_orig_mod."
            base_model_state_dict = {k.removeprefix(prefix): v for k, v in base_model_state_dict.items()}
            optimizer_state_dict = torch.load(f"{LOG_DIR}/{wandb.run.id}_optimizer.pt", map_location=torch.device(device))
            os.remove(f"{LOG_DIR}/{wandb.run.id}.pt")
            os.remove(f"{LOG_DIR}/{wandb.run.id}_optimizer.pt")
            wandb.finish()

            for noise_std in rel_config["noise_std"]:
                # train upscaled model
                config["epochs"] = config["epochs_af"]
                config["noise_std"] = noise_std
                config["lr"] = best_config["lr_af"]
                wandb.init(
                    project="weight_transfer2",
                    config=config,
                    reinit="finish_previous",
                    name=f"upscale_n={n}_noise={noise_std}_seed={seed}",
                    group=f"hyparam_transfer_upscale_fix_lr(not_t)",
                )
                fix_seed(config["seed"])
                train_loader, val_loader, input_dim, output_dim = get_data_loaders_MLP(config["dataset"], DATA_DIR, config["batch_size"])
                upscaled_model = load_upscale_and_train_model(
                    input_dim,
                    output_dim,
                    config,
                    train_loader,
                    val_loader,
                    base_model_state_dict,
                    optimizer_state_dict,
                    base_width=config["base_width"],
                )
                wandb.finish()