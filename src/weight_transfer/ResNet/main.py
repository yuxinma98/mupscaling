import argparse
import wandb
import torch
import math
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from weight_transfer import DATA_DIR, LOG_DIR, WANDB_ENTITY
from weight_transfer.train import fix_seed
from weight_transfer.ResNet.data import get_data_loaders_CNN
from weight_transfer.ResNet.dataset_configs import get_dataset_config
from weight_transfer.ResNet.train import create_and_train_model, load_upscale_and_train_model

# Enable TF32 for better performance on compatible GPUs (new API)
try:
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cuda.matmul.fp32_precision = "tf32"
except AttributeError:
    # Fallback to old API for older PyTorch versions
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def sweep_train_epochs(config=None):
    with wandb.init(config=config):
        cfg = wandb.config
        fix_seed(cfg.seed)
        train_loader, val_loader, output_dim = get_data_loaders_CNN(dataset, DATA_DIR, dataset_config["batch_size"])
        model = create_and_train_model(output_dim, dict(cfg), train_loader, val_loader, path=LOG_DIR)


def sweep_train_upscaled_epochs(config=None):
    with wandb.init(config=config):
        cfg = wandb.config
        fix_seed(cfg.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(f"{LOG_DIR}/{cfg.model}.pt", map_location=torch.device(device))
        prefix = "_orig_mod."
        state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        optimizer_state_dict = torch.load(f"{LOG_DIR}/{cfg.model}_optimizer.pt", map_location=torch.device(device))
        # model and per-layer learning rate
        train_loader, val_loader, output_dim = get_data_loaders_CNN(dataset, DATA_DIR, dataset_config["batch_size"])
        model = load_upscale_and_train_model(output_dim, dict(cfg), train_loader, val_loader, state_dict, optimizer_state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["CIFAR100"], required=True, help="Dataset to use")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "AdamW"], required=False, default="SGD", help="Optimizer to use")
    parser.add_argument("--read_wandb", action="store_true", help="Whether to read sweep/run info directly from wandb")
    args = parser.parse_args()

    small_model_best_lr_sweep_id = {"SGD": "ph7ofouv"}

    dataset = args.dataset
    dataset_config = get_dataset_config((dataset, args.optimizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sweep 1: Find best lr for small model
    sweep_config = {
        "name": "small_model_best_lr",
        "method": "grid",
        "parameters": {
            "dataset": {"values": [dataset]},
            "hidden_dim": {"values": [dataset_config["n"]]},
            "batch_size": {"values": [dataset_config["batch_size"]]},
            "epochs": {"values": [dataset_config["epochs"]]},
            "lr": {"values": [dataset_config["lr_bf"] * 2**k for k in range(0, 3)]},
            "init_std": {"values": [1]},
            "seed": {"values": [dataset_config["seed"]]},
            "weight_decay": {"values": [dataset_config["weight_decay"]]},
            "optimizer": {"values": [args.optimizer]},
            "momentum": {"values": [dataset_config.get("momentum", 0.0)]},
            "lr_scheduler": {"values": [dataset_config.get("lr_scheduler")]},
            "lr_end_factor": {"values": [dataset_config.get("lr_end_factor")]},
            "lr_total_iters": {"values": [dataset_config.get("lr_total_iters")]},
        },
        "metric": {"goal": "minimize", "name": "epoch/train_loss"},
    }
    if args.read_wandb and args.optimizer in small_model_best_lr_sweep_id:
        sweep_id = small_model_best_lr_sweep_id[args.optimizer]
    else:
        sweep_id = wandb.sweep(sweep_config, project="weight_transfer2")
        wandb.agent(sweep_id, function=sweep_train_epochs)
    api = wandb.Api()
    sweep = api.sweep(f"{WANDB_ENTITY}/weight_transfer2/{sweep_id}")
    runs = [run for run in sweep.runs if run.summary.get("min_train_loss") is not None and not math.isnan(float(run.summary.get("min_train_loss")))]
    runs = sorted(runs, key=lambda run: float(run.summary.get("min_train_loss")), reverse=False)
    best_run = runs[0]
    best_lr = best_run.config["lr"]
    print(f"Best learning rate: {best_lr}.")
    wandb.teardown()

    # Train base model
    config = dict(best_run.config)
    config["hidden_dim"] = dataset_config["N"]
    run_id = False
    if args.read_wandb:
        runs = api.runs(
            path=f"{WANDB_ENTITY}/weight_transfer2",
            filters={"group": f"base_model_{best_run.id}"},
        )
        if len(runs) > 0:
            run_id = sorted(runs, key=lambda run: float(run.summary.get("epoch/val_loss", {}).get("min", float("inf"))))[len(runs) // 2].id
    if not run_id:
        val_losses = {}
        for seed in range(1):
            config["seed"] = seed
            wandb.init(
                project="weight_transfer2", config=config, reinit="finish_previous", name=f"base_model_seed_{seed}", group=f"base_model_{best_run.id}"
            )
            fix_seed(config["seed"])
            train_loader, val_loader, output_dim = get_data_loaders_CNN(dataset, DATA_DIR, dataset_config["batch_size"])
            base_model = create_and_train_model(
                output_dim,
                config,
                train_loader,
                val_loader,
                path=LOG_DIR,
            )
            val_loss = wandb.run.summary.get("epoch/val_loss").get("min")
            if val_loss is not None:
                val_losses[wandb.run.id] = val_loss
            wandb.finish()
    runs = api.runs(f"{WANDB_ENTITY}/weight_transfer2", {"group": f"base_model_{best_run.id}"})
    run_id = sorted(runs, key=lambda run: float(run.summary.get("epoch/val_loss", {}).get("min", float("inf"))))[len(runs) // 2].id
    base_model_state_dict = torch.load(f"{LOG_DIR}/{run_id}.pt", map_location=torch.device(device))
    prefix = "_orig_mod."
    base_model_state_dict = {k.removeprefix(prefix): v for k, v in base_model_state_dict.items()}
    optimizer_state_dict = torch.load(f"{LOG_DIR}/{run_id}_optimizer.pt", map_location=torch.device(device))

    # Train large model from scratch
    config["hidden_dim"] = dataset_config["N"] * dataset_config["k"]

    if args.read_wandb:
        runs = api.runs(
            path=f"{WANDB_ENTITY}/weight_transfer2",
            filters={"group": f"large_model_from_scratch_{best_run.id}"},
        )
    if not args.read_wandb or len(runs) == 0:
        for seed in range(1):
            config["seed"] = seed
            wandb.init(
                project="weight_transfer2",
                config=config,
                reinit="finish_previous",
                name=f"large_model_from_scratch_seed_{seed}",
                group=f"large_model_from_scratch_{best_run.id}",
            )
            fix_seed(config["seed"])
            train_loader, val_loader, output_dim = get_data_loaders_CNN(dataset, DATA_DIR, dataset_config["batch_size"])
            large_model_from_scratch = create_and_train_model(
                output_dim,
                config,
                train_loader,
                val_loader,
            )
            wandb.finish()

    # Sweep 2: Find best noise and lr_af
    sweep_config["name"] = f"upscaled_model_tuning_{best_run.id}"
    sweep_config["parameters"] = {
        "dataset": {"values": [dataset]},
        "hidden_dim": {"values": [dataset_config["n"]]},
        "multiplier": {"values": [dataset_config["k"]]},
        "batch_size": {"values": [dataset_config["batch_size"]]},
        "epochs": {"values": [dataset_config["epochs"]]},
        "lr": {"values": [dataset_config["lr_af"] * 2**k for k in range(-2, 1)]},
        "noise_std": {"values": np.arange(0, 1, 0.1).tolist()},
        "seed": {"values": [dataset_config["seed"]]},
        "weight_decay": {"values": [dataset_config["weight_decay"]]},
        "model": {"values": [best_run.id]},
        "optimizer": {"values": [args.optimizer]},
        "momentum": {"values": [dataset_config.get("momentum", 0.0)]},
    }
    if args.read_wandb:
        sweeps = api.project(entity=WANDB_ENTITY, name="weight_transfer2").sweeps()
        sweep_id = False
        for sweep in sweeps:
            if sweep.name == f"upscaled_model_tuning_{best_run.id}":
                sweep_id = sweep.id
                print(f"Found Sweep! ID: {sweep.id} | URL: {sweep.url}")
                break
    if not args.read_wandb or not sweep_id:
        sweep_id = wandb.sweep(sweep_config, project="weight_transfer2")
        wandb.agent(sweep_id, function=sweep_train_upscaled_epochs)
    sweep = api.sweep(f"{WANDB_ENTITY}/weight_transfer2/{sweep_id}")
    runs = [run for run in sweep.runs if run.summary.get("last_train_loss") is not None and not math.isnan(float(run.summary.get("min_train_loss")))]
    runs = sorted(runs, key=lambda run: float(run.summary.get("last_train_loss")), reverse=False)
    best_run_af = runs[0]
    best_noise = best_run_af.config["noise_std"]
    best_lr_af = best_run_af.config["lr"]
    wandb.teardown()

    # Train large model from upscaled weights with best lr_af and t
    config["lr"] = best_lr_af
    config["noise_std"] = best_noise
    config["hidden_dim"] = dataset_config["N"]
    config["multiplier"] = dataset_config["k"]
    if args.read_wandb:
        runs = api.runs(
            path=f"{WANDB_ENTITY}/weight_transfer2",
            filters={"group": f"large_model_upscaled_{best_run.id}"},
        )

    # if not args.read_wandb or len(runs) == 0:
    for seed in range(1):
        config["seed"] = seed
        wandb.init(
            project="weight_transfer2",
            config=config,
            reinit="finish_previous",
            name=f"large_model_upscaled_seed_{seed}",
            group=f"large_model_upscaled_{best_run.id}",
        )
        fix_seed(config["seed"])
        train_loader, val_loader, output_dim = get_data_loaders_CNN(dataset, DATA_DIR, dataset_config["batch_size"])
        wide_model = load_upscale_and_train_model(output_dim, config, train_loader, val_loader, base_model_state_dict, optimizer_state_dict)
        wandb.finish()
