"""
Script to generate plots for hyperparameter transfer experiments.
Fetches runs from W&B and plots train loss vs learning rate for different hidden dimensions.
"""

import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import wandb
from pathlib import Path
from weight_transfer import PLOT_DIR, WANDB_ENTITY
import pandas as pd

Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
matplotlib.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{mathptmx}"
)

# Increase default font sizes for readability in smaller figures
matplotlib.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


def fetch_wandb_data(optimizer="SGD", dataset="ForestCoverType", group="hyparam_transfer"):
    """
    Fetch run data from W&B for the specified optimizer and dataset.
    
    Args:
        optimizer: Optimizer name (e.g., "SGD", "AdamW")
        dataset: Dataset name (e.g., "ForestCoverType")
    
    Returns:
        Dictionary mapping hidden dimensions to lists of learning rates and train losses
    """
    api = wandb.Api(timeout=60)

    try:
        runs = api.runs(
            path=f"{WANDB_ENTITY}/weight_transfer2",
            filters={
                "group": group,
                "config.dataset": dataset,
                "config.optimizer": optimizer
            },
            per_page=50,
        )

        data_by_n = {}

        for i, run in enumerate(runs):
            if 'basemodel' in run.name:
                continue
            n = run.config.get("hidden_dim") * run.config.get("multiplier")
            lr = run.config.get("lr")
            noise_std = run.config.get("noise_std")
            t = run.config.get("t")
            train_loss = run.summary.get("last_train_loss")

            try:
                history = run.scan_history(keys=["epoch/train_loss", "_step"])
                df = pd.DataFrame([row for row in history])
                if df.empty or "epoch/train_loss" not in df.columns:
                    continue
                df = df.sort_values("_step") if "_step" in df.columns else df.reset_index(drop=True)
                alpha = 0.1
                df["loss_ema"] = df["epoch/train_loss"].dropna().ewm(alpha=alpha, adjust=False).mean()
                last_ema_loss = df["loss_ema"].iloc[-1]
            except Exception:
                continue

            # Normalize numeric fields; skip runs with bad types
            try:
                lr = float(lr)
                last_ema_loss = float(last_ema_loss)
                noise_std = float(noise_std) if noise_std is not None else None
                t = float(t) if t is not None else None
            except (TypeError, ValueError):
                continue
            if n is None or lr is None or last_ema_loss is None or not np.isfinite(last_ema_loss) or not np.isfinite(lr):
                continue

            if n not in data_by_n:
                data_by_n[n] = {"lr": [], "seed": [], "noise_std": [], "train_loss": [], "t": []}

            data_by_n[n]["lr"].append(lr)
            data_by_n[n]["seed"].append(run.config.get("seed"))
            data_by_n[n]["noise_std"].append(noise_std)
            data_by_n[n]["t"].append(t)
            data_by_n[n]["train_loss"].append(last_ema_loss)

            # Every 50 runs, force Python to release memory
            if i % 50 == 0:
                gc.collect()

        return data_by_n

    except Exception as e:
        print(f"Critical Error: {e}")
        return {}


def fetch_wandb_data_transformer():
    """
    Fetch run data from W&B for the specified optimizer and dataset.

    Args:
        optimizer: Optimizer name (e.g., "SGD", "AdamW")
        dataset: Dataset name (e.g., "ForestCoverType")

    Returns:
        Dictionary mapping hidden dimensions to lists of learning rates and train losses
    """
    api = wandb.Api(timeout=60)
    runs = api.runs(
        path="weight-transfer/fineweb-edu",
        per_page=50,
    )
    fix_lr = {}
    fix_noise = {}

    for i, run in enumerate(runs):
        if not "upscale-L8-H8" in run.name:
            continue
        n_embed = run.config.get("n_embd")
        lr = run.config.get("learning_rate")
        lr_exp = run.config.get("learning_rate_exponent")
        noise_std = run.config.get("init_std")
        train_loss = run.summary.get("train/train_loss")
        val_loss = run.summary.get("val/val_loss")

        try:
            n = int(n_embed)
            lr = float(lr)
            noise_std = float(noise_std) if noise_std is not None else None
            train_loss = float(train_loss)
            val_loss = float(val_loss)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            continue

        if n not in fix_lr:
            fix_lr[n] = {"lr": [], "noise_std": [], "train_loss": [], "val_loss": []}
        if n not in fix_noise:
            fix_noise[n] = {"lr": [], "noise_std": [], "train_loss": [], "val_loss": []}

        if lr_exp == 9 and noise_std < 1:
            fix_lr[n]["lr"].append(lr)
            fix_lr[n]["noise_std"].append(noise_std)
            fix_lr[n]["train_loss"].append(train_loss)
            fix_lr[n]["val_loss"].append(val_loss)
        if noise_std == 0.005:
            fix_noise[n]["lr"].append(lr)
            fix_noise[n]["noise_std"].append(noise_std)
            fix_noise[n]["train_loss"].append(train_loss)
            fix_noise[n]["val_loss"].append(val_loss)

        # Every 50 runs, force Python to release memory
        if i % 50 == 0:
            gc.collect()

    return fix_lr, fix_noise


def plot_train_loss_vs_lr(data_by_n, output_file="train_loss_vs_lr.png", y_log=False, figsize=(3, 4)):
    """
    Plot train loss vs learning rate for different hidden dimensions.
    
    Args:
        data_by_n: Dictionary mapping hidden dimensions to data
        optimizer: Optimizer name for the plot title
        output_file: Output filename for the plot
    """
    plt.figure(figsize=figsize)

    # Prepare color mapping: color-blind friendly 'cividis' colormap.
    sorted_ns = sorted(data_by_n.keys())
    cmap = plt.cm.cividis
    # Sample values across the colormap; reverse so increasing n -> darker
    m = len(sorted_ns)
    color_vals = np.linspace(0.2, 0.8, m)[::-1]
    colors_by_n = {n: cmap(val) for n, val in zip(sorted_ns, color_vals)}

    all_lrs = []

    for n in sorted_ns:
        if data_by_n[n].get("lr"):  # Only plot if there's data
            # Group losses by learning rate to compute mean across seeds
            lr_to_losses = {}
            invalid_lrs = set()
            for lr, loss in zip(data_by_n[n]["lr"], data_by_n[n]["train_loss"]):
                try:
                    lr_val = float(lr)
                    loss_val = float(loss)
                except (TypeError, ValueError):
                    invalid_lrs.add(lr)
                    continue
                if not np.isfinite(loss_val) or not np.isfinite(lr_val):
                    invalid_lrs.add(lr)
                    continue
                lr_to_losses.setdefault(lr_val, []).append(loss_val)

            for lr in list(lr_to_losses.keys()):
                if lr in invalid_lrs:
                    lr_to_losses.pop(lr, None)

            lrs = sorted(lr_to_losses.keys())
            if not lrs:
                continue
            mean_losses = [np.mean(lr_to_losses[lr]) for lr in lrs]
            min_losses = [np.min(lr_to_losses[lr]) for lr in lrs]
            max_losses = [np.max(lr_to_losses[lr]) for lr in lrs]
            log_lrs = np.log2(lrs)

            all_lrs.extend(lrs)
            plt.fill_between(log_lrs, min_losses, max_losses, alpha=0.2, color=colors_by_n[n])
            plt.plot(log_lrs, mean_losses, marker="o", label=f"$N={n}$", color=colors_by_n[n], markersize=3)

            # Find the lr corresponding to the smallest mean_loss and add a red cross
            min_mean_loss_idx = np.argmin(mean_losses)
            best_lr = lrs[min_mean_loss_idx]
            best_log_lr = log_lrs[min_mean_loss_idx]
            best_mean_loss = mean_losses[min_mean_loss_idx]
            print(f"N={n}: best_lr={best_lr:.6f}, min_mean_loss={best_mean_loss:.6f}")
            plt.plot(best_log_lr, best_mean_loss, marker="x", color="red", markersize=8, markeredgewidth=2)

    if all_lrs:
        all_lrs = sorted({float(lr) for lr in all_lrs})
        loglr_start, loglr_end = np.ceil(np.log2(all_lrs[0])), np.floor(np.log2(all_lrs[-1]))
        ticks = np.arange(loglr_start, loglr_end + 1)
        plt.xticks(ticks[::2], ha="right")

    plt.xlabel(r"$\log_2(\text{learning rate }\overline{\gamma^\uparrow})$", fontsize=14)
    plt.ylabel("Training loss", fontsize=14)
    if all_lrs:
        plt.legend(fontsize=12)
    if y_log:
        plt.yscale("log")
        plt.gca().yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        plt.gca().yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to file in plots directory
    base_name = output_file.rsplit(".", 1)[0]
    ext = output_file.rsplit(".", 1)[1]
    output_file = f"{base_name}_{figsize[0]}x{figsize[1]}.{ext}"
    output_path = f"{PLOT_DIR}/{output_file}"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(f".{ext}", ".pdf"), dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()  # Close the figure to free memory


def plot_train_loss_vs_noise(data_by_n, output_file="train_loss_vs_t.png", x_log=False, y_log=False, figsize=(3, 4)):
    """
    Plot train loss vs t for different hidden dimensions.

    Args:
        data_by_n: Dictionary mapping hidden dimensions to data
        optimizer: Optimizer name for the plot title
        output_file: Output filename for the plot
    """
    plt.figure(figsize=figsize)

    # Reuse same color mapping logic as for lr plot
    sorted_ns = sorted(data_by_n.keys())
    cmap = plt.cm.cividis
    m = len(sorted_ns)
    color_vals = np.linspace(0.2, 0.8, m)[::-1]
    colors_by_n = {n: cmap(val) for n, val in zip(sorted_ns, color_vals)}

    all_noises = []
    has_data = False
    for n in sorted_ns:
        if data_by_n[n].get("t") or data_by_n[n].get("noise_std"):  # Only plot if there's data
            # Group losses by t to compute mean across seeds
            noise_to_losses = {}
            invalid_noises = set()
            key = "t" if data_by_n[n].get("t", [None])[0] else "noise_std"
            for noise, loss in zip(data_by_n[n][key], data_by_n[n]["train_loss"]):
                try:
                    noise_val = float(noise)
                    loss_val = float(loss)
                except (TypeError, ValueError):
                    invalid_noises.add(noise)
                    continue
                if not np.isfinite(loss_val) or not np.isfinite(noise_val):
                    invalid_noises.add(noise)
                    continue
                noise_to_losses.setdefault(noise_val, []).append(loss_val)

            for noise in list(noise_to_losses.keys()):
                if noise in invalid_noises:
                    noise_to_losses.pop(noise, None)

            noises = sorted(noise_to_losses.keys())
            if not noises:
                continue
            has_data = True
            mean_losses = [np.mean(noise_to_losses[noise]) for noise in noises]
            min_losses = [np.min(noise_to_losses[noise]) for noise in noises]
            max_losses = [np.max(noise_to_losses[noise]) for noise in noises]

            all_noises.extend(noises)

            if x_log:
                noises = np.log2(noises)

            plt.fill_between(noises, min_losses, max_losses, alpha=0.2, color=colors_by_n[n])
            plt.plot(noises, mean_losses, marker="o", label=f"$N={n}$", color=colors_by_n[n], markersize=3)

            # Find the lr corresponding to the smallest mean_loss and add a red cross
            min_mean_loss_idx = np.argmin(mean_losses)
            best_noise = noises[min_mean_loss_idx]
            best_mean_loss = mean_losses[min_mean_loss_idx]
            print(f"N={n}: best_noise={best_noise:.6f}, min_mean_loss={best_mean_loss:.6f}")
            plt.plot(best_noise, best_mean_loss, marker="x", color="red", markersize=8, markeredgewidth=2)

    if has_data and key == "t":
        plt.xlabel(r"$t$", fontsize=14)
    elif has_data and key == "noise_std":
        if x_log:
            plt.xlabel(r"$\log_2(\text{noise std }\overline{\sigma_{\Delta}})$", fontsize=14)
        else:
            plt.xlabel(r"$\text{noise std }\overline{\sigma}_{\Delta}$", fontsize=14)
    else:
        plt.xlabel("Parameter", fontsize=14)

    if all_noises:
        all_noises = sorted({float(noise) for noise in all_noises})
        if x_log:
            lognoise_start, lognoise_end = np.ceil(np.log2(all_noises[0])), np.floor(np.log2(all_noises[-1]))
            print(lognoise_start, lognoise_end)
            if lognoise_start <= lognoise_end:
                ticks = np.arange(lognoise_start, lognoise_end + 1)
                plt.xticks(ticks[::2], ha="right")

    plt.ylabel("Training loss", fontsize=14)
    if has_data:
        plt.legend(fontsize=12)
    if y_log:
        plt.yscale("log")
        plt.gca().yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        plt.gca().yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to file in plots directory
    base_name = output_file.rsplit(".", 1)[0]
    ext = output_file.rsplit(".", 1)[1]
    output_file = f"{base_name}_{figsize[0]}x{figsize[1]}.{ext}"
    output_path = f"{PLOT_DIR}/{output_file}"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(f".{ext}", ".pdf"), dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()  # Close the figure to free memory


def main():
    # MLP
    for dataset, optimizer in [("ForestCoverType", "SGD"), ("ForestCoverType", "AdamW")]:
        data_by_n = fetch_wandb_data(optimizer=optimizer, dataset=dataset, group="hyparam_transfer_upscale_fix_noise(not_t)")
        if data_by_n:
            output_file = f"hyparam_transfer_upscale_fix_noise_{dataset}_{optimizer}(not_t).png"
            plot_train_loss_vs_lr(data_by_n, output_file, y_log=True, figsize=(4, 4))

        data_by_n = fetch_wandb_data(optimizer=optimizer, dataset=dataset, group="hyparam_transfer_upscale_fix_lr(not_t)")
        if data_by_n:
            output_file = f"hyparam_transfer_upscale_fix_lr_{dataset}_{optimizer}(not_t).png"
            plot_train_loss_vs_noise(data_by_n, output_file, y_log=True, x_log=True if optimizer == "AdamW" else False, figsize=(4, 4))

    fix_lr, fix_noise = fetch_wandb_data_transformer()

    plot_train_loss_vs_lr(fix_noise, output_file="hyparam_transfer_fix_noise_transformer.png", figsize=(4, 4))
    plot_train_loss_vs_noise(fix_lr, output_file="hyparam_transfer_fix_lr_transformer.png", x_log=True, figsize=(4, 4))

if __name__ == "__main__":
    main()
