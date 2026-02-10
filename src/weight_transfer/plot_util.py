import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from weight_transfer import PLOT_DIR
import matplotlib
from pathlib import Path

matplotlib.use("Agg")  # Use non-interactive backend
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{mathptmx}"

METRIC_LABELS = {
    "epoch/train_loss": "Training Loss",
    "train/train_loss": "Training Loss",
    "epoch/val_loss": "Validation Loss",
    "val/val_loss": "Validation Loss",
    "epoch/val_acc": "Validation Accuracy",
}


def plot_multiple_run_groups(
    entity, project, run_id_groups, labels, metric_name="epoch/val_acc", fname=f"{PLOT_DIR}/multiple_groups.png", ylog=False, ylim=None, one_run=False
):
    """
    Fetches a specific metric from multiple groups of runs, calculates mean and std dev for each group,
    and plots them all in the same figure with different colors.

    Args:
        entity (str): WandB username/entity.
        project (str): WandB project name.
        run_id_groups (list of lists): Each element is a list of run ID strings.
                                       E.g., [['id1', 'id2'], ['id3', 'id4'], ['id5', 'id6']]
        labels (list of str): Labels for each group.
        metric_name (str): The specific metric to plot (y-axis).
        fname (str): Output filename for the plot.
    """
    api = wandb.Api()

    if len(labels) != len(run_id_groups):
        raise ValueError("Number of labels must match number of run_id_groups")

    # IBM color-blind friendly palette
    ibm_palette = [
        "#648fff",
        "#ffb000",
        "#dc267f",
        "#fe6100",
        "#785ef0",
        "#00b6cb",
        "#ffb1b0",
        "#0c7bdc",
    ]
    colors = [ibm_palette[i % len(ibm_palette)] for i in range(len(run_id_groups))]

    plt.figure(figsize=(8, 6))

    # Process each group
    for i, (run_ids, label, color) in enumerate(zip(run_id_groups, labels, colors)):
        all_runs_data = []

        if one_run:
            run_ids = [run_ids[0]]
        for run_id in run_ids:
            try:
                run = api.run(f"{entity}/{project}/{run_id}")
                history = pd.DataFrame([r for r in run.scan_history(keys=["epoch", metric_name])])
                if history.empty:
                    history = pd.DataFrame([r for r in run.scan_history(keys=["step", metric_name])])
                    key = "step"
                else:
                    key = "epoch"
                history = history.set_index(key)
                history = history.rename(columns={metric_name: run_id})
                all_runs_data.append(history)
            except Exception as e:
                print(f"Error fetching run {run_id}: {e}")

        if not all_runs_data:
            print(f"No valid data found for {label}. Skipping.")
            continue

        combined_df = pd.concat(all_runs_data, axis=1)
        mean_curve = combined_df.mean(axis=1)
        min_curve = combined_df.min(axis=1)
        max_curve = combined_df.max(axis=1)

        epochs = mean_curve.index

        plt.plot(epochs, mean_curve, label=label, color=color, linewidth=2)
        plt.fill_between(epochs, min_curve, max_curve, color=color, alpha=0.3)

    # Formatting
    plt.xlabel("Epoch" if key == "epoch" else "Step", fontsize=30)
    axis_label = METRIC_LABELS.get(metric_name, metric_name)
    plt.ylabel(axis_label, fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=25, loc="upper right" if metric_name != "epoch/val_acc" else "lower right")

    if ylog:
        plt.yscale("log")
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    if one_run:
        fname = fname.replace(".png", "_one_run.png")
    # Save the plot
    plt.savefig(fname, dpi=300)
    plt.savefig(fname.replace('.png', '.pdf'))
    print(f"\nPlot saved to {fname}")
    plt.close()
