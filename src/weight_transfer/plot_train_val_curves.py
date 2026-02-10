import argparse
import wandb
from weight_transfer import PLOT_DIR, WANDB_ENTITY
from weight_transfer.plot_util import plot_multiple_run_groups

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train MLP with Weight Transfer")
    argparser.add_argument("--model", type=str, required=True, choices=["MLP", "Resnet", "GPT"], help="Model type to use")
    argparser.add_argument("--optimizer", type=str, required=True, choices=["SGD", "AdamW"], help="Optimizer to use")
    argparser.add_argument("--one_run", action="store_true", help="Plot only one run from each group")
    best_run_ids = {"MLP": {"SGD": "3blo8di8", "AdamW": "5edndqdv"}, "Resnet": {"SGD": "jvshtfj5"}}

    args = argparser.parse_args()

    if args.model == "MLP" or args.model == "Resnet":
        best_run_id = best_run_ids[args.model][args.optimizer]
        run_ids = {}
        api = wandb.Api()
        runs = api.runs(path=f"{WANDB_ENTITY}/weight_transfer2", filters={"group": f"base_model_{best_run_id}"})
        # assert len(runs) == 5, f"Expected 5 runs for from-scratch group, got {len(runs)}"
        run_ids = {"Base model": [run.id for run in runs]}

        api = wandb.Api()
        runs = api.runs(path=f"{WANDB_ENTITY}/weight_transfer2", filters={"group": f"large_model_from_scratch_{best_run_id}"})
        # assert len(runs) == 5, f"Expected 5 runs for from-scratch group, got {len(runs)}"
        run_ids["Wide model (trained from scratch)"] = [run.id for run in runs]
        lr = runs[0].config["lr"]
        train_losses, val_losses, val_accs = [], [], []
        for run in runs:
            train_losses.append(run.summary["epoch/train_loss"]["min"])
            val_losses.append(run.summary["epoch/val_loss"]["min"])
            val_accs.append(run.summary["epoch/val_acc"]["max"])
        print(rf"From scratch & $\overline\gamma={lr:.3f}$ & {sum(train_losses)/5:.4f} & {sum(val_losses)/5:.4f} & {sum(val_accs)/5 * 100:.3f}\% \\")

        runs = api.runs(path=f"{WANDB_ENTITY}/weight_transfer2", filters={"group": f"large_model_upscaled_{best_run_id}"})
        if len(runs) > 0:
            assert len(runs) == 5, f"Expected 5 runs for default upscaling group, got {len(runs)}"
            run_ids["Wide model (upscaled)"] = [run.id for run in runs]
            lr = runs[0].config["lr"]
            sigma = runs[0].config.get("noise_std")
            train_losses, val_losses, val_accs = [], [], []
            for run in runs:
                train_losses.append(run.summary["epoch/train_loss"]["min"])
                val_losses.append(run.summary["epoch/val_loss"]["min"])
                val_accs.append(run.summary["epoch/val_acc"]["max"])
            print(
                rf"Default upscaling & $\overline\gamma_\mathrm{{af}}={lr:.3f},\ \overline{{\sigma}}^{{(\ell)}}={sigma:.0f}$ & {sum(train_losses)/5:.4f} & {sum(val_losses)/5:.4f} & {sum(val_accs)/5 * 100:.3f}\% \\"
            )

        for metric in ["epoch/train_loss", "epoch/val_loss", "epoch/val_acc"]:
            if args.model == "MLP" and "loss" in metric:
                ylim = (0, 0.3)
            elif args.model == "MLP" and metric == "epoch/val_acc":
                ylim = (0.9, 0.98)
            elif args.model == "Resnet" and metric == "epoch/train_loss":
                ylim = (0, 0.2)
            elif args.model == "Resnet" and metric == "epoch/val_loss":
                ylim = (1, 3)
            elif args.model == "Resnet" and metric == "epoch/val_acc":
                ylim = (0.5, 0.8)
            plot_multiple_run_groups(
                entity=WANDB_ENTITY,
                project="weight_transfer2",
                run_id_groups=run_ids.values(),
                labels=run_ids.keys(),
                metric_name=metric,
                fname=f"{PLOT_DIR}/{args.model}_{args.optimizer}_{metric.replace('/', '_')}.png",
                ylim=ylim,
                one_run=args.one_run,
            )

    if args.model == "GPT" and args.optimizer == "AdamW":
        run_ids = {"Base model": ["337r0917"], "Wide model (from scratch)": ["eookqilr"], "Wide model (upscaled)": ["i0ex8m18"]}

        for metric in ["train/train_loss", "val/val_loss"]:
            plot_multiple_run_groups(
                entity="weight-transfer",
                project="fineweb-edu",
                run_id_groups=run_ids.values(),
                labels=run_ids.keys(),
                metric_name=metric,
                fname=f"{PLOT_DIR}/GPT_{args.optimizer}_{metric.replace('/', '_')}.png",
                ylim=None,
                one_run=args.one_run,
            )
