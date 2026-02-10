import torch
import wandb
import os
import numpy as np
import random
from torch.optim.lr_scheduler import LinearLR

# Enable TF32 for better performance on compatible GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_weight_norms(model, epoch, step):
    """
    Logs the scaled weight norms of the model's parameters to wandb.
    """
    for name, p in model.named_parameters():
        norm = torch.norm(p).item()
        scaled_weight_entry = norm / np.sqrt(p.numel())  # \|W\|_F / sqrt(number of elements) = RMS of weight entries
        if p.infshape.ninf() == 2:  # weight matrix
            scaled_weight_entry *= p.shape[1]
        wandb.log({f"scaled_weight_entry/{name}": scaled_weight_entry, "epoch": epoch, "step": step})


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
    task="classification",
    stop_loss=None,
    device="cuda",
    start_epoch=1,
    path=None,
    log_weight_norms_flag=False,
    save_every_epochs=False,
    lr_scheduler=False,
    lr_end_factor=0.1,
    lr_total_iters=500,
):
    """
    if stop_loss is provided, training stops when training loss drops below stop_loss. Otherwise, train for 'epochs' epochs.
    """
    model.to(device)
    model = torch.compile(model)
    if task == "classification":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif task == "regression":
        loss_fn = torch.nn.MSELoss()
    elif task == "binary":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif task == "language":
        loss_fn = torch.nn.NLLLoss()

    if wandb.run is not None:
        wandb.run.define_metric("epoch/train_loss", summary="min")
        wandb.run.define_metric("epoch/val_loss", summary="min")
        if task == "language":
            wandb.run.define_metric("epoch/val_perplexity", summary="min")
        elif task == "classification":
            wandb.run.define_metric("epoch/val_acc", summary="max")

    if save_every_epochs and path is not None:
        run_id = wandb.run.id if wandb.run is not None else "offline"
        os.makedirs(os.path.join(path, run_id), exist_ok=True)

    if lr_scheduler:
        scheduler = LinearLR(optimizer, start_factor=1, end_factor=lr_end_factor, total_iters=lr_total_iters)

    # Log before training starts (so that epoch 0 is logged; the loss immediately after upscaling is also logged)
    step = (start_epoch - 1) * len(train_loader)
    with torch.no_grad():  # No gradient mode, but keep the model in the train mode
        train_loss = torch.tensor(0.0, device=device)
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            if task == "language":
                out = out.view(-1, out.size(-1))
            loss = loss_fn(out, y)
            train_loss += loss.detach() * y.size(0)
        train_loss /= len(train_loader.dataset)
        val_loss = torch.tensor(0.0, device=device)
        correct = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if task == "language":
                out = out.view(-1, out.size(-1))
            loss = loss_fn(out, y)
            val_loss += loss.detach() * y.size(0)
            if task == "classification":
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
        val_loss /= len(val_loader.dataset)
        if task == "classification":
            val_acc = correct / len(val_loader.dataset)
        else:
            val_acc = None
    print(f"Epoch {start_epoch-1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    if wandb.run is not None:
        log_dict = {"epoch/train_loss": train_loss, "epoch/val_loss": val_loss, "epoch": start_epoch - 1, "step": step}
        if task == "language":
            val_perplexity = torch.exp(val_loss).item()
            log_dict["epoch/val_perplexity"] = val_perplexity
        elif task == "classification":
            log_dict["epoch/val_acc"] = val_acc
        wandb.log(log_dict)
    if log_weight_norms_flag:  # Log initial weight norms before training
        log_weight_norms(model, start_epoch - 1, step)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        epoch_num = start_epoch + epoch
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            if task == "language":
                out = out.view(-1, out.size(-1))
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach() * y.size(0)
            step += 1
        train_loss /= len(train_loader.dataset)
        if log_weight_norms_flag:
            log_weight_norms(model, epoch_num, step)
        print(f"Epoch {epoch_num} | Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                if task == "language":
                    out = out.view(-1, out.size(-1))
                loss = loss_fn(out, y)
                val_loss += loss.detach() * y.size(0)
                if task == "classification":
                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
        val_loss /= len(val_loader.dataset)
        if task == "classification":
            val_acc = correct / len(val_loader.dataset)
        print(f"Epoch {epoch_num} | Val Loss: {val_loss:.4f}")

        if wandb.run is not None:
            log_dict = {"epoch/train_loss": train_loss, "epoch/val_loss": val_loss, "epoch": epoch_num, "step": step}
            if task == "language":
                val_perplexity = torch.exp(val_loss).item()
                log_dict["epoch/val_perplexity"] = val_perplexity
            elif task == "classification":
                log_dict["epoch/val_acc"] = val_acc
            wandb.log(log_dict)

        # Learning rate scheduler
        if lr_scheduler:
            scheduler.step()

        # Save checkpoint at every epoch if save_every_epochs is enabled
        if save_every_epochs and path is not None:
            run_id = wandb.run.id if wandb.run is not None else "offline"
            checkpoint_path = os.path.join(path, f"{run_id}/epoch_{epoch_num}.pt")
            torch.save(model.state_dict(), checkpoint_path)

        # Stop training if train loss blows up
        if torch.isnan(train_loss):
            print(f"Stopping early at epoch {epoch} due to NaN train loss.")
            break

        # If train_loss reaches stop_loss, stop training
        if stop_loss is not None and train_loss <= stop_loss:
            print(f"Stopping early at epoch {epoch} as train loss {train_loss:.4f} <= stop_loss {stop_loss}")
            break

    if wandb.run is not None:
        wandb.run.summary["min_train_loss"] = wandb.run.summary.get("epoch/train_loss", {}).get("min", None)
        # Also store the last epoch-level train loss
        last_train_loss = float(train_loss.detach().item() if isinstance(train_loss, torch.Tensor) else train_loss)
        wandb.run.summary["last_train_loss"] = last_train_loss

    if path is not None: # save the model at the end of training
        run_id = wandb.run.id if wandb.run is not None else "final_model"
        model_path = os.path.join(path, f"{run_id}.pt")
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), model_path.replace(".pt", "_optimizer.pt"))
    return model
