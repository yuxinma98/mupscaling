import os

def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}. " "Set it in your shell or .env before running.")
    return value


DATA_DIR = _require_env("WEIGHT_TRANSFER_DATA_DIR")
LOG_DIR = os.path.join(DATA_DIR, "log")
PLOT_DIR = os.path.expanduser(_require_env("WEIGHT_TRANSFER_PLOT_DIR"))
WANDB_ENTITY = _require_env("WANDB_ENTITY")
