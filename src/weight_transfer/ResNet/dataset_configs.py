"""
Dataset-specific configuration parameters for weight transfer experiments.
"""

DATASET_CONFIGS = {
    ("CIFAR100", "SGD"): {
        "n": 0.5,
        "N": 1,
        "k": 2,
        "seed": 0,
        "batch_size": 128,
        "epochs": 100,
        "lr_bf": 0.005,
        "lr_af": 0.005,
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "min_iters": 150,
        "lr_scheduler": False,
        "lr_end_factor": 0.01,
        "lr_total_iters": 250,
    },
}


def get_dataset_config(dataset_name):
    if dataset_name not in DATASET_CONFIGS:
        available_datasets = list(DATASET_CONFIGS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available_datasets}")

    return DATASET_CONFIGS[dataset_name].copy()
