"""
Dataset-specific configuration parameters for weight transfer experiments.
"""

DATASET_CONFIGS = {
    ("ForestCoverType", "SGD"): {
        "n": 100,
        "N": 500,
        "k": 4,
        "seed": 0,
        "batch_size": 2000,
        "num_layers": 4,
        "epochs": 500,
        "bias": True,
        "lr_bf": 0.1,
        "lr_af": 0.1,
        "weight_decay": 1e-4,
    },
    ("ForestCoverType", "AdamW"): {
        "n": 100,
        "N": 500,
        "k": 4,
        "seed": 0,
        "batch_size": 2000,
        "num_layers": 4,
        "epochs": 500,
        "bias": True,
        "lr_bf": 0.1,
        "lr_af": 0.1,
        "weight_decay": 1e-4,
    },
}


def get_dataset_config(dataset_name):
    if dataset_name not in DATASET_CONFIGS:
        available_datasets = list(DATASET_CONFIGS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available_datasets}")

    return DATASET_CONFIGS[dataset_name].copy()
