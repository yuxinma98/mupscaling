import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def get_data_loaders_MLP(dataset_name, data_dir, batch_size, **kwargs):
    if dataset_name == "ForestCoverType":
        covtype = fetch_covtype(data_home=data_dir)
        X, y = covtype.data, covtype.target
        y = y - 1  # Make labels 0-indexed
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        continuous_features = list(range(10))
        binary_features = list(range(10, 54))
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), continuous_features),
                ("cat", "passthrough", binary_features),  # Leave binary features as they are
            ]
        )
        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        val_ds = TensorDataset(X_val_tensor, y_val_tensor)
        input_dim = X.shape[1]  # 54 features
        output_dim = len(set(y))  # 7 classes
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=False)
    return train_loader, val_loader, input_dim, output_dim
