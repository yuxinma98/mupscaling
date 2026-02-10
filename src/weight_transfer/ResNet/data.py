from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loaders_CNN(dataset_name, data_dir, batch_size):
    if dataset_name == "CIFAR100":
        output_dim = 100
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        train_ds = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        val_ds = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=False)
    return train_loader, val_loader, output_dim