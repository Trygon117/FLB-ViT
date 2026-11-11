import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class AddGaussianNoise:
    def __init__(self, std: float = 0.0):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0.0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return tensor + noise


def get_cifar10_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    eval_noise_std: float = 0.0
) -> tuple[DataLoader, DataLoader]:
    # Standard CIFAR-10 channel-wise mean and standard deviation
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
        AddGaussianNoise(std=eval_noise_std)
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader