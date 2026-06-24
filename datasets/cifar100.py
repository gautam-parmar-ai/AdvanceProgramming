"""
datasets/cifar100.py

Handles:
- CIFAR-100 download
- train/test transforms
- dataloaders
- reproducibility
"""

import torch
import torchvision
import torchvision.transforms as transforms

from configs.config import Config


class CIFAR100DataModule:

    def __init__(self):

        self.batch_size = Config.batch_size

        # CIFAR-100 statistics
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def get_datasets(self):

        train_dataset = torchvision.datasets.CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=self.train_transform
        )

        test_dataset = torchvision.datasets.CIFAR100(
            root="./data",
            train=False,
            download=True,
            transform=self.test_transform
        )

        return train_dataset, test_dataset

    def get_full_dataloaders(self):

        train_dataset, test_dataset = self.get_datasets()

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        return train_loader, test_loader
    
    # from datasets.cifar100 import CIFAR100DataModule

# dm = CIFAR100DataModule()

# train_dataset, test_dataset = dm.get_datasets()

# print("Train samples:", len(train_dataset))
# print("Test samples:", len(test_dataset))