import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, random_split, DataLoader, TensorDataset, Dataset


def load_image_datasets(datapath, rank, clients_total, *transform_list):
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),  # Convert the images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the pixel values to the range [-1, 1],
        *transform_list
    ])

    # Download and load the MNIST dataset
    train_dataset = datasets.CIFAR10(root=datapath, train=True, transform=transform, download=True)
    train_size = len(train_dataset) // clients_total
    train_dataset = random_split(train_dataset, (train_size, len(train_dataset) - train_size))[rank-1]

    test_dataset = datasets.CIFAR10(root=datapath, train=False, transform=transform, download=True)
    test_size = len(test_dataset) // clients_total
    test_dataset = random_split(test_dataset, (test_size, len(test_dataset) - test_size))[rank-1]

    return train_dataset, test_dataset

def get_memory_size(tensor):
    return tensor.numel() * tensor.element_size()

def get_dataset_memory_size(dataset):
    total_memory_size = 0

    for sample in dataset:
        for item in sample:
            if isinstance(item, torch.Tensor):
                total_memory_size += get_memory_size(item)

    return total_memory_size

def prepare_data(
        train_dataset: Dataset,
        test_dataset: Dataset,
        clients_total: int, 
        rank: int, 
        datapath: str, 
        batch_size):

    torch.manual_seed(12082023)

    print(f"batch_size: {batch_size}")

    batched_train_dataset = DataLoader(train_dataset, batch_size=batch_size)
    batched_test_dataset = DataLoader(test_dataset, batch_size=batch_size)

    # Save the entire MNIST dataset using torch.save
    torch.save(batched_train_dataset, os.path.join(datapath,f"train_dataset_cifar_{rank}.pt"))
    torch.save(batched_test_dataset, os.path.join(datapath,f"test_dataset_cifar_{rank}.pt"))





    

