import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, random_split, DataLoader, TensorDataset, Dataset


def load_image_datasets(datapath, *transform_list):
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the pixel values to the range [-1, 1],
        *transform_list
    ])

    # Download and load the MNIST dataset
    train_dataset = datasets.CIFAR10(root=datapath, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=datapath, train=False, transform=transform, download=True)

    return train_dataset, test_dataset

def prepare_data(
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size):

    torch.manual_seed(12082023)

    batched_train_dataset = DataLoader(train_dataset, batch_size=batch_size)
    batched_test_dataset = DataLoader(test_dataset, batch_size=batch_size)

    return batched_train_dataset, batched_test_dataset





    

