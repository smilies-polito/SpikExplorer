import numpy as np
import torch
import tonic
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tonic import DiskCachedDataset, datasets as tonic_datasets, transforms as tonic_transforms


def load_mnist(batch_size):
    data_path = "./data/mnist"

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(
        data_path, train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        data_path, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        mnist_test, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return train_loader, test_loader


# TBD
def load_dvs(batch_size, n_time_bins=25):
    ### Transforms
    size = tonic.datasets.DVSGesture.sensor_size

    # Denoise transform removes outlier events with inactive surrounding pixels for 10ms
    denoise_transform = tonic_transforms.Denoise(filter_time=10000)

    # ToFrame transform bins events into 25 clusters of frames
    frame_transform = tonic_transforms.ToFrame(sensor_size=size, n_time_bins=n_time_bins)

    # Chain the transforms
    all_transform = tonic_transforms.Compose([denoise_transform, frame_transform])

    train_set = tonic.datasets.DVSGesture(save_to='./data', transform=all_transform, train=True)
    test_set = tonic.datasets.DVSGesture(save_to='./data', transform=all_transform, train=False)

    ### Caching and Dataloaders
    cached_trainset = tonic.DiskCachedDataset(train_set, cache_path='./cache/dvsgesture/train')
    cached_testset = tonic.DiskCachedDataset(test_set, cache_path='./cache/dvsgesture/test')

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(cached_trainset, batch_size=batch_size, shuffle=True, drop_last=True,
                                               collate_fn=tonic.collation.PadTensors(batch_first=False))
    test_loader = torch.utils.data.DataLoader(cached_testset, batch_size=batch_size, shuffle=True, drop_last=True,
                                              collate_fn=tonic.collation.PadTensors(batch_first=False))

    return train_loader, test_loader, size


def load_shd(batch_size, nums_bins=25):

    transform = tonic_transforms.Compose(
        [
            tonic_transforms.ToFrame(
                sensor_size=tonic.datasets.hsd.SHD.sensor_size,
                n_time_bins=nums_bins,
            )
        ]
    )

    train_set = tonic.datasets.hsd.SHD(save_to='./data', train=True, transform
    =transform)
    test_set = tonic.datasets.hsd.SHD(save_to='./data', train=False,
                                      transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                             drop_last=True)

    num_inputs = tonic.datasets.hsd.SHD.sensor_size[0]

    return train_loader, test_loader, num_inputs


def load_nmnist(batch_size):
    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_transform = transforms.Compose(
        [
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
        ]
    )

    trainset = tonic.datasets.NMNIST(
        save_to="./data", transform=frame_transform, train=True
    )
    testset = tonic.datasets.NMNIST(
        save_to="./data", transform=frame_transform, train=False
    )

    print(trainset)

    transform = tonic.transforms.Compose(
        [torch.from_numpy, transforms.RandomRotation([-10, 10])]
    )

    cached_trainset = DiskCachedDataset(
        trainset, transform=transform, cache_path="./cache/nmnist/train"
    )

    # no augmentations for the testset
    cached_testset = DiskCachedDataset(testset, cache_path="./cache/nmnist/test")
    print(cached_testset)
    train_loader = DataLoader(
        dataset=cached_trainset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=cached_testset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
    )

    return train_loader, test_loader

