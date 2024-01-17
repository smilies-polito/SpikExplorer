import numpy as np
import torch
import tonic
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tonic import DiskCachedDataset


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


class SHD(Dataset):
    def __init__(self, train: bool, dt: int, T: int):
        super(SHD, self).__init__()

        # dt = 60ms and T = 15
        assert dt == 60, "only SHD with dt=60ms is supported"
        self.train = train
        self.dt = dt
        self.T = T
        if train:
            X = np.load("./data/SHD/trainX_60ms.npy")[:, :T, :]
            y = np.load("./data/SHD/trainY_60ms.npy")
        else:
            X = np.load("./data/SHD/testX_60ms.npy")[:, :T, :]
            y = np.load("./data/SHD/testY_60ms.npy")

        self.len = 8156
        if train == False:
            self.len = 2264
        self.eventflow = X
        self.label = y

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.eventflow[idx, ...].astype(np.float32)
        y = self.label[idx].astype(np.float32)
        return x, y


def load_shd(batch_size):
    train_ds = SHD(train=True, dt=60, T=15)
    test_ds = SHD(train=False, dt=60, T=15)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=True)
    return train_dl, test_dl
"""
def load_shd(batch_size):
    sensor_size = tonic.datasets.SHD.sensor_size

    shd = tonic.datasets.SHD(
        save_to="./data", train=True
    )

    cache_dir = os.path.expanduser("./data")
    cache_subdir = "SHD"
    
    train_file = h5py.File(os.path.join(cache_dir, cache_subdir, "shd_train.h5"), "r")
    test_file = h5py.File(os.path.join(cache_dir, cache_subdir, "shd_test.h5"), "r")

    x_train = train_file["spikes"]
    y_train = train_file["labels"]
    x_test = test_file["spikes"]
    y_test = test_file["labels"]

    trainset = []
    for i in range(len(x_train)):
        trainset.append([(x_train["times"][i]), y_train[i]])

    testset = []
    for i in range(len(x_test)):
        testset.append([(x_test["times"][i]), y_test[i]])

    cached_trainset = DiskCachedDataset(trainset, cache_path="./cache/shd/train")
    cached_testset = DiskCachedDataset(testset, cache_path="./cache/shd/test")

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
"""
