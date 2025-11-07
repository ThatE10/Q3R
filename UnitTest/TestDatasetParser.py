import pytest
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace
import os

from main_helper import dataset_parser


def test_dataset_parser_cifar10():
    # Setup
    config = SimpleNamespace(
        dataset="CIFAR10",
        augmentation=True,
        batch_size=32,
        threads=2
    )

    # Execute
    size, labels, train_loader, val_loader = dataset_parser(config)

    # Assert
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert labels == 10
    assert len(size) == 4
    assert size[1:] == (3, 224, 224)  # C, H, W with augmentation

    # Test batch properties
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape[0] == config.batch_size
    assert batch_x.shape[1:] == (3, 224, 224)
    assert batch_y.shape[0] == config.batch_size


def test_dataset_parser_without_augmentation():
    config = SimpleNamespace(
        dataset="CIFAR10",
        augmentation=False,
        batch_size=32,
        threads=2
    )

    size, labels, train_loader, val_loader = dataset_parser(config)

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape[1:] == (3, 32, 32)  # Original CIFAR10 dimensions


def test_dataset_parser_mnist():
    config = SimpleNamespace(
        dataset="MNIST",
        augmentation=False,
        batch_size=32,
        threads=2
    )

    size, labels, train_loader, val_loader = dataset_parser(config)

    assert labels == 10
    assert size[1] == 1  # Single channel

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape[1] == 1  # Single channel


def test_invalid_dataset():
    config = SimpleNamespace(
        dataset="INVALID_DATASET",
        augmentation=False,
        batch_size=32,
        threads=2
    )

    with pytest.raises(ValueError, match="Dataset 'INVALID_DATASET' is not recognized"):
        dataset_parser(config)


def test_data_normalization():
    config = SimpleNamespace(
        dataset="CIFAR10",
        augmentation=True,
        batch_size=32,
        threads=2
    )

    _, _, train_loader, _ = dataset_parser(config)
    batch_x, _ = next(iter(train_loader))

    # Check if data is normalized (approximately between -2 and 2)
    assert torch.min(batch_x) > -3
    assert torch.max(batch_x) < 3


@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Cleanup downloaded data after tests
    if os.path.exists('../data'):
        import shutil
        shutil.rmtree('../data')


def test_batch_size_consistency():
    config = SimpleNamespace(
        dataset="CIFAR10",
        augmentation=True,
        batch_size=16,  # Different batch size
        threads=2
    )

    _, _, train_loader, val_loader = dataset_parser(config)

    assert train_loader.batch_size == 16
    assert val_loader.batch_size == 16