import torch
from cmws.examples.stacking_3d.data import StackingDataset, get_stacking_data_loader


def test_stacking_dataset():
    device = torch.device("cuda")

    # Train
    stacking_dataset = StackingDataset(device)
    assert list(stacking_dataset.obs.shape) == [stacking_dataset.num_train_data, 3, 32, 32]
    assert list(stacking_dataset.obs_id.shape) == [stacking_dataset.num_train_data]

    # Test
    stacking_dataset = StackingDataset(device, test=True)
    assert list(stacking_dataset.obs.shape) == [stacking_dataset.num_test_data, 3, 32, 32]
    assert list(stacking_dataset.obs_id.shape) == [stacking_dataset.num_test_data]


def test_stacking_data_loader():
    device = torch.device("cuda")
    batch_size = 7

    # Train
    data_loader = get_stacking_data_loader(device, batch_size)
    obs, obs_id = next(iter(data_loader))
    assert list(obs.shape) == [batch_size, 3, 32, 32]
    assert list(obs_id.shape) == [batch_size]

    # Train
    data_loader = get_stacking_data_loader(device, batch_size, test=True)
    obs, obs_id = next(iter(data_loader))
    assert list(obs.shape) == [batch_size, 3, 32, 32]
    assert list(obs_id.shape) == [batch_size]
