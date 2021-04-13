import torch
from cmws.examples.stacking.data import StackingDataset, get_stacking_data_loader


def test_stacking_dataset():
    device = torch.device("cuda")

    for dataset_type in ["stacking", "stacking_top_down"]:
        # Train
        stacking_dataset = StackingDataset(dataset_type, device)
        assert list(stacking_dataset.obs.shape) == [stacking_dataset.num_train_data, 3, 32, 32]
        assert list(stacking_dataset.obs_id.shape) == [stacking_dataset.num_train_data]

        # Test
        stacking_dataset = StackingDataset(dataset_type, device, test=True)
        assert list(stacking_dataset.obs.shape) == [stacking_dataset.num_test_data, 3, 32, 32]
        assert list(stacking_dataset.obs_id.shape) == [stacking_dataset.num_test_data]


def test_stacking_data_loader():
    device = torch.device("cuda")
    batch_size = 7

    for dataset_type in ["stacking", "stacking_top_down"]:
        # Train
        data_loader = get_stacking_data_loader(dataset_type, device, batch_size)
        obs, obs_id = next(iter(data_loader))
        assert list(obs.shape) == [batch_size, 3, 32, 32]
        assert list(obs_id.shape) == [batch_size]

        # Train
        data_loader = get_stacking_data_loader(dataset_type, device, batch_size, test=True)
        obs, obs_id = next(iter(data_loader))
        assert list(obs.shape) == [batch_size, 3, 32, 32]
        assert list(obs_id.shape) == [batch_size]
