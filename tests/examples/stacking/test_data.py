import torch
from cmws.examples.stacking.data import StackingDataset


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
