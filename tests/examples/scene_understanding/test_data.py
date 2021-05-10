import torch
from cmws.examples.scene_understanding.data import (
    SceneUnderstandingDataset,
    get_scene_understanding_data_loader,
)


def test_scene_understanding_dataset():
    device = torch.device("cuda")

    # Train
    scene_understanding_dataset = SceneUnderstandingDataset(device)
    assert list(scene_understanding_dataset.obs.shape) == [
        scene_understanding_dataset.num_train_data,
        3,
        32,
        32,
    ]
    assert list(scene_understanding_dataset.obs_id.shape) == [
        scene_understanding_dataset.num_train_data
    ]

    # Test
    scene_understanding_dataset = SceneUnderstandingDataset(device, test=True)
    assert list(scene_understanding_dataset.obs.shape) == [
        scene_understanding_dataset.num_test_data,
        3,
        32,
        32,
    ]
    assert list(scene_understanding_dataset.obs_id.shape) == [
        scene_understanding_dataset.num_test_data
    ]


def test_stacking_data_loader():
    device = torch.device("cuda")
    batch_size = 7

    # Train
    data_loader = get_scene_understanding_data_loader(device, batch_size)
    obs, obs_id = next(iter(data_loader))
    assert list(obs.shape) == [batch_size, 3, 32, 32]
    assert list(obs_id.shape) == [batch_size]

    # Train
    data_loader = get_scene_understanding_data_loader(device, batch_size, test=True)
    obs, obs_id = next(iter(data_loader))
    assert list(obs.shape) == [batch_size, 3, 32, 32]
    assert list(obs_id.shape) == [batch_size]
