import torch
from cmws.examples.csg.data import CSGDataset, get_csg_data_loader


def test_csg_dataset():
    device = torch.device("cuda")
    im_size = 64

    # Train
    csg_dataset = CSGDataset(device)
    assert list(csg_dataset.obs.shape) == [csg_dataset.num_train_data, im_size, im_size]
    assert list(csg_dataset.obs_id.shape) == [csg_dataset.num_train_data]

    # Test
    csg_dataset = CSGDataset(device, test=True)
    assert list(csg_dataset.obs.shape) == [csg_dataset.num_test_data, im_size, im_size]
    assert list(csg_dataset.obs_id.shape) == [csg_dataset.num_test_data]


def test_csg_data_loader():
    device = torch.device("cuda")
    batch_size = 7
    im_size = 64

    # Train
    data_loader = get_csg_data_loader(device, batch_size)
    obs, obs_id = next(iter(data_loader))
    assert list(obs.shape) == [batch_size, im_size, im_size]
    assert list(obs_id.shape) == [batch_size]

    # Train
    data_loader = get_csg_data_loader(device, batch_size, test=True)
    obs, obs_id = next(iter(data_loader))
    assert list(obs.shape) == [batch_size, im_size, im_size]
    assert list(obs_id.shape) == [batch_size]
