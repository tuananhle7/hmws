import torch
from cmws.examples.timeseries.data import TimeseriesDataset, get_timeseries_data_loader


def test_timeseries_dataset():
    device = torch.device("cuda")

    # Train
    timeseries_dataset = TimeseriesDataset(device)
    assert list(timeseries_dataset.obs.shape) == [timeseries_dataset.num_data, 256]
    assert list(timeseries_dataset.obs_id.shape) == [timeseries_dataset.num_data]

    # Test
    timeseries_dataset = TimeseriesDataset(device, test=True)
    assert list(timeseries_dataset.obs.shape) == [timeseries_dataset.num_data, 256]
    assert list(timeseries_dataset.obs_id.shape) == [timeseries_dataset.num_data]


def test_stacking_data_loader():
    device = torch.device("cuda")
    batch_size = 7

    # Train
    data_loader = get_timeseries_data_loader(device, batch_size)
    obs, obs_id = next(iter(data_loader))
    assert list(obs.shape) == [batch_size, 256]
    assert list(obs_id.shape) == [batch_size]

    # Train
    data_loader = get_timeseries_data_loader(device, batch_size, test=True)
    obs, obs_id = next(iter(data_loader))
    assert list(obs.shape) == [batch_size, 256]
    assert list(obs_id.shape) == [batch_size]
