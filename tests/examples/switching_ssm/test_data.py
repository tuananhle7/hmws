import cmws.util
from cmws.examples.switching_ssm.data import SLDSDataset


def test_timeseries_dataset():
    device = cmws.util.get_device()

    num_timesteps = 100
    obs_dim = 10

    # Train
    timeseries_dataset = SLDSDataset(device)
    assert list(timeseries_dataset.obs.shape) == [
        timeseries_dataset.num_data,
        num_timesteps,
        obs_dim,
    ]
    assert list(timeseries_dataset.obs_id.shape) == [timeseries_dataset.num_data]

    # Test
    timeseries_dataset = SLDSDataset(device, test=True)
    assert list(timeseries_dataset.obs.shape) == [
        timeseries_dataset.num_data,
        num_timesteps,
        obs_dim,
    ]
    assert list(timeseries_dataset.obs_id.shape) == [timeseries_dataset.num_data]
