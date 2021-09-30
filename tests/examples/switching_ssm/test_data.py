import cmws.util
from cmws.examples.switching_ssm.data import SLDSDataset, get_slds_data_loader


def test_slds_dataset():
    device = cmws.util.get_device()

    num_timesteps = 100
    obs_dim = 10

    # Train
    slds_dataset = SLDSDataset(device)
    assert list(slds_dataset.obs.shape) == [
        slds_dataset.num_data,
        num_timesteps,
        obs_dim,
    ]
    assert list(slds_dataset.obs_id.shape) == [slds_dataset.num_data]

    # Test
    slds_dataset = SLDSDataset(device, test=True)
    assert list(slds_dataset.obs.shape) == [
        slds_dataset.num_data,
        num_timesteps,
        obs_dim,
    ]
    assert list(slds_dataset.obs_id.shape) == [slds_dataset.num_data]


def test_slds_data_loader():
    device = cmws.util.get_device()

    num_timesteps = 100
    obs_dim = 10
    batch_size = 7

    # Train
    data_loader = get_slds_data_loader(device, batch_size)
    obs, obs_id = next(iter(data_loader))
    assert list(obs.shape) == [batch_size, num_timesteps, obs_dim]
    assert list(obs_id.shape) == [batch_size]

    # Train
    data_loader = get_slds_data_loader(device, batch_size, test=True)
    obs, obs_id = next(iter(data_loader))
    assert list(obs.shape) == [batch_size, num_timesteps, obs_dim]
    assert list(obs_id.shape) == [batch_size]
