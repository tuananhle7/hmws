import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from cmws import util
import cmws.examples.timeseries.plot

num_timesteps = 256


def standardize(x):
    x_np = np.array(x)
    if x_np.std() == 0:
        return x_np
    return list((x_np - x_np.mean()) / x_np.std())


def load_all_data():
    """
    https://github.com/insperatum/wsvae/blob/master/examples/timeseries/main-timeseries.py#L81

    Returns
        list of timeseries where each timeseries is a list of floats
    """
    # Init
    path = str(pathlib.Path(__file__).parent.joinpath("data.p"))

    # Read file
    with open(path, "rb") as f:
        d_in = pickle.load(f)

    data = [x for X in d_in for x in X["data"]]
    util.logging.info(f"Loaded {len(data)} timeseries")
    return data


def load_custom_data():
    data = []
    for filename in ["airlines.csv", "mauna.csv"]:
        path = str(pathlib.Path(__file__).parent.joinpath(filename))
        data.append(list(np.genfromtxt(path, delimiter=",")))
    return data


def standardize_data(data):
    return [standardize(x) for x in data]


def filter_data(data):
    return [x for x in data if len(x) >= num_timesteps]


def cut_data(data):
    return [x[:num_timesteps] for x in data]


def get_train_test_data():
    """
    https://github.com/insperatum/wsvae/blob/master/examples/timeseries/main-timeseries.py#L81
    """
    data = standardize_data(cut_data(filter_data(load_all_data())))
    custom_data = standardize_data(cut_data(filter_data(load_custom_data())))
    num_train_data, num_test_data = 2000, 2000
    num_non_custom_test_data = num_test_data - len(custom_data)

    train_data = data[:num_train_data]
    test_data = custom_data + data[num_train_data : (num_train_data + num_non_custom_test_data)]
    return train_data, test_data


class TimeseriesDataset(torch.utils.data.Dataset):
    """Loads or generates a dataset

    Args
        device
        test (bool; default: False)
        force_regenerate (bool; default: False): if False, the dataset is loaded if it exists
            if True, the dataset is regenerated regardless
        seed (int): only used for generation
    """

    def __init__(self, device, test=False, full_data=False):
        self.device = device
        self.test = test
        train_obs, test_obs = get_train_test_data()

        if self.test:
            self.obs = torch.tensor(test_obs, device=self.device).float()
            if not full_data:
                self.obs = self.obs[[99, 906, 920, 957, 697, 901, 1584]]
        else:
            self.obs = torch.tensor(train_obs, device=self.device).float()
            if not full_data:
                self.obs = self.obs[[62, 188, 269, 510, 711, 1262, 1790]]
        self.num_data = len(self.obs)
        self.obs_id = torch.arange(self.num_data, device=device)

    def __getitem__(self, idx):
        return self.obs[idx], self.obs_id[idx]

    def __len__(self):
        return self.num_data


def get_timeseries_data_loader(device, batch_size, test=False):
    if test:
        shuffle = True
    else:
        shuffle = False
    return torch.utils.data.DataLoader(
        TimeseriesDataset(device, test=test), batch_size=batch_size, shuffle=shuffle
    )


def plot_data():
    device = torch.device("cuda")

    # Plot train / test data
    timeseries_dataset = {}

    # Train
    timeseries_dataset["train"] = TimeseriesDataset(device)

    # Test
    timeseries_dataset["test"] = TimeseriesDataset(device, test=True)

    for mode in ["test", "train"]:
        start = 0
        end = 100

        while start < len(timeseries_dataset[mode]):
            obs, obs_id = timeseries_dataset[mode][start:end]
            path = f"./data/plots/{mode}/{start:05.0f}_{end:05.0f}.png"

            fig, axss = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(10 * 3, 10 * 2))

            for i in range(len(obs)):
                cmws.examples.timeseries.plot.plot_obs(axss.flat[i], obs[i])

            util.save_fig(fig, path)

            start = end
            end += 100

    # Plot all data
    all_data = standardize_data(filter_data(load_all_data()))
    start = 0
    end = 100
    while start < len(all_data):
        obs = all_data[start:end]
        path = f"./data/plots/all/{start:05.0f}_{end:05.0f}.png"

        fig, axss = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(10 * 3, 10 * 2))

        for i in range(len(obs)):
            ax = axss.flat[i]
            cmws.examples.timeseries.plot.plot_obs(ax, obs[i])
            ax.axvline(255, color="gray")
            ax.set_xlim(0, 355)

        util.save_fig(fig, path)

        start = end
        end += 100


if __name__ == "__main__":
    plot_data()
