import pathlib
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from cmws import util


def lukes_make_data():
    """
    https://github.com/insperatum/wsvae/blob/master/examples/timeseries/main-timeseries.py#L81
    """
    # Init
    path = "/om/user/lbh/wsvae/examples/timeseries/UCR_TS_Archive_2015/data.p"
    n_data = 2000
    n_timepoints = 256
    np.random.seed(0)
    random.seed(0)

    # Read file
    with open(path, "rb") as f:
        d_in = pickle.load(f)

    # Make arrays
    data = []
    testdata = []
    all_timeseries = [x for X in d_in for x in X["data"]]
    #     if args.shuffle: random.shuffle(all_timeseries)
    for x in all_timeseries:
        # if len(x)<n_timepoints: continue
        # lower = math.floor((len(x)-n_timepoints)/2)
        # upper = len(x) - math.ceil((len(x)-n_timepoints)/2)
        if len(x) < n_timepoints + 1:
            continue
        lower = 0
        upper = n_timepoints
        x = np.array(x)

        # Centre the timeseries
        if x.std() == 0:
            continue
        x = (x - x.mean()) / x.std()
        x = list(x)

        # Append
        data.append(x[lower:upper])
        testdata.append(x[upper : upper + 100])

        # Break
        if len(data) > n_data * 2:
            break

    print("Loaded", len(data), "timeseries")

    # Add more datasets
    # -- Airlines
    airlines = np.array(
        [
            112,
            115,
            118,
            125,
            132,
            130,
            129,
            125,
            121,
            128,
            135,
            141,
            148,
            148,
            148,
            142,
            136,
            127,
            119,
            111,
            104,
            111,
            118,
            116,
            115,
            120,
            126,
            133,
            141,
            138,
            135,
            130,
            125,
            137,
            149,
            159,
            170,
            170,
            170,
            164,
            158,
            145,
            133,
            123,
            114,
            127,
            140,
            142,
            145,
            147,
            150,
            164,
            178,
            170,
            163,
            167,
            172,
            175,
            178,
            188,
            199,
            199,
            199,
            191,
            184,
            173,
            162,
            154,
            146,
            156,
            166,
            168,
            171,
            175,
            180,
            186,
            193,
            187,
            181,
            182,
            183,
            200,
            218,
            224,
            230,
            236,
            242,
            225,
            209,
            200,
            191,
            181,
            172,
            183,
            194,
            195,
            196,
            196,
            196,
            216,
            236,
            235,
            235,
            232,
            229,
            236,
            243,
            253,
            264,
            268,
            272,
            254,
            237,
            224,
            211,
            195,
            180,
            190,
            201,
            202,
            204,
            196,
            188,
            211,
            235,
            231,
            227,
            230,
            234,
            249,
            264,
            283,
            302,
            297,
            293,
            276,
            259,
            244,
            229,
            216,
            203,
            216,
            229,
            235,
            242,
            237,
            233,
            250,
            267,
            268,
            269,
            269,
            270,
            292,
            315,
            339,
            364,
            355,
            347,
            329,
            312,
            293,
            274,
            255,
            237,
            257,
            278,
            281,
            284,
            280,
            277,
            297,
            317,
            315,
            313,
            315,
            318,
            346,
            374,
            393,
            413,
            409,
            405,
            380,
            355,
            330,
            306,
            288,
            271,
            288,
            306,
            310,
            315,
            308,
            301,
            328,
            356,
            352,
            348,
            351,
            355,
            388,
            422,
            443,
            465,
            466,
            467,
            435,
            404,
            375,
            347,
            326,
            305,
            320,
            336,
            338,
            340,
            329,
            318,
            340,
            362,
            355,
            348,
            355,
            363,
            399,
            435,
            463,
            491,
            498,
            505,
            454,
            404,
            381,
            359,
            334,
            310,
            323,
            337,
            348,
            360,
            351,
            342,
            374,
            406,
            401,
            396,
            408,
            420,
            446,
            472,
            510,
            548,
            553,
            559,
            511,
            463,
            435,
            407,
            384,
            362,
            383,
            405,
            411,
            417,
            404,
            391,
            405,
            419,
            440,
            461,
            466,
            472,
            503,
            535,
            578,
            622,
            614,
            606,
            557,
            508,
            484,
            461,
            425,
            390,
            411,
        ]
    ).astype(np.float32)
    airlines = (airlines - airlines.mean()) / airlines.std()
    airlines = airlines.tolist()
    data.append(airlines[:n_timepoints])
    testdata.append(airlines[n_timepoints:])

    # -- Mauna
    mauna = np.array(
        [
            -26.4529,
            -26.4529,
            -26.4529,
            -26.4529,
            -24.7129,
            -24.6629,
            -26.3029,
            -27.2329,
            -28.8229,
            -26.5829,
            -24.0029,
            -25.6129,
            -28.8229,
            -23.1329,
            -22.1329,
            -22.5729,
            -23.9829,
            -27.1629,
            -24.4629,
            -23.6229,
            -22.3829,
            -23.5829,
            -25.1529,
            -23.6029,
            -22.4729,
            -21.1529,
            -21.5529,
            -22.3029,
            -20.7729,
            -19.9229,
            -22.4229,
            -24.3929,
            -25.9529,
            -20.2729,
            -23.4629,
            -25.4629,
            -25.2929,
            -24.4829,
            -23.4529,
            -22.7229,
            -21.2729,
            -20.0029,
            -20.2929,
            -24.3529,
            -24.8629,
            -23.2929,
            -22.7429,
            -21.5429,
            -19.7729,
            -18.4129,
            -19.1229,
            -17.7429,
            -17.1629,
            -18.0729,
            -19.6129,
            -22.9029,
            -20.2029,
            -17.1429,
            -16.8029,
            -18.0229,
            -21.8329,
            -18.1629,
            -17.7429,
            -15.5029,
            -14.7829,
            -15.4629,
            -16.2729,
            -19.7829,
            -20.3829,
            -18.0429,
            -17.1029,
            -16.1829,
            -15.2329,
            -14.5029,
            -17.4729,
            -19.0629,
            -17.0329,
            -14.9829,
            -14.3829,
            -13.2429,
            -18.8029,
            -17.3629,
            -14.4129,
            -12.0929,
            -14.1129,
            -15.8429,
            -17.3229,
            -15.6629,
            -13.6229,
            -10.6629,
            -9.6829,
            -10.0929,
            -9.5129,
            -9.0729,
            -9.9129,
            -10.9829,
            -12.7629,
            -14.7929,
            -10.1229,
            -8.2029,
            -10.2529,
            -12.1029,
            -13.8229,
            -12.6729,
            -10.4229,
            -9.6029,
            -8.6629,
            -7.5829,
            -7.2929,
            -7.8229,
            -9.1129,
            -11.2229,
            -10.4829,
            -9.2429,
            -5.4229,
            -9.4129,
            -7.1929,
            -4.1529,
            -4.2729,
            -5.6229,
            -7.4829,
            -9.4029,
            -8.2429,
            -5.9329,
            -5.4029,
            -2.6929,
            -4.4329,
            -8.3029,
            -6.8729,
            -5.4329,
            -1.3929,
            -0.9929,
            -5.0629,
            -3.9529,
            -2.9329,
            0.3471,
            0.0871,
            -1.6729,
            -3.8029,
            -2.5529,
            -1.4129,
            0.5371,
            1.3971,
            1.1871,
            -2.3429,
            -4.1929,
            -1.6729,
            0.3571,
            0.9371,
            0.2271,
            0.8271,
            2.3471,
            3.1171,
            4.9171,
            5.2671,
            -0.8129,
            0.8171,
            3.8371,
            6.1871,
            6.7671,
            2.5271,
            0.9271,
            0.6371,
            4.7971,
            5.6971,
            7.3771,
            5.7771,
            2.6971,
            2.0071,
            4.7371,
            5.8571,
            6.3071,
            8.8271,
            9.0871,
            4.1971,
            8.2671,
            11.4271,
            12.0571,
            10.2271,
            8.2771,
            9.1771,
            10.5971,
            13.2571,
            13.5071,
            11.7371,
            9.5071,
            9.1371,
            10.3671,
            12.5371,
            13.2271,
            14.9971,
            12.6571,
            8.7971,
            12.0471,
            12.5571,
            13.5871,
            17.1771,
            11.8671,
            17.0871,
            14.8671,
            12.8371,
            13.2371,
            14.5371,
            14.9971,
            17.2971,
            18.1171,
            15.4071,
            11.8171,
            14.6371,
            16.7471,
            19.5171,
            18.7871,
            13.6771,
            15.8971,
            15.5871,
            17.3971,
            18.5371,
            19.8871,
            21.4871,
            17.4371,
            18.5971,
            20.1671,
            21.0171,
            21.8371,
            24.1871,
            18.6071,
            20.2671,
            23.9871,
            26.4471,
            27.1271,
            25.4771,
            22.0671,
            25.9871,
            28.9771,
            28.8371,
            24.5071,
            25.8471,
            26.9771,
            29.4971,
            24.4571,
            24.5671,
            26.1271,
            28.1171,
            29.9571,
            27.3871,
            25.7971,
            31.3571,
            33.9471,
            32.3371,
            30.8271,
            30.8371,
            32.1871,
            33.5371,
            33.5371,
            33.5371,
            33.5371,
        ]
    )
    mauna = (mauna - mauna.mean()) / mauna.std()
    mauna = mauna.tolist()
    data.append(mauna[:n_timepoints])
    testdata.append(mauna[n_timepoints:])

    data_novel = data[n_data:]
    data = data[:n_data]
    #     testdata_novel=testdata[n_data:]
    #     testdata=testdata[:n_data]
    return data, data_novel


class TimeseriesDataset(torch.utils.data.Dataset):
    """Loads or generates a dataset
    # Uses ~1.2M (test) / 120MB (train)

    Args
        device
        test (bool; default: False)
        force_regenerate (bool; default: False): if False, the dataset is loaded if it exists
            if True, the dataset is regenerated regardless
        seed (int): only used for generation
    """

    def __init__(self, device, test=False):
        self.device = device
        self.test = test
        train_obs, test_obs = lukes_make_data()

        if self.test:
            self.obs = torch.tensor(test_obs, device=self.device).float()
        else:
            self.obs = torch.tensor(train_obs, device=self.device).float()
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

    timeseries_dataset = {}

    # Train
    timeseries_dataset["train"] = TimeseriesDataset(device)

    # Test
    timeseries_dataset["test"] = TimeseriesDataset(device, test=True)

    for mode in ["train", "test"]:
        start = 0
        end = 100

        while start < len(timeseries_dataset[mode]):
            obs, obs_id = timeseries_dataset[mode][start:end]
            path = f"./data/plots/{mode}/{start:04.0f}_{end:04.0f}.pdf"

            fig, axss = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(10 * 3, 10 * 2))

            for i in range(len(obs)):
                axss.flat[i].plot(obs[i].cpu().numpy(), color="black")

            for ax in axss.flat:
                ax.set_ylim(-4, 4)
                ax.set_xticks([])
                ax.set_yticks([-4, 4])

            util.save_fig(fig, path)

            start = end
            end += 100


if __name__ == "__main__":
    plot_data()
