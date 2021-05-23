import os

import time
import cmws
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import torch
import numpy as np
from cmws import util, losses
from cmws.examples.timeseries import data, run
from cmws.examples.timeseries import util as timeseries_util
from cmws.examples.timeseries import lstm_util
import cmws.examples.timeseries.inference
import pathlib

def main(args):
    device = util.get_device()

    # Model
    model, optimizer, stats, run_args = timeseries_util.load_checkpoint(
                    args.checkpoint_path, device=device
                )
    generative_model, guide, memory = model["generative_model"], model["guide"], model["memory"]

    # Load Data
    train_data_iterator = cmws.examples.timeseries.data.get_timeseries_data_loader(
            device,
            args.batch_size,
            test=False,
            full_data=True,
            synthetic=False,
        )
    test_data_loader = cmws.examples.timeseries.data.get_timeseries_data_loader(
        device, args.batch_size, test=True, full_data=True, synthetic=False
    )

    # Calc log p
    for test_num_particles in [10, 100]:
        if hasattr(generative_model, 'log_eps'):
            print(f"eps = {generative_model.log_eps.exp()}")
        print(f"With {test_num_particles} particles...")

        log_p, kl = [], []
        for test_obs, test_obs_id in test_data_loader:
            print(".", end="", flush=True)
            log_p_, kl_ = losses.get_log_p_and_kl(
                generative_model, guide, test_obs, test_num_particles
            )
            log_p.append(log_p_)
            kl.append(kl_)
        log_p = torch.cat(log_p)
        kl = torch.cat(kl)
        print(f" test log p = {log_p.sum().item()}")

        log_p, kl = [], []
        for train_obs, train_obs_id in train_data_iterator:
            print(".", end="", flush=True)
            log_p_, kl_ = losses.get_log_p_and_kl(
                generative_model, guide, train_obs, test_num_particles
            )
            log_p.append(log_p_)
            kl.append(kl_)
        log_p = torch.cat(log_p)
        kl = torch.cat(kl)
        print(f" train log p = {log_p.sum().item()}")

        print()

def get_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--checkpoint-path", type=str, default=None, help=" ")
    parser.add_argument("--batch-size", type=int, default=100, help=" ")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    with torch.no_grad():
        main(args)