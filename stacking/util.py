import time
import torch
import imageio
import logging
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from models import stacking
import pyro
import collections
import os
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(pathname)s:%(lineno)d | %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


# Plotting
def save_fig(fig, path, dpi=100, tight_layout_kwargs={}):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(**tight_layout_kwargs)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    logging.info(f"Saved to {path}")
    plt.close(fig)


def make_gif(img_paths, gif_path, fps):
    images = []
    for img_path in img_paths:
        images.append(imageio.imread(img_path))
    imageio.mimsave(gif_path, images, duration=1 / fps)
    logging.info(f"Saved to {gif_path}")


# CUDA and seeds
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Paths
def get_path_base_from_args(args):
    return f"st_{args.num_primitives}"


def get_save_job_name_from_args(args):
    return get_path_base_from_args(args)


def get_save_dir_from_path_base(path_base):
    return f"save/{path_base}"


def get_save_dir(args):
    return get_save_dir_from_path_base(get_path_base_from_args(args))


def get_checkpoint_path(args, checkpoint_iteration=-1):
    return get_checkpoint_path_from_path_base(get_path_base_from_args(args), checkpoint_iteration)


def get_checkpoint_path_from_path_base(path_base, checkpoint_iteration=-1):
    checkpoints_dir = f"{get_save_dir_from_path_base(path_base)}/checkpoints"
    if checkpoint_iteration == -1:
        return f"{checkpoints_dir}/latest.pt"
    else:
        return f"{checkpoints_dir}/{checkpoint_iteration}.pt"


def get_checkpoint_paths(checkpoint_iteration=-1):
    save_dir = "./save/"
    for path_base in sorted(os.listdir(save_dir)):
        yield get_checkpoint_path_from_path_base(path_base, checkpoint_iteration)


# Init, saving, etc
def init(run_args, device):
    # Generative model
    generative_model = stacking.GenerativeModel(num_primitives=run_args.num_primitives).to(device)

    # Guide
    guide = stacking.Guide(num_primitives=run_args.num_primitives).to(device)

    # Model tuple
    model = (generative_model, guide)

    # Optimizer
    optimizer = pyro.optim.pytorch_optimizers.Adam({"lr": run_args.lr})

    # Stats
    stats = Stats([])

    return model, optimizer, stats


def save_checkpoint(path, model, optimizer, stats, run_args=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    generative_model, guide = model
    torch.save(
        {
            "generative_model_state_dict": generative_model.state_dict(),
            "guide_state_dict": guide.state_dict(),
            "optimizer_state_dict": optimizer.get_state(),
            "stats": stats,
            "run_args": run_args,
        },
        path,
    )
    logging.info(f"Saved checkpoint to {path}")


def load_checkpoint(path, device, num_tries=3):
    for i in range(num_tries):
        try:
            checkpoint = torch.load(path, map_location=device)
            break
        except Exception as e:
            logging.info(f"Error {e}")
            wait_time = 2 ** i
            logging.info(f"Waiting for {wait_time} seconds")
            time.sleep(wait_time)
    run_args = checkpoint["run_args"]
    model, optimizer, stats = init(run_args, device)

    generative_model, guide = model
    guide.load_state_dict(checkpoint["guide_state_dict"])
    generative_model.load_state_dict(checkpoint["generative_model_state_dict"])

    model = (generative_model, guide)
    optimizer.set_state(checkpoint["optimizer_state_dict"])
    stats = checkpoint["stats"]
    return model, optimizer, stats, run_args


Stats = collections.namedtuple("Stats", ["losses"])


def init_cnn(output_dim, hidden_dim=128):
    layers = []
    layers.append(nn.Conv2d(3, int(hidden_dim / 2), kernel_size=3, padding=2))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
    layers.append(nn.Conv2d(int(hidden_dim / 2), hidden_dim, kernel_size=3, padding=1))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
    layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=0))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=0))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)
