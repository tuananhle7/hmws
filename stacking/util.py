import subprocess
import getpass
import time
import torch
import imageio
import logging
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from models import stacking_pyro
from models import one_primitive
from models import two_primitives
from models import stacking
from models import stacking_top_down
import pyro
import collections
import os
import torch.nn as nn
import itertools
from tqdm import tqdm

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
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    images = []
    for img_path in tqdm(img_paths):
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
    if args.num_sleep_pretraining_iterations > 0:
        pretraining_suffix = "_pretraining"
    else:
        pretraining_suffix = ""
    return f"td{args.insomnia}{pretraining_suffix}"


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
    if run_args.model_type == "stacking_pyro":
        # Generative model
        generative_model = stacking_pyro.GenerativeModel(num_primitives=run_args.num_primitives).to(
            device
        )

        # Guide
        guide = stacking_pyro.Guide(num_primitives=run_args.num_primitives).to(device)
    elif run_args.model_type == "one_primitive":
        # Generative model
        generative_model = one_primitive.GenerativeModel().to(device)

        # Guide
        guide = one_primitive.Guide().to(device)
    elif run_args.model_type == "two_primitives":
        # Generative model
        generative_model = two_primitives.GenerativeModel().to(device)

        # Guide
        guide = two_primitives.Guide().to(device)
    elif run_args.model_type == "stacking":
        # Generative model
        generative_model = stacking.GenerativeModel(
            num_primitives=run_args.num_primitives, max_num_blocks=run_args.max_num_blocks
        ).to(device)

        # Guide
        guide = stacking.Guide(
            num_primitives=run_args.num_primitives, max_num_blocks=run_args.max_num_blocks
        ).to(device)

    elif run_args.model_type == "stacking_top_down":
        # Generative model
        generative_model = stacking_top_down.GenerativeModel(
            num_primitives=run_args.num_primitives, max_num_blocks=run_args.max_num_blocks
        ).to(device)

        # Guide
        guide = stacking_top_down.Guide(
            num_primitives=run_args.num_primitives, max_num_blocks=run_args.max_num_blocks
        ).to(device)

    # Model tuple
    model = (generative_model, guide)

    # Optimizer
    if "_pyro" in run_args.model_type:
        optimizer = pyro.optim.pytorch_optimizers.Adam({"lr": run_args.lr})
    else:
        parameters = itertools.chain(generative_model.parameters(), guide.parameters())
        optimizer = torch.optim.Adam(parameters, lr=run_args.lr)

    # Stats
    stats = Stats([], [], [], [])

    return model, optimizer, stats


def save_checkpoint(path, model, optimizer, stats, run_args=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    generative_model, guide = model
    torch.save(
        {
            "generative_model_state_dict": generative_model.state_dict(),
            "guide_state_dict": guide.state_dict(),
            "optimizer_state_dict": optimizer.get_state()
            if "_pyro" in run_args.model_type
            else optimizer.state_dict(),
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
    if "_pyro" in run_args.model_type:
        optimizer.set_state(checkpoint["optimizer_state_dict"])
    else:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    stats = checkpoint["stats"]
    return model, optimizer, stats, run_args


Stats = collections.namedtuple("Stats", ["losses", "sleep_pretraining_losses", "log_ps", "kls"])


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


def sqrt(x):
    """Safe sqrt"""
    if torch.any(x <= 0):
        logging.warn("Input to sqrt is <= 0")

    return torch.sqrt(torch.clamp(x, min=1e-8))


class JointDistribution:
    """p(x_{1:N}) = ∏_n p(x_n)
    Args:
        dists: list of distributions p(x_n)
    """

    def __init__(self, dists):
        self.dists = dists

    def sample(self, sample_shape=[]):
        return tuple([dist.sample(sample_shape) for dist in self.dists])

    def rsample(self, sample_shape=[]):
        return tuple([dist.rsample(sample_shape) for dist in self.dists])

    def log_prob(self, values):
        return sum([dist.log_prob(value) for dist, value in zip(self.dists, values)])


def get_num_elements(shape):
    return int(torch.tensor(shape).prod().long().item())


def pad_tensor(x, length, value):
    """Pads a tensor with a prespecified value

    Args
        x [*shape, max_length]
        length [*shape]
        value []

    Returns [*shape, max_length]
    """
    # Extract
    shape = length.shape
    max_length = x.shape[-1]
    device = x.device
    num_elements = get_num_elements(shape)

    # Index tensor [*shape, max_length]
    # [0, 1, 2, 3, 4, 5, 6, 7]
    index = (
        torch.arange(max_length, device=device)[None]
        .expand(num_elements, max_length)
        .view(*[*shape, max_length])
    )

    # Compute mask
    # --length       2
    # --index [0, 1, 2, 3, 4, 5, 6, 7]
    # --mask  [0, 0, 1, 1, 1, 1, 1, 1]
    mask = index >= length.unsqueeze(-1)

    # Compute result
    result = x.clone()
    result[mask] = value

    return result


class CategoricalPlusOne(torch.distributions.Categorical):
    # TODO: override other attributes
    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape=sample_shape) + 1

    def log_prob(self, value):
        return super().log_prob(value - 1)


def cancel_all_my_non_bash_jobs():
    logging.info("Cancelling all non-bash jobs.")
    jobs_status = (
        subprocess.check_output(f"squeue -u {getpass.getuser()}", shell=True)
        .decode()
        .split("\n")[1:-1]
    )
    non_bash_job_ids = []
    for job_status in jobs_status:
        if not ("bash" in job_status.split() or "zsh" in job_status.split()):
            non_bash_job_ids.append(job_status.split()[0])
    if len(non_bash_job_ids) > 0:
        cmd = "scancel {}".format(" ".join(non_bash_job_ids))
        logging.info(cmd)
        logging.info(subprocess.check_output(cmd, shell=True).decode())
    else:
        logging.info("No non-bash jobs to cancel.")


def get_max_gpu_memory_allocated_MB(device):
    return torch.cuda.max_memory_allocated(device=device) / 1e6 if device.type == "cuda" else 0


if __name__ == "__main__":
    # Test pad_tensor
    x = torch.rand(4)
    length = torch.randint(0, 4, ())
    value = -1
    print(f"x = {x}")
    print(f"length = {length}")
    print(f"pad_tensor(x, length, value) = {pad_tensor(x, length, value)}")
