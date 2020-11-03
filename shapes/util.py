import subprocess
import getpass
import itertools
import imageio
import scipy
import collections
from models import rectangles
from models import hearts
from models import heartangles
import os
import random
import numpy as np
import torch
import logging
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(pathname)s:%(lineno)d | %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


def save_fig(fig, path, dpi=100, tight_layout_kwargs={}):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(**tight_layout_kwargs)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    logging.info(f"Saved to {path}")
    plt.close(fig)


def init_cnn(output_dim, hidden_dim=128):
    layers = []
    layers.append(nn.Conv2d(1, int(hidden_dim / 2), kernel_size=3, padding=2))
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
    return nn.Sequential(*layers)


class MultilayerPerceptron(nn.Module):
    def __init__(self, dims, non_linearity, bias=True):
        """
        Args:
            dims: list of ints
            non_linearity: differentiable function
            bias (bool)

        Returns: nn.Module which represents an MLP with architecture
            x -> Linear(dims[0], dims[1]) -> non_linearity ->
            ...
            Linear(dims[-3], dims[-2]) -> non_linearity ->
            Linear(dims[-2], dims[-1]) -> y
        """

        super(MultilayerPerceptron, self).__init__()
        self.dims = dims
        self.non_linearity = non_linearity
        self.linear_modules = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.linear_modules.append(nn.Linear(in_dim, out_dim, bias=bias))

    def forward(self, x):
        temp = x
        for linear_module in self.linear_modules[:-1]:
            temp = self.non_linearity(linear_module(temp))
        return self.linear_modules[-1](temp)


def init_mlp(in_dim, out_dim, hidden_dim, num_layers, non_linearity=None, bias=True):
    """Initializes a MultilayerPerceptron.

    Args:
        in_dim: int
        out_dim: int
        hidden_dim: int
        num_layers: int
        non_linearity: differentiable function (tanh by default)
        bias (bool)

    Returns: a MultilayerPerceptron with the architecture
        x -> Linear(in_dim, hidden_dim) -> non_linearity ->
        ...
        Linear(hidden_dim, hidden_dim) -> non_linearity ->
        Linear(hidden_dim, out_dim) -> y
        where num_layers = 0 corresponds to
        x -> Linear(in_dim, out_dim) -> y
    """
    if non_linearity is None:
        non_linearity = nn.Tanh()
    dims = [in_dim] + [hidden_dim for _ in range(num_layers)] + [out_dim]
    return MultilayerPerceptron(dims, non_linearity, bias)


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
    return args.algorithm


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
    if run_args.model_type == "rectangles":
        # Generative model
        generative_model = rectangles.GenerativeModel().to(device)

        # Guide
        guide = rectangles.Guide().to(device)
    if run_args.model_type == "heartangles":
        # Generative model
        generative_model = heartangles.GenerativeModel().to(device)

        # Guide
        guide = heartangles.Guide().to(device)
    elif run_args.model_type == "hearts":
        # Generative model
        generative_model = hearts.GenerativeModel().to(device)

        # Guide
        guide = hearts.Guide().to(device)

    # Model tuple
    model = (generative_model, guide)

    # Optimizer
    if run_args.model_type == "rectangles":
        parameters = guide.parameters()
    elif run_args.model_type == "hearts" or run_args.model_type == "heartangles":
        parameters = itertools.chain(generative_model.parameters(), guide.parameters())
    optimizer = torch.optim.Adam(parameters, lr=run_args.lr)

    # Stats
    stats = Stats([])

    return model, optimizer, stats


def save_checkpoint(path, model, optimizer, stats, run_args=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    generative_model, guide = model
    torch.save(
        {
            "generative_model_state_dict": None
            if run_args.model_type == "rectangles"
            else generative_model.state_dict(),
            "guide_state_dict": guide.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stats": stats,
            "run_args": run_args,
        },
        path,
    )
    logging.info(f"Saved checkpoint to {path}")


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    run_args = checkpoint["run_args"]
    model, optimizer, stats = init(run_args, device)

    generative_model, guide = model
    guide.load_state_dict(checkpoint["guide_state_dict"])

    if run_args.model_type != "rectangles":
        generative_model.load_state_dict(checkpoint["generative_model_state_dict"])

    model = (generative_model, guide)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    stats = checkpoint["stats"]
    return model, optimizer, stats, run_args


Stats = collections.namedtuple("Stats", ["losses"])


def plot_normal2d(ax, mean, cov, num_points=100, confidence=0.95, **kwargs):
    # https://stats.stackexchange.com/questions/64680/how-to-determine-quantiles-isolines-of-a-multivariate-normal-distribution
    # plots a `confidence' probability ellipse
    const = -2 * np.log(1 - confidence)
    eigvals, eigvecs = scipy.linalg.eig(np.linalg.inv(cov))
    eigvals = np.real(eigvals)
    a = np.sqrt(const / eigvals[0])
    b = np.sqrt(const / eigvals[1])
    theta = np.linspace(-np.pi, np.pi, num=num_points)
    xy = eigvecs @ np.array([np.cos(theta) * a, np.sin(theta) * b]) + np.expand_dims(mean, -1)
    ax.plot(xy[0, :], xy[1, :], **kwargs)
    return ax


def make_gif(img_paths, gif_path, fps):
    images = []
    for img_path in img_paths:
        images.append(imageio.imread(img_path))
    imageio.mimsave(gif_path, images, duration=1 / fps)
    logging.info(f"Saved to {gif_path}")


def logit(z):
    return z.log() - (1 - z).log()


def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                                 exp(values[i_1, ..., i_N])
            log( ------------------------------------------------------------ )
                    sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
    # log_numerator = values
    return values - log_denominator


def exponentiate_and_normalize(values, dim=0):
    """Exponentiates and normalizes a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                            exp(values[i_1, ..., i_N])
            ------------------------------------------------------------
             sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    return torch.exp(lognormexp(values, dim=dim))


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
