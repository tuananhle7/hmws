import pyro
import subprocess
import getpass
import itertools
import imageio
import scipy
import collections
from models import rectangles
from models import hearts
from models import heartangles
from models import shape_program
from models import no_rectangle
from models import ldif_representation
from models import hearts_pyro
from models import ldif_representation_pyro
from models import neural_boundary
from models import neural_boundary_pyro
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
    return f"{args.model_type}_{args.algorithm}"


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
    elif run_args.model_type == "heartangles":
        # Generative model
        generative_model = heartangles.GenerativeModel().to(device)

        # Guide
        guide = heartangles.Guide().to(device)
    elif run_args.model_type == "hearts":
        # Generative model
        generative_model = hearts.GenerativeModel().to(device)

        # Guide
        guide = hearts.Guide().to(device)
    elif run_args.model_type == "shape_program":
        # Generative model
        generative_model = shape_program.GenerativeModel().to(device)

        # Guide
        guide = shape_program.Guide().to(device)
    elif run_args.model_type == "no_rectangle":
        # Generative model
        generative_model = no_rectangle.GenerativeModel().to(device)

        # Guide
        guide = no_rectangle.Guide().to(device)
    elif run_args.model_type == "ldif_representation":
        # Generative model
        generative_model = ldif_representation.GenerativeModel().to(device)

        # Guide
        guide = ldif_representation.Guide().to(device)
    elif run_args.model_type == "hearts_pyro":
        # Generative model
        generative_model = hearts_pyro.GenerativeModel().to(device)

        # Guide
        guide = hearts_pyro.Guide().to(device)
    elif run_args.model_type == "ldif_representation_pyro":
        # Generative model
        generative_model = ldif_representation_pyro.GenerativeModel().to(device)

        # Guide
        guide = ldif_representation_pyro.Guide().to(device)
    elif run_args.model_type == "neural_boundary":
        # Generative model
        generative_model = neural_boundary.GenerativeModel().to(device)

        # Guide
        guide = neural_boundary.Guide().to(device)
    elif run_args.model_type == "neural_boundary_pyro":
        # Generative model
        generative_model = neural_boundary_pyro.GenerativeModel().to(device)

        # Guide
        guide = neural_boundary_pyro.Guide().to(device)

    # Model tuple
    model = (generative_model, guide)

    # Optimizer
    if run_args.model_type == "rectangles":
        parameters = guide.parameters()
    else:
        parameters = itertools.chain(generative_model.parameters(), guide.parameters())

    if "_pyro" in run_args.model_type:
        optimizer = pyro.optim.pytorch_optimizers.Adam({"lr": run_args.lr})
    else:
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
            "optimizer_state_dict": optimizer.get_state()
            if "_pyro" in run_args.model_type
            else optimizer.state_dict(),
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
    if "_pyro" in run_args.model_type:
        optimizer.set_state(checkpoint["optimizer_state_dict"])
    else:
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


class RectanglePoseDistribution:
    def __init__(self, device):
        self.device = device
        self.lim = torch.tensor(0.8, device=self.device)

    def sample(self, sample_shape):
        """
        Args
            sample_shape

        Returns [*sample_shape, 4]
        """
        minus_lim = -self.lim
        padding = 0.2
        min_x = torch.distributions.Uniform(minus_lim, self.lim - padding).sample(sample_shape)
        max_x = torch.distributions.Uniform(min_x + padding, self.lim).sample()
        min_y = torch.distributions.Uniform(minus_lim, self.lim - padding).sample(sample_shape)
        max_y = torch.distributions.Uniform(min_y + padding, self.lim).sample()
        return torch.stack([min_x, min_y, max_x, max_y], dim=-1)

    def log_prob(self, xy_lims):
        """
        Args
            xy_lims [*shape, 4]

        Returns [*shape]
        """
        # HACK
        shape = xy_lims.shape[:-1]
        return torch.zeros(shape, device=xy_lims.device)
        # min_x, min_y, max_x, max_y = [xy_lims[..., i] for i in range(4)]
        # minus_one = -self.one
        # min_x_log_prob = torch.distributions.Uniform(minus_one, self.one).log_prob(min_x)
        # max_x_log_prob = torch.distributions.Uniform(min_x, self.one).log_prob(max_x)
        # min_y_log_prob = torch.distributions.Uniform(minus_one, self.one).log_prob(min_y)
        # max_y_log_prob = torch.distributions.Uniform(min_y, self.one).log_prob(max_y)
        # return min_x_log_prob + max_x_log_prob + min_y_log_prob + max_y_log_prob


class SquarePoseDistribution:
    def __init__(self, device):
        self.device = device
        self.lim = torch.tensor(0.8, device=self.device)

    def sample(self, sample_shape):
        """
        Args
            sample_shape

        Returns [*sample_shape, 4]
        """
        minus_lim = -self.lim
        padding = 0.2
        min_x = torch.distributions.Uniform(minus_lim, self.lim - padding).sample(sample_shape)
        min_y = torch.distributions.Uniform(minus_lim, self.lim - padding).sample(sample_shape)
        side = torch.distributions.Uniform(
            torch.zeros_like(min_x), self.lim - torch.max(min_x, min_y)
        ).sample()
        max_x = min_x + side
        max_y = min_y + side
        return torch.stack([min_x, min_y, max_x, max_y], dim=-1)

    def log_prob(self, xy_lims):
        """
        Args
            xy_lims [*shape, 4]

        Returns [*shape]
        """
        shape = xy_lims.shape[:-1]
        return torch.zeros(shape, device=xy_lims.device)


def heart_pose_to_str(heart_pose):
    """
    Args
        heart_pose
            raw_position [2]
            raw_scale []

    Returns (str)
    """
    raw_position, raw_scale = heart_pose
    position = raw_position.sigmoid() - 0.5
    scale = raw_scale.sigmoid() * 0.8 + 0.1
    return f"H(x={position[0].item():.1f},y={position[0].item():.1f},s={scale.item():.1f})"


def rectangle_pose_to_str(rectangle_pose):
    """
    Args
        rectangle_pose [4]

    Returns (str)
    """
    return (
        f"R(bl=({rectangle_lim_to_str(rectangle_pose[0].item())},"
        f"{rectangle_lim_to_str(rectangle_pose[1].item())}),"
        f"tr=({rectangle_lim_to_str(rectangle_pose[2].item())},"
        f"{rectangle_lim_to_str(rectangle_pose[3].item())}))"
    )


def rectangle_lim_to_str(rectangle_lim):
    """
    Args
        rectangle_lim (float)

    Returns (str)
    """
    if rectangle_lim > 1:
        return ">1"
    elif rectangle_lim < -1:
        return "<-1"
    else:
        return f"{rectangle_lim:.1f}"


def get_gaussian_kernel(dx, dy, kernel_size, scale, device):
    """
    Args
        dx (float)
        dy (float)
        kernel_size (int)
        scale (float)
        device

    Returns [kernel_size, kernel_size]
    """

    # Inputs to the kernel function
    kernel_x_lim = dx * kernel_size / 2
    kernel_y_lim = dy * kernel_size / 2
    x_range = torch.linspace(-kernel_x_lim, kernel_x_lim, steps=kernel_size, device=device)
    y_range = torch.linspace(-kernel_y_lim, kernel_y_lim, steps=kernel_size, device=device)
    # [kernel_size, kernel_size]
    kernel_x, kernel_y = torch.meshgrid(x_range, y_range)
    # [kernel_size, kernel_size, 2]
    kernel_xy = torch.stack([kernel_x, kernel_y], dim=-1)

    # Kernel function
    kernel_dist = torch.distributions.Independent(
        torch.distributions.Normal(
            torch.zeros((2,), device=device), torch.ones((2,), device=device) * scale
        ),
        reinterpreted_batch_ndims=1,
    )

    # Output from the kernel function
    # [kernel_size, kernel_size]
    log_kernel = kernel_dist.log_prob(kernel_xy)
    # normalize
    log_kernel = log_kernel - torch.logsumexp(log_kernel.view(-1), dim=0)

    return log_kernel.exp()


def smooth_image(image, kernel_size, scale):
    """
    Args
        image [*shape, num_rows, num_cols] (limits -1, 1)
        kernel_size (int; must be odd)
        scale (float)

    Returns [*shape, num_rows, num_cols]
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd. got {kernel_size}")

    # Extract
    device = image.device
    num_rows, num_cols = image.shape[-2:]
    shape = image.shape[:-2]
    num_samples = int(torch.tensor(shape).prod().long().item())
    image_flattened = image.view(num_samples, num_rows, num_rows)

    # Create gaussian kernel
    dx, dy = 2 / num_cols, 2 / num_rows
    kernel = get_gaussian_kernel(dx, dy, kernel_size, scale, device)

    # Run convolution
    return torch.nn.functional.conv2d(
        image_flattened[:, None], kernel[None, None], padding=kernel_size // 2
    ).view(*[*shape, num_rows, num_cols])


if __name__ == "__main__":
    batch_size, num_rows, num_cols = 1, 64, 64
    kernel_size, scale = 5, 1.0

    canvas = torch.zeros((num_rows, num_cols))
    import render

    image = render.render_rectangle(torch.tensor([-0.5, -0.5, 0.5, 0.5]), canvas)

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(image, vmin=0, vmax=1, cmap="Greys")
    axs[1].imshow(smooth_image(image, kernel_size, scale), vmin=0, vmax=1, cmap="Greys")
    save_fig(fig, "smoothing.png")
