import collections
import itertools
import logging
from pathlib import Path

import cmws
import numpy as np
import pyro
import scipy
import torch
from cmws.examples.csg.models import (
    heartangles,
    hearts,
    hearts_pyro,
    ldif_representation,
    ldif_representation_pyro,
    neural_boundary,
    neural_boundary_pyro,
    no_rectangle,
    rectangles,
    shape_program,
    shape_program_pyro,
    shape_program_pytorch,
)


# Init, saving, etc
def init(run_args, device):
    memory = None
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
        generative_model = neural_boundary_pyro.GenerativeModel(
            num_primitives=run_args.num_primitives, has_shape_scale=run_args.model_has_shape_scale
        ).to(device)

        # Guide
        guide = neural_boundary_pyro.Guide(
            num_primitives=run_args.num_primitives, has_shape_scale=run_args.model_has_shape_scale
        ).to(device)
    elif run_args.model_type == "shape_program_pyro":
        # Generative model
        generative_model = shape_program_pyro.GenerativeModel(
            num_primitives=run_args.num_primitives
        ).to(device)

        # Guide
        guide = shape_program_pyro.Guide(num_primitives=run_args.num_primitives).to(device)
    elif run_args.model_type == "shape_program_pytorch":
        # Generative model
        generative_model = shape_program_pytorch.GenerativeModel().to(device)

        # Guide
        guide = shape_program_pytorch.Guide().to(device)

        # Memory
        if "mws" in run_args.algorithm:
            memory = cmws.memory.Memory(
                10000,
                run_args.memory_size,
                [[], [generative_model.max_num_shapes]],
                [[0, 3], [0, generative_model.num_primitives]],
            ).to(device)

    # Model tuple
    model = {"generative_model": generative_model, "guide": guide, "memory": memory}

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
    stats = Stats([], [], [], [])

    return model, optimizer, stats


def save_checkpoint(path, model, optimizer, stats, run_args=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    generative_model, guide, memory = model["generative_model"], model["guide"], model["memory"]
    torch.save(
        {
            "generative_model_state_dict": None
            if run_args.model_type == "rectangles"
            else generative_model.state_dict(),
            "guide_state_dict": guide.state_dict(),
            "memory_state_dict": None if memory is None else memory.state_dict(),
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

    generative_model, guide, memory = model["generative_model"], model["guide"], model["memory"]
    guide.load_state_dict(checkpoint["guide_state_dict"])
    if run_args.model_type != "rectangles":
        generative_model.load_state_dict(checkpoint["generative_model_state_dict"])
    if memory is not None:
        memory.load_state_dict(checkpoint["memory_state_dict"])

    model = {"generative_model": generative_model, "guide": guide, "memory": None}
    if "_pyro" in run_args.model_type:
        optimizer.set_state(checkpoint["optimizer_state_dict"])
    else:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    stats = checkpoint["stats"]
    return model, optimizer, stats, run_args


Stats = collections.namedtuple("Stats", ["losses", "sleep_pretraining_losses", "log_ps", "kls"])


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
    def __init__(self, random_side, device):
        self.random_side = random_side
        self.device = device
        self.lim = torch.tensor(0.8, device=self.device)

    def sample(self, sample_shape):
        """
        Args
            sample_shape

        Returns [*sample_shape, 4]
        """
        minus_lim = -self.lim
        padding = 1.0
        min_x = torch.distributions.Uniform(minus_lim, self.lim - padding).sample(sample_shape)
        min_y = torch.distributions.Uniform(minus_lim, self.lim - padding).sample(sample_shape)
        if self.random_side:
            side = torch.distributions.Uniform(
                torch.zeros_like(min_x), self.lim - torch.max(min_x, min_y)
            ).sample()
        else:
            side = 0.5
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


def heart_pose_to_str(heart_pose, fixed_scale=False):
    """
    Args
        heart_pose
            raw_position [2]
            raw_scale []

    Returns (str)
    """
    if fixed_scale:
        raw_position = heart_pose
        position = raw_position.sigmoid() - 0.5
        return f"H(x={position[0].item():.1f},y={position[0].item():.1f})"
    else:
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
