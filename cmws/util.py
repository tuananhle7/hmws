import scipy
import torch
import imageio
import logging
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torch.nn as nn
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(pathname)s:%(lineno)d | %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


# Paths
def get_save_dir(experiment_name, config_name):
    if experiment_name == "":
        raise RuntimeError("Must specify experiment name")

    return f"save/{experiment_name}/{config_name}"


def get_logs_dir(experiment_name, config_name):
    return f"{get_save_dir(experiment_name, config_name)}/logs"


def get_checkpoint_path(experiment_name, config_name, checkpoint_iteration=-1):
    checkpoints_dir = f"{get_save_dir(experiment_name, config_name)}/checkpoints"
    if checkpoint_iteration == -1:
        return f"{checkpoints_dir}/latest.pt"
    else:
        return f"{checkpoints_dir}/{checkpoint_iteration}.pt"


def get_checkpoint_paths(experiment_name, checkpoint_iteration=-1):
    save_dir = f"./save/{experiment_name}"
    if Path(save_dir).exists():
        for config_name in sorted(os.listdir(save_dir)):
            yield get_checkpoint_path(experiment_name, config_name, checkpoint_iteration)
    else:
        return []


def sqrt(x):
    """Safe sqrt"""
    if torch.any(x <= 0):
        logging.warn("Input to sqrt is <= 0")

    return torch.sqrt(torch.clamp(x, min=1e-8))


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


class CategoricalPlusOne(torch.distributions.Categorical):
    # TODO: override other attributes
    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape=sample_shape) + 1

    def log_prob(self, value):
        return super().log_prob(value - 1)


def get_multivariate_normal_dist(loc, covariance_matrix, verbose=False):
    """Numerically stable multivariate normal distribution

    Args
        loc [*shape, dim]
        covariance_matrix [*shape, dim, dim]

    Returns distribution with batch_shape [*shape] and event_shape [dim]
    """
    jitter = 1e-2 * torch.eye(covariance_matrix.shape[-1], device=covariance_matrix.device) * (1 + covariance_matrix) 
    return torch.distributions.MultivariateNormal(loc, covariance_matrix + jitter)

    # import pdb; pdb.set_trace()
    # Extract
    shape = loc.shape[:-1]
    dim = loc.shape[-1]
    device = loc.device
    num_elements = get_num_elements(shape)

    # Extracting bad batch id
    def get_bad_batch_id(error_str):
        if num_elements == 1:
            return 0
        else:
            end = error_str.find("U(") - 2
            start = error_str.find("batch ") + len("batch ")
            return int(error_str[start:end])

    try:
        return torch.distributions.MultivariateNormal(loc, covariance_matrix)
    except RuntimeError as error_1:
        bad_batch_id = get_bad_batch_id(str(error_1))
        jitter = 1e-6 * torch.ones(num_elements, device=device)
        jitter_prev = torch.zeros(num_elements, device=device)
        exponents = torch.zeros(num_elements, device=device)

        while True:
            jitter_new = jitter[bad_batch_id] * (10 ** exponents[bad_batch_id])
            exponents[bad_batch_id] += 1
            if jitter_new > 1.0:
                raise error_1
            covariance_matrix.view((num_elements, dim, dim))[bad_batch_id].diagonal(
                dim1=-2, dim2=-1
            ).add_(jitter_new - jitter_prev[bad_batch_id])
            jitter_prev[bad_batch_id] = jitter_new
            try:
                if verbose:
                    logging.warn(
                        f"WARNING: cov not p.d., added jitter of {jitter_new} to batch "
                        f"{bad_batch_id} of the diagonal"
                    )
                return torch.distributions.MultivariateNormal(loc, covariance_matrix)
            except RuntimeError as error_2:
                bad_batch_id = get_bad_batch_id(str(error_2))
                error_1 = error_2
                continue
        raise error_1


def condition_mvn(multivariate_normal_dist, y):
    """Given a joint multivariate distribution
    p(y, x) = N([y, x] | [μ_y, μ_x], [[Σ_yy, Σ_yx], [Σ_xy, Σ_xx]]),
    return the conditional distribution conditioned on a subset of values y,
    p(x | y) = N(x | μ_{x|y}, Σ_{x|y})

    Reference
    https://www.robots.ox.ac.uk/~mosb/teaching/AIMS_CDT/Gaussian_identities.pdf

    Args
        multivariate_normal_dist: distribution with batch_shape [*shape] and event_shape [dim]
            with per element
            mean
                [mean_y
                 mean_x]
            and cov
                [cov_yy, cov_yx
                 cov_xy, cov_xx]

        y [*shape, dim_y]

    Returns distribution with batch_shape [*shape] and event_shape [dim - dim_y]
    """
    # Extract
    dim_y = y.shape[-1]
    loc, cov = multivariate_normal_dist.mean, multivariate_normal_dist.covariance_matrix

    # Extract component locs and covs
    loc_y = loc[..., :dim_y]
    loc_x = loc[..., dim_y:]
    cov_yy = cov[..., :dim_y, :dim_y]
    cov_yx = cov[..., :dim_y, dim_y:]
    cov_xx = cov[..., dim_y:, dim_y:]
    cov_xy = cov[..., dim_y:, :dim_y]

    # Compute new params
    # -- Compute matmul(inv(cov_yy), y - loc_y)
    # [*shape, dim_y]
    temp_1 = torch.solve((y - loc_y)[..., None], cov_yy)[0][..., 0]

    # -- Compute matmul(inv(cov_yy), cov_yx)
    # [*shape, dim_y, dim_x]
    temp_2 = torch.solve(cov_yx, cov_yy)[0]

    # -- Compute new loc
    # [*shape, dim_x]
    loc_new = loc_x + torch.einsum("...xy,...y->...x", cov_xy, temp_1)

    # -- Compute new cov
    # [*shape, dim_x, dim_x]
    cov_new = cov_xx - torch.einsum("...zy,...yx->...zx", cov_xy, temp_2)

    return get_multivariate_normal_dist(loc_new, covariance_matrix=cov_new)


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


def get_num_elements(shape):
    return int(torch.tensor(shape).prod().long().item())


def init_cnn(output_dim, input_num_channels=1, hidden_dim=128):
    layers = []
    layers.append(nn.Conv2d(input_num_channels, int(hidden_dim / 2), kernel_size=3, padding=2))
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


def pad_tensor(x, length, value):
    """Pads a tensor with a prespecified value

    Args
        x [*shape, max_length, ...]
        length [*shape]
        value []

    Returns [*shape, max_length, ...]
    """
    # Extract
    shape = length.shape
    max_length = x.shape[len(shape)]
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


def get_max_gpu_memory_allocated_MB(device):
    return torch.cuda.max_memory_allocated(device=device) / 1e6 if device.type == "cuda" else 0


def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810
    e.g.
    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


def cycle(iterable):
    """Returns an infinite iterator by cycling through the iterable.

    It's better than the itertools.cycle function because it resets to iterable each time it is
    exhausted. This is useful when using cycling through the torch.utils.data.DataLoader object.

    See https://stackoverflow.com/a/49987606
    """
    while True:
        for x in iterable:
            yield x


def tensors_to_list(tensors):
    """Convert tensor / list of tensors to a list of tensors

    Args
        tensors: tensor or list of tensors

    Returns list of tensors
    """
    if torch.is_tensor(tensors):
        return [tensors]
    else:
        return tensors


def logit(y):
    """Inverse of sigmoid
    https://github.com/pytorch/pytorch/issues/37060
    """
    return torch.log(y) - torch.log1p(-y)


if __name__ == "__main__":
    # Test pad_tensor
    x = torch.rand(4)
    length = torch.randint(0, 4, ())
    value = -1
    print(f"x = {x}")
    print(f"length = {length}")
    print(f"pad_tensor(x, length, value) = {pad_tensor(x, length, value)}")
