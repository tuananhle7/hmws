import models
import collections
from pathlib import Path
import torch
import logging
import sys
import matplotlib.pyplot as plt
import os
import losses
import numpy as np
import subprocess
import getpass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(pathname)s:%(lineno)d | %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


def get_path_base_from_args(args):
    if args.cmws_estimator == "exact":
        stuff = ""
    elif args.cmws_estimator == "is":
        stuff = args.num_particles
    elif args.cmws_estimator == "sgd":
        stuff = args.num_cmws_iterations

    return f"{args.cmws_estimator}{stuff}_{args.seed}"


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


def save_fig(fig, path, dpi=100, tight_layout_kwargs={}):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(**tight_layout_kwargs)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    logging.info("Saved to {}".format(path))
    plt.close(fig)


Stats = collections.namedtuple(
    "Stats", ["losses", "cmws_memory_errors", "locs_errors", "inference_errors"],
)


def init_memory(memory_size, support_size, device):
    discrete = torch.distributions.Categorical(
        logits=torch.zeros((support_size,), device=device)
    ).sample((memory_size,))
    continuous = torch.randn((memory_size,), device=device)
    return [discrete, continuous]


def init_discrete_memory(memory_size, support_size, device):
    if memory_size > support_size:
        raise ValueError("memory_size > support_size")
    return torch.arange(memory_size, device=device)


def init(run_args, device):
    generative_model = models.GenerativeModel(run_args.support_size, device)
    guide = models.Guide(run_args.support_size).to(device)
    optimizer = torch.optim.Adam(guide.parameters())
    if run_args.algorithm == "mws":
        memory = init_memory(run_args.memory_size, run_args.support_size, device)
    elif run_args.algorithm == "cmws":
        memory = init_discrete_memory(run_args.memory_size, run_args.support_size, device)
    else:
        memory = None

    stats = Stats([], [], [], [])

    return generative_model, guide, optimizer, memory, stats


def save_checkpoint(
    path,
    # models
    generative_model,
    guide,
    optimizer,
    memory,
    # stats
    stats,
    # run args
    run_args=None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # models
            "guide_state_dict": guide.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "memory": memory,
            # stats
            "stats": stats,
            # run args
            "run_args": run_args,
        },
        path,
    )
    logging.info(f"Saved checkpoint to {path}")


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    generative_model, guide, optimizer, _, _ = init(checkpoint["run_args"], device)

    guide.load_state_dict(checkpoint["guide_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    memory = checkpoint["memory"]
    stats = checkpoint["stats"]
    run_args = checkpoint["run_args"]
    return generative_model, guide, optimizer, memory, stats, run_args


def empirical_discrete_probs(data, support_size):
    discrete_probs = torch.zeros(support_size, device=data.device)
    for i in range(support_size):
        discrete_probs[i] = (data == i).sum()
    return discrete_probs / len(data)


def get_memory_prob(generative_model, guide, memory, num_particles=None, num_iterations=None):
    support_size = generative_model.support_size
    device = memory.device

    # [memory_size]
    memory_log_weight = losses.get_memory_log_weight(
        generative_model, guide, memory, num_particles=num_particles, num_iterations=num_iterations
    )

    memory_prob = torch.zeros(support_size, device=device)
    memory_prob[memory] = exponentiate_and_normalize(memory_log_weight).detach()

    return memory_prob


def get_cmws_memory_error(generative_model, guide, memory, num_particles=None, num_iterations=None):
    cmws_memory_prob = get_memory_prob(
        generative_model, guide, memory, num_particles=num_particles, num_iterations=num_iterations,
    )
    return torch.norm(cmws_memory_prob - generative_model.discrete_dist.probs, p=2).detach().item()


def get_locs_error(generative_model, guide, memory, num_particles=None, num_iterations=None):
    # probs = generative_model.discrete_dist.probs
    probs = get_memory_prob(
        generative_model, guide, memory, num_particles=num_particles, num_iterations=num_iterations,
    )
    return (probs * (generative_model.locs - guide.locs)).abs().sum().detach().item()
    # return torch.norm(generative_model.locs - guide.locs, p=2).detach().item()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def cancel_all_my_non_bash_jobs():
    print("Cancelling all non-bash jobs.")
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
        print(cmd)
        print(subprocess.check_output(cmd, shell=True).decode())
    else:
        print("No non-bash jobs to cancel.")
