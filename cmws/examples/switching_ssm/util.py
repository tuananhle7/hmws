import collections
import time
import logging
from pathlib import Path
import torch
import itertools
import cmws
from cmws.examples.switching_ssm.models import slds


# Init, saving, etc
def init(run_args, device):
    memory = None
    if run_args.model_type == "slds":
        # Generative model
        generative_model = slds.GenerativeModel(
            num_states=run_args.num_states,
            continuous_dim=run_args.continuous_dim,
            obs_dim=run_args.obs_dim,
            num_timesteps=run_args.num_timesteps,
        ).to(device)

        # Guide
        guide = slds.Guide(
            num_states=run_args.num_states,
            continuous_dim=run_args.continuous_dim,
            obs_dim=run_args.obs_dim,
            num_timesteps=run_args.num_timesteps,
        ).to(device)

        # Memory
        if "mws" in run_args.algorithm:
            memory = cmws.memory.Memory(
                1, run_args.memory_size, generative_model, check_unique=True,
            ).to(device)

    # Model dict
    model = {"generative_model": generative_model, "guide": guide, "memory": memory}

    # Optimizer
    parameters = itertools.chain(generative_model.parameters(), guide.parameters())
    optimizer = torch.optim.Adam(parameters, lr=run_args.lr)

    # Stats
    stats = Stats([], [], [], [])

    return model, optimizer, stats


def save_checkpoint(path, model, optimizer, stats, run_args=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    generative_model, guide, memory = model["generative_model"], model["guide"], model["memory"]
    torch.save(
        {
            "generative_model_state_dict": generative_model.state_dict(),
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

    generative_model, guide, memory = model["generative_model"], model["guide"], model["memory"]
    guide.load_state_dict(checkpoint["guide_state_dict"])
    generative_model.load_state_dict(checkpoint["generative_model_state_dict"])
    if memory is not None:
        memory.load_state_dict(checkpoint["memory_state_dict"])

    model = {"generative_model": generative_model, "guide": guide, "memory": memory}
    if "_pyro" in run_args.model_type:
        optimizer.set_state(checkpoint["optimizer_state_dict"])
    else:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    stats = checkpoint["stats"]
    return model, optimizer, stats, run_args


Stats = collections.namedtuple("Stats", ["losses", "sleep_pretraining_losses", "log_ps", "kls"])
