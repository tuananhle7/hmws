import collections
import itertools
import logging
import time
from pathlib import Path

import pyro
import torch
from cmws.examples.stacking_3d.models import stacking


# Init, saving, etc
def init(run_args, device):
    if run_args.model_type == "stacking":
        # Generative model
        generative_model = stacking.GenerativeModel(
            num_primitives=run_args.num_primitives, max_num_blocks=run_args.max_num_blocks
        ).to(device)

        # Guide
        guide = stacking.Guide(
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