import collections
import itertools
import logging
import time
from pathlib import Path

from cmws.examples.scene_understanding import data
import cmws
import torch
from cmws.examples.scene_understanding.models import scene_understanding
import os

# Init, saving, etc
def init(run_args, device):
    memory = None
    if run_args.model_type == "scene_understanding":
        # Generative model
        generative_model = scene_understanding.GenerativeModel(
            num_grid_rows=run_args.num_grid_rows,
            num_grid_cols=run_args.num_grid_cols,
            num_primitives=run_args.num_primitives,
            max_num_blocks=run_args.max_num_blocks,
            remove_color=(run_args.remove_color == 1) # map from int to bool
        ).to(device)

        # Guide
        guide = scene_understanding.Guide(
            num_grid_rows=run_args.num_grid_rows,
            num_grid_cols=run_args.num_grid_cols,
            num_primitives=run_args.num_primitives,
            max_num_blocks=run_args.max_num_blocks,
        ).to(device)

        # Memory
        if "mws" in run_args.algorithm:
            memory = cmws.memory.Memory(
                len(data.SceneUnderstandingDataset(device, test=False,remove_color=(run_args.remove_color == 1))),
                run_args.memory_size,
                generative_model,
            ).to(device)

    # Model tuple
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
            "optimizer_state_dict": optimizer.state_dict(),
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
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    stats = checkpoint["stats"]
    return model, optimizer, stats, run_args


Stats = collections.namedtuple("Stats", ["losses", "sleep_pretraining_losses", "log_ps", "kls"])
