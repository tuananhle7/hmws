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
            remove_color=run_args.remove_color, # map from int to bool
            mode=run_args.mode,
            shrink_factor=run_args.shrink_factor,
            learn_blur=run_args.learn_blur,
            blur_scale=run_args.blur_scale
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
                len(data.SceneUnderstandingDataset(device, test=False,remove_color=run_args.remove_color,
                                                   mode=run_args.mode,shrink_factor=run_args.shrink_factor)),
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


def importance_sample_memory(
        num_particles, obs, obs_id, generative_model, guide, memory, img_size=256
):
    # modifed from: https://github.com/tuananhle7/continuous_mws/blob/a43dd325e1e2c765d9811773ff5885b6f5f400e4/cmws/examples/timeseries/inference.py#L239
    """
    Args
        num_particles
        num_svi_iterations
        obs [batch_size, num_timesteps]
        obs_id [batch_size]
        generative_model
        guide
        memory
    Returns
        latent
            raw_expression [memory_size, batch_size, max_num_chars]
            eos [memory_size, batch_size, max_num_chars]
            raw_gp_params [memory_size, batch_size, max_num_chars, gp_params_dim]
        log_marginal_joint [memory_size, batch_size]
    """
    # Extract
    batch_size = obs.shape[0]
    memory_size = memory.size

    # Sample discrete latent
    # [memory_size, batch_size, ...]
    discrete_latent = memory.select(obs_id)

    # COMPUTE SCORES s_i = log p(d_i, x) for i  {1, ..., M}
    # [memory_size, batch_size]
    log_marginal_joint = cmws.losses.get_log_marginal_joint(
        generative_model, guide, discrete_latent, obs, num_particles
    )

    # Sample svi-optimized q(z_c | z_d, x)
    # -- Expand obs
    # [memory_size, batch_size, 3, img_size, img_size]
    obs_expanded = obs[None].expand([memory_size, batch_size, 3, img_size, img_size])
    # -- SVI
    continuous_latent = guide.sample_continuous(obs_expanded, discrete_latent)

    # Combine latents
    latent = discrete_latent[0], discrete_latent[1], continuous_latent

    return latent, log_marginal_joint


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

    try:
        mode = run_args.mode
    except:
        run_args.mode = "cube"  # hack to handle being modes run before primitive changes

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
