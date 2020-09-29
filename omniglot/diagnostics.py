import os
import util
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import data
import numpy as np
import models
import models_continuous
import math
import random
import rendering
from pathlib import Path

# hack for https://github.com/dmlc/xgboost/issues/1715
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = util.get_device()


def plot_losses(
    path,
    theta_losses,
    phi_losses,
    dataset_size,
    prior_losses=None,
    accuracies=None,
    novel_proportions=None,
    new_maps=None,
    batch_split=None,
):
    util.logging.info("plot_losses")
    loss_str = ""
    num_subplots = 2 + len(
        [x for x in [accuracies, novel_proportions, new_maps] if x is not None and len(x) > 0]
    )
    fig, axs = plt.subplots(1, num_subplots, figsize=(2 * num_subplots, 3), dpi=100)
    ax = axs[0]
    ax.plot(theta_losses)
    if prior_losses is not None:
        ax.plot(np.abs(prior_losses))  # abs because I accidentally saved them negative whoops
        ax.plot(
            np.ones(len(prior_losses)) * math.log(dataset_size), color="gray", linestyle="dashed",
        )

    ax.set_xlabel("iteration (batch_split={})".format(batch_split))
    ax.set_ylabel("theta loss")
    ax.set_xticks([0, len(theta_losses) - 1])
    if len(theta_losses) > 0:
        loss_str += "theta loss: {:.2f}\n".format(np.mean(theta_losses[:-100]))

    ax = axs[1]
    ax.plot(phi_losses)
    ax.set_xlabel("iteration")
    ax.set_ylabel("phi loss")
    ax.set_xticks([0, len(phi_losses) - 1])
    if len(phi_losses) > 0:
        loss_str += "phi loss: {:.2f}\n".format(np.mean(phi_losses[:-100]))

    plot_idx = 2
    if accuracies is not None and len(accuracies) > 0:
        ax = axs[plot_idx]
        ax.plot(accuracies)
        ax.set_xlabel("iteration")
        ax.set_ylabel("accuracy")
        ax.set_xticks([0, len(accuracies) - 1])
        if len(accuracies) > 0:
            loss_str += "accuracy: {:.2f}%\n".format(np.mean(accuracies[:-100]) * 100)
        plot_idx = plot_idx + 1

    if novel_proportions is not None and len(novel_proportions) > 0:
        ax = axs[plot_idx]
        ax.plot(novel_proportions)
        ax.set_xlabel("iteration")
        ax.set_ylabel("novel_proportion")
        ax.set_xticks([0, len(novel_proportions) - 1])
        if len(novel_proportions) > 0:
            loss_str += "novel_proportion: {:.2f}%\n".format(
                np.mean(novel_proportions[:-100]) * 100
            )
        plot_idx = plot_idx + 1

    if new_maps is not None and len(new_maps) > 0:
        ax = axs[plot_idx]
        ax.plot(new_maps)
        ax.set_xlabel("iteration")
        ax.set_ylabel("new MAP?")
        ax.set_ylim(0, 0.2)
        ax.set_xticks([0, len(new_maps) - 1])
        if len(new_maps) > 0:
            loss_str += "new_map: {:.2f}%\n".format(np.mean(new_maps[:-100]) * 100)
        plot_idx = plot_idx + 1

    loss_str += "iteration: {}".format(len(theta_losses))

    with open(path + ".txt", "w") as text_file:
        text_file.write(loss_str)

    for ax in axs:
        sns.despine(ax=ax, trim=True)

    util.save_fig(fig, path)


def plot_log_ps(path, log_ps):
    util.logging.info("plot_log_ps")
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
    ax.plot(log_ps)
    ax.set_xlabel("iteration")
    ax.set_ylabel("log p")
    sns.despine(ax=ax, trim=True)
    util.save_fig(fig, path)


def plot_kls(path, kls):
    util.logging.info("plot_kls")
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
    ax.plot(kls)
    ax.set_xlabel("iteration")
    ax.set_ylabel("KL")
    sns.despine(ax=ax, trim=True)
    util.save_fig(fig, path)


def sample_from_guide(guide, obs):
    """
    Args:
        guide
        obs [batch_size, num_rows, num_cols]

    Returns:
        ids_and_on_offs [batch_size, num_arcs, 2]
    """
    if isinstance(guide, models.Guide):
        return guide.sample(obs, 1)[0]
    elif isinstance(guide, models_continuous.Guide):
        return guide.sample(obs, 1)[0][0]


def sample_from_memory(memory, generative_model, obs, obs_id):
    """
    Args:
        memory
        generative_model
        obs [batch_size, num_rows, num_cols]
        obs_id

    Returns:
        ids_and_on_offs [batch_size, num_arcs, 2]
    """
    # [batch_size, memory_size, num_arcs, 2]
    memory_latent = memory[obs_id]
    memory_latent_transposed = memory_latent.transpose(
        0, 1
    ).contiguous()  # [memory_size, batch_size, num_arcs, 2]
    if isinstance(generative_model, models.GenerativeModel):
        latent = memory_latent_transposed
    elif isinstance(generative_model, models_continuous.GenerativeModel):
        motor_noise = torch.zeros(
            [*memory_latent_transposed.shape[:-1], 3], device=memory_latent_transposed.device,
        )
        latent = memory_latent_transposed, motor_noise
    memory_log_p = generative_model.get_log_prob(latent, obs)  # [memory_size, batch_size]
    dist = torch.distributions.Categorical(
        probs=util.exponentiate_and_normalize(memory_log_p.t(), dim=1)
    )
    sampled_memory_id = dist.sample()  # [batch_size]
    return torch.gather(
        memory_latent_transposed,
        0,
        sampled_memory_id[None, :, None, None].repeat(1, 1, generative_model.num_arcs, 2),
    )[
        0
    ]  # [batch_size, num_arcs, 2]


def plot_reconstructions(
    path,
    test,
    generative_model,
    guide,
    num_reconstructions,
    dataset,
    data_location,
    memory=None,
    resolution=28,
    data_size="small",
):
    util.logging.info("plot_reconstructions")
    (
        data_train,
        data_valid,
        data_test,
        target_train,
        target_valid,
        target_test,
    ) = data.load_binarized_omniglot_with_targets(location=args.data_location, data_size=data_size)

    if test:
        data_test = torch.tensor(data_test, device=device)
        # obs_id = torch.tensor(
        #     np.random.choice(np.arange(len(data_test)), num_reconstructions), device=device
        # ).long()
        num_reconstructions = min(len(data_test), num_reconstructions)
        obs_id = torch.arange(num_reconstructions, device=device).long()
        obs = data_test[obs_id].float().view(-1, 28, 28)
        obs_id = None
    else:
        data_train = torch.tensor(data_train, device=device)
        # obs_id = torch.tensor(
        #     np.random.choice(np.arange(len(data_train)), num_reconstructions), device=device
        # ).long()
        num_reconstructions = min(len(data_train), num_reconstructions)
        obs_id = torch.arange(num_reconstructions, device=device).long()
        obs = data_train[obs_id].float().view(-1, 28, 28)

    start_point = torch.tensor([0.5, 0.5], device=device)[None].expand(num_reconstructions, -1)
    if memory is None:
        # [batch_size, num_arcs, 2]
        ids_and_on_offs = sample_from_guide(guide, obs)
    else:
        # [batch_size, num_arcs, 2]
        ids_and_on_offs = sample_from_memory(memory, generative_model, obs, obs_id)

    ids = ids_and_on_offs[..., 0]
    on_offs = ids_and_on_offs[..., 1]
    num_arcs = ids.shape[1]

    reconstructed_images = []
    for arc_id in range(num_arcs):
        reconstructed_images.append(
            models.get_image_probs(
                ids[:, : (arc_id + 1)],
                on_offs[:, : (arc_id + 1)],
                start_point,
                generative_model.get_primitives(),
                generative_model.get_rendering_params(),
                resolution,
                resolution,
            ).detach()
        )

    fig, axss = plt.subplots(
        num_reconstructions,
        num_arcs + 1,
        figsize=((num_arcs + 1) * 2, num_reconstructions * 2),
        sharex=False,
        sharey=False,
    )
    for i, axs in enumerate(axss):
        obs_ = obs[i]
        axs[0].imshow(obs_.cpu(), "Greys", vmin=0, vmax=1)
        for j, ax in enumerate(axs[1:]):
            tmp = len(axs[1:])
            char_on = on_offs[i].cpu()
            for k in range(num_arcs):
                if char_on[-k]:
                    tmp = tmp - 1
                if tmp == j:
                    ax.imshow(reconstructed_images[-k][i].cpu(), "Greys", vmin=0, vmax=1)
                    break

            if j < num_arcs:
                ax.text(
                    0.99,
                    0.99,
                    f"{ids[i, j]} {'ON' if on_offs[i, j] else 'OFF'}",
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=ax.transAxes,
                )

    for axs in axss:
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            sns.despine(ax=ax, left=True, right=True, top=True, bottom=True)

    axss[0, 0].set_title(r"Image $x$")
    axss[0, 1].set_title("Sequential\nreconstruction ->")

    util.save_fig(fig, path)


def plot_primitives(path, generative_model):
    util.logging.info("plot_primitives")
    num_primitives = generative_model.num_primitives
    ids = torch.arange(num_primitives, device=device).long().unsqueeze(1)
    on_offs = torch.ones(num_primitives, device=device).long().unsqueeze(1)
    start_point = torch.Tensor([0.5, 0.5]).unsqueeze(0).repeat(num_primitives, 1).to(device)

    primitives = generative_model.get_primitives()
    rendering_params = generative_model.get_rendering_params()
    primitives_imgs = models.get_image_probs(
        ids,
        on_offs,
        start_point,
        primitives,
        rendering_params,
        generative_model.num_rows,
        generative_model.num_cols,
    ).detach()

    num_rows = math.floor(math.sqrt(num_primitives))
    num_cols = num_rows
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2), sharex=True, sharey=True
    )

    for i, axs in enumerate(axss):
        for j, ax in enumerate(axs):
            ax.imshow(primitives_imgs[i * num_cols + j].cpu(), "Greys", vmin=0, vmax=1)
            ax.text(
                0.99,
                0.99,
                "{}".format(i * num_cols + j),
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout(pad=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    util.logging.info(f"Saved to {path}")
    plt.close(fig)


def plot_prior(path, generative_model, num_samples, resolution=28):
    util.logging.info("plot_prior")
    try:
        latent, _ = generative_model.sample_latent_and_obs(num_samples=num_samples)
    except NotImplementedError:
        util.logging.info("Can't plot samples for this model")
        return
    generative_model.num_rows, generative_model.num_cols = resolution, resolution
    obs_probs = generative_model.get_obs_params(latent).sigmoid().detach()
    generative_model.num_rows, generative_model.num_cols = 28, 28

    num_rows = math.floor(math.sqrt(num_samples))
    num_cols = num_rows
    if num_rows * num_cols < num_samples:
        util.logging.info(
            "Plotting {} * {} samples instead of {} samples.".format(
                num_rows, num_cols, num_samples
            )
        )
    fig, axss = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows))
    for i, axs in enumerate(axss):
        for j, ax in enumerate(axs):
            ax.imshow(obs_probs[i * num_cols + j].cpu(), cmap="Greys", vmin=0, vmax=1)
            sns.despine(ax=ax, left=True, right=True, top=True, bottom=True)
            ax.set_xticks([])
            ax.set_yticks([])
            if isinstance(generative_model, models.GenerativeModel):
                ids_and_on_offs = latent
            elif isinstance(generative_model, models_continuous.GenerativeModel):
                ids_and_on_offs = latent[0]
            arc_ids = ids_and_on_offs[i * num_cols + j][:, 0]
            on_off_ids = ids_and_on_offs[i * num_cols + j][:, 1]

    util.save_fig(fig, path)


def plot_renderer(path):
    util.logging.info("plot_renderer")

    # Set up sweep
    num_interpolations = 20

    dx = torch.linspace(-0.4, 0.4, num_interpolations)
    dy = torch.linspace(-0.4, 0.4, num_interpolations)
    theta = torch.linspace(-0.99 * math.pi, 0.99 * math.pi, num_interpolations)
    sharpness = torch.linspace(-20, 20, num_interpolations)
    width = torch.linspace(0, 0.1, num_interpolations)
    scale = torch.linspace(0, 10, num_interpolations)
    bias = torch.linspace(-10, 10, num_interpolations)
    interpolations = [dx, dy, theta, sharpness, width]
    interpolations_str = ["dx", "dy", "theta", "sharpness", "width"]

    dx_constant = 0.2
    dy_constant = 0.2
    theta_constant = math.pi / 3
    sharpness_constant = 20.0
    width_constant = 0.01
    scale_constant = 3.0
    bias_constant = -3
    constants = [dx_constant, dy_constant, theta_constant, sharpness_constant, width_constant]

    # Plot
    num_rows, num_cols = 100, 100
    fig, axss = plt.subplots(
        7, num_interpolations, figsize=(num_interpolations * 2, 7 * 2), dpi=200
    )

    # Sweep over `dx`, `dy`, `theta`, `sharpness`, `width`
    for interpolation_id in range(5):
        # Render
        # [num_interpolations, 1, 7]
        arcs = torch.cat(
            # [x_start, y_start]
            [torch.tensor(0.5).repeat(num_interpolations, 2)]
            # keep properties before `interpolation_id` constant
            + [
                torch.tensor(constants[i]).repeat(num_interpolations, 1)
                for i in range(interpolation_id)
            ]
            # interpolate `interpolation_id`'s property
            + [interpolations[interpolation_id][:, None]]
            # keep properties after `interpolation_id` constant
            + [
                torch.tensor(constants[i]).repeat(num_interpolations, 1)
                for i in range(interpolation_id + 1, 5)
            ],
            dim=1,
        )[:, None, :]

        # [num_interpolations, 1]
        on_offs = torch.ones(num_interpolations, 1).long()

        # [2]
        rendering_params = torch.tensor([scale_constant, bias_constant])

        probs = rendering.get_probs(arcs, on_offs, rendering_params, num_rows, num_cols)

        # Plot rendered images
        for i in range(num_interpolations):
            ax = axss[interpolation_id, i]
            ax.imshow(probs[i], vmin=0, vmax=1, cmap="Greys")
            if interpolations_str[interpolation_id] == "theta":
                interpolation = f"{math.degrees(interpolations[interpolation_id][i].item()):.2f}°"
            else:
                interpolation = f"{interpolations[interpolation_id][i]:.2f}"
            ax.text(
                0.99,
                0.99,
                f"{interpolations_str[interpolation_id]} = {interpolation}",
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
            )

    # Sweep over `scale`, `bias`
    for i in range(num_interpolations):
        # Rendering params
        # [1, 1, 7]
        arcs = torch.cat(
            [torch.tensor(0.5).repeat(1, 2)]
            + [torch.tensor(constants[i]).repeat(1, 1) for i in range(5)],
            dim=1,
        )[:, None, :]

        # [1, 1]
        on_offs = torch.ones(1, 1).long()

        # Render `scale`
        # [2]
        rendering_params = torch.tensor([scale[i], bias_constant])
        prob = rendering.get_probs(arcs, on_offs, rendering_params, num_rows, num_cols)[0]
        ax = axss[5, i]
        ax.imshow(prob, vmin=0, vmax=1, cmap="Greys")
        ax.text(
            0.99,
            0.99,
            f"scale = {scale[i]:.2f}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
        )

        # Render `bias`
        # [2]
        rendering_params = torch.tensor([scale_constant, bias[i]])
        prob = rendering.get_probs(arcs, on_offs, rendering_params, num_rows, num_cols)[0]
        ax = axss[6, i]
        ax.imshow(prob, vmin=0, vmax=1, cmap="Greys")
        ax.text(
            0.99,
            0.99,
            f"bias = {bias[i]:.2f}",
            color="grey",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
        )

    # Save fig
    for axs in axss:
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

    util.save_fig(fig, path)


def plot_motor_noise(path):
    util.logging.info("plot_motor_noise")

    # Set up sweep
    num_interpolations = 5
    num_motor_noise_samples = 10

    dx = torch.linspace(-0.4, 0.4, num_interpolations)
    dy_constant = 0.2
    theta_constant = math.pi / 3
    sharpness_constant = 20.0
    width_constant = 0.01
    scale_constant = 3.0
    bias_constant = -3

    # [num_interpolations, 1 + num_motor_noise_samples, 7]
    arcs = torch.cat(
        [
            torch.ones((num_interpolations, 1 + num_motor_noise_samples, 2)) * 0.5,
            dx[:, None, None].expand(num_interpolations, 1 + num_motor_noise_samples, 1),
            torch.ones((num_interpolations, 1 + num_motor_noise_samples, 1)) * dy_constant,
            torch.ones((num_interpolations, 1 + num_motor_noise_samples, 1)) * theta_constant,
            torch.ones((num_interpolations, 1 + num_motor_noise_samples, 1)) * sharpness_constant,
            torch.ones((num_interpolations, 1 + num_motor_noise_samples, 1)) * width_constant,
        ],
        dim=-1,
    )
    # Add motor noise
    dx_motor_noise = torch.distributions.Normal(0, 0.05).sample(
        (num_interpolations, num_motor_noise_samples)
    )
    dy_motor_noise = torch.distributions.Normal(0, 0.05).sample(
        (num_interpolations, num_motor_noise_samples)
    )
    theta_motor_noise = torch.distributions.Normal(0, math.pi / 18).sample(
        (num_interpolations, num_motor_noise_samples)
    )
    arcs[:, 1:, 2:5] += torch.stack([dx_motor_noise, dy_motor_noise, theta_motor_noise], dim=-1)

    # [num_interpolations, 1 + num_motor_noise_samples, 100, 100]
    probs = rendering.get_probs(
        arcs.view(num_interpolations * (1 + num_motor_noise_samples), 1, 7),
        torch.ones(num_interpolations * (1 + num_motor_noise_samples), 1).long(),
        torch.tensor([scale_constant, bias_constant]),
        100,
        100,
    ).view(num_interpolations, 1 + num_motor_noise_samples, 100, 100)

    # Plot
    num_rows, num_cols = num_interpolations, 1 + num_motor_noise_samples
    fig, axss = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2), dpi=200)

    for i in range(num_interpolations):
        for j in range(1 + num_motor_noise_samples):
            ax = axss[i, j]
            ax.imshow(probs[i, j], vmin=0, vmax=1, cmap="Greys")

    axss[0, 0].set_title("Stroke type")
    axss[0, 1].set_title(f"Stroke tokens ->")
    # Save fig
    for axs in axss:
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

    util.save_fig(fig, path)


def plot_dataset(data_size, max_num_imgs=1000):
    util.logging.info("plot_dataset")
    (
        data_train,
        data_valid,
        data_test,
        target_train,
        target_valid,
        target_test,
    ) = data.load_binarized_omniglot_with_targets(location=args.data_location, data_size=data_size)

    for test, imgs in zip([True, False], [data_test[:max_num_imgs], data_train[:max_num_imgs]]):
        num_cols = 10
        num_rows = math.ceil(len(imgs) / 10)

        fig, axss = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))

        for i, (ax, img) in enumerate(zip(axss.flat, imgs)):
            ax.imshow(img, "Greys", vmin=0, vmax=1)
            ax.text(
                0.99,
                0.99,
                str(i),
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
            )

        for axs in axss:
            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])
                sns.despine(ax=ax, left=True, right=True, top=True, bottom=True)

        filename = "test.pdf" if test else "train.pdf"
        path = f"save/_data/{data_size}/{filename}"
        util.save_fig(fig, path)


def main(args):
    for data_size in ["mini", "full", "small"]:
        plot_dataset(data_size)
    plot_motor_noise("save/_model/motor_noise.pdf")
    plot_renderer("save/_model/renderer.pdf")
    dataset = "omniglot"

    if args.checkpoint_path is None:
        checkpoint_paths = list(util.get_checkpoint_paths())
    else:
        checkpoint_paths = [args.checkpoint_path]

    if args.shuffle:
        random.shuffle(checkpoint_paths)

    for checkpoint_path in checkpoint_paths:
        try:
            ((generative_model, guide), optimizer, memory, stats, run_args,) = util.load_checkpoint(
                checkpoint_path, device=device
            )
        except FileNotFoundError as e:
            print(e)
            if "No such file or directory" in str(e):
                print(e)
                continue

        iteration = len(stats.theta_losses)
        (
            data_train,
            data_valid,
            data_test,
            target_train,
            target_valid,
            target_test,
        ) = data.load_binarized_omniglot_with_targets(
            location=args.data_location, data_size=run_args.data_size
        )
        dataset_size = data_train.shape[0]

        diagnostics_dir = util.get_save_dir(run_args)
        Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)

        plot_losses(
            f"{diagnostics_dir}/losses.pdf",
            stats.theta_losses,
            stats.phi_losses,
            dataset_size,
            stats.prior_losses,
            stats.accuracies,
            stats.novel_proportions,
            stats.new_maps,
            1,
        )
        if args.loss_only:
            continue
        if len(stats.kls) > 0:
            plot_log_ps(f"{diagnostics_dir}/logp.pdf", stats.log_ps)
            plot_kls(f"{diagnostics_dir}/kl.pdf", stats.kls)

        for test in [False, True]:
            test_str = "test" if test else "train"

            # Guide
            plot_reconstructions(
                f"{diagnostics_dir}/reconstructions/{test_str}/guide/{iteration}.pdf",
                test,
                generative_model,
                guide,
                args.num_reconstructions,
                dataset,
                args.data_location,
                None,
                args.resolution,
                data_size=run_args.data_size,
            )

            # Memory
            if not test and memory is not None:
                plot_reconstructions(
                    f"{diagnostics_dir}/reconstructions/{test_str}/memory/{iteration}.pdf",
                    test,
                    generative_model,
                    guide,
                    args.num_reconstructions,
                    dataset,
                    args.data_location,
                    memory,
                    args.resolution,
                    data_size=run_args.data_size,
                )

        plot_primitives(f"{diagnostics_dir}/primitives/{iteration}.pdf", generative_model)
        plot_prior(
            f"{diagnostics_dir}/prior/{iteration}.pdf", generative_model, args.num_prior_samples,
        )
        print("-----------------")


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--repeat", action="store_true", help="")
    parser.add_argument("--shuffle", action="store_true", help="")

    # load checkpoint
    parser.add_argument("--checkpoint-path", type=str, default=None, help=" ")
    parser.add_argument("--data-location", default="local", help=" ")

    # plot
    parser.add_argument("--diagnostics-dir", default="diagnostics", help=" ")
    parser.add_argument("--loss-only", action="store_true")
    parser.add_argument("--resolution", type=int, default=28)
    parser.add_argument("--num-reconstructions", type=int, default=100, help=" ")
    parser.add_argument("--num-prior-samples", type=int, default=100)

    # for custom checkpoints
    parser.add_argument("--checkpoint-iteration", default=None, type=int)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    with torch.no_grad():
        if args.repeat:
            while True:
                main(args)
        else:
            main(args)
