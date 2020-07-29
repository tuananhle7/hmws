import os
import matplotlib.pyplot as plt
import torch
import util
import seaborn as sns
import argparse


def plot_memory(path, memory, support_size):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    ax = axs[0]
    # ax.hist(
    #     memory[0], bins=torch.linspace(-0.5, support_size + 0.5, support_size + 1), density=True
    # )
    ax.bar(torch.arange(support_size), util.empirical_discrete_probs(memory[0], support_size).cpu())
    ax.set_ylabel("$q_M(z_d)$")

    ax = axs[1]
    for i in range(support_size):
        if sum(memory[0] == i) > 0:
            sns.kdeplot(
                memory[1][memory[0] == i].cpu().detach().numpy(),
                ax=ax,
                label=f"$z_d = {i}$",
                color=f"C{i}",
                linewidth=3,
            )
        # ax.hist(memory[1][memory[0] == i], density=True, label=f"$z_d = {i}$")
    ax.set_xlim(-support_size, support_size)
    ax.set_ylabel("$q_M(z_c | z_d)$")
    ax.legend()
    util.save_fig(fig, path)


def plot_continuous_memory(
    path, generative_model, guide, memory, num_particles=None, num_iterations=None
):
    if len(torch.unique(memory)) != len(memory):
        raise RuntimeError("memory elements not unique")
    support_size = generative_model.support_size
    memory = torch.sort(memory)[0]
    device = memory.device

    xs = torch.linspace(-support_size, support_size, steps=1000, device=device)
    discrete = memory.clone().detach()  # torch.arange(support_size, device=guide.device)
    continuous_dist = guide.get_continuous_dist(discrete)
    # [memory_size, len(xs)]
    probss = continuous_dist.log_prob(xs[:, None].expand(-1, len(memory))).T.exp().detach()

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    ax = axs[0]
    support = torch.arange(support_size, device=device)
    ax.bar(
        support.cpu(),
        util.get_memory_prob(
            generative_model,
            guide,
            memory,
            num_particles=num_particles,
            num_iterations=num_iterations,
        ).cpu(),
    )
    ax.set_ylabel("$q_M(z_d)$")

    ax = axs[1]
    for i, (memory_element, probs) in enumerate(zip(memory, probss)):
        ax.plot(
            xs.cpu(),
            probs.cpu(),
            label=f"$z_d = {memory_element}$",
            color=f"C{memory_element}",
            linewidth=3,
        )
    ax.set_xlim(-support_size, support_size)
    ax.set_ylabel("$q(z_c | z_d)$")

    ax.legend()
    util.save_fig(fig, path)


def plot_stats(path, stats):
    num_rows = 4
    num_cols = 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 4), sharex=True)

    axs[0].plot(stats.losses)
    axs[0].set_ylabel("Loss")
    axs[1].plot(stats.cmws_memory_errors)
    axs[1].set_ylabel(f"$q(d)$ error")
    axs[2].plot(stats.locs_errors)
    axs[2].set_ylabel(f"$q(c | d)$ error")
    axs[3].plot(stats.inference_errors)
    axs[3].set_ylabel(f"$q(d, c)$ error")

    util.save_fig(fig, path)


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        util.logging.info("Using CUDA")
    else:
        device = torch.device("cpu")
        util.logging.info("Using CPU")

    if args.checkpoint_path is None:
        checkpoint_paths = list(util.get_checkpoint_paths())
    else:
        checkpoint_paths = [args.checkpoint_path]

    for checkpoint_path in checkpoint_paths:
        if os.path.isfile(checkpoint_path):
            (generative_model, guide, optimizer, memory, stats, run_args) = util.load_checkpoint(
                checkpoint_path, device=device
            )
        else:
            print(f"{checkpoint_path} doesn't exist... skipping")
            continue

        diagnostics_dir = util.get_save_dir(run_args)
        plot_stats(f"{diagnostics_dir}/stats.pdf", stats)
        generative_model.plot(f"{diagnostics_dir}/generative_model.pdf")
        guide.plot(f"{diagnostics_dir}/guide.pdf")
        if run_args.algorithm == "mws":
            plot_memory(f"{diagnostics_dir}/memory.pdf", memory, generative_model.support_size)
        elif run_args.algorithm == "cmws":
            plot_continuous_memory(
                f"{diagnostics_dir}/memory.pdf",
                generative_model,
                guide,
                memory,
                num_particles=run_args.num_particles,
                num_iterations=run_args.num_iterations,
            )


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint-path", type=str, default=None, help=" ")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
