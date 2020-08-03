import imageio
import util
import sweep
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np


def plot_animation_frame(checkpoint_iteration):
    sweep_argss = list(sweep.get_sweep_argss())
    sweep_args = sweep_argss[0]
    device = "cpu"
    checkpoint_path = util.get_checkpoint_path(sweep_args, checkpoint_iteration)
    (generative_model, guide, optimizer, memory, stats, run_args) = util.load_checkpoint(
        checkpoint_path, device=device
    )

    if checkpoint_iteration == -1:
        iteration = len(stats.losses)
    else:
        iteration = checkpoint_iteration
    path = f"./save/animation/{iteration}.png"

    num_rows = 1 + 3  # len(sweep_argss)
    num_cols = 2  # 3
    fig, axss = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 4), sharex="col")

    for axs in axss:
        for ax in axs:
            ax.tick_params(length=0)

    # GENERATIVE MODEL
    axs = axss[0]
    generative_model.plot_discrete(axs[0])
    generative_model.plot_continuous(axs[1])
    # axs[-1].set_axis_off()
    axs[1].legend(fontsize=12)

    # LABELS
    axs[0].set_ylabel("Ground\ntruth", fontsize=36)
    axs[0].set_title("$p(d)$", fontsize=36)
    axs[1].set_title("$p(c | d)$", fontsize=36)
    axss[1, 0].set_title("$q(d)$", fontsize=36)
    axss[1, 1].set_title("$q(c | d)$", fontsize=36)
    axss[3, 0].set_title("$q_{{memory}}(d)$", fontsize=36)
    # axss[4, -1].set_title("$q_{{memory}}(c | d)$", fontsize=36)
    i = 0
    for sweep_args in sweep_argss:
        if (
            sweep_args.seed != 0
            or (sweep_args.cmws_estimator == "is" and sweep_args.num_particles != 10)
            or (sweep_args.cmws_estimator == "sgd" and sweep_args.num_cmws_iterations != 10)
        ):
            continue
        # print(f"i = {i} | sweep_args = {sweep_args}")
        checkpoint_path = util.get_checkpoint_path(sweep_args, checkpoint_iteration)
        (generative_model, guide, optimizer, memory, stats, run_args) = util.load_checkpoint(
            checkpoint_path, device=device
        )
        axs = axss[i + 1]
        # axs[0].set_ylabel(util.get_path_base_from_args(run_args).upper(), fontsize=36)
        axs[0].set_ylabel(run_args.cmws_estimator.upper(), fontsize=36)
        if run_args.algorithm == "mws":
            # DISCRETE MEMORY
            support_size = generative_model.support_size
            axs[0].bar(
                torch.arange(support_size),
                util.empirical_discrete_probs(memory[0], support_size).cpu(),
            )

            # CONTINUOUS GUIDE
            ax = axs[1]
            xs = torch.linspace(-support_size, support_size, steps=1000, device=device)
            discrete = memory[0].clone().detach()  # torch.arange(support_size, device=guide.device)
            continuous_dist = guide.get_continuous_dist(discrete)
            # [memory_size, len(xs)]
            probss = (
                continuous_dist.log_prob(xs[:, None].expand(-1, len(memory[0]))).T.exp().detach()
            )
            for _, (memory_element, probs) in enumerate(zip(memory[0], probss)):
                ax.plot(
                    xs.cpu(),
                    probs.cpu(),
                    label=f"$d = {memory_element}$",
                    color=f"C{memory_element}",
                    linewidth=3,
                )
            ax.set_xlim(-support_size, support_size)

            # CONTINUOUS MEMORY
            ax = axs[-1]
            for j in range(support_size):
                if sum(memory[0] == j) > 0:
                    sns.kdeplot(
                        memory[1][memory[0] == j].cpu().detach().numpy(),
                        ax=ax,
                        color=f"C{j}",
                        linewidth=3,
                    )
            ax.set_xlim(-support_size, support_size)
        elif run_args.algorithm == "cmws":
            # DISCRETE MEMORY
            support_size = generative_model.support_size
            support = torch.arange(support_size, device=device)
            # [memory_size]
            memory_prob = util.get_memory_prob(
                generative_model,
                guide,
                memory,
                num_particles=run_args.num_particles,
                num_iterations=run_args.num_cmws_iterations,
            )
            axs[0].bar(support.cpu(), memory_prob.cpu())

            # CONTINUOUS GUIDE
            ax = axs[1]
            xs = torch.linspace(-support_size, support_size, steps=1000, device=device)
            discrete = memory.clone().detach()  # torch.arange(support_size, device=guide.device)
            continuous_dist = guide.get_continuous_dist(discrete)
            # [memory_size, len(xs)]
            probss = continuous_dist.log_prob(xs[:, None].expand(-1, len(memory))).T.exp().detach()
            for _, (memory_element, probs) in enumerate(zip(memory, probss)):
                ax.plot(
                    xs.cpu(),
                    probs.cpu(),
                    label=f"$d = {memory_element}$",
                    color=f"C{memory_element}",
                    linewidth=3,
                )
            ax.set_xlim(-support_size, support_size)

            # axs[-1].set_axis_off()
        else:
            # DISCRETE GUIDE
            guide.plot_discrete(axs[0])

            # CONTINUOUS GUIDE
            guide.plot_continuous(axs[1])

            # axs[-1].set_axis_off()

        i += 1
    fig.suptitle(f"Iteration {iteration}", fontsize=36)
    util.save_fig(fig, path, dpi=50, tight_layout_kwargs={"rect": [0, 0.03, 1, 0.95]})


if __name__ == "__main__":
    checkpoint_iterations = [int(i) for i in np.arange(0, 20000, 2000)] + [-1]
    # checkpoint_iterations = [
    #     0,
    #     100,
    #     200,
    #     500,
    #     1000,
    #     2000,
    #     3000,
    #     4000,
    #     5000,
    #     6000,
    #     7000,
    #     8000,
    #     9000,
    #     -1,
    # ]
    for checkpoint_iteration in checkpoint_iterations:
        plot_animation_frame(checkpoint_iteration)

    images = []
    for checkpoint_iteration in checkpoint_iterations:
        if checkpoint_iteration == -1:
            iteration = 20000
        else:
            iteration = checkpoint_iteration
        path = f"./save/animation/{iteration}.png"
        images.append(imageio.imread(path))
    imageio.mimsave("./save/animation/training_progress.gif", images, duration=0.5)
