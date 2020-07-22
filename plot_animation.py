import imageio
import util
import sweep
import matplotlib.pyplot as plt
import torch
import losses
import seaborn as sns


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

    num_rows = 1 + len(sweep_argss)
    num_cols = 4
    fig, axss = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 4))

    # GENERATIVE MODEL
    axs = axss[0]
    axs[0].set_axis_off()
    axs[-1].set_axis_off()

    axs[1].set_ylabel("Ground\ntruth", fontsize=36)
    generative_model.plot_discrete(axs[1])
    generative_model.plot_continuous(axs[2])
    axs[2].legend()

    # LABELS
    axs[1].set_title("$p(z_d)$", fontsize=36)
    axs[2].set_title("$p(z_c | z_d)$", fontsize=36)
    axss[1, 1].set_title("$q(z_d)$", fontsize=36)
    axss[1, 2].set_title("$q(z_c | z_d)$", fontsize=36)
    axss[3, 0].set_title("$q_{{memory}}(z_d)$", fontsize=36)
    axss[3, -1].set_title("$q_{{memory}}(z_c | z_d)$", fontsize=36)

    for i, sweep_args in enumerate(sweep_argss):
        checkpoint_path = util.get_checkpoint_path(sweep_args, checkpoint_iteration)
        (generative_model, guide, optimizer, memory, stats, run_args) = util.load_checkpoint(
            checkpoint_path, device=device
        )
        axs = axss[i + 1]
        if run_args.algorithm == "mws":
            axs[0].set_ylabel(run_args.algorithm.upper(), fontsize=36)

            # DISCRETE MEMORY
            support_size = generative_model.support_size
            axs[0].bar(
                torch.arange(support_size),
                util.empirical_discrete_probs(memory[0], support_size).cpu(),
            )

            # CONTINUOUS MEMORY
            ax = axs[-1]
            for j in range(support_size):
                if sum(memory[0] == j) > 0:
                    sns.kdeplot(
                        memory[1][memory[0] == j].cpu().detach().numpy(), ax=ax, color=f"C{j}",
                    )
            ax.set_xlim(-support_size, support_size)

            # CONTINUOUS GUIDE
            ax = axs[2]
            xs = torch.linspace(-support_size, support_size, steps=1000, device=device)
            discrete = memory[0].clone().detach()  # torch.arange(support_size, device=guide.device)
            continuous_dist = guide.get_continuous_dist(discrete)
            # [memory_size, len(xs)]
            probss = (
                continuous_dist.log_prob(xs[:, None].expand(-1, len(memory[0]))).T.exp().detach()
            )
            for i, (memory_element, probs) in enumerate(zip(memory[0], probss)):
                ax.plot(
                    xs.cpu(),
                    probs.cpu(),
                    label=f"$z_d = {memory_element}$",
                    color=f"C{memory_element}",
                )
            ax.set_xlim(-support_size, support_size)
        elif run_args.algorithm == "cmws":
            axs[0].set_ylabel(run_args.algorithm.upper(), fontsize=36)

            # DISCRETE MEMORY
            support_size = generative_model.support_size
            support = torch.arange(support_size, device=device)
            # [memory_size]
            memory_log_weight = losses.get_memory_log_weight(
                generative_model, guide, memory, run_args.num_particles
            )
            memory_prob = torch.zeros(support_size, device=device)
            memory_prob[memory] = util.exponentiate_and_normalize(memory_log_weight).detach()
            axs[0].bar(support.cpu(), memory_prob.cpu())
            axs[-1].set_axis_off()

            # CONTINUOUS GUIDE
            ax = axs[2]
            xs = torch.linspace(-support_size, support_size, steps=1000, device=device)
            discrete = memory.clone().detach()  # torch.arange(support_size, device=guide.device)
            continuous_dist = guide.get_continuous_dist(discrete)
            # [memory_size, len(xs)]
            probss = continuous_dist.log_prob(xs[:, None].expand(-1, len(memory))).T.exp().detach()
            for i, (memory_element, probs) in enumerate(zip(memory, probss)):
                ax.plot(
                    xs.cpu(),
                    probs.cpu(),
                    label=f"$z_d = {memory_element}$",
                    color=f"C{memory_element}",
                )
            ax.set_xlim(-support_size, support_size)
        else:
            axs[1].set_ylabel(run_args.algorithm.upper(), fontsize=36)
            axs[0].set_axis_off()
            axs[-1].set_axis_off()

            # CONTINUOUS GUIDE
            guide.plot_continuous(axs[2])

        # DISCRETE GUIDE
        guide.plot_discrete(axs[1])

    fig.suptitle(f"Iteration {iteration}", fontsize=36)
    util.save_fig(fig, path, dpi=50, tight_layout_kwargs={"rect": [0, 0.03, 1, 0.95]})


if __name__ == "__main__":
    for checkpoint_iteration in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, -1]:
        plot_animation_frame(checkpoint_iteration)

    images = []
    for checkpoint_iteration in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, -1]:
        if checkpoint_iteration == -1:
            iteration = 10000
        else:
            iteration = checkpoint_iteration
        path = f"./save/animation/{iteration}.png"
        images.append(imageio.imread(path))
    imageio.mimsave("./save/animation/training_progress.gif", images, duration=0.5)
