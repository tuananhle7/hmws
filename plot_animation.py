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

    path = f"./save/animation/{checkpoint_iteration}.png"

    num_rows = 1 + len(sweep_argss)
    num_cols = 4
    fig, axss = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 4))

    axss[0, 0].set_axis_off()
    axss[0, -1].set_axis_off()

    axss[0, 1].set_ylabel("Ground\ntruth", fontsize=36)
    generative_model.plot_discrete(axss[0, 1])
    axss[0, 1].set_title("$p(z_d)$", fontsize=36)
    generative_model.plot_continuous(axss[0, 2])
    axss[0, 2].set_title("$p(z_c | z_d)$", fontsize=36)

    axss[1, 1].set_title("$q(z_d)$", fontsize=36)
    axss[1, 2].set_title("$q(z_c | z_d)$", fontsize=36)

    axss[3, 0].set_title("$q_{{memory}}(z_d)$", fontsize=36)
    axss[3, -1].set_title("$q_{{memory}}(z_c | z_d)$", fontsize=36)

    for i, sweep_args in enumerate(sweep_argss):
        checkpoint_path = util.get_checkpoint_path(sweep_args, checkpoint_iteration)
        (generative_model, guide, optimizer, memory, stats, run_args) = util.load_checkpoint(
            checkpoint_path, device=device
        )
        if run_args.algorithm == "mws":
            support_size = generative_model.support_size
            axss[i + 1, 0].bar(
                torch.arange(support_size),
                util.empirical_discrete_probs(memory[0], support_size).cpu(),
            )

            ax = axss[i + 1, -1]
            for j in range(support_size):
                if sum(memory[0] == j) > 0:
                    sns.kdeplot(
                        memory[1][memory[0] == j].cpu().detach().numpy(), ax=ax, color=f"C{j}",
                    )
            ax.set_xlim(-support_size, support_size)
            axss[i + 1, 0].set_ylabel(run_args.algorithm.upper(), fontsize=36)
        elif run_args.algorithm == "cmws":
            support_size = generative_model.support_size
            support = torch.arange(support_size, device=device)
            # [memory_size]
            memory_log_weight = losses.get_memory_log_weight(
                generative_model, guide, memory, run_args.num_particles
            )
            memory_prob = torch.zeros(support_size, device=device)
            memory_prob[memory] = util.exponentiate_and_normalize(memory_log_weight).detach()
            axss[i + 1, 0].bar(support.cpu(), memory_prob.cpu())
            axss[i + 1, -1].set_axis_off()
            axss[i + 1, 0].set_ylabel(run_args.algorithm.upper(), fontsize=36)
        else:
            axss[i + 1, 1].set_ylabel(run_args.algorithm.upper(), fontsize=36)
            axss[i + 1, 0].set_axis_off()
            axss[i + 1, -1].set_axis_off()
        guide.plot_discrete(axss[i + 1, 1])
        guide.plot_continuous(axss[i + 1, 2])

    fig.suptitle(f"Iteration {checkpoint_iteration}", fontsize=36)
    util.save_fig(fig, path, tight_layout_kwargs={"rect": [0, 0.03, 1, 0.95]})


if __name__ == "__main__":
    for checkpoint_iteration in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, -1]:
        plot_animation_frame(checkpoint_iteration)
