from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import render
import util
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = util.init_mlp(2, 1, 100, 3)

    @property
    def device(self):
        return next(self.mlp.parameters()).device

    def get_dist(self, num_rows, num_cols):
        canvas_x, canvas_y = render.get_canvas_xy(num_rows, num_cols, self.device)
        mlp_input = torch.stack([canvas_x, canvas_y], dim=-1).view(num_rows * num_cols, 2)
        logits = self.mlp(mlp_input).view(num_rows, num_cols)
        return torch.distributions.Independent(
            torch.distributions.Bernoulli(logits=logits), reinterpreted_batch_ndims=2
        )


def main():
    # Data
    num_rows, num_cols, device = 64, 64, "cuda"
    canvas = torch.zeros((num_rows, num_cols), device=device)
    position = torch.tensor([0, 0], device=device)
    scale = torch.tensor(1.0, device=device)
    heart_canvas = render.render_heart((position, scale), canvas).detach()

    # Model
    model = Model().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Train
    losses = []
    frame_iterations = []
    num_iterations = 10000
    for iteration in tqdm(range(num_iterations)):
        loss = -model.get_dist(num_rows, num_cols).log_prob(heart_canvas)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if iteration % 100 == 0:
            # Plot
            fig, axs = plt.subplots(1, 5, figsize=(30, 6))
            axs[0].set_title(f"Image ({num_rows} x {num_cols})")
            axs[0].imshow(heart_canvas.cpu(), cmap="Greys", vmin=0, vmax=1)

            for i, resolution_multiplier in enumerate([1, 2, 4]):
                axs[1 + i].set_title(
                    f"Reconstruction ({resolution_multiplier * num_rows} x "
                    f"{resolution_multiplier * num_cols})"
                )
                axs[1 + i].imshow(
                    model.get_dist(
                        num_rows * resolution_multiplier, num_cols * resolution_multiplier
                    )
                    .base_dist.probs.cpu()
                    .detach()
                    .numpy(),
                    cmap="Greys",
                    vmin=0,
                    vmax=1,
                )

            axs[-1].set_title("Loss")
            axs[-1].plot(losses)
            axs[-1].set_xlabel("Iteration")
            util.save_fig(fig, f"save/occupancy/training/{iteration}.png")
            frame_iterations.append(iteration)

    # Make video
    util.make_gif(
        [f"save/occupancy/training/{frame_iteration}.png" for frame_iteration in frame_iterations],
        "save/occupancy/training.gif",
        fps=3,
    )


if __name__ == "__main__":
    main()
