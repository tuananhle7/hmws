import pyredner
import torch
import util
import matplotlib.pyplot as plt
import render


def plot(target_img, img, losses, iteration, path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(target_img.cpu().detach())
    axs[0].set_title("Target")
    axs[1].imshow(img.cpu().detach())
    axs[1].set_title(f"Rendered (iter {iteration})")
    axs[2].plot(losses)
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Loss")
    for ax in axs[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    util.save_fig(fig, path)


def get_position(raw_position, true_position):
    if raw_position.ndim == 0:
        position = true_position.clone().detach()
        position[0] = raw_position.sigmoid() * 1.6 - 0.8
        return position
    else:
        pass


if __name__ == "__main__":
    pyredner.set_use_gpu(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    im_size = 32
    true_size = torch.tensor(0.2, device=device)
    true_position = torch.tensor([0.0, 0.6, -1.0], device=device)
    true_color = torch.tensor([1.0, 0, 0], device=device)
    target_img = render.render_cube(true_size, true_color, true_position, im_size)

    raw_size = torch.tensor(0.0, device=device, requires_grad=True)
    raw_color = torch.randn(3, device=device, requires_grad=True)
    raw_x_position = torch.zeros((), device=device, requires_grad=True)

    optimizer = torch.optim.Adam([raw_size, raw_x_position, raw_color], lr=5e-2)
    num_iterations = 100
    losses = []

    for i in range(num_iterations):
        optimizer.zero_grad()
        img = render.render_cube(
            raw_size.exp(),
            torch.softmax(raw_color, 0),
            get_position(raw_x_position, true_position),
            im_size
        )
        loss = (img - target_img).pow(2).sum()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        plot(target_img, img, losses, i, f"save/cube/{i}.png")

        print(f"Iter. {i} | Loss {losses[-1]}")

    util.make_gif(
        [f"save/cube/{i}.png" for i in range(num_iterations)], "save/cube/reconstruction.gif", 10
    )
