import torch


def heart_occupancy_function(x, y):
    """
    Args
        x [*shape]
        y [*shape]

    Returns bool tensor [*shape]
    """
    # Rescale so that it is inside [-1, 1]
    x = x / 0.5
    y = y / 0.5

    return (x ** 2 + y ** 2 - 1) ** 3 - x ** 2 * y ** 3 < 0


def render_heart(position_scale, canvas):
    """Draws a heart on a canvas whose xy limits are [-1, 1].

    Args
        position_scale
            position [2]
            scale [] (scalar)
        canvas [num_rows, num_cols]

    Returns
        new_canvas [num_rows, num_cols]
    """
    # Extract
    position, scale = position_scale
    position_x, position_y = position
    num_rows, num_cols = canvas.shape
    device = canvas.device

    # Create xy points on the canvas
    x_range = torch.linspace(-1, 1, steps=num_cols, device=device)
    y_range = torch.linspace(-1, 1, steps=num_rows, device=device).flip(dims=[0])
    # [num_cols, num_rows]
    canvas_x, canvas_y = torch.meshgrid(x_range, y_range)
    # [num_rows, num_cols]
    canvas_x, canvas_y = canvas_x.T, canvas_y.T

    # Draw heart
    new_canvas = canvas.clone()
    new_canvas[
        heart_occupancy_function((canvas_x - position_x) / scale, (canvas_y - position_x) / scale)
    ] = 1.0

    return new_canvas


def render_rectangle(xy_lims, canvas):
    """Draws a rectangle on a canvas whose xy limits are [-1, 1].

    Args
        xy_lims [4] (min_x, min_y, max_x, max_y)
        canvas [num_rows, num_cols]

    Returns
        new_canvas [num_rows, num_cols]
    """
    # Extract
    # top_left_x, top_left_y, width, height = tlwh
    # min_x, min_y, max_x, max_y = top_left_x, top_left_x + width, top_left_y - height, top_left_y
    min_x, min_y, max_x, max_y = xy_lims
    num_rows, num_cols = canvas.shape
    device = xy_lims.device

    # Create xy points on the canvas
    x_range = torch.linspace(-1, 1, steps=num_cols, device=device)
    y_range = torch.linspace(-1, 1, steps=num_rows, device=device).flip(dims=[0])
    # [num_cols, num_rows]
    canvas_x, canvas_y = torch.meshgrid(x_range, y_range)
    # [num_rows, num_cols]
    canvas_x, canvas_y = canvas_x.T, canvas_y.T

    # Draw on canvas
    new_canvas = canvas.clone()
    new_canvas[
        (canvas_x >= min_x) & (canvas_x <= max_x) & (canvas_y >= min_y) & (canvas_y <= max_y)
    ] = 1.0

    return new_canvas


if __name__ == "__main__":
    num_rows, num_cols, device = 64, 64, "cuda"
    canvas = torch.zeros((num_rows, num_cols), device=device)

    # rectangle
    xy_lims = torch.tensor([-0.9, 0.9, 1, 0.1], device=device)
    # canvas = render_rectangle(xy_lims, canvas)

    # heart
    position = torch.tensor([-0.5, -0.5], device=device)
    scale = torch.tensor(0.5, device=device)
    canvas = render_heart((position, scale), canvas)
    import matplotlib.pyplot as plt
    import util

    fig, ax = plt.subplots(1, 1)
    ax.imshow(canvas.cpu(), cmap="Greys", vmin=0, vmax=1)
    util.save_fig(fig, "test.png")
