import torch
import torch.nn as nn
import torch.nn.functional as F
from cmws import util


class Square:
    def __init__(self, name, color, size):
        self.name = name
        self.color = color
        self.size = size

    @property
    def device(self):
        return self.size.device

    def __repr__(self):
        return f"{self.name}(color={self.color.tolist()}, size={self.size.item():.1f})"


class LearnableSquare(nn.Module):
    def __init__(self, name=None, fixed_color=False):
        super().__init__()
        if name is None:
            self.name = "LearnableSquare"
        else:
            self.name = name
        self.fixed_color = fixed_color
        if not self.fixed_color:
            self.raw_color = nn.Parameter(torch.randn((3,)))
        self.raw_size = nn.Parameter(torch.randn(()))

    @property
    def device(self):
        return self.raw_size.device

    @property
    def size(self):
        min_size = 0.01
        max_size = 1.0
        return self.raw_size.sigmoid() * (max_size - min_size) + min_size

    @property
    def color(self):
        if self.fixed_color:
            return torch.zeros((3,), device=self.device)
        else:
            return self.raw_color.sigmoid()

    def __repr__(self):
        return f"{self.name}(color={self.color.tolist()}, size={self.size.item():.1f})"


def get_min_edge_distance(square_size, location, point):
    """Computes shortest distance from a point to the square edge. (batched)
    Negative if it's inside the square.
    Positive if it's outside the square.

    Args
        square_size [] or [*location_shape]
        location [*location_shape, 2]
        point [*point_shape, 2]

    Returns [*location_shape, *point_shape]
    """
    # Extract
    device = location.device
    # [*location_shape]
    min_x, min_y = location[..., 0], location[..., 1]
    max_x = min_x + square_size
    max_y = min_y + square_size
    location_shape = min_x.shape
    num_locations = int(torch.tensor(location_shape).prod().long().item())
    # [*point_shape]
    x, y = point[..., 0], point[..., 1]
    point_shape = x.shape
    num_points = int(torch.tensor(point_shape).prod().long().item())

    # Flatten
    # [num_locations, 1]
    min_x, min_y, max_x, max_y = [tmp.view(-1)[:, None] for tmp in [min_x, min_y, max_x, max_y]]
    # [1, num_points]
    x, y = [tmp.view(-1)[None] for tmp in [x, y]]

    # Determine which area the point is in
    # [num_locations, num_points]
    # --High level areas
    up = y >= max_y
    middle = (y >= min_y) & (y < max_y)
    bottom = y < min_y
    left = x < min_x
    center = (x >= min_x) & (x < max_x)
    right = x >= max_x

    # --Use high level areas to define smaller sectors which we're going to work with
    area_1 = left & up
    area_2 = center & up
    area_3 = right & up
    area_4 = left & middle
    area_5 = center & middle
    area_6 = right & middle
    area_7 = left & bottom
    area_8 = center & bottom
    area_9 = right & bottom

    # Compute min distances
    # --Init the results
    # [num_locations, num_points]
    min_edge_distance = torch.zeros((num_locations, num_points), device=device)

    # --Compute distances for points in the corners (areas 1, 3, 7, 9)
    min_edge_distance[area_1] = util.sqrt((x - min_x) ** 2 + (y - max_y) ** 2)[area_1]
    min_edge_distance[area_3] = util.sqrt((x - max_x) ** 2 + (y - max_y) ** 2)[area_3]
    min_edge_distance[area_7] = util.sqrt((x - min_x) ** 2 + (y - min_y) ** 2)[area_7]
    min_edge_distance[area_9] = util.sqrt((x - max_x) ** 2 + (y - min_y) ** 2)[area_9]

    # --Compute distances for points in the outside strips (areas 2, 4, 6, 8)
    min_edge_distance[area_2] = (y - max_y)[area_2]
    min_edge_distance[area_4] = (min_x - x)[area_4]
    min_edge_distance[area_6] = (x - max_x)[area_6]
    min_edge_distance[area_8] = (min_y - y)[area_8]

    # --Compute distances for points inside the square
    min_edge_distance[area_5] = -torch.min(
        torch.stack([y - min_y, max_y - y, x - min_x, max_x - x]), dim=0
    )[0][area_5]

    return min_edge_distance.view(*[*location_shape, *point_shape])


def get_render_log_prob(min_edge_distance, blur=1e-4):
    """
    Returns the (log) probability map used for soft rasterization as specified by
    equation (1) of
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Soft_Rasterizer_A_Differentiable_Renderer_for_Image-Based_3D_Reasoning_ICCV_2019_paper.pdf

    Also visualized here https://www.desmos.com/calculator/5z95dy2mny

    Args
        min_edge_distance [*shape]
        blur [] (default 1e-4): this is the σ in equation (1)

    Returns [*shape]
    """
    return F.logsigmoid(-torch.sign(min_edge_distance) * min_edge_distance ** 2 / blur)


def get_canvas_xy(num_rows, num_cols, device):
    """Create xy points on the canvas

    Args
        num_rows (int)
        num_cols (int)

    Returns
        canvas_x [num_rows, num_cols]
        canvas_y [num_rows, num_cols]
    """

    x_range = torch.linspace(-1, 1, steps=num_cols, device=device)
    y_range = torch.linspace(-1, 1, steps=num_rows, device=device).flip(dims=[0])
    # [num_cols, num_rows]
    canvas_x, canvas_y = torch.meshgrid(x_range, y_range)
    # [num_rows, num_cols]
    canvas_x, canvas_y = canvas_x.T, canvas_y.T

    return canvas_x, canvas_y


def init_canvas(device, num_channels=3, num_rows=32, num_cols=32, shape=[]):
    """Return a white canvas of shape [*shape, num_channels, num_rows, num_cols]"""
    return torch.ones(*[*shape, num_channels, num_rows, num_cols], device=device)


def render_square(square, location, canvas, draw_on_top=False):
    """Draws a square on a canvas whose xy limits are [-1, 1].

    Args
        square
        location [2]
        canvas [num_channels, num_rows, num_cols]
        draw_on_top (bool): draw squares on top of the canvas, instead of adding it to the canvas

    Returns
        new_canvas [num_channels, num_rows, num_cols]
    """
    # Extract
    # []
    min_x, min_y = location
    max_x = min_x + square.size
    max_y = min_y + square.size
    num_channels, num_rows, num_cols = canvas.shape
    device = location.device

    # Canvas xy
    # [num_rows, num_cols]
    canvas_x, canvas_y = get_canvas_xy(num_rows, num_cols, device)

    # Draw on canvas
    new_canvas = canvas.clone()
    for channel_id in range(num_channels):
        if draw_on_top:
            new_canvas[
                channel_id,
                (canvas_x >= min_x)
                & (canvas_x <= max_x)
                & (canvas_y >= min_y)
                & (canvas_y <= max_y),
            ] = square.color[channel_id]
        else:
            new_canvas[
                channel_id,
                (canvas_x >= min_x)
                & (canvas_x <= max_x)
                & (canvas_y >= min_y)
                & (canvas_y <= max_y),
            ] -= (1 - square.color[channel_id])
    new_canvas = new_canvas.clamp(0, 1)

    return new_canvas


def soft_render_square(
    square, location, background, background_depth=-1e-3, color_sharpness=1e-4, blur=1e-4
):
    """Draws a square on a canvas whose xy limits are [-1, 1].

    Follows equations (2) and (3) in
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Soft_Rasterizer_A_Differentiable_Renderer_for_Image-Based_3D_Reasoning_ICCV_2019_paper.pdf

    Args
        square
        location [*shape, 2]
        background [num_channels, num_rows, num_cols] or [*shape, num_channels, num_rows, num_cols]
            this is the background color C_b in equation (2)
        background_weight [] (default 1.): ϵ in equation (3)
        color_sharpness [] (default 1e-4): γ in equation (3)
        blur [] (default 1e-4): this is the σ in equation (1)

    Returns
        new_canvas [*shape, num_channels, num_rows, num_cols]
    """
    # Extract
    shape = location.shape[:-1]

    # Init
    device = location.device
    if background.ndim > 3:
        num_channels, num_rows, num_cols = background.shape[-3:]
        expanded_background = True
        assert background.shape[:-3] == shape
    else:
        num_channels, num_rows, num_cols = background.shape
        expanded_background = False

    # Canvas xy
    # [num_rows, num_cols]
    canvas_x, canvas_y = get_canvas_xy(num_rows, num_cols, device)
    canvas_xy = torch.stack([canvas_x, canvas_y], dim=-1)

    # Get render log prob
    # --Foreground object (treat depth z = -1) [*shape, num_rows, num_cols]
    depth = 0
    square_render_log_prob = (
        get_render_log_prob(get_min_edge_distance(square.size, location, canvas_xy), blur=blur)
        + depth / color_sharpness
    )
    # --Background [*shape, num_rows, num_cols]
    background_render_log_prob = (
        torch.ones_like(square_render_log_prob) * background_depth / color_sharpness
    )

    # Compute color weight (equation (3))
    # [*shape, num_rows, num_cols]
    square_weight, background_weight = F.softmax(
        torch.stack([square_render_log_prob, background_render_log_prob]), dim=0
    )

    # Flatten
    # [num_samples, num_rows, num_cols]
    square_weight_flattened = square_weight.view(-1, num_rows, num_cols)
    background_weight_flattened = background_weight.view(-1, num_rows, num_cols)
    if expanded_background:
        background_flattened = background.view(-1, num_channels, num_rows, num_cols)
    else:
        background_flattened = background[None]

    return (
        square_weight_flattened[:, None] * square.color[None, :, None, None]
        + background_weight_flattened[:, None] * background_flattened
    ).view(*[*shape, num_channels, num_rows, num_cols])


def soft_render_square_batched(
    square_size,
    square_color,
    location,
    background,
    background_depth=-1e-3,
    color_sharpness=1e-4,
    blur=1e-4,
):
    """Draws a square on a canvas whose xy limits are [-1, 1].

    Follows equations (2) and (3) in
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Soft_Rasterizer_A_Differentiable_Renderer_for_Image-Based_3D_Reasoning_ICCV_2019_paper.pdf

    Args
        square_size [*shape] or []
        square_color [*shape, 3] or [3]
        location [*shape, 2]
        background [num_channels, num_rows, num_cols] or [*shape, num_channels, num_rows, num_cols]
            this is the background color C_b in equation (2)
        background_weight [] (default 1.): ϵ in equation (3)
        color_sharpness [] (default 1e-4): γ in equation (3)
        blur [] (default 1e-4): this is the σ in equation (1)

    Returns
        new_canvas [*shape, num_channels, num_rows, num_cols]
    """
    # Extract
    shape = location.shape[:-1]

    # Init
    device = location.device
    if background.ndim > 3:
        num_channels, num_rows, num_cols = background.shape[-3:]
        expanded_background = True
        assert background.shape[:-3] == shape
    else:
        num_channels, num_rows, num_cols = background.shape
        expanded_background = False

    # Canvas xy
    # [num_rows, num_cols]
    canvas_x, canvas_y = get_canvas_xy(num_rows, num_cols, device)
    canvas_xy = torch.stack([canvas_x, canvas_y], dim=-1)

    # Get render log prob
    # --Foreground object (treat depth z = -1) [*shape, num_rows, num_cols]
    depth = 0
    square_render_log_prob = (
        get_render_log_prob(get_min_edge_distance(square_size, location, canvas_xy), blur=blur)
        + depth / color_sharpness
    )
    # --Background [*shape, num_rows, num_cols]
    background_render_log_prob = (
        torch.ones_like(square_render_log_prob) * background_depth / color_sharpness
    )

    # Compute color weight (equation (3))
    # [*shape, num_rows, num_cols]
    square_weight, background_weight = F.softmax(
        torch.stack([square_render_log_prob, background_render_log_prob]), dim=0
    )

    # Flatten
    # [num_samples, num_rows, num_cols]
    square_weight_flattened = square_weight.view(-1, num_rows, num_cols)
    background_weight_flattened = background_weight.view(-1, num_rows, num_cols)
    if expanded_background:
        background_flattened = background.view(-1, num_channels, num_rows, num_cols)
    else:
        background_flattened = background[None]
    if square_color.ndim == 1:
        square_color_expanded = square_color[None, :, None, None]
    else:
        square_color_expanded = square_color.reshape(-1, 3)[:, :, None, None]

    return (
        square_weight_flattened[:, None] * square_color_expanded
        + background_weight_flattened[:, None] * background_flattened
    ).view(*[*shape, num_channels, num_rows, num_cols])


def render_square_batched(
    square_size, square_color, location, background,
):
    """Draws a square on a canvas whose xy limits are [-1, 1].

    Follows equations (2) and (3) in
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Soft_Rasterizer_A_Differentiable_Renderer_for_Image-Based_3D_Reasoning_ICCV_2019_paper.pdf

    Args
        square_size [*shape] or []
        square_color [*shape, 3] or [3]
        location [*shape, 2]
        background [num_channels, num_rows, num_cols] or [*shape, num_channels, num_rows, num_cols]
            this is the background color C_b in equation (2)
        background_weight [] (default 1.): ϵ in equation (3)
        color_sharpness [] (default 1e-4): γ in equation (3)
        blur [] (default 1e-4): this is the σ in equation (1)

    Returns
        new_canvas [*shape, num_channels, num_rows, num_cols]
    """
    # Extract
    shape = location.shape[:-1]
    device = location.device
    num_elements = int(torch.tensor(shape).prod().long().item())
    num_channels, num_rows, num_cols = background.shape[-3:]
    num_points = num_rows * num_cols

    # Canvas xy
    # --Compute
    # [num_rows, num_cols]
    canvas_x, canvas_y = get_canvas_xy(num_rows, num_cols, device)
    # --Flatten
    # [1, num_points]
    x, y = [tmp.reshape(-1)[None] for tmp in [canvas_x, canvas_y]]

    # Compute boundaries
    # --Compute
    # [*shape]
    min_x, min_y = location[..., 0], location[..., 1]
    max_x = min_x + square_size
    max_y = min_y + square_size
    # --Flatten
    # [num_elements, 1]
    min_x, min_y, max_x, max_y = [tmp.view(-1)[:, None] for tmp in [min_x, min_y, max_x, max_y]]

    # Draw on canvas
    # --Expand background
    if background.ndim > 3:
        canvas = background.clone().view(num_elements, num_channels, num_points)
        assert background.shape[:-3] == shape
    else:
        canvas = (
            background.clone()
            .view(1, num_channels, num_points)
            .expand(num_elements, num_channels, num_points)
        )

    # --Expand square_color
    if square_color.ndim == 1:
        square_color_expanded = square_color[None, :, None].expand(
            num_elements, num_channels, num_points
        )
    else:
        square_color_expanded = square_color.reshape(-1, 3, 1).expand(
            num_elements, num_channels, num_points
        )

    # --Compute a mask that indicates whether a point is inside a square
    # [num_elements, num_channels, num_points]
    inside_square = ((x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y))[:, None, :].expand(
        num_elements, num_channels, num_points
    )

    # --Draw inside the square
    canvas[inside_square] = square_color_expanded[inside_square]

    return canvas.view(*[*shape, num_channels, num_rows, num_cols])


def render(primitives, stacking_program, raw_locations, num_channels=3, num_rows=32, num_cols=32):
    # Init
    device = primitives[0].device

    # Convert
    locations = convert_raw_locations(raw_locations, stacking_program, primitives)

    # Render
    canvas = init_canvas(device, num_channels, num_rows, num_cols)
    for primitive_id, location in zip(stacking_program, locations):
        primitive = primitives[primitive_id]
        canvas = render_square(primitive, location, canvas)

    return canvas


def render_batched(
    primitives,
    num_blocks,
    stacking_program,
    raw_locations,
    num_channels=3,
    num_rows=32,
    num_cols=32,
):
    """
    Args
        primitives (list [num_primitives])
        num_blocks [*shape]
        stacking_program (tensor [*shape, max_num_blocks])
        raw_locations (tensor [*shape, max_num_blocks])

    Returns [*shape, num_channels, num_rows, num_cols]
    """
    # Extract
    device = primitives[0].device
    shape = stacking_program.shape[:-1]
    max_num_blocks = stacking_program.shape[-1]

    # [num_primitives]
    square_size = torch.stack([primitive.size for primitive in primitives])
    # [num_primitives, 3]
    square_color = torch.stack([primitive.color for primitive in primitives])

    # Convert [*shape, max_num_blocks, 2]
    locations = convert_raw_locations_batched(raw_locations, stacking_program, primitives)

    # Render
    canvas = init_canvas(device, num_channels, num_rows, num_cols, shape)
    for block_id in range(max_num_blocks):
        # Determine whether this block is drawn
        # [*shape, 1, 1, 1]
        is_drawn = (block_id < num_blocks).float()[..., None, None, None]

        # Draw the block
        canvas = render_square_batched(
            square_size[stacking_program[..., block_id]],
            square_color[stacking_program[..., block_id]],
            locations[..., block_id, :],
            canvas,
        ) * is_drawn + canvas * (1 - is_drawn)

    return canvas


def soft_render(
    primitives,
    stacking_program,
    raw_locations,
    raw_color_sharpness,
    raw_blur,
    num_channels=3,
    num_rows=32,
    num_cols=32,
):
    """
    Args
        primitives (list [num_primitives])
        stacking_program (tensor [num_blocks])
        raw_locations (tensor [num_blocks])
        raw_color_sharpness []
        raw_blur []

    Returns [num_channels, num_rows, num_cols]
    """
    # Init
    device = primitives[0].device

    # Convert
    locations = convert_raw_locations(raw_locations, stacking_program, primitives)

    # Render
    canvas = init_canvas(device, num_channels, num_rows, num_cols)
    for primitive_id, location in zip(stacking_program, locations):
        primitive = primitives[primitive_id]
        canvas = soft_render_square(
            primitive,
            location,
            canvas,
            color_sharpness=get_color_sharpness(raw_color_sharpness),
            blur=get_blur(raw_blur),
        )

    return canvas


def soft_render_batched(
    primitives,
    stacking_program,
    raw_locations,
    raw_color_sharpness,
    raw_blur,
    num_channels=3,
    num_rows=32,
    num_cols=32,
):
    """
    Args
        primitives (list [num_primitives])
        stacking_program (tensor [*shape, num_blocks])
        raw_locations (tensor [*shape, num_blocks])
        raw_color_sharpness []
        raw_blur []

    Returns [*shape, num_channels, num_rows, num_cols]
    """
    # Extract
    device = primitives[0].device
    shape = stacking_program.shape[:-1]
    num_blocks = stacking_program.shape[-1]
    # [num_primitives]
    square_size = torch.stack([primitive.size for primitive in primitives])
    # [num_primitives, 3]
    square_color = torch.stack([primitive.color for primitive in primitives])

    # Convert
    locations = convert_raw_locations_batched(raw_locations, stacking_program, primitives)

    # Render
    canvas = init_canvas(device, num_channels, num_rows, num_cols, shape)
    for block_id in range(num_blocks):
        canvas = soft_render_square_batched(
            square_size[stacking_program[..., block_id]],
            square_color[stacking_program[..., block_id]],
            locations[..., block_id, :],
            canvas,
            color_sharpness=get_color_sharpness(raw_color_sharpness),
            blur=get_blur(raw_blur),
        )

    return canvas


def soft_render_variable_num_blocks(
    primitives,
    num_blocks,
    stacking_program,
    raw_locations,
    raw_color_sharpness,
    raw_blur,
    num_channels=3,
    num_rows=32,
    num_cols=32,
):
    """
    Args
        primitives (list [num_primitives])
        num_blocks [*shape]
        stacking_program (tensor [*shape, max_num_blocks])
        raw_locations (tensor [*shape, max_num_blocks])
        raw_color_sharpness []
        raw_blur []

    Returns [*shape, num_channels, num_rows, num_cols]
    """
    # Extract
    device = primitives[0].device
    shape = stacking_program.shape[:-1]
    max_num_blocks = stacking_program.shape[-1]

    # [num_primitives]
    square_size = torch.stack([primitive.size for primitive in primitives])
    # [num_primitives, 3]
    square_color = torch.stack([primitive.color for primitive in primitives])

    # Convert [*shape, max_num_blocks, 2]
    locations = convert_raw_locations_batched(raw_locations, stacking_program, primitives)

    # Render
    canvas = init_canvas(device, num_channels, num_rows, num_cols, shape)
    for block_id in range(max_num_blocks):
        # Determine whether this block is drawn
        # [*shape, 1, 1, 1]
        is_drawn = (block_id < num_blocks).float()[..., None, None, None]

        # Draw the block
        canvas = soft_render_square_batched(
            square_size[stacking_program[..., block_id]],
            square_color[stacking_program[..., block_id]],
            locations[..., block_id, :],
            canvas,
            color_sharpness=get_color_sharpness(raw_color_sharpness),
            blur=get_blur(raw_blur),
        ) * is_drawn + canvas * (1 - is_drawn)

    return canvas


def convert_raw_locations(raw_locations, stacking_program, primitives):
    """
    Args
        raw_locations (tensor [num_blocks])
        stacking_program (tensor [num_blocks])
        primitives (list [num_primitives])

    Returns [num_blocks, 2]
    """
    # Extract
    device = primitives[0].device

    # Sample the bottom
    y = torch.tensor(-1.0, device=device)
    min_x = -0.8
    max_x = 0.8
    locations = []
    for primitive_id, raw_location in zip(stacking_program, raw_locations):
        size = primitives[primitive_id].size

        min_x = min_x - size
        x = raw_location.sigmoid() * (max_x - min_x) + min_x
        locations.append(torch.stack([x, y]))

        y = y + size
        min_x = x
        max_x = min_x + size
    return torch.stack(locations)


def convert_raw_locations_batched(raw_locations, stacking_program, primitives):
    """
    Args
        raw_locations (tensor [*shape, num_blocks])
        stacking_program (tensor [*shape, num_blocks])
        primitives (list [num_primitives])

    Returns [*shape, num_blocks, 2]
    """
    # Extract
    shape = raw_locations.shape[:-1]
    num_samples = util.get_num_elements(shape)
    num_blocks = raw_locations.shape[-1]

    # Flatten
    # [num_samples, num_blocks]
    raw_locations_flattened = raw_locations.view(num_samples, num_blocks)
    stacking_program_flattened = stacking_program.reshape(num_samples, num_blocks)

    locations_batched = []
    for sample_id in range(num_samples):
        locations_batched.append(
            convert_raw_locations(
                raw_locations_flattened[sample_id],
                stacking_program_flattened[sample_id],
                primitives,
            )
        )
    return torch.stack(locations_batched).view(*[*shape, num_blocks, 2])


def get_color_sharpness(raw_color_sharpness):
    return raw_color_sharpness.exp()


def get_blur(raw_blur):
    return raw_blur.exp()


def convert_raw_locations_top_down(raw_locations, stacking_program, primitives):
    """
    Args
        raw_locations (tensor [num_blocks])
        stacking_program (tensor [num_blocks])
        primitives (list [num_primitives])

    Returns [num_blocks, 2]
    """
    # Sample the bottom
    min_x = -0.8
    max_x = 0.8
    locations = []
    for primitive_id, raw_location in zip(stacking_program, raw_locations):
        size = primitives[primitive_id].size

        min_x = min_x - size
        x = raw_location.sigmoid() * (max_x - min_x) + min_x
        y = -size / 2.0
        locations.append(torch.stack([x, y]))

        min_x = x
        max_x = min_x + size
    return torch.stack(locations)


def convert_raw_locations_batched_top_down(raw_locations, stacking_program, primitives):
    """
    Args
        raw_locations (tensor [*shape, num_blocks])
        stacking_program (tensor [*shape, num_blocks])
        primitives (list [num_primitives])

    Returns [*shape, num_blocks, 2]
    """
    # Extract
    shape = raw_locations.shape[:-1]
    num_samples = util.get_num_elements(shape)
    num_blocks = raw_locations.shape[-1]

    # Flatten
    # [num_samples, num_blocks]
    raw_locations_flattened = raw_locations.view(num_samples, num_blocks)
    stacking_program_flattened = stacking_program.reshape(num_samples, num_blocks)

    locations_batched = []
    for sample_id in range(num_samples):
        locations_batched.append(
            convert_raw_locations_top_down(
                raw_locations_flattened[sample_id],
                stacking_program_flattened[sample_id],
                primitives,
            )
        )
    return torch.stack(locations_batched).view(*[*shape, num_blocks, 2])


def soft_render_top_down(
    primitives,
    num_blocks,
    stacking_program,
    raw_locations,
    raw_color_sharpness,
    raw_blur,
    num_channels=3,
    num_rows=32,
    num_cols=32,
):
    """
    Args
        primitives (list [num_primitives])
        num_blocks [*shape]
        stacking_program (tensor [*shape, max_num_blocks])
        raw_locations (tensor [*shape, max_num_blocks])
        raw_color_sharpness []
        raw_blur []

    Returns [*shape, num_channels, num_rows, num_cols]
    """
    # Extract
    device = primitives[0].device
    shape = stacking_program.shape[:-1]
    max_num_blocks = stacking_program.shape[-1]

    # [num_primitives]
    square_size = torch.stack([primitive.size for primitive in primitives])
    # [num_primitives, 3]
    square_color = torch.stack([primitive.color for primitive in primitives])

    # Convert [*shape, max_num_blocks, 2]
    locations = convert_raw_locations_batched_top_down(raw_locations, stacking_program, primitives)

    # Render
    canvas = init_canvas(device, num_channels, num_rows, num_cols, shape)
    for block_id in range(max_num_blocks):
        # Determine whether this block is drawn
        # [*shape, 1, 1, 1]
        is_drawn = (block_id < num_blocks).float()[..., None, None, None]

        # Draw the block
        canvas = soft_render_square_batched(
            square_size[stacking_program[..., block_id]],
            square_color[stacking_program[..., block_id]],
            locations[..., block_id, :],
            canvas,
            color_sharpness=get_color_sharpness(raw_color_sharpness),
            blur=get_blur(raw_blur),
        ) * is_drawn + canvas * (1 - is_drawn)

    return canvas


def render_batched_top_down(
    primitives,
    num_blocks,
    stacking_program,
    raw_locations,
    num_channels=3,
    num_rows=32,
    num_cols=32,
):
    """
    Args
        primitives (list [num_primitives])
        num_blocks [*shape]
        stacking_program (tensor [*shape, max_num_blocks])
        raw_locations (tensor [*shape, max_num_blocks])

    Returns [*shape, num_channels, num_rows, num_cols]
    """
    # Extract
    device = primitives[0].device
    shape = stacking_program.shape[:-1]
    max_num_blocks = stacking_program.shape[-1]

    # [num_primitives]
    square_size = torch.stack([primitive.size for primitive in primitives])
    # [num_primitives, 3]
    square_color = torch.stack([primitive.color for primitive in primitives])

    # Convert [*shape, max_num_blocks, 2]
    locations = convert_raw_locations_batched_top_down(raw_locations, stacking_program, primitives)

    # Render
    canvas = init_canvas(device, num_channels, num_rows, num_cols, shape)
    for block_id in range(max_num_blocks):
        # Determine whether this block is drawn
        # [*shape, 1, 1, 1]
        is_drawn = (block_id < num_blocks).float()[..., None, None, None]

        # Draw the block
        canvas = render_square_batched(
            square_size[stacking_program[..., block_id]],
            square_color[stacking_program[..., block_id]],
            locations[..., block_id, :],
            canvas,
        ) * is_drawn + canvas * (1 - is_drawn)

    return canvas


def render_top_down(
    primitives, stacking_program, raw_locations, num_channels=3, num_rows=32, num_cols=32
):
    # Init
    device = primitives[0].device

    # Convert
    locations = convert_raw_locations_top_down(raw_locations, stacking_program, primitives)

    # Render
    canvas = init_canvas(device, num_channels, num_rows, num_cols)
    for primitive_id, location in zip(stacking_program, locations):
        primitive = primitives[primitive_id]
        canvas = render_square(primitive, location, canvas, draw_on_top=True)

    return canvas
