import pyredner
import torch
import random
import torch.nn as nn
import util


class Cube:
    def __init__(self, name, color, size):
        self.name = name
        self.color = color
        self.size = size

    @property
    def device(self):
        return self.size.device

    def __repr__(self):
        return f"{self.name}(color={self.color.tolist()}, size={self.size.item():.1f})"


class LearnableCube(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        if name is None:
            self.name = "LearnableCube"
        else:
            self.name = name
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
        return self.raw_color.sigmoid()

    def __repr__(self):
        return f"{self.name}(color={self.color.tolist()}, size={self.size.item():.1f})"


def get_cube_mesh(position, size):
    """Computes a cube mesh
    Adapted from https://github.com/mikedh/trimesh/blob/master/trimesh/creation.py#L566

    Args
        position [3]
        size []

    Returns
        vertices [num_vertices, 3]
        faces [num_faces, 3]
    """
    # Extract
    device = position.device

    # vertices of the cube
    centered_vertices = (
        torch.tensor(
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            dtype=torch.float,
            device=device,
        ).view(-1, 3)
        - 0.5
    )
    translation = position.clone()
    translation[-1] += size / 2
    vertices = centered_vertices * size + translation[None]

    # hardcoded face indices
    faces = torch.tensor(
        [
            1,
            3,
            0,
            4,
            1,
            0,
            0,
            3,
            2,
            2,
            4,
            0,
            1,
            7,
            3,
            5,
            1,
            4,
            5,
            7,
            1,
            3,
            7,
            2,
            6,
            4,
            2,
            2,
            7,
            6,
            6,
            5,
            4,
            7,
            5,
            6,
        ],
        dtype=torch.int32,
        device=device,
    ).view(-1, 3)

    return vertices, faces


def render_cube(size, color, position, im_size=32, render_seed=None):
    """Renders a cube given cube specs

    Args
        size []
        color [3]
        position [3]
        im_size (int)
        render_seed (int) -- to be passed into the redner renderer

    Returns rgb image [im_size, im_size, 3]
    """
    return render_cubes(size[None], color[None], position[None], im_size, render_seed)


def render_cubes(sizes, colors, positions, im_size=32, render_seed=None):
    """Renders cubes given cube specs

    Args
        sizes [num_cubes]
        colors [num_cubes, 3]
        positions [num_cubes, 3]
        im_size (int)
        render_seed (int) -- to be passed into the redner renderer

    Returns rgb image [im_size, im_size, 3]
    """
    # Extract
    device = sizes.device
    num_objects = len(sizes)
    if render_seed is None:
        render_seed = random.randrange(1000000)

    # Set pyredner stuff
    pyredner.set_print_timing(False)
    pyredner.set_use_gpu("cuda" in str(device))

    # Camera
    cam = pyredner.Camera(
        position=torch.tensor([0.0, -2.0, -0.2]),
        look_at=torch.tensor([0.0, 1.0, -0.6]),
        up=torch.tensor([0.0, 1.0, 0.0]),
        fov=torch.tensor([45.0]),  # in degree
        clip_near=1e-2,  # needs to > 0
        resolution=(im_size, im_size),
        fisheye=False,
    )

    # Material
    object_materials = [pyredner.Material(diffuse_reflectance=color) for color in colors]
    white_material = pyredner.Material(
        diffuse_reflectance=torch.tensor([1.0, 1.0, 1.0], device=device)
    )
    materials = [white_material] + object_materials

    # Shape
    shape_objects = []
    for i, (position, size) in enumerate(zip(positions, sizes)):
        cube_vertices, cube_faces = get_cube_mesh(position, size)
        shape_objects.append(
            pyredner.Shape(
                vertices=cube_vertices,
                indices=cube_faces,
                uvs=None,
                normals=None,
                material_id=1 + i,
            )
        )
    shape_light_top = pyredner.Shape(
        vertices=torch.tensor(
            [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], device=device
        ),
        indices=torch.tensor([[0, 2, 1], [1, 2, 3]], dtype=torch.int32, device=device),
        uvs=None,
        normals=None,
        material_id=0,
    )
    shape_light_front = pyredner.Shape(
        vertices=torch.tensor(
            [[-1.0, -2.0, -1.0], [1.0, -2.0, -1.0], [-1.0, -2.0, 1.0], [1.0, -2.0, 1.0]],
            device=device,
        ),
        indices=torch.tensor([[0, 2, 1], [1, 2, 3]], dtype=torch.int32, device=device),
        uvs=None,
        normals=None,
        material_id=0,
    )

    shapes = shape_objects + [shape_light_top, shape_light_front]

    # Light
    light_top = pyredner.AreaLight(shape_id=num_objects, intensity=torch.ones(3) * 2)
    light_front = pyredner.AreaLight(shape_id=num_objects + 1, intensity=torch.ones(3) * 2)
    area_lights = [light_top, light_front]

    # Scene
    scene = pyredner.Scene(cam, shapes, materials, area_lights)
    scene_args = pyredner.RenderFunction.serialize_scene(scene=scene, num_samples=32, max_bounces=1)

    # Render
    render = pyredner.RenderFunction.apply
    img = render(render_seed, *scene_args)

    return img


def convert_raw_locations(raw_locations, stacking_program, primitives):
    """
    Args
        raw_locations (tensor [num_blocks])
        stacking_program (tensor [num_blocks])
        primitives (list [num_primitives])

    Returns [num_blocks, 3]
    """
    # Extract
    device = primitives[0].device

    # Sample the bottom
    y = torch.tensor(0.0, device=device)
    z = torch.tensor(-1.0, device=device)
    min_x = -0.8
    max_x = 0.8
    locations = []
    for primitive_id, raw_location in zip(stacking_program, raw_locations):
        size = primitives[primitive_id].size

        min_x = min_x - size
        x = raw_location.sigmoid() * (max_x - min_x) + min_x
        locations.append(torch.stack([x, y, z]))

        z = z + size
        min_x = x
        max_x = min_x + size
    return torch.stack(locations)


def convert_raw_locations_batched(raw_locations, stacking_program, primitives):
    """
    Args
        raw_locations (tensor [*shape, num_blocks])
        stacking_program (tensor [*shape, num_blocks])
        primitives (list [num_primitives])

    Returns [*shape, num_blocks, 3]
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
    return torch.stack(locations_batched).view(*[*shape, num_blocks, 3])


def render(
    primitives, num_blocks, stacking_program, raw_locations, im_size=32,
):
    """
    Args
        primitives (list [num_primitives])
        num_blocks [*shape]
        stacking_program (tensor [*shape, max_num_blocks])
        raw_locations (tensor [*shape, max_num_blocks])
        im_size

    Returns [*shape, num_channels=3, im_size, im_size]
    """
    # Extract
    shape = stacking_program.shape[:-1]
    max_num_blocks = stacking_program.shape[-1]
    num_elements = util.get_num_elements(shape)
    num_channels = 3

    # [num_primitives]
    square_size = torch.stack([primitive.size for primitive in primitives])
    # [num_primitives, 3]
    square_color = torch.stack([primitive.color for primitive in primitives])

    # Convert [*shape, max_num_blocks, 3]
    locations = convert_raw_locations_batched(raw_locations, stacking_program, primitives)

    # Flatten
    num_blocks_flattened = num_blocks.reshape(num_elements)
    stacking_program_flattened = stacking_program.reshape((num_elements, max_num_blocks))
    locations_flattened = locations.view((num_elements, max_num_blocks, 3))

    # Render
    imgs = []
    for i in range(num_elements):
        num_blocks_i = num_blocks_flattened[i]
        sizes_i = square_size[stacking_program_flattened[i, :num_blocks_i]]
        colors_i = square_color[stacking_program_flattened[i, :num_blocks_i]]
        positions_i = locations_flattened[i, :num_blocks_i]
        imgs.append(render_cubes(sizes_i, colors_i, positions_i, im_size).permute(2, 0, 1))

    return torch.stack(imgs).view(*[*shape, num_channels, im_size, im_size])
