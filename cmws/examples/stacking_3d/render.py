import random

import torch
import torch.nn as nn
from cmws import util
import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.structures.meshes import (
    Meshes,
    join_meshes_as_batch,
    join_meshes_as_scene,
)


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
    # Modified from PyTorch3D tutorial
    # https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/render_textured_meshes.ipynb

    # Extract
    device = sizes.device
    num_objects = len(sizes)

    # Create camera
    R, T = look_at_view_transform(2.7, 0, 180)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Settings for rasterizer (optional blur)
    raster_settings = RasterizationSettings(
        image_size=im_size, # crisper objects + texture w/ higher resolution
        blur_radius=0.0,
        faces_per_pixel=2, # increase at cost of GPU memory
    )

    # Add light from the front
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Compose renderer and shader
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    # Combine obj meshes into single mesh from rendering
    # https://github.com/facebookresearch/pytorch3d/issues/15
    vertices = []
    faces = []
    textures = []
    vert_offset = 0 # offset by vertices from prior meshes
    for i, (position, size) in enumerate(zip(positions, sizes)):
        cube_vertices, cube_faces = get_cube_mesh(position, size)
        # TODO: learn entire texture vector
        # For now, apply same color to each mesh vertex (v \in V)
        texture = torch.ones_like(cube_vertices) * colors[i] # [V, 3]
        # Offset faces (account for diff indexing, b/c treating as one mesh)
        cube_faces = cube_faces + vert_offset
        vert_offset = cube_vertices.shape[0]
        vertices.append(cube_vertices)
        faces.append(cube_faces)
        textures.append(texture)

    # Concatenate data into single mesh
    vertices = torch.cat(vertices)
    faces = torch.cat(faces)
    textures = torch.cat(textures)[None]  # (1, num_verts, 3)
    textures = TexturesVertex(verts_features=textures)
    # each elmt of verts array is diff mesh in batch
    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)

    # Render image
    img = renderer(mesh)   # (B, H, W, 4)

    # Remove alpha channel and return (im_size, im_size, 3)
    img = img[0, ..., :3]#.detach().squeeze().cpu().numpy()

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
