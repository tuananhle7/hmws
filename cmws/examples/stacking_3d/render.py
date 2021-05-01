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
    TexturesVertex,
    BlendParams
)
from pytorch3d.structures.meshes import (
    Meshes,
    join_meshes_as_batch,
    join_meshes_as_scene,
)
import numpy as np


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


def render_cube(size, color, position, im_size=32):
    """Renders a cube given cube specs

    Args
        size []
        color [3]
        position [3]
        im_size (int)

    Returns rgb image [im_size, im_size, 3]
    """
    num_cubes = torch.IntTensor([1])
    imgs = render_cubes(num_cubes, size[None,None], color[None,None], position[None, None], im_size)
    return imgs[0]


def render_cubes(num_cubes, sizes, colors, positions, im_size=32, sigma=1e-10, gamma=1e-6):
    """Renders cubes given cube specs

    Args
        num_cubes [*shape]
        sizes [*shape,max_num_cubes]
        colors [*shape, max_num_cubes, 3]
        positions [*shape, max_num_cubes, 3]
        im_size (int)

    Returns rgb image [*shape, im_size, im_size, 3]
    """
    # Modified from PyTorch3D tutorial
    # https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/render_textured_meshes.ipynb

    # Extract
    device = sizes.device

    # Create camera
    R, T = look_at_view_transform(1.7, 0, 180,
                                  at=((0.0, 0.0, -0.5),))
    # R, T = look_at_view_transform(3.5, 0, 0,
    #                               up=((0.0, 0.0, 0.0),),
    #                               at=((0.0, 0.0, -0.5),))
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T,
                                    )  # fov=45.0)

    # Settings for rasterizer (optional blur)
    # https://github.com/facebookresearch/pytorch3d/blob/1c45ec9770ee3010477272e4cd5387f9ccb8cb51/pytorch3d/renderer/mesh/shader.py
    # implements eqs from SoftRasterizer paper
    blend_params = BlendParams(sigma=sigma, gamma=gamma) #,background_color=(0.0, 0.0, 0.0))
    raster_settings = RasterizationSettings(
        image_size=im_size,  # crisper objects + texture w/ higher resolution
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=1,  # increase at cost of GPU memory,
        bin_size=None
    )

    # Add light from the front
    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])  

    # Compose renderer and shader
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params
        )
    )

    # create one mesh per elmt in batch
    meshes = []
    for batch_idx, n_cubes in enumerate(num_cubes):
        # Combine obj meshes into single mesh from rendering
        # https://github.com/facebookresearch/pytorch3d/issues/15
        vertices = []
        faces = []
        textures = []
        vert_offset = 0 # offset by vertices from prior meshes
        for i, (position, size,color) in enumerate(zip(positions[batch_idx, :n_cubes, :], sizes[batch_idx, :n_cubes],
                                                       colors[batch_idx, :n_cubes, :])):
            cube_vertices, cube_faces = get_cube_mesh(position, size)
            # For now, apply same color to each mesh vertex (v \in V)
            texture = torch.ones_like(cube_vertices) * color# [V, 3]
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
        meshes.append(mesh)

    batched_mesh = join_meshes_as_batch(meshes)

    # Render image
    img = renderer(batched_mesh)   # (B, H, W, 4)

    # Remove alpha channel and return (B, im_size, im_size, 3)
    img = img[:, ..., :3]#.detach().squeeze().cpu().numpy()

    return img


# def convert_raw_locations(raw_locations, stacking_program, primitives):
#     """
#     Args
#         raw_locations (tensor [num_blocks])
#         stacking_program (tensor [num_blocks])
#         primitives (list [num_primitives])
#
#     Returns [num_blocks, 3]
#     """
#     # Extract
#     device = primitives[0].device
#
#     # Sample the bottom
#     y = torch.tensor(0.0, device=device)
#     z = torch.tensor(-1.0, device=device)
#     min_x = -0.8
#     max_x = 0.8
#     locations = []
#     for primitive_id, raw_location in zip(stacking_program, raw_locations):
#         size = primitives[primitive_id].size
#
#         min_x = min_x - size
#         x = raw_location.sigmoid() * (max_x - min_x) + min_x
#         locations.append(torch.stack([x, y, z]))
#
#         z = z + size
#         min_x = x
#         max_x = min_x + size
#     return torch.stack(locations)

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
    y = torch.tensor(-1.0, device=device)
    z = torch.tensor(0.0, device=device)
    min_x = -0.8
    max_x = 0.8
    locations = []
    for primitive_id, raw_location in zip(stacking_program, raw_locations):
        size = primitives[primitive_id].size

        min_x = min_x - size
        x = raw_location.sigmoid() * (max_x - min_x) + min_x
        locations.append(torch.stack([x, y, z]))

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

def convert_raw_gamma(gamma):
    return torch.abs(gamma* 1e-2)

def convert_raw_sigma(sigma):
    return torch.abs(sigma * 1e-2)

def render(
    primitives, num_blocks, stacking_program, raw_locations, im_size=32,
    sigma=1e-10, gamma=1e-6
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

    if torch.is_tensor(gamma):
        gamma = convert_raw_gamma(gamma)
    if torch.is_tensor(sigma):
        sigma = convert_raw_sigma(sigma)

    if torch.is_tensor(gamma):gamma = convert_raw_gamma(gamma)
    if torch.is_tensor(sigma):sigma = convert_raw_sigma(sigma)

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

    imgs = render_cubes(num_blocks_flattened, square_size[stacking_program_flattened], square_color[stacking_program_flattened], locations_flattened, im_size,
                        sigma,gamma)
    imgs = imgs.permute(0, 3, 1, 2)
    imgs = imgs.view(*[*shape, num_channels, *imgs.shape[-2:]])

    return imgs

