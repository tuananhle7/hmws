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
    imgs = render_cubes(num_cubes[None], size[None,None,None], color[None,None,None], position[None,None,None], im_size)
    return imgs[0]


def render_cubes(num_cubes, sizes, colors, positions, im_size=32, sigma=1e-10, gamma=1e-6, remove_color=False):
    """Renders cubes given cube specs

    Args
        num_cubes [batch_size, num_cells]
        sizes [batch_size, num_cells, max_num_cubes]
        colors [batch_size, num_cells, max_num_cubes, 3]
        positions [batch_size, num_cells, max_num_cubes, 3]
        im_size (int)
        remove_color (bool) if True, make all blocks the same color (ignores colors tensor)
    Returns rgb image [batch_size, im_size, im_size, 3]
    """
    # Modified from PyTorch3D tutorial
    # https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/render_textured_meshes.ipynb

    # Extract
    device = sizes.device

    # Create camera
    R, T = look_at_view_transform(4., 0.1, 180,
                                  at=((-.15, 0.0, 0.0),),)
    # R, T = look_at_view_transform(3.5, 0, 0,
    #                               up=((0.0, 0.0, 0.0),),
    #                               at=((0.0, 0.0, -0.5),))
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T,
                                    )#fov=90.0)  # fov=45.0)

    # Settings for rasterizer (optional blur)
    # https://github.com/facebookresearch/pytorch3d/blob/1c45ec9770ee3010477272e4cd5387f9ccb8cb51/pytorch3d/renderer/mesh/shader.py
    # implements eqs from SoftRasterizer paper
    blend_params = BlendParams(sigma=sigma, gamma=gamma) #,background_color=(0.0, 0.0, 0.0))
    raster_settings = RasterizationSettings(
        image_size=im_size,  # crisper objects + texture w/ higher resolution
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=3,  # increase at cost of GPU memory,
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

    # extract info
    num_batches, num_cells = num_cubes.shape

    # create one mesh per elmt in batch
    meshes = []
    empty_idxs = set() # patch: fill-in rendered images with blank images for blockless grids
    for batch_idx in range(num_batches):
        # Combine obj meshes into single mesh from rendering
        # https://github.com/facebookresearch/pytorch3d/issues/15
        vertices = []
        faces = []
        textures = []
        vert_offset = 0  # offset by vertices from prior meshes
        for cell_idx in range(num_cells):
            n_cubes = num_cubes[batch_idx][cell_idx]
            if n_cubes == 0: continue # empty cell
            for i, (position, size,color) in enumerate(zip(positions[batch_idx, cell_idx, :n_cubes, :], sizes[batch_idx, cell_idx, :n_cubes],
                                                           colors[batch_idx, cell_idx, :n_cubes, :])):
                cube_vertices, cube_faces = get_cube_mesh(position, size)
                # For now, apply same color to each mesh vertex (v \in V)
                if remove_color: texture = torch.ones_like(cube_vertices)
                else: texture = torch.ones_like(cube_vertices) * color# [V, 3]
                # Offset faces (account for diff indexing, b/c treating as one mesh)
                cube_faces = cube_faces + vert_offset
                vert_offset += cube_vertices.shape[0]
                vertices.append(cube_vertices)
                faces.append(cube_faces)
                textures.append(texture)

        # Concatenate data into single mesh
        # first check that entire grid has at least one block
        if len(vertices) == 0:
            # no cubes were in the entire grid -- error case
            print("No cubes in entire grid!!")
            singleton = torch.zeros_like(torch.empty(8, 3)).to(device)
            blank_texture = TexturesVertex(verts_features=singleton[None])
            mesh = Meshes(verts=torch.tensor([]).to(device), faces=torch.tensor([]).to(device),textures=blank_texture)
            empty_idxs.add(batch_idx)
        else:
            vertices = torch.cat(vertices).to(device)
            faces = torch.cat(faces).to(device)
            textures = torch.cat(textures)[None].to(device)  # (1, num_verts, 3)
            textures = TexturesVertex(verts_features=textures)
            # each elmt of verts array is diff mesh in batch
            mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
        meshes.append(mesh)

    batched_mesh = join_meshes_as_batch(meshes)

    # Render image
    rendered_scenes = renderer(batched_mesh)   # (B, H, W, 4)

    # TODO: clean up code -- patch to handle rendering a blank img if not blocks in full cell
    if len(empty_idxs) == 0: imgs = rendered_scenes
    else:
        imgs = torch.ones_like(torch.empty(num_batches, im_size, im_size, 4))
        rendered_scene_idx = 0
        for batch_idx in range(num_batches):
            if batch_idx in empty_idxs:
                imgs[batch_idx] = torch.ones_like(torch.empty(im_size, im_size, 4))
            else: imgs[batch_idx] = rendered_scenes[rendered_scene_idx]


    # Remove alpha channel and return (B, im_size, im_size, 3)
    imgs = imgs[:, ..., :3]#.detach().squeeze().cpu().numpy()

    return imgs.to(device)


def convert_raw_locations(raw_locations, stacking_program, primitives, cell_idx, num_rows, num_cols):
    """
    Args
        raw_locations (tensor [num_blocks])
        stacking_program (tensor [num_blocks])
        primitives (list [num_primitives])
        cell_idx  (tuple of ints (x_cell, z_cell))
        num_rows (scalar integer)
        num_cols (scalar integer)
    Returns [num_blocks, 3]
    """

    # Extract
    device = primitives[0].device

    # Map cell idx to position in n x n grid, where grid is in x-z space
    # (x_cell, z_cell) = np.unravel_index(int(cell_idx), (num_rows, num_cols))
    (x_cell, z_cell) = cell_idx

    # Sample the bottom and adjust coords based on grid cell
    y = torch.tensor(-1.0, device=device)

    # adjust z based on cell idx -- place at 0 point within cell
    # cells = unit sized 1 x y x 1 (for x,y,z - y can be > 1 based on vertical stacking)
    # -z = closer to the camera, +z = farther away
    z_spacing = 1 # spacing in the z direction between cells
    z = torch.tensor(0.0 + z_spacing*z_cell , device=device)
    # z = torch.tensor(num_rows - (z_spacing*z_cell + 1), device=device)


    # each box is width 1.6 (+/- 0.8)
    min_x = -0.8
    max_x = 0.8
    screen_width = (max_x - min_x) * num_cols#num_rows

    # get pairs of [min_x, max_x] for screen
    x_bounds = np.linspace(-(screen_width / 2), (screen_width / 2), num_rows * 2)

    cell_x_min = x_bounds[x_cell*2]
    cell_x_max = x_bounds[(x_cell*2)+1]

    min_x = cell_x_min
    max_x = cell_x_max

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
        raw_locations (tensor [*shape, num_grid_rows, num_grid_cols, max_num_blocks])
        stacking_program (tensor [*shape, num_grid_rows, num_grid_cols, max_num_blocks])
        primitives (list [num_primitives])

    Returns [*shape, num_grid_rows * num_grid_cols * max_num_blocks, 3]
    """
    # Extract
    shape = raw_locations.shape[:-3]
    num_samples = util.get_num_elements(shape)
    num_grid_rows, num_grid_cols, max_num_blocks = raw_locations.shape[-3:]

    # Flatten
    # [num_samples, num_blocks]
    raw_locations_flattened = raw_locations.reshape(num_samples, num_grid_rows, num_grid_cols, max_num_blocks)
    stacking_program_flattened = stacking_program.reshape(num_samples, num_grid_rows, num_grid_cols, max_num_blocks)

    locations_batched = []
    for sample_id in range(num_samples):
        for row in range(num_grid_rows):
            for col in range(num_grid_cols):
                locations_batched.append(
                    convert_raw_locations(
                        raw_locations_flattened[sample_id, row, col],
                        stacking_program_flattened[sample_id, row, col],
                        primitives,
                        (int(col), int(row)),
                        num_grid_rows,
                        num_grid_cols
                    )
                )
    return torch.stack(locations_batched).view(*[*shape, num_grid_rows * num_grid_cols * max_num_blocks, 3])

def convert_raw_gamma(gamma):
    return torch.abs(gamma* 1e-4)

def convert_raw_sigma(sigma):
    return torch.abs(sigma * 1e-4)

def render(
    primitives, num_blocks, stacking_program, raw_locations, im_size=32,
    sigma=1e-10, gamma=1e-6
):
    """
    Args
        primitives (list [num_primitives])
        num_blocks [*shape, num_grid_rows, num_grid_cols]
        stacking_program (tensor [*shape, num_grid_rows, num_grid_cols, max_num_blocks])
        raw_locations (tensor [*shape, num_grid_rows, num_grid_cols, max_num_blocks])
        im_size
        sigma, gamma (blending parameters)

    Returns [*shape, num_channels=3, im_size, im_size]
    """

    # TODO: adjust input to include cell idxs + more blocks

    if torch.is_tensor(gamma):gamma = convert_raw_gamma(gamma)
    if torch.is_tensor(sigma):sigma = convert_raw_sigma(sigma)

    # Extract
    shape = stacking_program.shape[:-3]
    num_grid_rows, num_grid_cols, max_num_blocks = stacking_program.shape[-3:]
    num_elements = util.get_num_elements(shape)
    num_channels = 3

    # [num_primitives]
    square_size = torch.stack([primitive.size for primitive in primitives])
    # [num_primitives, 3]
    square_color = torch.stack([primitive.color for primitive in primitives])

    # Convert [*shape, max_num_blocks, 3]
    locations = convert_raw_locations_batched(raw_locations, stacking_program, primitives)

    # Flatten
    # num_blocks_flattened = torch.sum(num_blocks.reshape((num_elements, num_grid_rows * num_grid_cols)),axis=1)
    num_blocks_flattened = num_blocks.reshape((num_elements, num_grid_rows * num_grid_cols))

    stacking_program_flattened = stacking_program.reshape((num_elements, num_grid_rows * num_grid_cols, max_num_blocks))
    locations_flattened = locations.view((num_elements, num_grid_rows * num_grid_cols, max_num_blocks, 3))

    imgs = render_cubes(num_blocks_flattened, square_size[stacking_program_flattened], square_color[stacking_program_flattened], locations_flattened, im_size,
                        sigma,gamma)
    imgs = imgs.permute(0, 3, 1, 2)

    imgs = imgs.view(*[*shape, num_channels, *imgs.shape[-2:]])

    return imgs

