import pyredner
import torch
import random


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

    # Camera
    cam = pyredner.Camera(
        position=torch.tensor([0.5, 0.0, 0.4]),
        look_at=torch.tensor([0.5, 1.0, 0.2]),
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
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]], device=device
        ),
        indices=torch.tensor([[0, 2, 1], [1, 2, 3]], dtype=torch.int32, device=device),
        uvs=None,
        normals=None,
        material_id=0,
    )
    shape_light_front = pyredner.Shape(
        vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], device=device
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
