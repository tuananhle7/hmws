import pyredner
import torch
import util
import matplotlib.pyplot as plt


# Adapted from https://github.com/mikedh/trimesh/blob/master/trimesh/creation.py#L566
def create_box(position, size):
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


def render_cube(render_seed, size, color, position):
    device = size.device

    # Camera
    cam = pyredner.Camera(
        position=torch.tensor([0.5, 0.0, 0.4]),
        look_at=torch.tensor([0.5, 1.0, 0.2]),
        up=torch.tensor([0.0, 1.0, 0.0]),
        fov=torch.tensor([45.0]),  # in degree
        clip_near=1e-2,  # needs to > 0
        resolution=(256, 256),
        fisheye=False,
    )

    # Material
    object_material = pyredner.Material(diffuse_reflectance=color)
    white_material = pyredner.Material(
        diffuse_reflectance=torch.tensor([1.0, 1.0, 1.0], device=device)
    )
    materials = [white_material, object_material]

    # Shape
    box_vertices, box_faces = create_box(position, size)
    shape_object = pyredner.Shape(
        vertices=box_vertices, indices=box_faces, uvs=None, normals=None, material_id=1
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

    shapes = [shape_object, shape_light_top, shape_light_front]

    # Light
    light_top = pyredner.AreaLight(shape_id=1, intensity=torch.ones(3) * 2)
    light_front = pyredner.AreaLight(shape_id=2, intensity=torch.ones(3) * 2)
    area_lights = [light_top, light_front]

    # Scene
    scene = pyredner.Scene(cam, shapes, materials, area_lights)
    scene_args = pyredner.RenderFunction.serialize_scene(scene=scene, num_samples=32, max_bounces=1)

    # Render
    render = pyredner.RenderFunction.apply
    img = render(render_seed, *scene_args)

    return img


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


def get_position(raw_xy_position):
    position = torch.zeros(3, device=raw_xy_position.device)
    position[:2] = raw_xy_position.sigmoid() * 0.8 + 0.1
    return position


if __name__ == "__main__":
    # pyredner.set_use_gpu(False)
    # device = "cpu"
    pyredner.set_use_gpu(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    true_size = torch.tensor(0.1, device=device)
    true_position = torch.tensor([0.5, 0.8, 0.0], device=device)
    true_color = torch.tensor([1.0, 0, 0], device=device)
    target_img = render_cube(0, true_size, true_color, true_position)

    raw_size = torch.tensor(-1.0, device=device, requires_grad=True)
    raw_color = torch.randn(3, device=device, requires_grad=True)
    raw_xy_position = torch.zeros(2, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([raw_size, raw_xy_position, raw_color], lr=5e-2)
    num_iterations = 100
    losses = []

    for i in range(num_iterations):
        optimizer.zero_grad()
        img = render_cube(
            # i + 1, raw_size.exp(), torch.softmax(raw_color, 0), get_position(raw_xy_position)
            i + 1,
            raw_size.exp(),
            torch.softmax(raw_color, 0),
            true_position,
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
