import pyredner
import torch
import util
import matplotlib.pyplot as plt
import trimesh


def get_transform(translation):
    """Computes transformation matrix from a translation vector

    Args [3]

    Returns [4, 4]
    """

    device = translation.device
    transform = torch.eye(4, device=device)
    transform[:3, -1] = translation

    return transform


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
    translation = position.clone()
    translation[-1] += size / 2
    transform = get_transform(translation=translation)
    box = trimesh.creation.box(extents=[size, size, size], transform=transform)
    box.fix_normals()
    # box.invert()
    box_vertices = torch.tensor(box.vertices, device=device, dtype=torch.float32)
    box_faces = torch.tensor(box.faces, device=device, dtype=torch.int32)

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
    axs[0].imshow(target_img.detach())
    axs[0].set_title("Target")
    axs[1].imshow(img.detach())
    axs[1].set_title(f"Rendered (iter {iteration})")
    axs[2].plot(losses)
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Loss")
    for ax in axs[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    util.save_fig(fig, path)


if __name__ == "__main__":
    pyredner.set_use_gpu(False)
    device = "cpu"
    # pyredner.set_use_gpu(torch.cuda.is_available())
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    true_size = torch.tensor(0.1)
    true_position = torch.tensor([0.5, 0.8, 0.0])
    true_color = torch.tensor([1.0, 0, 0])
    target_img = render_cube(0, true_size, true_color, true_position)
    raw_color = torch.randn(3, requires_grad=True)

    optimizer = torch.optim.Adam([raw_color], lr=5e-2)
    num_iterations = 100
    losses = []

    for i in range(num_iterations):
        optimizer.zero_grad()
        img = render_cube(i + 1, true_size, torch.softmax(raw_color, 0), true_position)
        loss = (img - target_img).pow(2).sum()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        plot(target_img, img, losses, i, f"save/cube/{i}.png")

        print(f"Iter. {i} | Loss {losses[-1]}")

    util.make_gif(
        [f"save/cube/{i}.png" for i in range(num_iterations)], "save/cube/reconstruction.gif", 10
    )
