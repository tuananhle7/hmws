import logging
import sys
from pathlib import Path
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(pathname)s:%(lineno)d | %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


def make_gif(img_paths, gif_path, fps):
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    images = []
    for img_path in tqdm(img_paths):
        images.append(imageio.imread(img_path))
    imageio.mimsave(gif_path, images, duration=1 / fps)
    logging.info(f"Saved to {gif_path}")


# Plotting
def save_fig(fig, path, dpi=100, tight_layout_kwargs={}):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(**tight_layout_kwargs)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    logging.info(f"Saved to {path}")
    plt.close(fig)
