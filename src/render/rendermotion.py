import numpy as np
import imageio
import os
import argparse
from tqdm import tqdm
from .renderer import get_renderer
from skimage.transform import resize


def get_rotation(theta=np.pi / 3):
    import src.utils.rotation_conversions as geometry
    import torch

    axis = torch.tensor([0, 1, 0], dtype=torch.float)
    axisangle = theta * axis
    matrix = geometry.axis_angle_to_matrix(axisangle)
    return matrix.numpy()


def render_video(
    meshes,
    key,
    action,
    renderer,
    savepath,
    background,
    cam=(0.75, 0.75, 0, 0.10),
    color=[0.11, 0.53, 0.8],
):
    writer = imageio.get_writer(savepath, fps=30)
    # center the first frame
    meshes = meshes - meshes[0].mean(axis=0)
    # matrix = get_rotation(theta=np.pi/4)
    # meshes = meshes[45:]
    # meshes = np.einsum("ij,lki->lkj", matrix, meshes)
    imgs = []
    for mesh in tqdm(meshes, desc=f"Visualize {key}, action {action}"):
        img = renderer.render(background, mesh, cam, color=color)
        # img = resize(img, (480, 800))
        imgs.append(img)
        # show(img)

    imgs = np.array(imgs)
    masks = ~(imgs / 255.0 > 0.95).all(-1)

    coords = np.argwhere(masks.sum(axis=0))
    y1, x1 = coords.min(axis=0)
    y2, x2 = y1 + 800, x1 + 480

    for cimg in imgs[:, y1:y2, x1:x2]:
        cimg = (resize(cimg, (800, 480)) * 256).astype(np.uint8)
        writer.append_data(cimg)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    opt = parser.parse_args()
    filename = opt.filename
    savefolder = os.path.splitext(filename)[0]
    os.makedirs(savefolder, exist_ok=True)

    output = np.load(filename)

    # output = {f"generation_{key}": output[key] for key in range(len(output))}
    output = {f"generation_{key}": output[key] for key in range(len(output))}

    width = 1024
    height = 1024

    background = np.zeros((height, width, 3))
    renderer = get_renderer(width, height)

    # if str(action) == str(1) and str(key) == "generation_4":
    for key in output:
        vidmeshes = output[key]
        for action in range(len(vidmeshes)):
            meshes = vidmeshes[action].transpose(2, 0, 1)
            action = [6, 12, 27, 36, 39][action]
            path = os.path.join(savefolder, "action{}_{}.mp4".format(action, key))
            render_video(meshes, key, action, renderer, path, background)


if __name__ == "__main__":
    main()
