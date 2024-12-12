import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests
from pathlib import Path
from PIL import Image


def download_model(model_url: str = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
                   config_url: str =  "https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/sam2/configs/sam2/sam2_hiera_l.yaml",
                   force: bool = False
                   ) -> None:
    """Check if there are files inside the models folder, if not download the 
    model and config files. Use force=True to force download."""
    if not force and len(list(Path("../models").glob("*.pt"))) > 0:
        print("Model files already exist. Use force=True to redownload.")
        return
    
    response = requests.get(model_url, stream=True)
    with open("../models/sam2_hiera_large.pt", 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {model_url.split('/')[-1]}")

    response = requests.get(config_url, stream=True)
    with open("../models/sam2_hiera_l.yaml", 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {config_url.split('/')[-1]}")


def get_image(image_name, image_dir):
    """Get image from the image directory. Some image 
    files have <space> in their names, and <space> replaced
    by <_> in the csv table. This function tries to find the
    image file."""
    # some image files have <space> in their names,
    # but <space> replaced by <_> in the csv table.
    image_file = image_dir.joinpath(image_name)
    if not image_file.exists():
        # try to find the image
        image_file = None
        for file in image_dir.glob("*.jpg"):
            if image_name.replace("_", " ") == file.name.replace("_", " "):
                image_file = image_dir.joinpath(file.name)
                break

    if image_file is not None:
        test_image = Image.open(image_file)
        return np.array(test_image)
    else:
        return None


def get_point_coord(point: str):
    """Get point coordinates from string."""
    # to convert string coordinates into numbers
    point = point.strip("[]").replace('"', '').replace("'", "")
    y, x = point.split(",")

    return float(y), float(x)


def get_polygon(mask):
    """Get polygon coordinates from mask."""
    mask_img = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS
    )
    # contours shape: n,1,2 in x,y coords
    # transform to n,2 in y,x coords
    return contours[0][:, 0, [1, 0]]


def show_mask(mask, ax, random_color=False):
    """Plot mask into axes."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_mask_and_label(mask, species, out_file, input_point, input_label, image, saveit = True):
    """Plot mask, point and species label. You can also save it."""
    plt.figure(figsize=(40, 22))
    plt.imshow(image)
    show_mask(mask, plt.gca())

    if (input_point is not None) and (input_label is not None):
        show_points(input_point, input_label, plt.gca(), marker_size=100)

    #print(f"Score: {score:.3f}")
    plt.axis("off")
    plt.title(species[0], fontsize=40)
    if saveit:
        plt.savefig(out_file,  bbox_inches='tight', pad_inches=0)
        

def show_points(coords, labels, ax, marker_size=20):
    """Plot all points."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1],
        color="limegreen", marker="o", s=marker_size,
        edgecolor="white", linewidth=0.9, alpha=0.85, zorder=999
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1],
        color="red", marker="o", s=marker_size,
        edgecolor="white", linewidth=0.9, alpha=0.85, zorder=999
    )


def show_box(box, ax):
    """Plot bounding box."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_res(masks, scores, input_point, input_label, input_box, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())

        print(f"Score: {score:.3f}")
        plt.axis("off")


def show_res_multi(masks, scores, image, input_box=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    ax.imshow(image)
    img_area = image.shape[0] * image.shape[1]
    for mask in masks:
        # skip very large masks which probably covers background
        if mask.sum() < img_area / 2:
            show_mask(mask, ax, random_color=True)

    if input_box is not None:
        for box in input_box:
            show_box(box, ax)

    # for score in scores:
    #     print(f"Score: {score.item():.3f}")
    # ax.axis("off")


def save_image_masks(masks, image_name, results_dir):
    """Save each mask as a separate png in results_dir."""
    save_dir = results_dir.joinpath(image_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, mask in enumerate(masks):
        # mask is 3D: 1, y, x
        mask_img = mask[0].astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_img)
        mask_img.save(save_dir.joinpath(f"{i:03d}.png"))