import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_point_coord(point: str):
    # to convert string coordinates into numbers
    point = point.strip("[]").replace('"', "").replace("'", "")
    y, x = point.split(",")

    return float(y), float(x)


def get_polygon(mask):
    mask_img = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS
    )
    # contours shape: n,1,2 in x,y coords
    # transform to n,2 in y,x coords
    return contours[0][:, 0, [1, 0]]


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels == 0]
    neg_points = coords[labels == 1]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="limegreen",
        marker="o",
        s=marker_size,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.85,
        zorder=999,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="o",
        s=marker_size,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.85,
        zorder=999,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )

def plot_mask_and_label(mask, species, out_file, input_point, input_label, image, saveit = True):
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


def show_res(masks, scores, species, input_points, input_labels, input_box, image, saveit = True):
    for i, (mask, score, specie, input_point, input_label) in enumerate(zip(masks, scores, species, input_points, input_labels)):
        print(specie)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())

        #print(f"Score: {score:.3f}")
        plt.axis("off")
        plt.title(specie[0])
        if saveit:
            plt.savefig("masked_image_"+str(i)+'.png')


def show_res_multi(masks, scores, image, input_box=None, ax=None, saveit = False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    ax.imshow(image)
    for mask in masks:
        show_mask(mask, ax, random_color=True)

    if input_box is not None:
        for box in input_box:
            show_box(box, ax)

    # for score in scores:
    #     print(f"Score: {score.item():.3f}")
    # ax.axis("off")
    if saveit:
        plt.savefig('foo.png')