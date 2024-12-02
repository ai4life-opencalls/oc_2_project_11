import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def bit_get(val, idx):
    """Gets the bit value.
    Args:
        val: Input value, int or numpy int array.
        idx: Which bit of the input val.
    Returns:
        The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


def get_color_map(num_colors=100):
    """Creates a color map for visualizing segmentation results.
    """
    num_colors += 1  # to omit the first color which is black
    colors = np.zeros((num_colors, 3), dtype=int)
    indices = np.arange(num_colors, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colors[:, channel] |= bit_get(indices, channel) << shift
        indices >>= 3

    # make sure colors' max is 255 (to have brighter colors)
    colors = (
        (colors - colors.min()) * (255 / colors.max() - colors.min())
    ).astype("uint8")[1:]  # omit the black color
    assert colors.max() == 255

    norm = mpl.colors.Normalize(0, 255)
    cm = mpl.colors.LinearSegmentedColormap.from_list("happy", norm(colors), num_colors - 1)

    return cm, colors



if __name__ == "__main__":
    a = np.linspace(0, 50, 100).reshape(10, 10)
    plt.imshow(a, cmap="jet")

    cm, colors = get_color_map(30)

    plt.imshow(a, cmap=cm)
