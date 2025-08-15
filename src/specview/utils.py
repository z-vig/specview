# Dependencies
from matplotlib.axes import Axes


def set_image_axis(ax: Axes) -> None:
    """
    Sets a matplotlib axis up to display an image.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False
    )
