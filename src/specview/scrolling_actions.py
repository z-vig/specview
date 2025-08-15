# Dependencies
from matplotlib.axes import Axes
import numpy as np


def apply_zoom(
    event,
    image: np.ndarray,
    in_axis: Axes,
    zoom_step: float = 1.3
):
    ax = event.inaxes
    if ax == in_axis:
        img = image
    else:
        return

    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()

    xdata = event.xdata
    ydata = event.ydata

    x_left = xdata - cur_xlim[0]
    x_right = cur_xlim[1] - xdata
    y_bottom = ydata - cur_ylim[0]
    y_top = cur_ylim[1] - ydata

    if event.button == 'up':
        scale = 1 / zoom_step
    elif event.button == 'down':
        scale = zoom_step
    else:
        return

    new_xlim = [
        xdata - x_left * scale,
        xdata + x_right * scale
    ]
    new_ylim = [
        ydata - y_bottom * scale,
        ydata + y_top * scale
    ]

    # Clamp to image bounds
    new_xlim[0] = max(0, new_xlim[0])
    new_xlim[1] = min(img.shape[1], new_xlim[1])
    new_ylim[0] = max(0, new_ylim[0])
    new_ylim[1] = min(img.shape[0], new_ylim[1])

    ax.set_xlim(new_xlim)
    ax.set_ylim(new_ylim)
