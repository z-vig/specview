# Standard Libraries
from typing import Any

# Dependencies
from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.axes import Axes


def scrolling_zoom(
    img_axis: Axes, event: Event, zoom_speed: float = 1.3
) -> Any:
    # Checking if the Event is the right type.
    if not isinstance(event, MouseEvent):
        return

    # Checking if the event is in the image axis.
    ax = event.inaxes
    if ax is None:
        return
    if ax != img_axis:
        return

    # Current cursor position in data coordinates
    fx = event.xdata
    fy = event.ydata

    if fx is None or fy is None:
        return

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    xdata = event.xdata
    ydata = event.ydata

    if xdata is None:
        return
    if ydata is None:
        return

    if event.button == "down":
        scale = 1 / zoom_speed
    elif event.button == "up":
        scale = zoom_speed
    else:
        return

    # Distances from cursor to each side
    dx0 = fx - x0
    dx1 = x1 - fx
    dy0 = fy - y0
    dy1 = y1 - fy

    # Scale distances
    dx0 /= scale
    dx1 /= scale
    dy0 /= scale
    dy1 /= scale

    ax.set_xlim(fx - dx0, fx + dx1)
    ax.set_ylim(fy - dy0, fy + dy1)
