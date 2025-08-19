# Standard Libraries
from time import time
from typing import Tuple

# Dependencies
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.widgets import RangeSlider
from matplotlib.image import AxesImage
from matplotlib.axes import Axes

# Relative Imports
from .rgb_image import ThreeBandRGB
from .utils import percentiles


class ImageUnboundError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def add_slider(image_data: np.ndarray, image_object: AxesImage, axis: Axes):
    finite_img = image_data[np.isfinite(image_data)]
    norm = mcolors.Normalize(*percentiles(finite_img, 1, 99))

    if norm.vmin is not None and norm.vmax is not None:
        targ_slider = RangeSlider(
            ax=axis,
            label="",
            valmin=finite_img.min(),
            valmax=finite_img.max(),
            valinit=(norm.vmin, norm.vmax),
        )
        image_object.norm = norm
    else:
        raise

    def update(_val):
        # Preventing GUI loop event flooding from slider.
        now = time()
        last_slider_update = 0

        if now - last_slider_update < 0.05:
            return
        last_slider_update = now

        vmin, vmax = map(float, targ_slider.val)
        if vmin < vmax:  # guard against inversion
            image_object.norm.vmin = vmin
            image_object.norm.vmax = vmax
            axis.figure.canvas.draw_idle()

    targ_slider.on_changed(update)

    return targ_slider


def add_rgb_slider(
    rgb: ThreeBandRGB, image_object: AxesImage, axes: Tuple[Axes, Axes, Axes]
) -> Tuple[RangeSlider, RangeSlider, RangeSlider]:
    rax: Axes = axes[0]
    gax: Axes = axes[1]
    bax: Axes = axes[2]

    rslide = RangeSlider(
        rax,
        "Red Channel",
        rgb.rlow,
        rgb.rhigh,
        valinit=percentiles(rgb.red, 1, 99),
    )
    gslide = RangeSlider(
        gax,
        "Green Channel",
        rgb.glow,
        rgb.ghigh,
        valinit=percentiles(rgb.green, 1, 99),
    )
    bslide = RangeSlider(
        bax,
        "Blue Channel",
        rgb.blow,
        rgb.bhigh,
        valinit=percentiles(rgb.blue, 1, 99),
    )

    def update(_val):
        rgb.rlow, rgb.rhigh = rslide.val
        rgb.glow, rgb.ghigh = gslide.val
        rgb.blow, rgb.bhigh = bslide.val

        image_object.set_data(rgb.scaled_rgb)
        image_object.axes.figure.canvas.draw_idle()

    rslide.on_changed(update)
    gslide.on_changed(update)
    bslide.on_changed(update)

    return rslide, gslide, bslide
