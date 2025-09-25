# Dependencies
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, MouseEvent

# Top-Level Imports
from specview.gui_state_classes import ImageState, PixelCoordinate


def crosshairs(
    img_axis: Axes,
    spec_axis: Axes,
    cube: np.ndarray,
    wvl: np.ndarray,
    state: ImageState,
    event: Event,
):
    if not isinstance(event, MouseEvent):
        return
    if event.xdata is None or event.ydata is None:
        return

    if state.crosshair.is_active and event.inaxes == img_axis:
        state.crosshair.xline.set_visible(True)
        state.crosshair.yline.set_visible(True)
        state.crosshair.scan_spectrum.set_visible(True)

        pixel_coords = PixelCoordinate(
            event.xdata,
            event.ydata,
        )

        data_coords = PixelCoordinate(
            event.xdata + state.img_offsets.x_offset,
            event.ydata + state.img_offsets.y_offset,
        )

        state.crosshair.xline.set_ydata([pixel_coords.y, pixel_coords.y])
        state.crosshair.yline.set_xdata([pixel_coords.x, pixel_coords.x])

        state.crosshair.scan_spectrum.set_ydata(data_coords.pull_data(cube))

        if spec_axis is not None:
            spec_axis.relim()
            spec_axis.autoscale_view()
    else:
        state.crosshair.xline.set_visible(False)
        state.crosshair.yline.set_visible(False)
        state.crosshair.scan_spectrum.set_visible(False)
