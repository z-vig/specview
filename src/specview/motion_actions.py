# Relative Imports
from .gui_states import PanningState, CrosshairState

# Dependencies
import numpy as np


def apply_panning(
    event, panning_state: PanningState, sensitivity: float = 0.2
):
    if not panning_state.is_panning:
        return
    if panning_state.pan_start is None or panning_state.pan_ax is None:
        return
    if event.inaxes != panning_state.pan_ax:
        return
    if event.x is None or event.y is None:
        return

    ax = panning_state.pan_ax
    inv = ax.transData.inverted()

    start_pt = inv.transform(panning_state.pan_start)
    end_pt = inv.transform((event.x, event.y))

    dx_data = (start_pt[0] - end_pt[0]) * sensitivity
    dy_data = (start_pt[1] - end_pt[1]) * sensitivity

    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()

    ax.set_xlim(cur_xlim[0] + dx_data, cur_xlim[1] + dx_data)
    ax.set_ylim(cur_ylim[0] + dy_data, cur_ylim[1] + dy_data)

    panning_state.pan_start = (event.x, event.y)


def apply_crosshairs(event, state: CrosshairState, cube: np.ndarray):
    if (
        state.is_scanning
        and event.inaxes == state.crosshair_ax
        and event.ydata > 0
        and event.ydata < cube.shape[0]
    ):
        state.xcrosshair.set_visible(True)
        state.ycrosshair.set_visible(True)

        state.xcrosshair.set_xdata([event.xdata])
        state.ycrosshair.set_ydata([event.ydata])

        pixel_coords = (int(event.ydata), int(event.xdata))
        if all([i > 0 for i in pixel_coords]):
            state.data_line.set_ydata(cube[*pixel_coords, :])

        if state.scanning_ax is not None:
            state.scanning_ax.relim()
            state.scanning_ax.autoscale_view()
    else:
        state.xcrosshair.set_visible(False)
        state.ycrosshair.set_visible(False)
