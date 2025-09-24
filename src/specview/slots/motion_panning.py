# Dependencies
from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.axes import Axes

# Top-Level Dependencies
from specview.gui_state_classes import ImageState


def motion_panning(
    img_axis: Axes, state: ImageState, event: Event, sensitivity: float = 0.5
):
    if not isinstance(event, MouseEvent):
        return
    if not state.panning.is_active:
        return
    if event.inaxes != img_axis:
        return
    if event.xdata is None or event.ydata is None:
        return

    start_pt = state.panning.start_coord
    end_pt = (
        event.xdata + state.img_offsets.x_offset,
        event.ydata + state.img_offsets.y_offset,
    )

    dx_data = (start_pt[0] - end_pt[0]) * sensitivity
    dy_data = (start_pt[1] - end_pt[1]) * sensitivity

    cur_xlim = img_axis.get_xlim()
    cur_ylim = img_axis.get_ylim()

    img_axis.set_xlim(cur_xlim[0] + dx_data, cur_xlim[1] + dx_data)
    img_axis.set_ylim(cur_ylim[0] + dy_data, cur_ylim[1] + dy_data)

    state.panning.start_coord = end_pt
