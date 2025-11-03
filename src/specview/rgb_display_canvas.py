# Standard Libraries
from typing import Any, TypedDict

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backend_bases import Event, MouseEvent, KeyEvent
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

# Top-Level Dependencies
from specview.slots import scrolling_zoom, motion_panning, crosshairs
from specview.gui_state_classes import ImageState, PixelCoordinate
from specview.utils import square_image
from specview.spectral_display_canvas import SpectralCanvas


class RGBCanvasSettings(TypedDict):
    zoom_speed: float


class RGBCanvas(FigureCanvasQTAgg):
    """
    Widget for displaying and interacting with a 2D image that is
    representative of a spectral cube. The following implemented interactions
    are:

    - Scrolling: Zoom in/out
    - Panning: Hold down middle mouse button

    Parameters
    ----------
    img_data: np.ndarray
        A 2D Array that is representative of a spectral cube.
    settings: ImageCanvasSettings
        Settings class to help initialize aspects of the widget. Settings
        include:

        - zoom_speed: float // Speed of zoom in/out while scrolling.

    shared_gui_state: SharedState
        An instance of a shared GUI state class, holding information to be
        passed between the image window and the spectral window, such as image
        coordinates.
    parent: QMainWindow
        Parent Qt window.
    """

    def __init__(
        self,
        img_data: np.ndarray,
        settings: RGBCanvasSettings,
        connected_spectral_canvas: SpectralCanvas,
        parent,
    ):
        self.settings = settings
        self.parent = parent
        self.spec_canv = connected_spectral_canvas
        fig, self.img_axis = plt.subplots(1, 1, tight_layout=True)

        offsets, self.square_img = square_image(img_data, rgb=True)

        self.state = ImageState()
        self.state.lasso.set_pixel_coords(self.square_img)
        self.state.img_offsets = offsets

        # ========= Plotting the image using plt.imshow ==========
        self.img_axis.imshow(self.square_img)
        self.img_axis.set_axis_off()
        self.img_data = img_data
        # =========================================================

        super().__init__(fig)

        self.selector = LassoSelector(
            self.img_axis,
            onselect=self.onselect,
            useblit=True,
            props={"color": "red"},
        )
        self.selector.set_active(False)

        # Connecting all slots to matplotlib signals
        self.mpl_connect("scroll_event", self.on_scroll)
        self.mpl_connect("button_press_event", self.on_button_press)
        self.mpl_connect("button_release_event", self.on_button_release)
        self.mpl_connect("key_press_event", self.on_key_press)
        self.mpl_connect("key_release_event", self.on_key_release)
        self.mpl_connect("motion_notify_event", self.on_motion)

    def on_scroll(self, event: Event) -> Any:
        scrolling_zoom(self.img_axis, event, self.settings["zoom_speed"])
        self.draw_idle()

    def on_button_press(self, event: Event) -> Any:
        if not isinstance(event, MouseEvent):
            return
        if event.xdata is None or event.ydata is None:
            return

        if event.button == 2:
            self.state.panning.is_active = True
            self.state.panning.start_coord = (
                event.xdata + self.state.img_offsets.x_offset,
                event.ydata + self.state.img_offsets.y_offset,
            )

        if event.button == 1:
            if self.state.lasso.is_active:
                return
            if event.inaxes == self.img_axis:
                self.parent.show_spectral_window()
                c = PixelCoordinate(
                    x=event.xdata + self.state.img_offsets.x_offset,
                    y=event.ydata + self.state.img_offsets.y_offset,
                )
                print(c)
                self.spec_canv.add_spectrum(c)

    def on_button_release(self, event: Event) -> Any:
        if not isinstance(event, MouseEvent):
            return

        if event.button == 2:
            self.state.panning.is_active = False

    def on_motion(self, event: Event) -> Any:
        if not isinstance(event, MouseEvent):
            return
        if event.xdata is None or event.ydata is None:
            return

        motion_panning(self.img_axis, self.state, event)
        crosshairs(
            self.img_axis,
            self.spec_canv.spec_axis,
            self.spec_canv.cube,
            self.state,
            event,
        )
        self.draw_idle()
        self.spec_canv.draw_idle()

        # print(
        #     "Data Coords: ("
        #     f"{event.xdata + self.state.img_offsets.x_offset},"
        #     f"{event.ydata + self.state.img_offsets.y_offset}), "
        #     f" Figure Coords: {event.x}, {event.y}"
        # )

    def on_key_press(self, event: Event) -> Any:
        if not isinstance(event, KeyEvent):
            return

        if event.key == "c":
            self.parent.show_spectral_window()
            self.state.crosshair.is_active = not self.state.crosshair.is_active
            if self.state.crosshair.is_active:
                self.state.crosshair.add_to_axes(
                    self.img_axis,
                    self.spec_canv.spec_axis,
                    self.spec_canv.wvl,
                    self.spec_canv.cube,
                )
            elif not self.state.crosshair.is_active:
                self.state.crosshair.scan_spectrum.remove()
                self.spec_canv.spec_axis.relim()
                self.spec_canv.spec_axis.autoscale_view()

        if event.key == "control":
            self.state.lasso.is_active = True
            self.selector.set_active(True)

    def on_key_release(self, event: Event) -> Any:
        if not isinstance(event, KeyEvent):
            return
        if event.key == "control":
            self.state.lasso.is_active = False
            self.selector.set_active(False)
            print("Lasso is inactive")

    def onselect(self, vertices):
        self.parent.show_spectral_window()
        path = Path(vertices)
        inside = path.contains_points(self.state.lasso.pixel_coords)
        idxs = np.nonzero(inside)[0]
        h, w, _ = self.square_img.shape
        rows, cols = np.unravel_index(idxs, (h, w))
        selected_pixels = [
            PixelCoordinate(
                x + self.state.img_offsets.x_offset,
                y + self.state.img_offsets.y_offset,
            )
            for x, y in zip(cols, rows)
        ]
        self.spec_canv.add_lasso(selected_pixels)
        print(f"{len(selected_pixels)} pixels in area")
