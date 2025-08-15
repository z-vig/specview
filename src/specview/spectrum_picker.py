# Standard Libraries

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider

# Relative Imports
from .utils import set_image_axis
from .scrolling_actions import apply_zoom
from .motion_actions import apply_panning
from .gui_states import PanningState


class SpectralPicker:
    def __init__(
        self,
        display_image: np.ndarray,
        **mpl_kwargs
    ) -> None:
        self.display = display_image

        self.display_fig = plt.figure(figsize=(8, 8))
        gpsc = self.display_fig.add_gridspec(nrows=12, ncols=12)
        self.display_ax = self.display_fig.add_subplot(gpsc[0:11, 0:12])
        self.slider_ax = self.display_fig.add_subplot(gpsc[11:12, 0:12])

        box = self.slider_ax.get_position()
        self.slider_ax.set_position((
            box.x0 + 0.15 * box.width,  # x (move right slightly)
            box.y0 + 0.15 * box.height,  # y (move up slightly)
            0.7 * box.width,  # width
            0.2 * box.height  # height (shrink)
        ))
        self.slider_ax.set_facecolor("none")  # Make background transparent

        self.display_ax.set_box_aspect(1)
        set_image_axis(self.display_ax)

        default_mpl = {
            'cmap': "Grays_r"
        }
        mpl_kwargs = {**default_mpl, **mpl_kwargs}

        self.image_obj = self.display_ax.imshow(self.display, **mpl_kwargs)
        finite_img = self.display[np.isfinite(self.display)]
        self.targ_slider = RangeSlider(
            ax=self.slider_ax,
            label="",
            valmin=finite_img.min(),
            valmax=finite_img.max(),
            valinit=(
                float(np.quantile(finite_img, 0.05)),
                float(np.quantile(finite_img, 0.95))
            )
        )
        self.targ_slider.on_changed(self.update)

        self._state = PanningState()

        self.display_fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.display_fig.canvas.mpl_connect(
            "button_press_event", self.on_button_press
        )
        self.display_fig.canvas.mpl_connect(
            "button_release_event", self.on_button_release
        )
        self.display_fig.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )

    def update(self, _val):
        self.image_obj.set_clim(*self.targ_slider.val)

    def on_scroll(self, event):
        apply_zoom(event, self.display, self.display_ax)

    def on_button_press(self, event):
        if event.button == 2 and event.inaxes in self.display_ax:
            self._state.is_panning = True
            self._state.pan_start = (event.x, event.y)
            self._state.pan_ax = event.inaxes

    def on_button_release(self, event):
        if event.button == 2:
            self._state.is_panning = False
            self._state.pan_start = None
            self._state.pan_ax = None

    def on_motion(self, event):
        apply_panning(event, self._state)
        self.display_fig.canvas.draw_idle()
