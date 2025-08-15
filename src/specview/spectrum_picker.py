# Standard Libraries
from typing import Optional

# Dependencies
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Button, LassoSelector
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.path import Path
from matplotlib.patches import Rectangle
from tkinter.filedialog import asksaveasfilename
import h5py as h5  # type: ignore

# Relative Imports
from .utils import set_image_axis
from .scrolling_actions import apply_zoom
from .motion_actions import apply_panning
from .gui_states import ImagePlotState, SpectrumPlotState
from .spectrum_plot_objects import PlottedMeanSpectrum, PlottedSingleSectrum


class PlotNotInitializedError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class SpectrumDislpay:
    def __init__(
        self,
        wvl: npt.NDArray[np.floating],
        spectral_cube: npt.NDArray[np.floating],
    ):
        self.wvl = wvl
        self.data = spectral_cube
        self.fig: Optional[Figure] = None
        self.plot_ax: Optional[Axes] = None
        self._plot_state = SpectrumPlotState()

    def show(self):
        self.fig = plt.figure(tight_layout=True)
        gridspec = self.fig.add_gridspec(nrows=9, ncols=12)
        self.plot_ax = self.fig.add_subplot(gridspec[0:8, 0:10])
        self.button_ax = self.fig.add_subplot(gridspec[8:9, 1:5])
        self.save_button_ax = self.fig.add_subplot(gridspec[8:9, 5:9])
        self.button = Button(self.button_ax, "Clear Spectra")
        self.save_button = Button(self.save_button_ax, "Save Plot Data")
        self.button.on_clicked(self.clear_spectra)
        self.save_button.on_clicked(self.save_current_spectra)

        plt.show(block=False)

    def clear_spectra(self, event):
        [i.plot_obj.remove() for i in self._plot_state.cached_plots]
        self._plot_state.cached_plots = []
        self._plot_state.single_spec_count = 0
        self._plot_state.mean_spec_count = 0

        if self.fig is not None:
            self.fig.canvas.draw_idle()
        else:
            raise PlotNotInitializedError("Figure was not initialized.")

    def plot(self, x: int, y: int):
        self._plot_state.single_spec_count += 1
        dat = self.data[y, x, :]

        if self.plot_ax is not None:
            (sp,) = self.plot_ax.plot(self.wvl, dat)
        else:
            raise PlotNotInitializedError("Axis was not initialized.")

        print(
            f"Plotted from X: {x}, Y: {y} ("
            f"{self._plot_state.single_spec_count} in total)."
        )

        if self._plot_state.currently_plotted.name != "NULL":
            self._plot_state.cached_plots.append(
                self._plot_state.currently_plotted
            )
        self._plot_state.currently_plotted = PlottedSingleSectrum(
            name=f"SPECTRUM_{self._plot_state.single_spec_count:02d}",
            plot_obj=sp,
            wvl=self.wvl,
            data=dat,
            pixel_coord=(x, y),
        )

        if self.fig is not None:
            self.fig.canvas.draw_idle()
        else:
            raise PlotNotInitializedError("Figure was not initialized.")

    def plot_average(
        self,
        x_vals: npt.NDArray[np.int16],
        y_vals: npt.NDArray[np.int16],
        include_error_bars: bool = False,
    ):
        self._plot_state.mean_spec_count += 1
        all_spectra = self.data[x_vals, y_vals, :]
        nspec = all_spectra.shape[0]
        mean_spec = np.mean(all_spectra, axis=0)
        err_spec = np.std(all_spectra, axis=0, ddof=1)
        if self.plot_ax is not None:
            if include_error_bars:
                (sp,) = self.plot_ax.errorbar(self.wvl, mean_spec, err_spec)
            else:
                (sp,) = self.plot_ax.plot(self.wvl, mean_spec)
        else:
            raise PlotNotInitializedError("Axis was not initialized.")

        print(
            f"Averaged {nspec} spectra. ({self._plot_state.mean_spec_count}"
            " mean spectra in total.)"
        )

        save_pixel_coords = np.hstack((y_vals, x_vals))

        if self._plot_state.currently_plotted.name != "NULL":
            self._plot_state.cached_plots.append(
                self._plot_state.currently_plotted
            )
        self._plot_state.currently_plotted = PlottedMeanSpectrum(
            name=f"AREA_{self._plot_state.mean_spec_count:02d}",
            plot_obj=sp,
            wvl=self.wvl,
            data=mean_spec,
            data_err=err_spec,
            pixel_coords=save_pixel_coords,
            total=nspec,
        )

        if self.fig is not None:
            self.fig.canvas.draw_idle()
        else:
            raise PlotNotInitializedError("Figure was not initialized.")

    def save_current_spectra(self, event):
        self._plot_state.cached_plots.append(
            self._plot_state.currently_plotted
        )
        file = asksaveasfilename(
            filetypes=[("HDF5", "*.hdf5")],
            defaultextension="*.hdf5",
            initialdir="./",
        )

        with h5.File(file, "w") as f:
            for obj in self._plot_state.cached_plots:
                f[obj.name] = obj.data
                if isinstance(obj, PlottedMeanSpectrum):
                    f[f"{obj.name}_error"] = obj.data_err
                    f[f"{obj.name}_coords"] = obj.pixel_coords
                elif isinstance(obj, PlottedSingleSectrum):
                    f[obj.name].attrs["coordinate"] = obj.pixel_coord
            f.attrs["wvls"] = self.wvl

    def close(self):
        plt.close(self.fig)

    def toggle(self):
        if self.is_open():
            self.close()
        else:
            self.show()

    def is_open(self):
        return self.fig is not None and plt.fignum_exists(self.fig.number)


class ImageDisplay:
    def __init__(
        self,
        display_image: np.ndarray,
        specdisplay: SpectrumDislpay,
        **mpl_kwargs,
    ) -> None:
        self.display = display_image
        self.specdisplay = specdisplay

        self.display_fig = plt.figure(figsize=(8, 8))
        gpsc = self.display_fig.add_gridspec(nrows=12, ncols=12)
        self.display_ax = self.display_fig.add_subplot(gpsc[0:11, 0:12])
        self.slider_ax = self.display_fig.add_subplot(gpsc[11:12, 0:12])

        box = self.slider_ax.get_position()
        self.slider_ax.set_position(
            (
                box.x0 + 0.15 * box.width,  # x (move right slightly)
                box.y0 + 0.15 * box.height,  # y (move up slightly)
                0.7 * box.width,  # width
                0.2 * box.height,  # height (shrink)
            )
        )
        self.slider_ax.set_facecolor("none")  # Make background transparent

        self.display_ax.set_box_aspect(1)
        set_image_axis(self.display_ax)

        default_mpl = {"cmap": "Grays_r"}
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
                float(np.quantile(finite_img, 0.95)),
            ),
        )
        self.targ_slider.on_changed(self.update)

        self._state = ImagePlotState()

        y, x = np.mgrid[
            0 : self.display.shape[0],  # noqa: E203
            0 : self.display.shape[1],  # noqa: E203
        ]
        self.pixel_coords = np.column_stack((x.ravel(), y.ravel()))

        self.selector = LassoSelector(self.display_ax, onselect=self.onselect)
        self.selector.set_active(False)

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
        self.display_fig.canvas.mpl_connect(
            "key_press_event", self.on_key_press
        )
        self.display_fig.canvas.mpl_connect(
            "key_release_event", self.on_key_release
        )

    def update(self, _val):
        self.image_obj.set_clim(*self.targ_slider.val)

    def on_scroll(self, event):
        apply_zoom(event, self.display, self.display_ax)

    def on_button_press(self, event):
        if (event.button == 2) and (event.inaxes == self.display_ax):
            self._state.panning.is_panning = True
            self._state.panning.pan_start = (event.x, event.y)
            self._state.panning.pan_ax = event.inaxes
        if (
            (event.button == 1)
            and (event.inaxes == self.display_ax)
            and self._state.collect_spectra
            and (self._state.collect_area is False)
        ):
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            self.specdisplay.plot(x, y)
            [p.remove() for p in self.display_ax.patches]

            rect = Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            self.display_ax.add_patch(rect)
            self.display_fig.canvas.draw_idle()

    def on_button_release(self, event):
        if event.button == 2:
            self._state.panning.is_panning = False
            self._state.panning.pan_start = None
            self._state.panning.pan_ax = None

    def on_key_press(self, event):
        if event.key == "shift":
            if self._state.collect_spectra is False:
                self._state.collect_spectra = True

                self.specdisplay.toggle()
            elif self._state.collect_spectra is True:
                self._state.collect_spectra = False
                [p.remove() for p in self.display_ax.patches]

                self.specdisplay.toggle()
                self.display_fig.canvas.draw_idle()

            self.display_fig.canvas.draw_idle()

        if event.key == "control" and self._state.collect_spectra:
            self._state.collect_area = True
            self.selector.set_active(True)

    def on_key_release(self, event):
        if event.key == "control" and self._state.collect_spectra:
            self._state.collect_area = False
            self.selector.set_active(False)

    def on_motion(self, event):
        apply_panning(event, self._state.panning, sensitivity=0.4)
        self.display_fig.canvas.draw_idle()

    def onselect(self, vertices):
        path = Path(vertices)
        inside = path.contains_points(self.pixel_coords)
        idxs = np.nonzero(inside)[0]
        cols, rows = np.unravel_index(
            idxs, (self.display.shape[0], self.display.shape[1])
        )
        selected_pixels = list(zip(rows, cols))
        self.specdisplay.plot_average(
            cols.astype(np.int16), rows.astype(np.int16)
        )
        print(f"{len(selected_pixels)} pixels in area")


class SpectrumPicker:
    def __init__(
        self,
        display_image: npt.NDArray[np.floating],
        spectral_cube: npt.NDArray[np.floating],
        wvl: npt.NDArray[np.floating],
    ):
        self.img = display_image
        self.cube = spectral_cube
        self.wvl = wvl

        self.specdisplay = SpectrumDislpay(self.wvl, self.cube)

        ImageDisplay(self.img, self.specdisplay)
