# Standard Libraries
from typing import Optional, Sequence
from pathlib import Path as FilePath
from datetime import datetime

# Dependencies
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, LassoSelector
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tkinter.filedialog import asksaveasfilename
import h5py as h5  # type: ignore

# Relative Imports
from .utils import set_image_axis, find_wvl
from .scrolling_actions import apply_zoom
from .motion_actions import apply_panning, apply_crosshairs
from .color_slider import add_slider, add_rgb_slider
from .gui_states import ImagePlotState, SpectrumPlotState
from .spectrum_plot_objects import PlottedMeanSpectrum, PlottedSingleSectrum
from .rgb_image import ThreeBandRGB


class PlotNotInitializedError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class SpectrumDislpay:
    def __init__(
        self,
        wvl: npt.NDArray[np.floating],
        spectral_cube: npt.NDArray[np.floating],
        comparison_cubes: Sequence[npt.NDArray[np.floating]] = [],
    ):
        self.wvl = wvl
        self.data = spectral_cube
        self.comp = comparison_cubes
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
            for comparison in self.comp:
                _ = self.plot_ax.plot(
                    self.wvl,
                    comparison[y, x, :],
                    color=sp.get_color(),
                    alpha=0.5,
                )
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

        if not FilePath(file).is_file():
            open_flag = "w"
        else:
            open_flag = "r+"

        with h5.File(file, open_flag) as f:
            g = f.create_group(
                f"save_{datetime.now().strftime("%m%d%YT%H%M%S")}"
            )
            for obj in self._plot_state.cached_plots:
                g.create_dataset(obj.name, data=obj.data)
                if isinstance(obj, PlottedMeanSpectrum):
                    g.create_dataset(f"{obj.name}_error", data=obj.data_err)
                    g.create_dataset(
                        f"{obj.name}_coords", data=obj.pixel_coords
                    )
                elif isinstance(obj, PlottedSingleSectrum):
                    g[obj.name].attrs["coordinate"] = obj.pixel_coord
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

        self.display_fig = plt.figure(figsize=(8, 8), tight_layout=True)
        gpsc = self.display_fig.add_gridspec(nrows=9, ncols=12)
        self.display_ax = self.display_fig.add_subplot(gpsc[0:7, 0:7])
        self.scanning_ax = self.display_fig.add_subplot(gpsc[0:5, 7:12])
        self.slider_ax = self.display_fig.add_subplot(gpsc[7:8, 0:12])

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
        self.scanning_ax.set_box_aspect(1)
        set_image_axis(self.display_ax)

        default_mpl = {"cmap": "Grays_r"}
        mpl_kwargs = {**default_mpl, **mpl_kwargs}

        self.image_obj = self.display_ax.imshow(self.display, **mpl_kwargs)

        self.targ_slider = add_slider(
            self.display, self.image_obj, self.slider_ax
        )

        self._state = ImagePlotState()
        self._state.crosshair.init_crosshairs(
            self.display_ax, self.scanning_ax, self.specdisplay.wvl
        )

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

        plt.show()

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
                self._state.crosshair.is_scanning = True
                self.specdisplay.toggle()
            elif self._state.collect_spectra is True:
                self._state.collect_spectra = False
                self._state.crosshair.is_scanning = False
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
        apply_crosshairs(event, self._state.crosshair, self.specdisplay.data)
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


class RGBImageDisplay:
    def __init__(
        self,
        rgbimage: ThreeBandRGB,
        specdisplay: SpectrumDislpay,
        band_indicators: Sequence[int],
        **mpl_kwargs,
    ) -> None:
        self.rgb = rgbimage
        self.specdisplay = specdisplay

        # Setting up axes
        self.fig = plt.figure()
        gpsc = self.fig.add_gridspec(nrows=11, ncols=12)
        self.image_ax = self.fig.add_subplot(gpsc[0:8, 0:7])
        self.scanning_ax = self.fig.add_subplot(gpsc[0:5, 7:12])
        self.red_ax = self.fig.add_subplot(gpsc[8:9, 0:12])
        self.green_ax = self.fig.add_subplot(gpsc[9:10, 0:12])
        self.blue_ax = self.fig.add_subplot(gpsc[10:11, 0:12])

        self.image_ax.set_box_aspect(1)
        self.scanning_ax.set_box_aspect(1)
        set_image_axis(self.image_ax)

        default_mpl = {"cmap": "Grays_r"}
        mpl_kwargs = {**default_mpl, **mpl_kwargs}

        self.image_obj = self.image_ax.imshow(
            self.rgb.scaled_rgb, **mpl_kwargs
        )

        self.rslide, self.gslide, self.bslide = add_rgb_slider(
            self.rgb,
            self.image_obj,
            (self.red_ax, self.green_ax, self.blue_ax),
        )

        self._state = ImagePlotState()
        self._state.crosshair.init_crosshairs(
            self.image_ax, self.scanning_ax, self.specdisplay.wvl
        )
        [
            band.init_indicator(c, self.scanning_ax, idx)
            for band, idx, c in zip(
                [
                    self._state.r_indicator,
                    self._state.g_indicator,
                    self._state.b_indicator,
                ],
                band_indicators,
                ["r", "g", "b"],
            )
        ]
        self.band_indicator_mirrors: list[None | Line2D] = [None, None, None]

        y, x = np.mgrid[
            0 : self.rgb.scaled_rgb.shape[0],  # noqa: E203
            0 : self.rgb.scaled_rgb.shape[1],  # noqa: E203
        ]
        self.pixel_coords = np.column_stack((x.ravel(), y.ravel()))

        self.selector = LassoSelector(self.image_ax, onselect=self.onselect)
        self.selector.set_active(False)

        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.fig.canvas.mpl_connect(
            "button_release_event", self.on_button_release
        )
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)

        plt.show()

    def on_scroll(self, event):
        apply_zoom(event, self.rgb.scaled_rgb, self.image_ax)

    def on_button_press(self, event):
        if (event.button == 2) and (event.inaxes == self.image_ax):
            self._state.panning.is_panning = True
            self._state.panning.pan_start = (event.x, event.y)
            self._state.panning.pan_ax = event.inaxes
        if (
            (event.button == 1)
            and (event.inaxes == self.image_ax)
            and self._state.collect_spectra
            and (self._state.collect_area is False)
        ):
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            self.specdisplay.plot(x, y)
            [p.remove() for p in self.image_ax.patches]

            rect = Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            self.image_ax.add_patch(rect)

        if event.inaxes == self.scanning_ax:
            r_press, _ = self._state.r_indicator.line_obj.contains(event)
            g_press, _ = self._state.g_indicator.line_obj.contains(event)
            b_press, _ = self._state.b_indicator.line_obj.contains(event)

            if r_press:
                self._state.r_indicator.pressed = True
            if g_press:
                self._state.g_indicator.pressed = True
            if b_press:
                self._state.b_indicator.pressed = True

        self.fig.canvas.draw_idle()

    def on_button_release(self, event):
        if event.button == 2:
            self._state.panning.is_panning = False
            self._state.panning.pan_start = None
            self._state.panning.pan_ax = None
        self._state.r_indicator.pressed = False
        self._state.g_indicator.pressed = False
        self._state.b_indicator.pressed = False

        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == "shift":
            if self._state.collect_spectra is False:
                self._state.collect_spectra = True
                self._state.crosshair.is_scanning = True
                self.specdisplay.toggle()
            elif self._state.collect_spectra is True:
                self._state.collect_spectra = False
                self._state.crosshair.is_scanning = False
                [p.remove() for p in self.image_ax.patches]

                self.specdisplay.toggle()
                self.fig.canvas.draw_idle()

            self.fig.canvas.draw_idle()

        if event.key == "control" and self._state.collect_spectra:
            self._state.collect_area = True
            self.selector.set_active(True)

    def on_key_release(self, event):
        if event.key == "control" and self._state.collect_spectra:
            self._state.collect_area = False
            self.selector.set_active(False)

        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        apply_panning(event, self._state.panning, sensitivity=0.4)
        apply_crosshairs(event, self._state.crosshair, self.specdisplay.data)
        if event.inaxes == self.scanning_ax:
            if self._state.r_indicator.pressed:
                self._state.r_indicator.line_obj.set_xdata([event.xdata])
                self.rgb.ridx, _ = find_wvl(self.specdisplay.wvl, event.xdata)
                self.image_obj.set_data(self.rgb.scaled_rgb)
            if self._state.g_indicator.pressed:
                self._state.g_indicator.line_obj.set_xdata([event.xdata])
                self.rgb.gidx, _ = find_wvl(self.specdisplay.wvl, event.xdata)
                self.image_obj.set_data(self.rgb.scaled_rgb)
            if self._state.b_indicator.pressed:
                self._state.b_indicator.line_obj.set_xdata([event.xdata])
                self.rgb.bidx, _ = find_wvl(self.specdisplay.wvl, event.xdata)
                self.image_obj.set_data(self.rgb.scaled_rgb)

            if self.specdisplay.plot_ax is not None:
                for (n, i), idx, c in zip(
                    enumerate(self.band_indicator_mirrors),
                    [self.rgb.ridx, self.rgb.gidx, self.rgb.bidx],
                    ["r", "g", "b"],
                ):
                    if i is not None:
                        i.remove()
                    self.band_indicator_mirrors[n] = (
                        self.specdisplay.plot_ax.axvline(
                            self.specdisplay.wvl[idx], color=c
                        )
                    )

        self.fig.canvas.draw_idle()
        if self.specdisplay.fig is not None:
            self.specdisplay.fig.canvas.draw_idle()

    def onselect(self, vertices):
        path = Path(vertices)
        inside = path.contains_points(self.pixel_coords)
        idxs = np.nonzero(inside)[0]
        cols, rows = np.unravel_index(idxs, self.rgb.scaled_rgb.shape[:2])
        selected_pixels = list(zip(rows, cols))
        self.specdisplay.plot_average(
            cols.astype(np.int16), rows.astype(np.int16)
        )
        print(f"{len(selected_pixels)} pixels in area")
