# Standard Libraries
from pathlib import Path as FilePath
from typing import Any
from tempfile import NamedTemporaryFile as tf

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backend_bases import Event, PickEvent
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib import colormaps
import PyQt6.QtWidgets as qtw
from tkinter.filedialog import askdirectory, askopenfilename
import spectralio as sio

# Top-Level Imports
from specview.gui_state_classes import (
    SpectralState,
    PixelCoordinate,
    PlottedSpectrum,
    PlottedSingleSpectrum,
    PlottedMeanSpectrum,
)

from specview.spectrum_edit_window import SpectrumEditWindow


class SpectralCanvas(FigureCanvasQTAgg):
    def __init__(
        self,
        cube: np.ndarray,
        wvl: np.ndarray,
        parent: qtw.QWidget,
        crs: str,
        gtrans: tuple[float, float, float, float, float, float],
        colormap: str = "tab10",
    ):
        fig, self.spec_axis = plt.subplots(1, 1, tight_layout=True)
        self.legend: None | Legend = None
        self.spec_axis.margins(0, 0)
        super().__init__(fig)

        self.cube = cube
        self.wvl = wvl
        self.state = SpectralState()
        self.parent = parent
        self.crs = crs
        self.gtrans = gtrans

        self.cmap = colormaps.get_cmap(colormap)

        self.mpl_connect("pick_event", self.on_pick)

    def on_pick(self, event: Event) -> Any:
        if not isinstance(event, PickEvent):
            return

        highlight_selection: None | Line2D = None
        if isinstance(event.artist, Line2D):
            _xdata, _ydata = event.artist.get_data()
            (highlight_selection,) = self.spec_axis.plot(
                _xdata, _ydata, label="_hightlight", color="r", zorder=2
            )

        editable_spectrum: None | PlottedSpectrum = None
        for i in self.state.spectral_catalog:
            if event.artist == i.plot_obj:
                editable_spectrum = i

        if editable_spectrum is None:
            raise ValueError(
                "The spectrum selected for editing does not exist."
            )
        self.edit_window = SpectrumEditWindow(editable_spectrum)
        self.edit_window.show()

        def delete_highlight():
            if highlight_selection is not None:
                highlight_selection.remove()
            self.draw_idle()

        self.edit_window.closed.connect(delete_highlight)

        def modify_legend():
            if self.legend is not None:
                self.legend.remove()
            handle_list = [i.plot_obj for i in self.state.spectral_catalog]
            lbl_list = [i.name for i in self.state.spectral_catalog]
            self.legend = self.spec_axis.legend(
                handles=handle_list, labels=lbl_list, bbox_to_anchor=(1, 1)
            )
            self.draw_idle()

        self.edit_window.name_changed.connect(modify_legend)

        def delete_spectrum():
            for spec in self.state.spectral_catalog:
                if spec == editable_spectrum:
                    spec.plot_obj.remove()
                    if isinstance(spec, PlottedMeanSpectrum):
                        spec.errorbar_caps[0].remove()
                        spec.errorbar_caps[1].remove()
                        spec.errorbar_lines.remove()
                    self.state.spectral_catalog.remove(spec)
                    break
            modify_legend()
            self.draw_idle()
            self.edit_window.close()

        self.edit_window.spectrum_deleted.connect(delete_spectrum)

        self.draw_idle()

    def add_spectrum(self, c: PixelCoordinate):
        spectrum_data = c.pull_data(self.cube)
        (line,) = self.spec_axis.plot(
            self.wvl,
            spectrum_data,
            color=self.cmap(self.state.nspectra / self.cmap.N),
            zorder=1,
        )
        line.set_picker(True)

        spectrum = PlottedSingleSpectrum(
            f"SPECTRUM_{self.state.nspectra+1:02d}",
            line,
            self.wvl,
            spectrum_data,
            c,
        )
        self.state.nspectra += 1
        self.state.spectral_catalog.append(spectrum)
        self.state.current_spectrum = spectrum

        if self.legend is not None:
            self.legend.remove()

        handle_list = [i.plot_obj for i in self.state.spectral_catalog]
        lbl_list = [i.name for i in self.state.spectral_catalog]

        self.legend = self.spec_axis.legend(
            handles=handle_list, labels=lbl_list, bbox_to_anchor=(1, 1)
        )

        self.spec_axis.relim()
        self.spec_axis.autoscale_view()
        self.draw_idle()

    def add_lasso(self, c_list: list[PixelCoordinate]):
        all_spectra = np.asarray([i.pull_data(self.cube) for i in c_list])

        mean_spec = np.mean(all_spectra, axis=0)
        std_spec = np.std(all_spectra, axis=0, ddof=1)
        line, errorbar_caps, errorbar_lines = self.spec_axis.errorbar(
            self.wvl,
            mean_spec,
            yerr=std_spec,
            color=self.cmap(self.state.nspectra / self.cmap.N),
            marker=".",
            capsize=2,
            zorder=1,
        )
        line.set_picker(True)

        spectrum = PlottedMeanSpectrum(
            f"SPECTURM_{self.state.nspectra+1:02d}_AREA",
            line,
            self.wvl,
            mean_spec,
            std_spec,
            errorbar_caps,
            errorbar_lines[0],
            c_list,
            len(c_list),
        )

        self.state.nspectra += 1
        self.state.spectral_catalog.append(spectrum)
        self.state.current_spectrum = spectrum

        if self.legend is not None:
            self.legend.remove()

        handle_list = [i.plot_obj for i in self.state.spectral_catalog]
        lbl_list = [i.name for i in self.state.spectral_catalog]

        self.legend = self.spec_axis.legend(
            handles=handle_list, labels=lbl_list, bbox_to_anchor=(1, 1)
        )

        self.spec_axis.autoscale()
        self.draw_idle()

    def clear_spectra(self):
        for i in self.state.spectral_catalog:
            i.plot_obj.remove()
            if isinstance(i, PlottedMeanSpectrum):
                i.errorbar_caps[0].remove()
                i.errorbar_caps[1].remove()
                i.errorbar_lines.remove()
        self.state = SpectralState()
        (null_line,) = self.spec_axis.plot([], color=(0, 0, 0, 0))
        self.legend = self.spec_axis.legend(
            handles=[null_line], labels=[""], bbox_to_anchor=(1, 1)
        )
        self.draw_idle()

    def save_spectra(self):
        save_dir = askdirectory(initialdir="./")
        for obj in self.state.spectral_catalog:
            file = FilePath(save_dir, obj.name).with_suffix(".spec")
            if isinstance(obj, PlottedSingleSpectrum):
                wvlmodel = sio.WvlModel.fromarray(obj.wvl, "nm")
                tmp = tf(delete=False, suffix=".geodata")
                sio.write_geodata(self.crs, self.gtrans, tmp.name)
                sio.write_spec1D(
                    obj.data.tolist(),
                    wvlmodel,
                    obj.name,
                    file,
                    location=obj.pixel_coord.as_tuple(),
                    location_type="pixel",
                    geodata_fp=tmp.name,
                )
                tmp.close()
            elif isinstance(obj, PlottedMeanSpectrum):
                crd_arr = obj.coords_as_array().astype(int)
                crd_list = [
                    tuple(crd_arr[n, :]) for n in range(crd_arr.shape[0])
                ]
                wvlmodel = sio.WvlModel.fromarray(self.wvl, "nm")
                sio.write_group(
                    self.cube[crd_arr[:, 1], crd_arr[:, 0], :],
                    crd_list,
                    wvlmodel,
                    name=obj.name,
                    fp=file,
                )
            else:
                raise ValueError("Obj is not of the correct type")

    def set_cube(self) -> None:
        new_cube = askopenfilename(
            initialdir="./",
            filetypes=[("geospcub", ".geospcub")],
        )
        if new_cube == "":
            return
        cube_data = sio.read_spec3D(new_cube, kind="geospcub")
        self.cube = cube_data.load_raster()
        self.wvl = cube_data.wavelength.asarray()
        self.crs = cube_data.geodata.crs
        self.gtrans = cube_data.geodata.geotransform.togdal()
