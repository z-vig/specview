# Standard Libraries
from pathlib import Path as FilePath
from datetime import datetime

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5  # type: ignore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib import colormaps
import PyQt6.QtWidgets as qtw
from tkinter.filedialog import asksaveasfilename

# Top-Level Imports
from specview.gui_state_classes import (
    SpectralState,
    PixelCoordinate,
    PlottedSingleSpectrum,
    PlottedMeanSpectrum,
)


class SpectralCanvas(FigureCanvasQTAgg):
    def __init__(
        self,
        cube: np.ndarray,
        wvl: np.ndarray,
        parent: qtw.QWidget,
        colormap: str = "tab10",
    ):
        fig, self.spec_axis = plt.subplots(1, 1)
        self.spec_axis.margins(0, 0)
        super().__init__(fig)

        self.cube = cube
        self.wvl = wvl
        self.state = SpectralState()
        self.parent = parent

        self.cmap = colormaps.get_cmap(colormap)

    def add_spectrum(self, c: PixelCoordinate):
        spectrum_data = c.pull_data(self.cube)
        (line,) = self.spec_axis.plot(
            self.wvl,
            spectrum_data,
            color=self.cmap(self.state.nspectra / self.cmap.N),
        )

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
            linestyle="",
            capsize=2,
        )

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
        self.draw_idle()

    def save_spectra(self):
        file = asksaveasfilename(
            filetypes=[("SPEC", "*.spec")],
            defaultextension="*.spec",
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
            for obj in self.state.spectral_catalog:
                g.create_dataset(obj.name, data=obj.data)
                if isinstance(obj, PlottedMeanSpectrum):
                    g.create_dataset(f"{obj.name}_error", data=obj.data_err)
                    g.create_dataset(
                        f"{obj.name}_coords", data=obj.coords_as_array()
                    )
                elif isinstance(obj, PlottedSingleSpectrum):
                    g[obj.name].attrs[
                        "coordinate"
                    ] = obj.pixel_coord.as_tuple()
            f.attrs["wvls"] = self.wvl
