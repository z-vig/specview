# Standard Libraries
from pathlib import Path as FilePath

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib import colormaps
import PyQt6.QtWidgets as qtw
from tkinter.filedialog import askdirectory

# Top-Level Imports
from specview.gui_state_classes import (
    SpectralState,
    PixelCoordinate,
    PlottedSingleSpectrum,
    PlottedMeanSpectrum,
)
from specview.save_models import FullSpectrum, SpectrumCollection
from specview.utils import forward_geotransform


class SpectralCanvas(FigureCanvasQTAgg):
    def __init__(
        self,
        cube: np.ndarray,
        wvl: np.ndarray,
        parent: qtw.QWidget,
        gtrans: tuple[float, float, float, float, float, float],
        colormap: str = "tab10",
    ):
        fig, self.spec_axis = plt.subplots(1, 1)
        self.spec_axis.margins(0, 0)
        super().__init__(fig)

        self.cube = cube
        self.wvl = wvl
        self.state = SpectralState()
        self.parent = parent
        self.gtrans = gtrans

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
        save_dir = askdirectory(initialdir="./")

        for obj in self.state.spectral_catalog:
            file = FilePath(save_dir, obj.name).with_suffix(".spec")
            if isinstance(obj, PlottedSingleSpectrum):
                long, lat = forward_geotransform(
                    obj.pixel_coord.x, obj.pixel_coord.y, self.gtrans
                )
                spec_model = FullSpectrum(
                    wavelength=obj.wvl.tolist(),
                    spectrum=obj.data.tolist(),
                    pixel_row=int(obj.pixel_coord.y),
                    pixel_col=int(obj.pixel_coord.x),
                    latitude=lat,
                    longitude=long,
                )
            elif isinstance(obj, PlottedMeanSpectrum):
                crd_arr = obj.coords_as_array().astype(int)
                collection_list = []
                for n in range(crd_arr.shape[0]):
                    long, lat = forward_geotransform(
                        crd_arr[n, 0], crd_arr[n, 1], self.gtrans
                    )
                    print(crd_arr[n, :])
                    single_spec = FullSpectrum(
                        wavelength=obj.wvl.tolist(),
                        spectrum=self.cube[
                            crd_arr[n, 1], crd_arr[n, 0], :
                        ].tolist(),
                        pixel_row=crd_arr[n, 1],
                        pixel_col=crd_arr[n, 0],
                        latitude=lat,
                        longitude=long,
                    )
                    collection_list.append(single_spec)
                spec_model = SpectrumCollection[FullSpectrum](
                    wavelength=obj.wvl.tolist(),
                    mean=obj.data.tolist(),
                    error=obj.data_err.tolist(),
                    component_list=collection_list,
                )
            else:
                raise ValueError("Obj is not of the correct type")

            with open(file, "w") as f:
                f.write(spec_model.model_dump_json(indent=2))
