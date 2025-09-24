# Standard Libraries
import sys

# Dependencies
import numpy as np
import PyQt6.QtWidgets as qtw

# Top-Level Imports
from specview.image_display_canvas import ImageCanvas, ImageCanvasSettings
from specview.spectral_display_canvas import SpectralCanvas


class SpectralWindow(qtw.QMainWindow):
    def __init__(
        self, cube: np.ndarray, wvl: np.ndarray, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        spec_canvas = SpectralCanvas(parent=self)

        layout = qtw.QVBoxLayout()
        layout.addWidget(spec_canvas)

        widget = qtw.QWidget()
        widget.setLayout(layout)
        self.setWindowTitle("Spectral Window")
        self.setCentralWidget(widget)


class MainWindow(qtw.QMainWindow):
    def __init__(
        self,
        cube: np.ndarray,
        wvl: np.ndarray,
        display_image: np.ndarray,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cube = cube
        self.wvl = wvl

        default_settings = ImageCanvasSettings(zoom_speed=1.3)

        im_canvas = ImageCanvas(
            display_image, settings=default_settings, parent=self
        )

        layout = qtw.QVBoxLayout()
        layout.addWidget(im_canvas)

        widget = qtw.QWidget()
        widget.setLayout(layout)
        self.setWindowTitle("SpecView Version 0.2")
        self.setCentralWidget(widget)

        self.w = None

    def open_spectral_window(self):
        if self.w is None:
            self.w = SpectralWindow(self.cube, self.wvl)
        self.w.show()


def open_specview(
    cube: np.ndarray, wvl: np.ndarray, display_image: np.ndarray
):
    app = qtw.QApplication(sys.argv)
    w = MainWindow(cube, wvl, display_image)
    w.show()
    app.exec()


def main():
    import h5py as h5  # type: ignore
    from pathlib import Path

    print("Launching Specview...")

    root = Path("D:/moon_data/m3/Gruithuisen_Region/M3G20090208T175211/")
    p = Path(root, "pipeline_cache_175211.hdf5")
    with h5.File(p) as f:
        g = f["thermal_correction"]
        cube: np.ndarray = g["data"][...]  # type: ignore
        wvl: np.ndarray = g.attrs["wavelengths"][...]  # type: ignore

    open_specview(cube, wvl, cube[:, :, 10])
