# Standard Libraries
import sys

# Dependencies
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QMainWindow,
    QHBoxLayout,
    QApplication,
    QPushButton,
    QVBoxLayout,
)

# Top-Level Imports
from specview.image_display_canvas import ImageCanvas, ImageCanvasSettings
from specview.spectral_display_canvas import SpectralCanvas


class SpectralWindow(QWidget):
    def __init__(
        self,
        cube: np.ndarray,
        wvl: np.ndarray,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.spec_canvas = SpectralCanvas(cube, wvl, parent=self)

        button_layout = QHBoxLayout()
        btn = QPushButton("Clear Spectra")
        btn.pressed.connect(self.spec_canvas.clear_spectra)
        button_layout.addWidget(btn)

        btn = QPushButton("Save Spectra")
        btn.pressed.connect(self.spec_canvas.save_spectra)
        button_layout.addWidget(btn)

        layout = QVBoxLayout()
        layout.addWidget(self.spec_canvas)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.setWindowTitle("Spectral Window")


class MainWindow(QMainWindow):
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

        self.spec_window = SpectralWindow(cube, wvl)

        im_canvas = ImageCanvas(
            display_image,
            settings=default_settings,
            connected_spectral_canvas=self.spec_window.spec_canvas,
            parent=self,
        )

        im_canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        im_canvas.setFocus()

        layout = QVBoxLayout()
        layout.addWidget(im_canvas)

        widget = QWidget()
        widget.setLayout(layout)
        self.setWindowTitle("SpecView Version 0.2")
        self.setCentralWidget(widget)

        self.w = None

    def show_spectral_window(self):
        if not self.spec_window.isVisible():
            self.spec_window.show()


def open_specview(
    cube: np.ndarray, wvl: np.ndarray, display_image: np.ndarray
):
    app = QApplication(sys.argv)
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
