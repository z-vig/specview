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
from specview.rgb_display_canvas import RGBCanvas, RGBCanvasSettings
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

        self.spec_window = SpectralWindow(cube, wvl)

        if display_image.ndim == 2:
            default_settings = ImageCanvasSettings(zoom_speed=1.3)
            im_canvas = ImageCanvas(
                display_image,
                settings=default_settings,
                connected_spectral_canvas=self.spec_window.spec_canvas,
                parent=self,
            )
        elif display_image.ndim == 3:
            default_settings = RGBCanvasSettings(zoom_speed=1.3)
            if display_image.shape[2] != 3:
                raise ValueError("Display image has an invalid 3rd dimension.")
            im_canvas = RGBCanvas(
                display_image,
                settings=default_settings,
                connected_spectral_canvas=self.spec_window.spec_canvas,
                parent=self,
            )

        else:
            raise ValueError(
                f"Invalid display image ({display_image.ndim} dimensions)."
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
    """
    Opens a SpecView GUI for viewing spectral data.

    Parameters
    ----------
    cube: np.ndarray
        3D Spectral image cube.
    wvl: np.ndarray
        Wavelength array corresponding to the cube.
    display_image: np.ndarray
        2D array that will be shown as the front image of the spectral cube.

    GUI Controls
    ------------
    - Image zoom // scroll wheel
    - Image panning // middle mouse button
    - Spectrum selection // left click
    - Toggle browse mode // press c
    """
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
