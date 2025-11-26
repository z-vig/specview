# Standard Libraries
import sys
from typing import Optional

# Dependencies
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QWidget,
    QMainWindow,
    QApplication,
    QVBoxLayout,
    QToolBar,
    QStatusBar,
)
from rasterio.crs import CRS  # type: ignore
import spectralio as sio

# Top-Level Imports
from specview.image_display_canvas import ImageCanvas, ImageCanvasSettings
from specview.rgb_display_canvas import RGBCanvas, RGBCanvasSettings
from specview.spectral_display_canvas import SpectralCanvas


class SpectralWindow(QMainWindow):
    def __init__(
        self,
        cube: np.ndarray,
        wvl: np.ndarray,
        crs: str,
        geotrans: tuple[float, float, float, float, float, float],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.spec_canvas = SpectralCanvas(
            cube, wvl, parent=self, crs=crs, gtrans=geotrans
        )

        layout = QVBoxLayout()
        layout.addWidget(self.spec_canvas)

        toolbar = QToolBar("Main Toolbar")

        self.addToolBar(toolbar)
        widget = QWidget()
        widget.setLayout(layout)

        clear_action = QAction("Clear", self)
        clear_action.setStatusTip("Clear all currently viewed spectra.")
        clear_action.triggered.connect(self.spec_canvas.clear_spectra)
        toolbar.addAction(clear_action)

        save_action = QAction("Save", self)
        save_action.setStatusTip("Save the currently viewed spectra.")
        save_action.triggered.connect(self.spec_canvas.save_spectra)
        toolbar.addAction(save_action)

        self.setStatusBar(QStatusBar(self))

        self.setWindowTitle("Spectral Window")
        self.setCentralWidget(widget)


class MainWindow(QMainWindow):
    def __init__(
        self,
        cube: np.ndarray,
        wvl: np.ndarray,
        display_image: np.ndarray,
        crs: str,
        geotrans: tuple[float, float, float, float, float, float],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cube = cube
        self.wvl = wvl

        self.spec_window = SpectralWindow(cube, wvl, crs, geotrans)

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
    cube: np.ndarray,
    wvl: np.ndarray,
    display_image: np.ndarray,
    geodata: Optional[str],
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
    if geodata is None:
        crs = CRS.from_authority("ESRI", "104903").to_wkt()
        assert isinstance(crs, str)
        geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    else:
        geodat = sio.read_geodata(geodata_fp=geodata)
        crs = geodat.crs
        geotransform = geodat.geotransform.togdal()

    app = QApplication(sys.argv)
    w = MainWindow(cube, wvl, display_image, crs, geotransform)
    w.show()
    app.exec()
