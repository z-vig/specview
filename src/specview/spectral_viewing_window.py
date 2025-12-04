from typing import Optional
from PyQt6 import QtWidgets as qtw
from PyQt6 import QtGui
from PyQt6 import QtCore
from pyqtgraph.dockarea import Dock, DockArea  # type: ignore
import rasterio as rio  # type: ignore
from pathlib import Path
import numpy as np
import pyqtgraph as pg  # type: ignore

from .base_window import BaseWindow
from .image_display_widget import ImagePickerWidget
from .spectral_display_widget import SpectralDisplayWidget


class SpecViewWindow(BaseWindow):
    def __init__(
        self,
        wvl: Optional[np.ndarray],
        image_data: Optional[np.ndarray] = None,
        cube_data: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.polygon_cache: dict[str, qtw.QGraphicsPolygonItem | None] = {}

        dock_area = DockArea()
        self.setCentralWidget(dock_area)

        # ---- Connecting menu actions to slots ----
        self.open_display.triggered.connect(self.open_image)
        self.clear_spectra.triggered.connect(self.empty_cache)

        # ---- Image Dock ----
        self.imview_dock = Dock(name="Image Window")
        dock_area.addDock(self.imview_dock, "top")

        self.img_picker = ImagePickerWidget()
        self.imview_dock.addWidget(self.img_picker)

        # ---- Spectral Display Dock ----
        self.spec_dock = Dock(name="Spectral Display Window")
        dock_area.addDock(self.spec_dock, "bottom")

        self.spectral_display = SpectralDisplayWidget()
        self.spec_dock.addWidget(self.spectral_display)

        # ---- Linking Image and Spectral Display ----
        def intercept_pixel_coord(y: int, x: int):
            spectrum_name = self.spectral_display.add_spectrum((y, x))
            self.polygon_cache[spectrum_name] = None

        def cache_spectrum(
            plot: pg.PlotDataItem, err: pg.ErrorBarItem
        ) -> None:
            name = plot.name()
            if name is None:
                return
            self.state.spectrum_cache[name] = (plot, err)
            n_cached = len(self.state.spectrum_cache)
            self.clear_spectra.setStatusTip(
                f"Clear {n_cached} spectra from memory."
            )

        def update_cache(
            plot: pg.PlotDataItem,
            err: pg.ErrorBarItem,
            old_name: str,
            new_name: str,
        ):
            del self.state.spectrum_cache[old_name]
            self.state.spectrum_cache[new_name] = (plot, err)
            self.polygon_cache[new_name] = self.polygon_cache.pop(old_name)

        def intercept_roi(in_coords: np.ndarray, verts: np.ndarray):
            group_name = self.spectral_display.add_group(in_coords)
            pts = [QtCore.QPointF(*verts[n, :]) for n in range(verts.shape[0])]
            poly = QtGui.QPolygonF(pts)
            poly_item = qtw.QGraphicsPolygonItem(poly)
            poly_item.setPen(pg.mkPen("k", width=2))
            poly_item.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 30)))
            self.polygon_cache[group_name] = poly_item
            self.img_picker.imview.getView().addItem(poly_item)

        def remove_spectrum(name):
            poly = self.polygon_cache[name]
            if poly is not None:
                self.img_picker.imview.getView().removeItem(poly)

            del self.polygon_cache[name]
            del self.state.spectrum_cache[name]

            self.clear_spectra.setStatusTip(
                f"Clear {len(self.state.spectrum_cache)} spectra from memory."
            )

        # Image Picker Connections
        self.img_picker.pixel_picked.connect(intercept_pixel_coord)
        self.img_picker.lasso_finished.connect(intercept_roi)

        # Spectral Display Connections
        self.spectral_display.data_added.connect(cache_spectrum)
        self.spectral_display.data_updated.connect(update_cache)
        self.spectral_display.data_removed.connect(remove_spectrum)

        # Status Bar Connections
        def cursor_message(x: float, y: float, val: float) -> None:
            if x == -999 and y == -999 and val == -999:
                self.status_bar.clearMessage()
                return
            self.status_bar.showMessage(
                f"x={x:.1f}   y={y:.1f}   value={val:.5f}"
            )

        self.img_picker.mouse_moved.connect(cursor_message)

        if image_data is not None:
            self.set_window_size(image_data)
            self.set_image(image_data)
        else:
            self.resize(600, 600)

        if cube_data is not None and wvl is not None:
            self.set_cube(wvl, cube_data)

    def open_image(self, fp: str | Path, bbl: list[bool] | None = None):
        with rio.open(fp, "r") as f:
            arr: np.ndarray = f.read()
        arr[arr == -999] = np.nan
        if arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))
            if bbl is not None:
                arr = arr[:, :, np.asarray(bbl, dtype=bool)]
        self.set_image(arr)

    def set_image(self, data: np.ndarray) -> None:
        self.img_picker.set_image(data)
        self.set_window_size(data)

    def set_cube(self, wvl: np.ndarray, data: np.ndarray) -> None:
        self.spectral_display.set_cube(wvl, data)

    def empty_cache(self) -> None:
        for plot, err in self.state.spectrum_cache.values():
            self.spectral_display.spec_plot.removeItem(plot)
            self.spectral_display.spec_plot.removeItem(err)
            self.spectral_display.spec_legend.removeItem(plot)
        for poly in self.polygon_cache.values():
            if poly is None:
                continue
            self.img_picker.imview.getView().removeItem(poly)
        self.spectral_display.plot_reset.emit()
