# Standard Libraries
# from typing import Any, TypedDict, Optional

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import PyQt6.QtWidgets as qtw

# Top-Level Imports
from specview.gui_state_classes import SpectralState


class SpectralCanvas(FigureCanvasQTAgg):
    def __init__(
        self,
        init_spectrum: np.ndarray,
        wvl: np.ndarray,
        parent: qtw.QMainWindow,
    ):
        fig, self.spec_axis = plt.subplots(1, 1)
        super().__init__(fig)

        self.wvl = wvl
        self.state = SpectralState(current_spectrum=init_spectrum)

        # ========= Plotting spectrum using plt.plot ==========
        self.spec_axis.plot(self.wvl, self.state.current_spectrum)
        self.spec_axis.set_xlabel("Wavelength (nm)")
        self.spec_axis.set_ylabel("Reflectance")
