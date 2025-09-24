# Dependencies
import numpy as np
from PyQt6.QtCore import Qt
import PyQt6.QtWidgets as qtw
from matplotlib.backends.backend_qt import NavigationToolbar2QT


class MainWindow(qtw.QMainWindow):
    def __init__(self, cube: np.ndarray, wvl: np.ndarray, display_image: np.ndarray):
        layout = qtw.QVBoxLayout()

        widget = qtw.QWidget()
        widget.setLayout(layout)
        self.setWindowTitle("SpecView Version 0.2")
        self.setCentralWidget(widget)

def open_specview():