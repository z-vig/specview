# Standard Libraries
# import sys
# from typing import Optional

# Dependencies
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QLabel,
    QPushButton,
)

# Top-level Imports
from specview.gui_state_classes import PlottedSpectrum


class SpectrumEditWindow(QWidget):
    name_changed = pyqtSignal()
    spectrum_deleted = pyqtSignal()
    closed = pyqtSignal()

    def __init__(self, edit_spec: PlottedSpectrum, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.edit_spec = edit_spec
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Spectrum Name:"))

        self.line_edit_widget = QLineEdit()
        self.line_edit_widget.setMaxLength(80)
        self.line_edit_widget.setPlaceholderText(f"{self.edit_spec.name}")
        self.line_edit_widget.returnPressed.connect(self.set_spectrum_name)

        layout.addWidget(self.line_edit_widget)

        delete_button = QPushButton("Delete Spectrum")
        delete_button.pressed.connect(self.delete_spectrum)
        layout.addWidget(delete_button)

        self.setLayout(layout)
        self.setWindowTitle("Spectrum Editor")

    def set_spectrum_name(self):
        new_name = self.line_edit_widget.text()
        self.edit_spec.name = new_name
        self.name_changed.emit()

    def delete_spectrum(self):
        self.spectrum_deleted.emit()

    def closeEvent(self, a0):
        self.closed.emit()
        if a0 is not None:
            a0.accept()
