# Dependencies
import arguably

# PyQt6 Imports
from PyQt6.QtWidgets import QApplication

# Local Imports
from .spectral_viewing_window import CubeViewWindow


@arguably.command
def cubeview():
    app = QApplication([])
    main = CubeViewWindow()
    main.show()
    app.exec()


def main():
    arguably.run()
