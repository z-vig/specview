# Built-ins
# from pathlib import Path

# Dependencies
import arguably

# import spectralio as sio
# import numpy as np
# import rasterio as rio  # type: ignore

# PyQt6 Imports
from PyQt6.QtWidgets import QApplication

# Local Imports
from .spectral_viewing_window import SpecViewWindow


@arguably.command
def hypview():
    app = QApplication([])
    # root = Path("D:/moon_data/m3/Gruithuisen_Region/Gruithuisen_Mosaics/")
    # fp = Path(root, "global_mode/M3G_GDOMES_RFL.bsq")
    # with rio.open(fp) as f:
    #     cube = f.read()
    # wvl = sio.read_wvl("D:/moon_data/m3/M3G.wvl").asarray()
    # cube = np.transpose(cube, (1, 2, 0))
    # cube[cube == -999] = np.nan
    main = SpecViewWindow(
        image_data="D:/moon_data/m3/Gruithuisen_Region/Gruithuisen_Mosaics/global_mode/M3G_GDOMES_RFL.geospcub",  # noqa
        cube_data="D:/moon_data/m3/Gruithuisen_Region/Gruithuisen_Mosaics/global_mode/M3G_GDOMES_RFL.geospcub",  # noqa
        wvl="D:/moon_data/m3/M3G.wvl",
    )
    main.show()
    app.exec()


def main():
    arguably.run()
