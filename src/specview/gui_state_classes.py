# Standard Libraries
from dataclasses import dataclass, field

# Dependencies
import numpy as np

# Top-Level Imports
from specview.utils import SquareOffset


# ===================
# Image Display Canvas state classes
# ===================
@dataclass
class PanningData:
    is_active: bool = False
    start_coord: tuple = (0, 0)


@dataclass
class ImageState:
    img_offsets: SquareOffset = field(default_factory=SquareOffset)
    panning: PanningData = field(default_factory=PanningData)
    spectral_viewer_open: bool = False


# ===================
# Spectral Display Canvas state classes
# ===================
@dataclass
class SpectralState:
    current_spectrum: np.ndarray
