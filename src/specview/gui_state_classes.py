# Standard Libraries
from dataclasses import dataclass, field

# Dependencies
import numpy as np
from matplotlib.lines import Line2D

# Top-Level Imports
from specview.utils import SquareOffset


@dataclass
class PixelCoordinate:
    x: float | int
    y: float | int

    def __post_init__(self):
        self.x = int(round(self.x, 0))
        self.y = int(round(self.y, 0))

    def pull_data(self, cube: np.ndarray):
        if not isinstance(self.x, int) or not isinstance(self.y, int):
            raise ValueError("Pixel Coordinates are not integers.")
        return cube[self.y, self.x, :]

    def as_tuple(self) -> tuple[int, int]:
        if not isinstance(self.x, int) or not isinstance(self.y, int):
            raise ValueError("Pixel Coordinates are not integers.")
        return (self.x, self.y)


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
class PlottedSpectrum:
    name: str
    plot_obj: Line2D
    wvl: np.ndarray
    data: np.ndarray

    @classmethod
    def null(cls):
        return cls("NULL", Line2D([], []), np.empty(0), np.empty(0))


@dataclass
class PlottedSingleSpectrum(PlottedSpectrum):
    pixel_coord: PixelCoordinate


@dataclass
class PlottedMeanSpectrum(PlottedSpectrum):
    data_err: np.ndarray
    pixel_coords: np.ndarray
    total: int


@dataclass
class SpectralState:
    nspectra: int = 0
    current_spectrum: PlottedSpectrum = field(
        default_factory=PlottedSpectrum.null
    )
    spectral_catalog: list[PlottedSpectrum] = field(default_factory=list)
