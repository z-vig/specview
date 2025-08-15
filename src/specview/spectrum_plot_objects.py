# Standard Libraries
from dataclasses import dataclass

# Dependencies
import numpy as np
from matplotlib.lines import Line2D


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
class PlottedSingleSectrum(PlottedSpectrum):
    pixel_coord: tuple[int, int]


@dataclass
class PlottedMeanSpectrum(PlottedSpectrum):
    data_err: np.ndarray
    pixel_coords: np.ndarray
    total: int
