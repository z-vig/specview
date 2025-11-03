# Standard Libraries
from dataclasses import dataclass, field

# Dependencies
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

# Top-Level Imports
from specview.utils import SquareOffset


@dataclass
class PixelCoordinate:
    x: float | int
    y: float | int

    def __post_init__(self):
        self.x = int(round(self.x, 0))
        self.y = int(round(self.y, 0))

    @classmethod
    def zero_point(cls):
        return cls(0, 0)

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
class CrosshairData:
    is_active: bool = False
    is_init: bool = False
    coord: PixelCoordinate = field(default_factory=PixelCoordinate.zero_point)
    xline: Line2D = Line2D([], [])
    yline: Line2D = Line2D([], [])
    scan_spectrum: Line2D = Line2D([], [])

    def add_to_axes(
        self,
        img_axis: Axes,
        spec_axis: Axes,
        wvl: np.ndarray,
        cube: np.ndarray,
    ):
        print(img_axis.dataLim)
        self.xline.set_xdata([img_axis.dataLim.xmin, img_axis.dataLim.xmax])
        self.xline.set_ydata([0, 0])

        self.yline.set_xdata([0, 0])
        self.yline.set_ydata([img_axis.dataLim.ymin, img_axis.dataLim.ymax])

        self.xline.set_color("red")
        self.yline.set_color("red")

        self.scan_spectrum.set_xdata(wvl)
        self.scan_spectrum.set_ydata(self.coord.pull_data(cube))
        self.scan_spectrum.set_alpha(0.5)
        self.scan_spectrum.set_color("k")

        img_axis.add_line(self.xline)
        img_axis.add_line(self.yline)
        spec_axis.add_line(self.scan_spectrum)

        self.is_init = True


@dataclass
class LassoData:
    is_active: bool = False
    pixel_coords: np.ndarray = field(default_factory=lambda: np.empty(0))

    def set_pixel_coords(self, display_image: np.ndarray):
        if display_image.ndim > 2:
            h, w, _ = display_image.shape
        else:
            h, w = display_image.shape
        y, x = np.mgrid[0:h, 0:w]
        self.pixel_coords = np.column_stack((x.ravel(), y.ravel()))


@dataclass
class ImageState:
    img_offsets: SquareOffset = field(default_factory=SquareOffset)
    panning: PanningData = field(default_factory=PanningData)
    crosshair: CrosshairData = field(default_factory=CrosshairData)
    lasso: LassoData = field(default_factory=LassoData)
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
    errorbar_caps: tuple[Line2D, Line2D]
    errorbar_lines: Line2D
    pixel_coords: list[PixelCoordinate]
    total: int

    def coords_as_array(self):
        coord_array = np.empty((self.total, 2))
        for n, crd in enumerate(self.pixel_coords):
            coord_array[n, :] = crd.as_tuple()
        return coord_array


@dataclass
class SpectralState:
    nspectra: int = 0
    current_spectrum: PlottedSpectrum = field(
        default_factory=PlottedSpectrum.null
    )
    spectral_catalog: list[PlottedSpectrum] = field(default_factory=list)
