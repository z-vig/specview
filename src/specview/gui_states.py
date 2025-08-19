# Standard Libraries
from typing import Optional, Tuple
from dataclasses import dataclass, field

# Dependencies
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np

# Relative Imports
from .spectrum_plot_objects import PlottedSpectrum


class NoAxisError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass
class PanningState:
    is_panning: bool = False
    pan_start: Optional[Tuple[float, float]] = None
    pan_ax: Optional[Axes] = None


@dataclass
class CrosshairState:
    is_scanning: bool = False
    crosshair_ax: Optional[Axes] = None
    scanning_ax: Optional[Axes] = None
    wvl: Optional[np.ndarray] = None
    xcrosshair: Line2D = field(init=False)
    ycrosshair: Line2D = field(init=False)
    data_line: Line2D = field(init=False)

    def init_crosshairs(
        self, crosshair_axis: Axes, scanning_axis: Axes, wvl: np.ndarray
    ):
        self.crosshair_ax = crosshair_axis
        self.scanning_ax = scanning_axis
        self.wvl = wvl
        self.xcrosshair = self.crosshair_ax.axvline(0, color="red")
        self.ycrosshair = self.crosshair_ax.axhline(0, color="red")
        (self.data_line,) = self.scanning_ax.plot(
            self.wvl, np.ones(self.wvl.shape)
        )


@dataclass
class BandIndicatorState:
    color: str = ""
    line_obj: Line2D = field(init=False)
    pressed: bool = False

    def init_indicator(
        self,
        color: str,
        scanning_axis: Axes,
        xval: int,
    ):
        self.color = color
        self.line_obj = scanning_axis.axvline(xval, color=self.color)


@dataclass
class ImagePlotState:
    panning: PanningState = field(default_factory=PanningState)
    crosshair: CrosshairState = field(default_factory=CrosshairState)
    r_indicator: BandIndicatorState = field(default_factory=BandIndicatorState)
    g_indicator: BandIndicatorState = field(default_factory=BandIndicatorState)
    b_indicator: BandIndicatorState = field(default_factory=BandIndicatorState)
    collect_spectra: bool = False
    collect_area: bool = False


@dataclass
class SpectrumPlotState:
    currently_plotted: PlottedSpectrum = field(
        default_factory=lambda: PlottedSpectrum.null()
    )
    cached_plots: list[PlottedSpectrum] = field(default_factory=list)
    plot_activate: bool = False
    single_spec_count: int = 0
    mean_spec_count: int = 0
