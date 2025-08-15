# Standard Libraries
from typing import Optional, Tuple
from dataclasses import dataclass, field

# Dependencies
from matplotlib.axes import Axes

# Relative Imports
from .spectrum_plot_objects import PlottedSpectrum


@dataclass
class PanningState:
    is_panning: bool = False
    pan_start: Optional[Tuple[float, float]] = None
    pan_ax: Optional[Axes] = None


@dataclass
class ImagePlotState:
    panning: PanningState = field(default_factory=PanningState)
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
