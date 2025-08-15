# Standard Libraries
from typing import Optional, Tuple
from dataclasses import dataclass

# Dependencies
from matplotlib.axes import Axes


@dataclass
class PanningState:
    is_panning: bool = False
    pan_start: Optional[Tuple[float, float]] = None
    pan_ax: Optional[Axes] = None
