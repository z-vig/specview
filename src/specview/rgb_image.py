# Standard Libraries
from dataclasses import dataclass, field
from typing import TypedDict, Optional, Literal

# Dependencies
import numpy as np
import matplotlib.colors as mcolors

# Relative
from .utils import extrema


class RGBNorms(TypedDict):
    r: mcolors.Normalize
    g: mcolors.Normalize
    b: mcolors.Normalize


class RGBCache(TypedDict):
    r: Optional[np.ndarray]
    g: Optional[np.ndarray]
    b: Optional[np.ndarray]
    rgb: Optional[np.ndarray]


@dataclass
class ThreeBandRGB:
    # User-set parameters
    data_cube: np.ndarray
    _ridx: int
    _gidx: int
    _bidx: int

    # Initialized post-init
    _rlow: float = field(init=False)
    _rhigh: float = field(init=False)
    _glow: float = field(init=False)
    _ghigh: float = field(init=False)
    _blow: float = field(init=False)
    _bhigh: float = field(init=False)
    _norms: RGBNorms = field(init=False)
    _cache: RGBCache = field(init=False)

    def __post_init__(self) -> None:
        self._rlow, self._rhigh = extrema(self.data_cube[..., self._ridx])
        self._glow, self._ghigh = extrema(self.data_cube[..., self._gidx])
        self._blow, self._bhigh = extrema(self.data_cube[..., self._bidx])
        self._norms: RGBNorms = {
            "r": mcolors.Normalize(self.rlow, self.rhigh, clip=True),
            "g": mcolors.Normalize(self._glow, self._ghigh, clip=True),
            "b": mcolors.Normalize(self._blow, self._bhigh, clip=True),
        }
        self._cache = {"r": None, "g": None, "b": None, "rgb": None}

    # Color boundary properties
    @property
    def rlow(self) -> float:
        return self._rlow

    @rlow.setter
    def rlow(self, value: float) -> None:
        self._rlow = value
        self._update_norm("r", self._rlow, self._rhigh)

    @property
    def rhigh(self) -> float:
        return self._rhigh

    @rhigh.setter
    def rhigh(self, value: float) -> None:
        self._rhigh = value
        self._update_norm("r", self._rlow, self._rhigh)

    @property
    def glow(self) -> float:
        return self._glow

    @glow.setter
    def glow(self, value: float) -> None:
        self._glow = value
        self._update_norm("g", self._glow, self._ghigh)

    @property
    def ghigh(self) -> float:
        return self._ghigh

    @ghigh.setter
    def ghigh(self, value: float) -> None:
        self._ghigh = value
        self._update_norm("g", self._glow, self._ghigh)

    @property
    def blow(self) -> float:
        return self._glow

    @blow.setter
    def blow(self, value: float) -> None:
        self._blow = value
        self._update_norm("b", self._blow, self._bhigh)

    @property
    def bhigh(self) -> float:
        return self._ghigh

    @bhigh.setter
    def bhigh(self, value: float) -> None:
        self._bhigh = value
        self._update_norm("b", self._blow, self._bhigh)

    # Band index properties
    @property
    def ridx(self) -> int:
        return self._ridx

    @ridx.setter
    def ridx(self, value) -> None:
        self._ridx = value
        self._update_band("r", self._ridx)

    @property
    def gidx(self) -> int:
        return self._gidx

    @gidx.setter
    def gidx(self, value) -> None:
        self._gidx = value
        self._update_band("g", self._gidx)

    @property
    def bidx(self) -> int:
        return self._bidx

    @bidx.setter
    def bidx(self, value) -> None:
        self._bidx = value
        self._update_band("b", self._bidx)

    # Internal Helpers
    def _update_band(
        self, color: Literal["r", "g", "b"], bandidx: int
    ) -> None:
        low, high = extrema(self.data_cube[..., bandidx])
        if color == "r":
            self._rlow, self._rhigh = low, high
        elif color == "g":
            self._glow, self.ghigh = low, high
        elif color == "b":
            self.blow, self.bhigh = low, high

        self._update_norm(color, low, high)

    def _update_norm(
        self, color: Literal["r", "g", "b"], low: float, high: float
    ):
        self._norms[color] = mcolors.Normalize(low, high, clip=True)
        self._cache[color] = None
        self._cache["rgb"] = None

    # Exposed properties
    @property
    def norms(self) -> RGBNorms:
        return self._norms

    @property
    def red(self) -> np.ndarray:
        if self._cache["r"] is None:
            self._cache["r"] = self.data_cube[:, :, self.ridx]
            assert isinstance(self._cache["r"], np.ndarray)
        return self._cache["r"]

    @property
    def green(self) -> np.ndarray:
        if self._cache["g"] is None:
            self._cache["g"] = self.data_cube[:, :, self.gidx]
            assert isinstance(self._cache["g"], np.ndarray)
        return self._cache["g"]

    @property
    def blue(self) -> np.ndarray:
        if self._cache["b"] is None:
            self._cache["b"] = self.data_cube[:, :, self.bidx]
            assert isinstance(self._cache["b"], np.ndarray)
        return self._cache["b"]

    @property
    def scaled_rgb(self) -> np.ndarray:
        if self._cache["rgb"] is None:
            scaled_r = self.norms["r"](self.red, clip=True)
            scaled_g = self.norms["g"](self.green, clip=True)
            scaled_b = self.norms["b"](self.blue, clip=True)
            self._cache["rgb"] = np.stack(
                [scaled_r, scaled_g, scaled_b], axis=2, dtype=np.float32
            )
            assert isinstance(self._cache["rgb"], np.ndarray)

        return self._cache["rgb"]
