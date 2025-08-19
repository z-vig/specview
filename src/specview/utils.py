# Standard Libraries
from typing import Tuple

# Dependencies
from matplotlib.axes import Axes
import numpy as np


def set_image_axis(ax: Axes) -> None:
    """
    Sets a matplotlib axis up to display an image.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )


def extrema(arr: np.ndarray) -> Tuple[float, float]:
    finite_arr = arr[np.isfinite(arr)]
    min = float(np.min(finite_arr))
    max = float(np.max(finite_arr))
    return (min, max)


def percentiles(
    arr: np.ndarray, low: float = 5, high: float = 95
) -> Tuple[float, float]:
    finite_arr = arr[np.isfinite(arr)]
    low_pct = float(np.percentile(finite_arr, low))
    high_pct = float(np.percentile(finite_arr, high))
    return (low_pct, high_pct)


def find_wvl(wvls: np.ndarray, targetwvl: float):
    """
        findλ(λ.targetλ)

    Given a list of wavelengths, `wvls`, find the index of a `targetwvl` and
    the actual wavelength closest to your target.

    Parameters
    ----------
    wvls: np.ndarray
        Wavelength array to search in.
    targetwvl:
        Wavelength to search for.

    Returns
    -------
    idx: int
        Index of the found wavelength.
    wvl: float
        Actual wavelength that is closest to the target wavelength (at idx).
    """

    idx = np.argmin(np.abs(wvls - targetwvl))
    return idx, wvls[idx]
