# Standard Libraries
from typing import NamedTuple, Tuple

# Dependencies
from matplotlib.axes import Axes
import numpy as np


class SquareOffset(NamedTuple):
    x_offset: int = 0
    y_offset: int = 0


def square_image(img: np.ndarray) -> tuple[SquareOffset, np.ndarray]:
    """
    Pads a rectangular image to make it square, for the purpose of reshaping
    an image axis to be better suited for GUI applications.

    Parameters
    ----------
    img: np.ndarray
        Image array.

    Returns
    -------
    offset: SquareOffset
        NamedTuple of offset values added due to padding. The 2 attributes are
        `x_offset` and `y_offset` representing the horizontal and vertical
        offsets respectively. One will always be zero.
    square_img: np.ndarray
        Image with np.nan values padding the sides to fit the square
    """
    long_axis = np.argmax(img.shape)
    short_axis = np.argmin(img.shape)
    padding_length = (img.shape[long_axis] - img.shape[short_axis]) // 2
    square_padding = np.full(
        (img.shape[long_axis], padding_length),
        np.nan,
    )
    square_img = np.concat(
        [square_padding, img, square_padding], axis=short_axis
    )

    if long_axis < short_axis:
        offset = SquareOffset(-padding_length, 0)
    else:
        offset = SquareOffset(0, -padding_length)

    return offset, square_img


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
