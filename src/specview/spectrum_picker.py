# Standard Libraries
from enum import Enum, auto
from typing import Any, Optional, Tuple, Sequence

# Dependencies
import numpy as np
import numpy.typing as npt
import matplotlib as mpl

# Relative Imports
from .spectrum_picker_classes import (
    SpectrumDislpay,
    ImageDisplay,
    RGBImageDisplay,
)
from .rgb_image import ThreeBandRGB


class InvalidPickerModeError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class PickerMode(Enum):
    IMAGE_ONLY = auto()
    AUTO_RGB = auto()
    SINGLE_BAND = auto()
    THREE_BAND = auto()


def classify_picker_mode(
    display_image: Optional[npt.NDArray[np.floating[Any]]],
    display_bands: Optional[int | Tuple[int, int, int]],
):
    # Case 1: display_image provided, no bands given --> Greyscale
    if display_image is not None and display_bands is None:
        return PickerMode.IMAGE_ONLY
    # Case 2: no display_image, no bands given --> Evenly spaced RGB
    if display_image is None and display_bands is None:
        return PickerMode.AUTO_RGB
    # Case 3: no display_image, single band given --> Greyscale from cube
    if display_image is None and isinstance(display_bands, int):
        return PickerMode.SINGLE_BAND
    # Case 4: no display_image, 3 bands given --> RGB from cube
    if display_image is None and isinstance(display_bands, Sequence):
        return PickerMode.THREE_BAND
    raise InvalidPickerModeError(
        f"{type(display_image)} and {type(display_bands)} is an invalid type"
        "combination for a Picker Mode."
    )


def open_spectrum_picker(
    spectral_cube: npt.NDArray[np.floating[Any]],
    wvl: npt.NDArray[np.floating[Any]],
    display_image: Optional[npt.NDArray[np.floating[Any]]] = None,
    display_bands: Optional[int | Tuple[int, int, int]] = None,
    comparison_cubes: Sequence[npt.NDArray[np.floating[Any]]] = [],
) -> None:
    """
    Opens a spectrum picker GUI using the current backend.

    Parameters
    ----------
    spectral_cube: ndarray of floats
        Main spectral data cube to examine. The cube must be in the bands-last
        configuration.
    wvl: ndarray of floats
        Wavelength values corresponding tothe spectral domain of the cube.
    display_image: ndarray of floats, optional
        An image (2D array) to display as the spectral picking surface rather
        than (a) band(s) from the spectral cube. If None (default), the picking
        surface image will be derived from the spectral cube.
    display_bands: int or tuple of 3 integers, optional
        If a single `int` is passed, that band index will be chosen as the
        display image derived from the spectral cube. If three integers are
        passed, an RGB image of those three indices derived from the cube will
        be displayed. If both `display_image` and `display_bands` are None
        (default), three equally spaced bands will be chosen from the spectral
        cube to be displayed as an RGB image.
    comparison_cubes: Sequence of ndarrays
        Spectra from these cubes will be plotted alongside the main spectral
        cube.
    """
    mpl.use("qtagg")
    picker_mode = classify_picker_mode(display_image, display_bands)
    bands: tuple[int, ...] = ()
    img = None
    rgb = None

    match picker_mode:
        case PickerMode.IMAGE_ONLY:
            img = display_image

        case PickerMode.AUTO_RGB:
            idx = np.linspace(wvl.min(), wvl.max(), 3, dtype=int)
            rgb = ThreeBandRGB(spectral_cube, *idx)
            bands = tuple(idx)

        case PickerMode.SINGLE_BAND:
            img = spectral_cube[..., display_bands]

        case PickerMode.THREE_BAND:
            assert isinstance(display_bands, Sequence)
            rgb = ThreeBandRGB(spectral_cube, *display_bands)
            bands = tuple(display_bands)

    specdisplay = SpectrumDislpay(wvl, spectral_cube, comparison_cubes)

    if rgb is not None:
        RGBImageDisplay(rgb, specdisplay, bands)
    elif img is not None:
        ImageDisplay(img, specdisplay)
