"""
File Opening Utilities

This module provides a centralized, type-safe dispatch for opening three
categories of files for the operation ofthe SpecView GUI.

Supported File Types
--------------------


Available Helper Functions::

    from pathlib import Path

    wvl = open_wvl(Path("wvlvals.wvl"))
    img = open_cube(Path("cube.geospcub"))

"""

# Built-Ins
from pathlib import Path
from typing import Protocol

# Dependencies
import numpy as np
import spectralio as sio
import re
import pandas as pd


class WvlHandler(Protocol):
    """
    Protocol for handling wavelength (or other context data) files.
    """

    def __call__(self, path: Path) -> np.ndarray: ...

    """
    Handle a file at the given path.

    Parameters
    ----------
    path: Path
        Path to an existing file whose suffix is one of the following

        - .wvl
        - .hdr
        - .txt
        - .csv

    Returns
    -------
    wvl_array: np.ndarray
        A 1D numpy array that holds the wavelength values.

    Raises
    ------
    OSError
        If the file cannot be read.
    """


def open_wvl_file(path: Path) -> np.ndarray:
    """Reads .wvl files"""
    wvl = sio.read_wvl(path)
    return wvl.asarray()


def open_hdr_file(path: Path) -> np.ndarray:
    """Read an ENVI .hdr file"""
    wvl_pattern = re.compile(r"wavelength\s=\s\{([^}]*)\}")
    with open(path, "r") as f:
        file_contents = f.read()
    result = re.findall(wvl_pattern, file_contents)
    if len(result) == 0:
        raise OSError("Unable to open .hdr file. Is there a wavelength field?")
    vals = np.asarray([float(i) for i in result[0].split(",")])
    return vals


def open_txt_file(path: Path) -> np.ndarray:
    """
    Read a .txt file. Wavelength values should be seperated by commas.
    """
    with open(path, "r") as f:
        contents = f.read()
    vals = contents.split(",")
    if vals[-1] == " ":
        vals = vals[:-1]
    return np.asarray([float(i) for i in vals])


def open_csv_file(path: Path) -> np.ndarray:
    """
    Opens a csv file where there is one row of headers and at least one is
    "wavelength". Make sure there are no spaces around the commas!
    """
    df = pd.read_csv(path)
    wvl_col_idx = [i.lower() for i in df.columns.to_list()].index("wavelength")
    wvl = df.iloc[:, wvl_col_idx].to_numpy()
    return wvl


HANDLERS: dict[str, WvlHandler] = {
    """
    Mapping from lowercase string file extensions to handler functions.
    """
    ".wvl": open_wvl_file,
    ".hdr": open_hdr_file,
    ".txt": open_txt_file,
    ".csv": open_csv_file,
}


def open_wvl(path: str | Path) -> np.ndarray:
    """
    Open a file that stores wavelength information.

    This function inspects the files extension name and passes it to the
    appropriate handler for that file type.

    Parameters
    ----------
    path: str or Path
        Path to file containing wavelength data that is to be opened.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file does not have a valid extension.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    handler = HANDLERS.get(suffix)

    if handler is None:
        raise ValueError(f"Unsupported file type: {suffix}")

    return handler(path)
