# Standard Libraries
from typing import TypeVar, Generic

# Dependencies
from pydantic import BaseModel


T = TypeVar("T", bound="BaseSpectrum")


class BaseSpectrum(BaseModel):
    name: str
    pixel_row: int
    pixel_col: int
    latitude: float
    longitude: float


class FullSpectrum(BaseSpectrum):
    wavelength: list[float]
    spectrum: list[float]


class SpectrumCollection(BaseModel, Generic[T]):
    wavelength: list[float]
    mean: list[float]
    error: list[float]
    component_list: list[T]
