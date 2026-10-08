"""Image loading, channel selection and normalisation.

Georeferenced inputs (xarray objects, GeoTIFF and NetCDF files) are converted by
:mod:`shades2shapes.geo`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from skimage import io as skio
from skimage.color import rgb2gray

from .geo import from_xarray, is_geo_file, is_xarray, open_geo

ArrayLike = Union[str, Path, np.ndarray, "xarray.DataArray", "xarray.Dataset"]

CHANNELS = {"r": 0, "red": 0, "g": 1, "green": 1, "b": 2, "blue": 2}


def load_image(source: ArrayLike) -> np.ndarray:
    """Load an image as a float array in [0, 1].

    Parameters
    ----------
    source : str, pathlib.Path or numpy.ndarray
        Image file or array. An alpha channel is dropped; 8- and 16-bit integer
        values are scaled to [0, 1].

    Returns
    -------
    numpy.ndarray
        Array of shape (H, W) or (H, W, 3).
    """
    if isinstance(source, (str, Path)):
        arr = skio.imread(str(source))
    else:
        arr = np.asarray(source)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape {arr.shape}")
    return _to_float(arr)


def _to_float(arr):
    arr = np.asarray(arr, dtype=float)
    n_bad = int((~np.isfinite(arr)).sum())
    if n_bad:
        raise ValueError(f"The image has {n_bad} no-data (NaN or infinite) values, which are not "
                         "supported yet: fill them (e.g. da.fillna(...)) or crop the image to a "
                         "valid area.")
    if arr.max() > 1.0:
        arr = arr / (65535.0 if arr.max() > 255 else 255.0)
    return arr


def select_channel(img: np.ndarray, channel: str = "auto",
                   names=None) -> tuple[np.ndarray, str]:
    """Pick the band to analyse.

    Parameters
    ----------
    img : numpy.ndarray
        Image of shape (H, W) or (H, W, n_bands).
    channel : str
        ``'auto'`` (band with the highest 5-95 % percentile spread), ``'r'``, ``'g'``,
        ``'b'`` (first three bands), ``'gray'`` (luminance; mean of the bands if there
        are not three), a band name from `names`, or a 0-based band index given as a
        string.
    names : list of str, optional
        Band names along the last axis (or the name of a 2-D image).

    Returns
    -------
    band : numpy.ndarray
    name : str
        Name of the band used (``'R'``, ``'G'``, ``'B'``, ``'gray'``, ``'band<i>'`` or
        the band name).
    """
    if img.ndim == 2:
        return img, (names[0] if names else "gray")
    n = img.shape[-1]

    def label(i):
        return names[i] if names else ("RGB"[i] if n == 3 else f"band{i}")

    channel = str(channel)
    if names and channel.lower() in [str(b).lower() for b in names]:
        i = [str(b).lower() for b in names].index(channel.lower())
        return img[..., i], label(i)
    channel = channel.lower()
    if channel == "auto":
        spreads = [np.subtract(*np.percentile(img[..., i], [95, 5])) for i in range(n)]
        i = int(np.argmax(spreads))
        return img[..., i], label(i)
    if channel in ("gray", "grey", "luminance", "l"):
        return (rgb2gray(img) if n == 3 else img.mean(axis=-1)), "gray"
    if channel in CHANNELS and CHANNELS[channel] < n:
        i = CHANNELS[channel]
        return img[..., i], label(i)
    if channel.isdigit() and int(channel) < n:
        i = int(channel)
        return img[..., i], (names[i] if names else f"band{i}")
    raise ValueError(f"Unknown channel '{channel}'"
                     + (f"; bands are {list(names)}" if names else ""))


def normalize(band: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    """Percentile stretch to [0, 1], so that images are comparable.

    Parameters
    ----------
    band : numpy.ndarray
    low, high : float
        Percentiles mapped to 0 and 1; values outside are clipped.

    Returns
    -------
    numpy.ndarray
    """
    lo, hi = np.percentile(band, [low, high])
    if hi <= lo:
        return np.zeros_like(band)
    return np.clip((band - lo) / (hi - lo), 0, 1)


def prepare(source: ArrayLike, channel: str = "auto", stretch: bool = True):
    """Load an image, pick the channel and normalise it.

    Parameters
    ----------
    source : str, pathlib.Path, numpy.ndarray, xarray.DataArray or xarray.Dataset
        Image file (GeoTIFF and NetCDF files are read with :func:`~shades2shapes.geo.open_geo`),
        array, or georeferenced image (:func:`~shades2shapes.geo.from_xarray`).
    channel : str
        See :func:`select_channel`.
    stretch : bool
        Apply :func:`normalize`.

    Returns
    -------
    band : numpy.ndarray
        Analysed band.
    rgb : numpy.ndarray or None
        Colour image, if the input has three bands.
    name : str
        Name of the band used.
    geo : GeoInfo or None
        Georeferencing (:class:`~shades2shapes.geo.GeoInfo`), for georeferenced inputs.
    """
    names = geo = None
    if is_xarray(source) or is_geo_file(source):
        if not is_xarray(source):
            source = open_geo(source)
        img, names, geo = from_xarray(source, channel)
        img = _to_float(img)
    else:
        img = load_image(source)
    band, name = select_channel(img, channel, names)
    if stretch:
        band = normalize(band)
    rgb = img if img.ndim == 3 and img.shape[-1] == 3 else None
    return band, rgb, name, geo
