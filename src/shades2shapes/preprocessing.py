"""Image loading, channel selection and normalisation."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from skimage import io as skio
from skimage.color import rgb2gray

ArrayLike = Union[str, Path, np.ndarray]

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
    arr = arr.astype(float)
    if arr.max() > 1.0:
        arr = arr / (65535.0 if arr.max() > 255 else 255.0)
    return arr


def select_channel(img: np.ndarray, channel: str = "auto") -> tuple[np.ndarray, str]:
    """Pick the band to analyse.

    Parameters
    ----------
    img : numpy.ndarray
        Image of shape (H, W) or (H, W, 3).
    channel : str
        ``'auto'`` (band with the highest 5-95 % percentile spread), ``'r'``, ``'g'``,
        ``'b'``, ``'gray'`` (luminance), or a band index given as a string.

    Returns
    -------
    band : numpy.ndarray
    name : str
        Name of the band used (``'R'``, ``'G'``, ``'B'``, ``'gray'`` or ``'band<i>'``).
    """
    if img.ndim == 2:
        return img, "gray"
    channel = str(channel).lower()
    if channel == "auto":
        spreads = [np.subtract(*np.percentile(img[..., i], [95, 5])) for i in range(3)]
        i = int(np.argmax(spreads))
        return img[..., i], "RGB"[i]
    if channel in ("gray", "grey", "luminance", "l"):
        return rgb2gray(img), "gray"
    if channel in CHANNELS:
        i = CHANNELS[channel]
        return img[..., i], "RGB"[i]
    if channel.isdigit():
        i = int(channel)
        return img[..., i], f"band{i}"
    raise ValueError(f"Unknown channel '{channel}'")


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

    Returns
    -------
    band : numpy.ndarray
        Analysed band.
    rgb : numpy.ndarray or None
        Colour image, if the input has three bands.
    name : str
        Name of the band used.
    """
    img = load_image(source)
    band, name = select_channel(img, channel)
    if stretch:
        band = normalize(band)
    rgb = img if img.ndim == 3 else None
    return band, rgb, name
