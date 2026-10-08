"""Georeferenced inputs: xarray objects, GeoTIFF and NetCDF files.

:func:`~shades2shapes.diagnose` accepts an :class:`xarray.DataArray` or
:class:`xarray.Dataset` (for instance a satellite scene read with
:func:`rioxarray.open_rasterio`), or the path of a GeoTIFF or NetCDF file. The image
is converted to the north-up ``(row, column, band)`` layout used by the analysis, and
its georeferencing is kept in a :class:`GeoInfo`. This gives the pixel size (so that
lengths are also reported in physical units), the map coordinates of the eddies and
georeferenced output maps (:meth:`~shades2shapes.Diagnosis.to_xarray`).

Requires the optional dependencies ``xarray`` and ``rioxarray``::

    pip install shades2shapes[geo]

Without rioxarray, DataArrays still work, but the CRS is unknown: the pixel size is
then taken from the ``units`` attribute of the x coordinate, if any.

No-data pixels (NaN, e.g. land or clouds) are not supported yet: fill them or crop
the image to a valid area first.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

#: Extensions of the files opened with :func:`rioxarray.open_rasterio`.
RASTER_SUFFIXES = (".tif", ".tiff", ".jp2", ".vrt")
#: Extensions of the files opened with :func:`xarray.open_dataset`.
NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf")

_X_NAMES = ("x", "lon", "longitude", "easting")
_Y_NAMES = ("y", "lat", "latitude", "northing")
_TO_METRES = {"m": 1.0, "metre": 1.0, "meter": 1.0, "metres": 1.0, "meters": 1.0,
              "km": 1e3, "kilometre": 1e3, "kilometer": 1e3, "cm": 1e-2, "mm": 1e-3}


def _import_xarray():
    try:
        import xarray as xr
    except ImportError as err:
        raise ImportError("Georeferenced inputs need xarray and rioxarray: "
                          "pip install shades2shapes[geo]") from err
    try:
        import rioxarray  # noqa: F401  (registers the .rio accessor)
    except ImportError:
        pass
    return xr


def _has_rioxarray():
    try:
        import rioxarray  # noqa: F401
    except ImportError:
        return False
    return True


def is_xarray(obj) -> bool:
    """True if `obj` is an xarray object (checked without importing xarray)."""
    return type(obj).__module__.split(".")[0] == "xarray"


def is_geo_file(path) -> bool:
    """True if `path` is a NetCDF file, or a GeoTIFF/JPEG2000 file and rioxarray is installed.

    Without rioxarray, TIFF files are read as plain images.
    """
    if not isinstance(path, (str, Path)):
        return False
    suffix = Path(path).suffix.lower()
    return suffix in NETCDF_SUFFIXES or (suffix in RASTER_SUFFIXES and _has_rioxarray())


def open_geo(path):
    """Open a GeoTIFF (or other raster) or NetCDF file.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Rasters are opened with ``rioxarray.open_rasterio(path, masked=True)``, so
        no-data values become NaN; the bands are named after the band descriptions
        when the file has them, otherwise numbered from 1. NetCDF files are opened with
        :func:`xarray.open_dataset`.
    """
    xr = _import_xarray()
    path = Path(path)
    if path.suffix.lower() in NETCDF_SUFFIXES:
        return xr.open_dataset(path)
    if not _has_rioxarray():
        raise ImportError("Reading GeoTIFF files needs rioxarray: pip install shades2shapes[geo]")
    import rioxarray
    da = rioxarray.open_rasterio(path, masked=True)
    desc = da.attrs.get("long_name")
    if (isinstance(desc, (list, tuple)) and "band" in da.dims
            and len(desc) == da.sizes["band"] and len(set(desc)) == len(desc)):
        da = da.assign_coords(band=[str(d) for d in desc])
    return da


@dataclass
class GeoInfo:
    """Georeferencing of an analysed image.

    The image is stored north-up: row 0 is the northernmost (largest y) and column 0
    the westernmost (smallest x).

    Attributes
    ----------
    x : numpy.ndarray
        Map coordinate of the pixel centres, by column.
    y : numpy.ndarray
        Map coordinate of the pixel centres, by row (decreasing).
    x_dim : str
        Name of the x dimension of the input (e.g. ``'x'`` or ``'lon'``).
    y_dim : str
        Name of the y dimension of the input (e.g. ``'y'`` or ``'lat'``).
    crs : str, optional
        Coordinate reference system (e.g. ``'EPSG:32631'``), when known (needs
        rioxarray).
    geographic : bool
        The coordinates are longitudes and latitudes in degrees.
    units : str, optional
        Unit of the map coordinates (``'metre'``, ``'degree'``...), when known.
    """

    x: np.ndarray
    y: np.ndarray
    x_dim: str = "x"
    y_dim: str = "y"
    crs: Optional[str] = None
    geographic: bool = False
    units: Optional[str] = None

    @property
    def resolution(self):
        """Pixel spacing ``(dx, dy)`` in map units; dy is negative (north-up)."""
        def step(c):
            return float((c[-1] - c[0]) / (len(c) - 1)) if len(c) > 1 else float("nan")
        return step(self.x), step(self.y)

    @property
    def transform(self):
        """Affine transform ``(a, b, c, d, e, f)`` of the pixel corners, as in rasterio:
        ``x = a * col + b * row + c`` and ``y = d * col + e * row + f``."""
        dx, dy = self.resolution
        return (dx, 0.0, float(self.x[0]) - dx / 2, 0.0, dy, float(self.y[0]) - dy / 2)

    def xy(self, col, row):
        """Map coordinates of (fractional) pixel indices; integers are pixel centres.

        Parameters
        ----------
        col, row : float or numpy.ndarray

        Returns
        -------
        x, y : float or numpy.ndarray
        """
        dx, dy = self.resolution
        return self.x[0] + dx * np.asarray(col, float), self.y[0] + dy * np.asarray(row, float)

    def lonlat(self, col, row):
        """Longitude and latitude (degrees, WGS84) of pixel indices.

        Returns
        -------
        lon, lat : float or numpy.ndarray
            Both None when the CRS is unknown.
        """
        x, y = self.xy(col, row)
        if self.geographic:
            return x, y
        if self.crs is None:
            return None, None
        from pyproj import Transformer  # installed with rioxarray
        return Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True).transform(x, y)

    def pixel_size(self, unit="km"):
        """Pixel size in `unit`, or None (with a warning) when it cannot be derived.

        Non-square pixels give the geometric mean of the two sides (with a warning).
        Geographic coordinates (degrees) give None: reproject the image first, e.g.
        ``da.rio.reproject(da.rio.estimate_utm_crs())``, or pass ``pixel_size``.

        Parameters
        ----------
        unit : str
            Unit of the result: ``'m'``, ``'km'``, ``'cm'`` or ``'mm'``.

        Returns
        -------
        float or None
        """
        if self.geographic:
            warnings.warn("The image is in geographic coordinates (degrees): the pixel size is "
                          "not set. Reproject it (da.rio.reproject(da.rio.estimate_utm_crs())) "
                          "or pass pixel_size.", stacklevel=3)
            return None
        to_m = _TO_METRES.get(str(self.units).lower())
        if to_m is None:
            warnings.warn(f"Unknown unit of the map coordinates ({self.units}): the pixel size "
                          "is not set; pass pixel_size.", stacklevel=3)
            return None
        if unit not in _TO_METRES:
            warnings.warn(f"Unknown unit '{unit}': the pixel size is not set; pass pixel_size.",
                          stacklevel=3)
            return None
        dx, dy = (abs(v) for v in self.resolution)
        if not np.isfinite(dx * dy):
            return None
        if abs(dx - dy) > 0.01 * max(dx, dy):
            warnings.warn(f"Non-square pixels ({dx:g} x {dy:g} {self.units}): using their "
                          "geometric mean as pixel size.", stacklevel=3)
        return float(f"{np.sqrt(dx * dy) * to_m / _TO_METRES[unit]:.12g}")

    def to_dict(self):
        """CRS, dimension names, units, transform and size, as a plain dict."""
        return {"crs": self.crs, "x_dim": self.x_dim, "y_dim": self.y_dim, "units": self.units,
                "geographic": self.geographic, "transform": list(self.transform),
                "width": len(self.x), "height": len(self.y)}


def _spatial_dims(da):
    try:
        return da.rio.x_dim, da.rio.y_dim
    except Exception:
        pass
    low = {str(d).lower(): d for d in da.dims}
    x = next((low[n] for n in _X_NAMES if n in low), None)
    y = next((low[n] for n in _Y_NAMES if n in low), None)
    if x is None or y is None:
        if da.ndim == 2:
            return da.dims[1], da.dims[0]
        raise ValueError(f"Cannot identify the spatial dimensions among {da.dims}: rename them "
                         "to 'x' and 'y' (or 'lon' and 'lat').")
    return x, y


def _geoinfo(da, x_dim, y_dim):
    if x_dim not in da.coords or y_dim not in da.coords:
        return None
    x = np.asarray(da[x_dim].values, dtype=float)
    y = np.asarray(da[y_dim].values, dtype=float)
    for c, name in ((x, x_dim), (y, y_dim)):
        if len(c) > 2 and not np.allclose(np.diff(c), np.diff(c).mean(), rtol=1e-3):
            warnings.warn(f"Coordinate '{name}' is not regularly spaced: map coordinates "
                          "are approximate.", stacklevel=4)
    crs = units = None
    geographic = False
    rio_crs = None
    try:
        rio_crs = da.rio.crs
    except Exception:
        pass
    if rio_crs is not None:
        crs = rio_crs.to_string()
        geographic = bool(rio_crs.is_geographic)
        lin = None if geographic else rio_crs.linear_units
        units = "degree" if geographic else (lin if lin and lin != "unknown" else None)
    else:
        u = str(da[x_dim].attrs.get("units", "")).lower()
        geographic = u.startswith("degree") or str(x_dim).lower() in ("lon", "longitude")
        units = "degree" if geographic else (u or None)
    return GeoInfo(x=x, y=y, x_dim=str(x_dim), y_dim=str(y_dim), crs=crs,
                   geographic=geographic, units=units)


def from_xarray(obj, channel="auto"):
    """Convert an xarray image to an array, its band names and its georeferencing.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        A DataArray with two spatial dimensions and at most one other (bands), or a
        Dataset. Other dimensions of size 1 (e.g. ``time``) are dropped. For a Dataset,
        `channel` names the variable to analyse; it can be omitted if the Dataset has
        a single image variable. To combine variables into bands, stack them first:
        ``ds[["B4", "B3", "B2"]].to_array("band")``.
    channel : str
        If it is the name of a band (a value of the band coordinate, e.g. ``'B8'``, or
        the 1-based band number of a GeoTIFF) or of a Dataset variable, only that band
        is returned.

    Returns
    -------
    img : numpy.ndarray
        Array of shape (H, W) or (H, W, n_bands), north-up.
    names : list of str or None
        Band names along the last axis (a single name when `img` is 2-D and a band was
        selected), or None.
    geo : GeoInfo or None
        None when the spatial dimensions have no coordinates.
    """
    xr = _import_xarray()
    names = None
    channel = str(channel)
    if isinstance(obj, xr.Dataset):
        if channel in obj.data_vars:
            da, names = obj[channel], [channel]
        else:
            images = [v for v in obj.data_vars if obj[v].ndim >= 2]
            if len(images) != 1:
                raise ValueError(f"The Dataset has several variables {images}: choose one "
                                 "with channel='<name>', or stack bands with "
                                 "ds[[...]].to_array('band').")
            da = obj[images[0]]
            names = [str(images[0])]
        gm = da.attrs.get("grid_mapping")      # CF grid mapping variable, e.g. spatial_ref
        if gm in obj.variables and gm not in da.coords:
            da = da.assign_coords({gm: obj[gm]})
    else:
        da = obj

    x_dim, y_dim = _spatial_dims(da)
    for d in [d for d in da.dims if d not in (x_dim, y_dim)]:
        if da.sizes[d] == 1:
            if d in da.coords and str(da[d].values[0]).lower() == channel.lower():
                names = [str(da[d].values[0])]          # band already selected
            da = da.isel({d: 0}, drop=True)
    extra = [d for d in da.dims if d not in (x_dim, y_dim)]
    if len(extra) > 1:
        raise ValueError(f"Too many dimensions {da.dims}: keep two spatial dimensions and at "
                         "most one band dimension, e.g. da.isel(time=0).")
    if extra:
        bdim = extra[0]
        n = da.sizes[bdim]
        bands = ([str(v) for v in da[bdim].values] if bdim in da.coords
                 else [str(i) for i in range(n)])
        low = [b.lower() for b in bands]
        if channel.lower() in low:
            da = da.isel({bdim: low.index(channel.lower())}, drop=True)
            names = [bands[low.index(channel.lower())]]
            da = da.transpose(y_dim, x_dim)
        else:
            names = bands
            da = da.transpose(y_dim, x_dim, bdim)
    else:
        da = da.transpose(y_dim, x_dim)

    # north-up, west to east
    if da.sizes[y_dim] > 1 and y_dim in da.coords and da[y_dim][1] > da[y_dim][0]:
        da = da.isel({y_dim: slice(None, None, -1)})
    if da.sizes[x_dim] > 1 and x_dim in da.coords and da[x_dim][1] < da[x_dim][0]:
        da = da.isel({x_dim: slice(None, None, -1)})
    geo = _geoinfo(da, x_dim, y_dim)
    return np.asarray(da.values, dtype=float), names, geo


def fill_nodata(da, valid=None, sigma=2.0):
    """Fill no-data pixels from their valid neighbours (normalised convolution).

    Each invalid pixel gets the Gaussian-weighted mean of the valid pixels around it,
    at scale `sigma`, then ``4 * sigma`` and ``16 * sigma`` px for larger gaps;
    pixels still out of reach get the median. Filled areas are smooth, so they add
    no filaments, but keep them small (clouds, glint, small islands): crop out land.

    Parameters
    ----------
    da : xarray.DataArray
        Image, with or without a band dimension.
    valid : numpy.ndarray of bool, optional
        Valid pixels, shape of the two spatial dimensions. NaN and infinite values are
        always invalid.
    sigma : float
        Smallest smoothing scale (px).

    Returns
    -------
    filled : xarray.DataArray
    fraction : float
        Fraction of pixels filled.
    """
    from scipy import ndimage as ndi
    _import_xarray()
    x_dim, y_dim = _spatial_dims(da)
    other = [d for d in da.dims if d not in (x_dim, y_dim)]
    da = da.transpose(*other, y_dim, x_dim)
    vals = np.asarray(da.values, dtype=float)
    ok = np.isfinite(vals).all(axis=tuple(range(len(other))))
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    if ok.all():
        return da, 0.0
    if not ok.any():
        raise ValueError("No valid pixel to fill from.")
    w = ok.astype(float)
    flat = vals.reshape(-1, *ok.shape)
    out = np.empty_like(flat)
    for i, v in enumerate(flat):
        v0 = np.where(ok, v, 0.0)
        f = np.where(ok, v, np.nan)
        for s in (sigma, 4 * sigma, 16 * sigma):
            todo = np.isnan(f)
            if not todo.any():
                break
            den = ndi.gaussian_filter(w, s)
            num = ndi.gaussian_filter(v0, s)
            reach = todo & (den > 1e-3)
            f[reach] = num[reach] / den[reach]
        f[np.isnan(f)] = np.median(v[ok])
        out[i] = f
    return da.copy(data=out.reshape(vals.shape)), float((~ok).mean())


def source_name(obj) -> str:
    """File name of an xarray object, from its encoding, or ``'<xarray>'``."""
    src = getattr(obj, "encoding", {}).get("source")
    return str(src) if src else "<xarray>"
