"""Multi-level (pyramid) analysis: the same area diagnosed at several resolutions.

:func:`~shades2shapes.diagnose_pyramid` runs :func:`~shades2shapes.diagnose` on every level of an image
pyramid, for instance a Zarr store whose groups ``0, 1, 2...`` are 2, 4... times
coarser, and returns a :class:`PyramidDiagnosis`. It gives a table of metrics per level
and matches the eddies detected at different levels: a structure found at several
resolutions, at the same place and with the same radius, is more robust than one seen
at a single level.

Inputs can be a Zarr pyramid, a list or dict of levels (xarray objects or arrays), or a
single image, which is then coarsened by factors of 2.

Requires xarray (``pip install shades2shapes[geo]``); Zarr stores also need ``zarr``.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .diagnose import Diagnosis, DiagnosisConfig, diagnose
from .geo import _import_xarray, _spatial_dims, fill_nodata, is_xarray

#: Metrics shown by :meth:`~shades2shapes.PyramidDiagnosis.summary` and
#: :meth:`~shades2shapes.PyramidDiagnosis.plot_metrics`.
PYRAMID_METRICS = ("filament_fraction", "skeleton_density_phys", "branch_length_mean_phys",
                   "junction_endpoint_ratio", "coherence_filaments", "orientation_order",
                   "glcm_correlation_decay", "spectral_slope", "fractal_dimension", "n_eddies")


# ---------------------------------------------------------------------- opening
def pyramid_levels(path):
    """Group names of the levels of a Zarr pyramid, finest first.

    The levels are read from the ``multiscales`` attribute of the root group, either
    as ``{"layout": [{"group": "0"}, ...]}`` or as OME-NGFF
    ``{"datasets": [{"path": "0"}, ...]}``; without it, the subgroups named by
    integers are used.

    Parameters
    ----------
    path : str or pathlib.Path
        Zarr store.

    Returns
    -------
    list of str
    """
    import zarr
    g = zarr.open_group(str(path), mode="r")
    ms = dict(g.attrs).get("multiscales")
    if ms:
        ms = ms[0] if isinstance(ms, list) else ms
        items = ms.get("layout") or ms.get("datasets") or []
        names = [str(it.get("group", it.get("path"))) for it in items]
        if names:
            return names
    names = sorted((k for k in g.group_keys() if k.isdigit()), key=int)
    if not names:
        raise ValueError(f"{path} is not a pyramid: no 'multiscales' attribute and no "
                         "groups named 0, 1, 2...")
    return names


def open_pyramid(path, levels=None):
    """Open the levels of a Zarr pyramid as xarray Datasets (lazily).

    Parameters
    ----------
    path : str or pathlib.Path
        Zarr store.
    levels : sequence of str or int, optional
        Levels to open (default: all, see :func:`pyramid_levels`).

    Returns
    -------
    dict
        ``{level name: xarray.Dataset}``, finest first.
    """
    xr = _import_xarray()
    names = pyramid_levels(path) if levels is None else [str(v) for v in levels]
    out = {}
    for n in names:
        try:
            ds = xr.open_zarr(str(path), group=n, consolidated=False)
        except Exception:
            ds = xr.open_zarr(str(path), group=n)
        out[n] = ds
    return out


def coarsen_pyramid(da, factors=(1, 2, 4, 8, 16)):
    """Build a pyramid by block-averaging an image (NaN ignored).

    Parameters
    ----------
    da : xarray.DataArray, xarray.Dataset or numpy.ndarray
        Image. Arrays of shape (H, W) or (H, W, bands) get pixel coordinates.
    factors : sequence of int
        Coarsening factors.

    Returns
    -------
    dict
        ``{"x<factor>": coarsened image}``.
    """
    xr = _import_xarray()
    if not is_xarray(da):
        a = np.asarray(da)
        da = xr.DataArray(a, dims=("y", "x", "band")[:a.ndim])
    x_dim, y_dim = _spatial_dims(da if isinstance(da, xr.DataArray)
                                 else next(iter(da.data_vars.values())))
    out = {}
    for f in factors:
        f = int(f)
        out[f"x{f}"] = da if f == 1 else da.coarsen({x_dim: f, y_dim: f},
                                                    boundary="trim").mean()
    return out


# ---------------------------------------------------------------------- helpers
def _select(ds, variable, channel):
    """DataArray to analyse from a level, with its grid mapping attached."""
    xr = _import_xarray()
    if isinstance(ds, xr.Dataset):
        if variable is None:
            images = [v for v in ds.data_vars if ds[v].ndim >= 2]
            if len(images) != 1:
                raise ValueError(f"Choose the variable to analyse among {images} "
                                 "(variable=...).")
            variable = images[0]
        da = ds[variable]
        gm = da.attrs.get("grid_mapping")
        if gm in ds.variables and gm not in da.coords:
            da = da.assign_coords({gm: ds[gm]})
    else:
        da = ds
    # keep only the requested band (with its name) to load and fill less data
    x_dim, y_dim = _spatial_dims(da)
    extra = [d for d in da.dims if d not in (x_dim, y_dim)]
    if len(extra) == 1 and extra[0] in da.coords:
        names = [str(v).lower() for v in da[extra[0]].values]
        if str(channel).lower() in names:
            da = da.isel({extra[0]: [names.index(str(channel).lower())]})
    return da


def _bbox_index(da, bbox, bbox_crs):
    """isel slices of the spatial dims covering `bbox` = (xmin, ymin, xmax, ymax)."""
    x_dim, y_dim = _spatial_dims(da)
    xmin, ymin, xmax, ymax = bbox
    if bbox_crs is not None:
        crs = da.rio.crs if hasattr(da, "rio") else None
        if crs is None:
            raise ValueError("bbox_crs needs a georeferenced image with a known CRS.")
        from pyproj import Transformer
        t = Transformer.from_crs(bbox_crs, crs, always_xy=True)
        xs, ys = t.transform([xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax])
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    x = np.asarray(da[x_dim].values)
    y = np.asarray(da[y_dim].values)
    ix = np.nonzero((x >= xmin) & (x <= xmax))[0]
    iy = np.nonzero((y >= ymin) & (y <= ymax))[0]
    if len(ix) == 0 or len(iy) == 0:
        raise ValueError(f"bbox {bbox} does not intersect the image.")
    return {x_dim: slice(ix[0], ix[-1] + 1), y_dim: slice(iy[0], iy[-1] + 1)}


def _apply_transform(da, transform):
    if transform is None:
        return da
    if transform == "log10":
        pos = da.where(da > 0)
        floor = float(pos.min()) if np.isfinite(pos.min()) else 1e-6
        return np.log10(da.clip(min=floor))
    if callable(transform):
        return transform(da)
    raise ValueError(f"Unknown transform {transform!r}: use 'log10' or a function.")


# ---------------------------------------------------------------------- result
@dataclass
class PyramidDiagnosis:
    """Result of :func:`diagnose_pyramid`.

    Attributes
    ----------
    diagnoses : dict
        ``{level name: Diagnosis}`` for the levels analysed, finest first.
    levels : pandas.DataFrame
        One row per level (also the skipped ones): pixel size, shape, fraction of
        filled no-data pixels and status.
    eddy_tracks : pandas.DataFrame
        Eddies matched across levels, one row per structure: number and names of the
        levels where it is found, mean centre (map coordinates and lon/lat), mean
        radius in map units and in pixels of the finest level, and best significance.
        Sorted by number of levels, then significance.
    eddies : pandas.DataFrame
        All the eddies, one row per detection, with their ``track`` number.
    """

    diagnoses: dict
    levels: pd.DataFrame
    eddy_tracks: pd.DataFrame
    eddies: pd.DataFrame = field(repr=False, default=None)

    def metrics_table(self, metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Metrics per level (rows) for the analysed levels.

        Parameters
        ----------
        metrics : sequence of str, optional
            Metric names (default: all scalar metrics).

        Returns
        -------
        pandas.DataFrame
        """
        rows = {}
        for name, d in self.diagnoses.items():
            m = d.scalar_metrics()
            rows[name] = {k: m.get(k) for k in metrics} if metrics else m
        df = pd.DataFrame(rows).T
        df.insert(0, "pixel_size", [d.config.pixel_size for d in self.diagnoses.values()])
        df.index.name = "level"
        return df

    def summary(self) -> str:
        """Human-readable report: levels, main metrics per level and eddy tracks."""
        unit = next((d.config.unit for d in self.diagnoses.values()), "")
        L = ["shades2shapes pyramid diagnosis",
             f"  {len(self.diagnoses)} level(s) analysed of {len(self.levels)}"]
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            L.append("\nLevels")
            L.append(self.levels.to_string())
            L.append(f"\nMain metrics per level (pixel size and lengths in {unit})")
            t = self.metrics_table([m for m in PYRAMID_METRICS])
            L.append(t.dropna(axis=1, how="all").T.to_string(
                float_format=lambda v: f"{v:.4g}"))
            L.append("\nEddy tracks (same structure at several levels)")
            if len(self.eddy_tracks):
                L.append(self.eddy_tracks.to_string(float_format=lambda v: f"{v:.6g}"))
            else:
                L.append("  none detected")
        L.append("\nMost metrics depend on the pixel size (finer levels resolve smaller "
                 "filaments, and noise): compare images at the same level, and use the "
                 "trends across levels to see at which scales the structures appear.")
        return "\n".join(L)

    def to_dict(self):
        """Serialisable version: levels, eddy tracks and each level's diagnosis."""
        return {"levels": json.loads(self.levels.reset_index().to_json(orient="records")),
                "eddy_tracks": json.loads(self.eddy_tracks.to_json(orient="records")),
                "diagnoses": {k: d.to_dict() for k, d in self.diagnoses.items()}}

    def save(self, outdir, plots: bool = True, maps: bool = False):
        """Write the results to a directory.

        Files: ``summary.txt``, ``levels.csv``, ``metrics_by_level.csv``,
        ``eddy_tracks.csv``, ``eddies.csv``, ``pyramid.json`` and, with `plots`,
        ``pyramid.png``, ``metrics_by_level.png`` and ``level<name>_diagnosis.png``;
        with `maps`, ``level<name>_maps.nc`` (:meth:`~shades2shapes.Diagnosis.to_xarray`).

        Note that `plots` switches matplotlib to the non-interactive ``Agg`` backend.

        Returns
        -------
        pathlib.Path
        """
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.txt").write_text(self.summary())
        self.levels.to_csv(out / "levels.csv")
        self.metrics_table().to_csv(out / "metrics_by_level.csv")
        self.eddy_tracks.to_csv(out / "eddy_tracks.csv", index=False)
        if self.eddies is not None:
            self.eddies.to_csv(out / "eddies.csv", index=False)
        (out / "pyramid.json").write_text(json.dumps(self.to_dict(), indent=2, default=float))
        if plots:
            self.plot(out / "pyramid.png")
            self.plot_metrics(out / "metrics_by_level.png")
            for name, d in self.diagnoses.items():
                d.plot(out / f"level{name}_diagnosis.png")
        if maps:
            for name, d in self.diagnoses.items():
                d.to_xarray().to_netcdf(out / f"level{name}_maps.nc")
        return out

    def plot(self, path=None, dpi=90):
        """Figure with one row per level: band, filament mask, coherence and eddies.

        Panels share map coordinates (km for metric CRSs), so the levels can be compared
        directly; eddies are labelled by track number.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib
        if path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(self.diagnoses)
        d0 = next(iter(self.diagnoses.values()))
        H, W = d0.shape
        fig, ax = plt.subplots(n, 3, figsize=(15, (max(1.0, 4.4 * H / W) + 0.7) * n + 0.5),
                               squeeze=False)
        tracks = self.eddies
        for i, (name, d) in enumerate(self.diagnoses.items()):
            ext, scale, unit = _extent(d)
            ax[i, 0].imshow(d.band, cmap="viridis", extent=ext)
            ax[i, 1].imshow(d.mask, cmap="gray", extent=ext)
            ax[i, 2].imshow(d.coherence, cmap="magma", vmin=0, vmax=1, extent=ext)
            px = self.levels.loc[name, "pixel_size"]
            ax[i, 0].set_ylabel(f"level {name}\n{px:.4g} {d.config.unit}" if px
                                else f"level {name}")
            sel = tracks[tracks["level"] == name] if tracks is not None and len(tracks) else []
            for _, e in (sel.iterrows() if len(sel) else []):
                cx, cy, r = _eddy_xyr(d, e)
                ax[i, 2].add_patch(plt.Circle((cx * scale, cy * scale), r * scale, fill=False,
                                              color="cyan", lw=1.5))
                ax[i, 2].text(cx * scale, (cy + r) * scale, f"T{int(e['track'])}", color="cyan",
                              ha="center", va="bottom", fontsize=8, weight="bold")
            for a in ax[i]:
                a.set_xlim(ext[0], ext[1])
                a.set_ylim(ext[2], ext[3])
        for a, t in zip(ax[0], ("Analysed band (normalised)", "Filament mask",
                                "Coherence + eddies (T = track)")):
            a.set_title(t)
        for a in ax[-1]:
            a.set_xlabel(unit)
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
        return fig

    def plot_metrics(self, path=None, metrics: Sequence[str] = PYRAMID_METRICS, dpi=100):
        """Main metrics as a function of the pixel size (log axis).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib
        if path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = self.metrics_table(metrics).astype(float)
        cols = [c for c in metrics if c in t and t[c].notna().any()]
        x = t["pixel_size"] if t["pixel_size"].notna().all() else pd.Series(
            range(len(t)), index=t.index)
        nc = 4
        nr = int(np.ceil(len(cols) / nc))
        fig, ax = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr), squeeze=False)
        for a, c in zip(ax.ravel(), cols):
            a.plot(x, t[c], "o-")
            if t["pixel_size"].notna().all():
                a.set_xscale("log")
                a.set_xticks(list(x), [f"{v:.3g}" for v in x])
                a.minorticks_off()
            else:
                a.set_xticks(list(x), list(t.index))
            a.set_title(c, fontsize=9)
        for a in ax.ravel()[len(cols):]:
            a.axis("off")
        unit = next((d.config.unit for d in self.diagnoses.values()), "")
        for a in ax[-1]:
            a.set_xlabel(f"pixel size ({unit})" if t["pixel_size"].notna().all() else "level")
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
        return fig


def _extent(d):
    """imshow extent, scale factor from map units to plot units, and axis label."""
    H, W = d.shape
    g = d.geo
    if g is None:
        return [-0.5, W - 0.5, H - 0.5, -0.5], 1.0, "px"
    scale, unit = (1e-3, "km") if str(g.units).lower() in ("metre", "meter", "m") else (
        1.0, g.units or "map units")
    dx, dy = g.resolution
    x0, x1 = g.x[0] - dx / 2, g.x[-1] + dx / 2
    y0, y1 = g.y[-1] + dy / 2, g.y[0] - dy / 2
    return [x0 * scale, x1 * scale, y0 * scale, y1 * scale], scale, unit


def _eddy_xyr(d, e):
    """Eddy centre and radius in map units (pixels without georeferencing)."""
    if d.geo is None:
        return e["x"], e["y"], e["radius"]
    return e["x_map"], e["y_map"], e["radius_map"]


def _match_eddies(rows, max_shift=0.5, max_ratio=2.0):
    """Group eddies of different levels into tracks (greedy, finest level first)."""
    tracks = []      # list of lists of row indices
    for i, e in enumerate(rows):
        best, best_d = None, np.inf
        for t in tracks:
            if any(rows[j]["level"] == e["level"] for j in t):
                continue
            ref = rows[t[-1]]
            dist = np.hypot(e["cx"] - ref["cx"], e["cy"] - ref["cy"])
            ratio = max(e["r"], ref["r"]) / max(min(e["r"], ref["r"]), 1e-12)
            if dist <= max_shift * max(e["r"], ref["r"]) and ratio <= max_ratio and dist < best_d:
                best, best_d = t, dist
        if best is None:
            tracks.append([i])
        else:
            best.append(i)
    return tracks


# ---------------------------------------------------------------------- main
PyramidSource = Union[str, Path, dict, list, tuple, "xarray.DataArray", "xarray.Dataset",
                      np.ndarray]


def diagnose_pyramid(source: PyramidSource, levels: Optional[Sequence] = None,
                     variable: Optional[str] = None, bbox: Optional[Sequence[float]] = None,
                     bbox_crs: Optional[str] = None, mask: Optional[str] = None,
                     fill_nodata_pixels: bool = True, max_nodata: float = 0.25,
                     transform: Union[None, str, Callable] = None, min_size: int = 64,
                     factors: Sequence[int] = (1, 2, 4, 8, 16),
                     max_shift: float = 0.5, max_ratio: float = 2.0,
                     config: Optional[DiagnosisConfig] = None, **diag_kwargs) -> PyramidDiagnosis:
    """Diagnose the same area at every level of an image pyramid.

    Parameters
    ----------
    source : str, pathlib.Path, dict, list, xarray.DataArray, xarray.Dataset or numpy.ndarray
        A Zarr pyramid (path of the store, see :func:`~shades2shapes.multiscale.pyramid_levels`); the levels
        themselves, as a dict ``{name: image}`` or a list (finest first) of xarray
        objects or arrays; or a single image, coarsened by `factors`
        (:func:`~shades2shapes.multiscale.coarsen_pyramid`).
    levels : sequence of str or int, optional
        Levels to analyse (default: all).
    variable : str, optional
        Variable of the Datasets to analyse (e.g. ``"Rrs"``); optional if there is only
        one image variable.
    bbox : sequence of float, optional
        Area to analyse, ``(xmin, ymin, xmax, ymax)``, in map coordinates of the image
        (or in `bbox_crs`). The same area is cut from every level. Default: whole image.
    bbox_crs : str, optional
        CRS of `bbox`, e.g. ``"EPSG:4326"`` for longitudes and latitudes.
    mask : str, optional
        Variable of the Datasets flagging invalid pixels (non-zero = invalid, e.g.
        ``"mask"`` for clouds and land). NaN pixels are always invalid.
    fill_nodata_pixels : bool
        Fill invalid pixels from their neighbours (:func:`~shades2shapes.geo.fill_nodata`)
        instead of failing. The filled fraction is reported per level.
    max_nodata : float
        Levels with a larger fraction of invalid pixels in `bbox` are skipped.
    transform : {'log10'} or callable, optional
        Applied to the image before the analysis, e.g. ``'log10'`` for reflectances or
        concentrations that span orders of magnitude.
    min_size : int
        Levels whose smaller side is under `min_size` px are skipped.
    factors : sequence of int
        Coarsening factors, when `source` is a single image.
    max_shift : float
        Eddies of two levels belong to the same track if their centres are closer than
        `max_shift` x the larger radius...
    max_ratio : float
        ...and their radii differ by less than this factor.
    config : DiagnosisConfig, optional
        Options of :func:`~shades2shapes.diagnose`, for every level.
    **diag_kwargs
        Fields of :class:`~shades2shapes.DiagnosisConfig`, e.g. ``channel="665"``,
        ``unit="km"``. `pixel_size` is not allowed: it comes from each level.

    Returns
    -------
    PyramidDiagnosis

    Examples
    --------
    >>> import shades2shapes as s2s
    >>> p = s2s.diagnose_pyramid("S2_GRS.zarr", variable="Rrs", channel="665",
    ...                          mask="mask", transform="log10",
    ...                          bbox=(612000, 4790600, 678000, 4800600))
    >>> print(p.summary())
    >>> p.eddy_tracks            # structures found at several resolutions
    >>> p.save("pyramid/")
    """
    xr = _import_xarray()
    if "pixel_size" in diag_kwargs:
        raise TypeError("pixel_size is derived from each level; it cannot be given.")
    channel = diag_kwargs.get("channel", config.channel if config else "auto")

    if isinstance(source, (str, Path)):
        src = open_pyramid(source, levels)
    elif isinstance(source, dict):
        src = {str(k): v for k, v in source.items()}
    elif isinstance(source, (list, tuple)):
        src = {str(i): v for i, v in enumerate(source)}
    else:
        src = coarsen_pyramid(source, factors)
    if levels is not None and not isinstance(source, (str, Path)):
        want = [str(v) for v in levels]
        src = {k: v for k, v in src.items() if k in want}

    base = Path(source).name if isinstance(source, (str, Path)) else "<pyramid>"
    info, diags, rows = [], {}, []
    for name, lev in src.items():
        if not is_xarray(lev):
            a = np.asarray(lev)
            lev = xr.DataArray(a, dims=("y", "x", "band")[:a.ndim])
        da = _select(lev, variable, channel)
        bad = None
        if mask is not None:
            if not isinstance(lev, xr.Dataset) or mask not in lev:
                raise ValueError(f"mask variable '{mask}' not found in level {name}")
            bad = lev[mask]
        if bbox is not None:
            idx = _bbox_index(da, bbox, bbox_crs)
            da = da.isel(idx)
            bad = bad.isel(idx) if bad is not None else None
        x_dim, y_dim = _spatial_dims(da)
        H, W = da.sizes[y_dim], da.sizes[x_dim]
        row = {"level": name, "pixel_size": None, "shape": f"{W}x{H}",
               "nodata_fraction": None, "status": "ok"}
        info.append(row)
        if min(H, W) < min_size:
            row["status"] = f"skipped: smaller than {min_size} px"
            continue
        da = da.load()
        valid = None
        if bad is not None:
            valid = (np.asarray(bad.transpose(y_dim, x_dim).values) == 0)
        finite = np.isfinite(da.transpose(..., y_dim, x_dim).values)
        finite = finite.reshape(-1, H, W).all(axis=0)
        ok = finite if valid is None else finite & valid
        row["nodata_fraction"] = round(float((~ok).mean()), 4)
        if not ok.all():
            if not fill_nodata_pixels:
                row["status"] = "skipped: no-data pixels (fill_nodata_pixels=False)"
                continue
            if (~ok).mean() > max_nodata:
                row["status"] = f"skipped: more than {max_nodata:.0%} no-data"
                continue
            da, _ = fill_nodata(da, ok)
        da = _apply_transform(da, transform)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*geographic coordinates.*")
            d = diagnose(da, config=config, **diag_kwargs)
        d.source = f"{base}[{name}]"
        diags[name] = d
        row["pixel_size"] = d.config.pixel_size
        res = abs(d.geo.resolution[0]) if d.geo is not None else 1.0
        for k, e in enumerate(d.eddies, 1):
            r = e.to_dict()
            r.pop("profile", None)
            r.update(level=name, rank=k, radius_map=e.radius * res,
                     cx=e.x_map if e.x_map is not None else e.x * res,
                     cy=e.y_map if e.y_map is not None else e.y * res,
                     r=e.radius * res)
            rows.append(r)
    if not diags:
        raise ValueError("No level could be analysed:\n" + pd.DataFrame(info).to_string())

    levels_df = pd.DataFrame(info).set_index("level")
    eddies = pd.DataFrame(rows)
    track_rows = []
    if len(rows):
        tracks = _match_eddies(rows, max_shift, max_ratio)
        tracks.sort(key=lambda t: (-len(t), -max(rows[j]["significance"] for j in t)))
        eddies["track"] = 0
        for tid, t in enumerate(tracks, 1):
            eddies.loc[t, "track"] = tid
            sub = eddies.loc[t]
            finest = next(iter(diags))
            res0 = abs(diags[finest].geo.resolution[0]) if diags[finest].geo is not None else 1.0
            tr = {"track": tid, "n_levels": len(t), "levels": ",".join(sub["level"]),
                  "x_map": sub["cx"].mean(), "y_map": sub["cy"].mean(),
                  "radius_map": sub["r"].mean(), "radius_px_finest": sub["r"].mean() / res0,
                  "max_significance": sub["significance"].max(),
                  "mean_alignment": sub["alignment"].mean()}
            if sub["lon"].notna().all():
                tr["lon"], tr["lat"] = sub["lon"].mean(), sub["lat"].mean()
            track_rows.append(tr)
        eddies = eddies.drop(columns=["cx", "cy", "r"])
        first = ["track", "level", "rank"]
        eddies = eddies[first + [c for c in eddies.columns if c not in first]]
        order = {k: i for i, k in enumerate(diags)}
        eddies = eddies.sort_values(["track", "level"], key=lambda c: c.map(order)
                                    if c.name == "level" else c).reset_index(drop=True)
    tracks_df = pd.DataFrame(track_rows)
    return PyramidDiagnosis(diagnoses=diags, levels=levels_df, eddy_tracks=tracks_df,
                            eddies=eddies)
