"""Detection of eddies (vortex-like spiral organisation of filaments).

For a candidate centre c and radius R, the *tangential alignment index* is the
coherence-weighted mean of cos 2(theta_filament - theta_tangential) over the
filament pixels within R of c:  +1 = circular, 0 = random, -1 = radial.

It is evaluated for every pixel at once with FFT convolutions, at several radii.
Candidates must (i) be strongly aligned, (ii) be surrounded by filaments in most
angular sectors (a single curved filament is not an eddy) and (iii) be
statistically significant: significance = alignment * sqrt(n_eff), where n_eff
is the weighted filament area in units of the orientation-field correlation
area. Centres are local maxima of significance; the radius is the scale that
maximises it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import fftconvolve
from skimage.feature import peak_local_max


@dataclass
class Eddy:
    """An eddy detected by :func:`detect_eddies`.

    Attributes
    ----------
    x : float
        Centre column (px).
    y : float
        Centre row (px).
    radius : float
        Radius (px) that maximises the significance.
    alignment : float
        Tangential alignment index within `radius`: +1 circular, 0 random, -1 radial.
    significance : float
        ``alignment * sqrt(n_eff)``, with n_eff the weighted filament area in units of
        the orientation correlation area.
    support : float
        Weighted filament coverage of the disk.
    coverage : float
        Fraction of the 8 angular sectors around the centre that contain filaments.
    profile : dict
        Alignment per annulus, ``{"r0-r1": value}`` (``None`` where too few filament pixels).
    """

    x: float
    y: float
    radius: float        # radius (px) that maximises the significance
    alignment: float     # tangential alignment index within `radius` (-1..1)
    significance: float  # alignment * sqrt(n_eff)
    support: float       # weighted filament coverage of the disk
    coverage: float      # fraction of angular sectors containing filaments
    profile: dict        # alignment per annulus {"r0-r1": value}

    def to_dict(self):
        """Fields as a plain dict."""
        return asdict(self)


def _disk(R, r_inner):
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    r = np.hypot(xx, yy)
    return yy, xx, ((r >= r_inner) & (r <= R)).astype(float)


def alignment_map(theta, weights, R, r_inner=3, n_sectors=8, sector_min=None):
    """Alignment index, weighted support and angular coverage for every centre.

    All centres are evaluated at once by FFT convolution with an annulus of radii
    `r_inner` to `R`.

    Parameters
    ----------
    theta : numpy.ndarray
        Filament orientation (radians).
    weights : numpy.ndarray
        Pixel weights, e.g. coherence x ridge strength.
    R : int
        Outer radius (px).
    r_inner : int
        Inner radius (px); the centre itself is excluded.
    n_sectors : int
        Number of angular sectors used for the coverage.
    sector_min : float, optional
        Weighted filament fraction a sector needs to count as occupied (default: a
        quarter of the image-mean weight).

    Returns
    -------
    alignment, weight_sum, coverage : numpy.ndarray
        Tangential alignment index, sum of weights in the annulus, and fraction of
        occupied sectors, for every centre.
    """
    yy, xx, disk = _disk(R, r_inner)
    ang = np.arctan2(yy, xx)
    # tangential direction phi = ang + pi/2  ->  e^{-2i phi} = -e^{-2i ang}
    k = -np.exp(-2j * ang) * disk  # symmetric under d -> -d, so conv == correlation
    num = fftconvolve(weights * np.exp(2j * theta), k, mode="same")
    den = fftconvolve(weights, disk, mode="same")
    align = np.real(num) / np.maximum(den, 1e-9)
    if sector_min is None:
        sector_min = 0.25 * float(weights.mean())
    # sectors of (pixel - centre); convolution flips the kernel, hence -yy, -xx
    sec = ((np.arctan2(-yy, -xx) + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
    coverage = np.zeros(weights.shape)
    for i in range(n_sectors):
        kd = disk * (sec == i)
        coverage += fftconvolve(weights, kd, mode="same") / max(kd.sum(), 1) >= sector_min
    return align, den, coverage / n_sectors


def radial_profile(theta, mask, x, y, edges):
    """Tangential alignment of filament pixels in annuli around ``(x, y)``.

    Parameters
    ----------
    theta : numpy.ndarray
        Filament orientation (radians).
    mask : numpy.ndarray of bool
        Filament mask.
    x, y : float
        Centre (px).
    edges : sequence of float
        Annulus edges (px).

    Returns
    -------
    dict
        ``{"r0-r1": alignment}``; ``None`` for annuli with 20 filament pixels or fewer.
    """
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - y, xx - x)
    a = np.cos(2 * (theta - (np.arctan2(yy - y, xx - x) + np.pi / 2)))
    out = {}
    for r0, r1 in zip(edges[:-1], edges[1:]):
        sel = mask & (r >= r0) & (r < r1)
        out[f"{int(r0)}-{int(r1)}"] = round(float(a[sel].mean()), 4) if sel.sum() > 20 else None
    return out


def detect_eddies(theta, coherence, mask, ridges=None, radii=None, sigma=4.0,
                  min_alignment=0.5, min_coverage=0.75, min_significance=1.5,
                  min_relative_significance=0.5, min_inside=0.8, max_eddies=10):
    """Find eddies from the filament orientation field.

    Parameters
    ----------
    theta, coherence : numpy.ndarray
        Filament orientation and coherence
        (:func:`~shades2shapes.features.orientation_field`).
    mask : numpy.ndarray of bool
        Filament mask, used for the radial profiles.
    ridges : numpy.ndarray, optional
        Ridge response (:func:`~shades2shapes.features.ridge_response`). When given,
        weights are coherence x ridge strength, which keeps faint filaments (e.g. inner
        spiral arms) that a binary mask misses. Otherwise coherence x mask.
    radii : sequence of int, optional
        Candidate radii (px). Default: geometric series from 12 px to a third of the
        image size.
    sigma : float
        Structure-tensor scale used for the orientation field; sets the correlation
        area ``(2 sigma)**2`` used in n_eff.
    min_alignment : float
        Minimum tangential alignment index (0-1).
    min_coverage : float
        Minimum fraction of the 8 angular sectors containing filaments.
    min_significance : float
        Minimum ``alignment * sqrt(n_eff)``.
    min_relative_significance : float
        Secondary eddies must reach this fraction of the dominant eddy's significance
        (0 disables).
    min_inside : float
        Minimum fraction of the disk lying inside the image.
    max_eddies : int
        Maximum number of eddies returned.

    Returns
    -------
    list of Eddy
        Non-overlapping eddies, most significant first.
    """
    H, W = mask.shape
    if radii is None:
        rmax = max(16, int(min(H, W) / 3))
        radii = np.unique(np.geomspace(12, rmax, 8).astype(int))
    radii = [int(r) for r in radii]
    if ridges is not None:
        rn = np.clip(ridges / (np.percentile(ridges, 99) + 1e-12), 0, 1)
        weights = coherence * rn
    else:
        weights = coherence * mask
    corr_area = (2 * sigma) ** 2
    best_sig = np.full((H, W), -np.inf)
    best = {k: np.zeros((H, W)) for k in ("R", "align", "den", "cov")}
    for R in radii:
        align, den, cov = alignment_map(theta, weights, R)
        sig = align * np.sqrt(np.maximum(den, 0) / corr_area)
        _, _, disk = _disk(R, 3)
        inside = fftconvolve(np.ones((H, W)), disk, mode="same") / disk.sum()
        ok = (align >= min_alignment) & (cov >= min_coverage) & (inside >= min_inside - 1e-6)
        sig = np.where(ok, sig, -np.inf)
        upd = sig > best_sig
        best_sig[upd] = sig[upd]
        for k, v in (("R", R), ("align", align), ("den", den), ("cov", cov)):
            best[k][upd] = v if np.isscalar(v) else v[upd]
    sig_f = np.where(np.isfinite(best_sig), best_sig, 0.0)
    peaks = peak_local_max(sig_f, min_distance=max(3, min(radii) // 2),
                           threshold_abs=min_significance, exclude_border=False)
    cands = sorted(((sig_f[r, c], c, r) for r, c in peaks), reverse=True)
    if cands and min_relative_significance > 0:
        cands = [cd for cd in cands if cd[0] >= min_relative_significance * cands[0][0]]
    accepted = []
    for s, x, y in cands:
        R = best["R"][y, x]
        if any(np.hypot(x - e.x, y - e.y) < max(R, e.radius) for e in accepted):
            continue
        edges = np.unique(np.r_[3, np.linspace(0, 1.5 * R, 6)[1:]].astype(int))
        _, _, disk = _disk(int(R), 3)
        accepted.append(Eddy(
            x=float(x), y=float(y), radius=float(R),
            alignment=round(float(best["align"][y, x]), 4),
            significance=round(float(s), 3),
            support=round(float(best["den"][y, x] / disk.sum()), 4),
            coverage=float(best["cov"][y, x]),
            profile=radial_profile(theta, mask, x, y, edges)))
        if len(accepted) >= max_eddies:
            break
    return accepted
