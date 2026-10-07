"""Low-level spatial-pattern metrics: filaments, network topology, orientation,
texture and multiscale complexity. All functions take a single-band float
image in [0, 1] (or masks derived from it)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
from skimage.feature import (graycomatrix, graycoprops, local_binary_pattern,
                             structure_tensor, structure_tensor_eigenvalues)

GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


# --------------------------------------------------------------------------- filaments
def ridge_response(img, sigmas=(1, 2, 3), bright_ridges=True):
    """Sato tubeness filter: high values on filament-like (ridge) structures.

    Parameters
    ----------
    img : numpy.ndarray
        Single-band image in [0, 1].
    sigmas : sequence of float
        Filter scales (px), roughly the half-widths of the filaments to detect.
    bright_ridges : bool
        Detect bright filaments on a dark background (else dark on bright).

    Returns
    -------
    numpy.ndarray
    """
    return filters.sato(img, sigmas=list(sigmas), black_ridges=not bright_ridges)


def filament_mask(ridges, method="otsu", min_size=30, block_size=51, offset=0.0):
    """Binary filament mask from a ridge response.

    Parameters
    ----------
    ridges : numpy.ndarray
        Ridge response (:func:`ridge_response`).
    method : {'otsu', 'local'}
        Global Otsu threshold, or local (adaptive) threshold, which keeps faint
        filaments better. The local threshold is also floored at half the Otsu value.
    min_size : int
        Connected objects of this size (px) or smaller are removed (0 keeps all).
    block_size : int
        Window size (px) of the local threshold (made odd).
    offset : float
        Offset subtracted from the local threshold.

    Returns
    -------
    numpy.ndarray of bool
    """
    if method == "local":
        bs = block_size if block_size % 2 else block_size + 1
        thr = filters.threshold_local(ridges, bs, offset=offset)
        mask = (ridges > thr) & (ridges > filters.threshold_otsu(ridges) * 0.5)
    else:
        mask = ridges > filters.threshold_otsu(ridges)
    if min_size:
        lab = measure.label(mask, connectivity=2)
        sizes = np.bincount(lab.ravel())
        keep = sizes > min_size
        keep[0] = False
        mask = keep[lab]
    return mask


# --------------------------------------------------------------------------- network
def skeleton_graph(mask):
    """Skeletonise a mask and classify skeleton pixels.

    Skeleton pixels with three or more 8-neighbours are junctions; pixels with one
    neighbour are endpoints. Branches are the skeleton with junctions removed.

    Parameters
    ----------
    mask : numpy.ndarray of bool
        Filament mask.

    Returns
    -------
    dict
        ``skeleton``, ``junctions``, ``endpoints`` (boolean arrays) and ``branches``
        (integer labels of the branches).
    """
    skel = morphology.skeletonize(mask)
    nb = ndi.convolve(skel.astype(np.int16), np.ones((3, 3), np.int16), mode="constant") - 1
    nb = nb * skel
    junction_px = (nb >= 3) & skel
    endpoint_px = (nb == 1) & skel
    branches = skel & ~ndi.binary_dilation(junction_px, structure=np.ones((3, 3)))
    branch_lab = measure.label(branches, connectivity=2)
    return dict(skeleton=skel, junctions=junction_px, endpoints=endpoint_px, branches=branch_lab)


def network_metrics(mask, min_branch=3, min_cell=10):
    """Topology of the filament network.

    Parameters
    ----------
    mask : numpy.ndarray of bool
        Filament mask.
    min_branch : int
        Branches shorter than this (px) are ignored in the branch statistics.
    min_cell : int
        Smallest area (px) of a closed mesh cell. Cells touching the image border are
        not counted.

    Returns
    -------
    dict
        ``filament_fraction``, ``skeleton_length_px``, ``skeleton_density`` (skeleton px
        per 1000 px²), ``n_components``, ``n_junctions``, ``n_endpoints``,
        ``junction_endpoint_ratio``, ``junctions_per_100px``, ``n_branches``,
        ``branch_length_mean``, ``branch_length_median``, ``n_cells``,
        ``cell_area_median`` and ``cell_area_mean``.
    """
    H, W = mask.shape
    g = skeleton_graph(mask)
    skel = g["skeleton"]
    n_skel = int(skel.sum())
    n_junc = int(measure.label(g["junctions"], connectivity=2).max())
    n_end = int(g["endpoints"].sum())
    blen = np.bincount(g["branches"].ravel())[1:]
    blen = blen[blen >= min_branch]
    holes = measure.label(~ndi.binary_dilation(skel), connectivity=1)
    cells = [r.area for r in measure.regionprops(holes)
             if r.area >= min_cell and r.bbox[0] > 0 and r.bbox[1] > 0
             and r.bbox[2] < H and r.bbox[3] < W]
    return {
        "filament_fraction": float(mask.mean()),
        "skeleton_length_px": n_skel,
        "skeleton_density": n_skel / (H * W) * 1000.0,  # px per 1000 px^2
        "n_components": int(measure.label(skel, connectivity=2).max()),
        "n_junctions": n_junc,
        "n_endpoints": n_end,
        "junction_endpoint_ratio": n_junc / max(n_end, 1),
        "junctions_per_100px": 100.0 * n_junc / max(n_skel, 1),
        "n_branches": int(len(blen)),
        "branch_length_mean": float(blen.mean()) if len(blen) else 0.0,
        "branch_length_median": float(np.median(blen)) if len(blen) else 0.0,
        "n_cells": int(len(cells)),
        "cell_area_median": float(np.median(cells)) if cells else 0.0,
        "cell_area_mean": float(np.mean(cells)) if cells else 0.0,
    }


# --------------------------------------------------------------------------- orientation
def orientation_field(img, sigma=4.0):
    """Local filament orientation and coherence from the structure tensor.

    Parameters
    ----------
    img : numpy.ndarray
        Single-band image.
    sigma : float
        Smoothing scale (px) of the tensor.

    Returns
    -------
    theta : numpy.ndarray
        Filament orientation (radians, axial: theta and theta + pi are equivalent).
    coherence : numpy.ndarray
        (l1 - l2) / (l1 + l2) in [0, 1]: 0 isotropic, 1 perfectly oriented.
    """
    Arr, Arc, Acc = structure_tensor(img, sigma=sigma, order="rc")
    l1, l2 = structure_tensor_eigenvalues([Arr, Arc, Acc])
    coherence = (l1 - l2) / (l1 + l2 + 1e-12)
    grad_theta = 0.5 * np.arctan2(2 * Arc, Acc - Arr)
    return grad_theta + np.pi / 2, coherence


def orientation_order(theta, weights=None):
    """Axial order parameter ``R = abs(weighted mean of exp(2i theta))``.

    Parameters
    ----------
    theta : numpy.ndarray
        Orientations (radians).
    weights : numpy.ndarray, optional
        Weights, e.g. coherence. Default: uniform.

    Returns
    -------
    float
        R in [0, 1]: 0 no preferred direction, 1 all filaments parallel.
    """
    w = np.ones_like(theta) if weights is None else weights
    return float(np.abs(np.sum(w * np.exp(2j * theta))) / (w.sum() + 1e-12))


def dominant_orientation(theta, weights=None):
    """Dominant (weighted mean axial) filament orientation.

    Parameters
    ----------
    theta : numpy.ndarray
        Orientations (radians).
    weights : numpy.ndarray, optional
        Weights, e.g. coherence. Default: uniform.

    Returns
    -------
    float
        Angle in degrees in [0, 180): 0 = horizontal, counted counter-clockwise as
        displayed (image rows pointing down).
    """
    w = np.ones_like(theta) if weights is None else weights
    z = np.sum(w * np.exp(2j * theta))
    return float(np.degrees(-0.5 * np.angle(z)) % 180)


# --------------------------------------------------------------------------- texture
def glcm_metrics(img, distances=(1, 3, 5), levels=64):
    """Haralick texture properties of the grey-level co-occurrence matrix.

    Properties are averaged over 4 directions (0, 45, 90, 135 degrees).

    Parameters
    ----------
    img : numpy.ndarray
        Single-band image in [0, 1].
    distances : sequence of int
        Pixel distances.
    levels : int
        Number of grey levels after quantisation.

    Returns
    -------
    dict
        ``glcm_<prop>_d<distance>`` for contrast, homogeneity, energy, correlation,
        dissimilarity and asm; ``glcm_contrast_anisotropy`` (max / min contrast over
        directions at the first distance) and ``glcm_correlation_decay`` (correlation
        at the last distance / first distance).
    """
    q = np.clip((img * (levels - 1)).round(), 0, levels - 1).astype(np.uint8)
    glcm = graycomatrix(q, list(distances), list(GLCM_ANGLES), levels=levels,
                        symmetric=True, normed=True)
    out = {}
    for prop in ("contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM"):
        v = graycoprops(glcm, prop)
        for i, d in enumerate(distances):
            out[f"glcm_{prop.lower()}_d{d}"] = float(v[i].mean())
    contrast_ang = graycoprops(glcm, "contrast")[0]
    out["glcm_contrast_anisotropy"] = float(contrast_ang.max() / max(contrast_ang.min(), 1e-12))
    c = graycoprops(glcm, "correlation")
    out["glcm_correlation_decay"] = float(c[-1].mean() / max(c[0].mean(), 1e-12))
    return out


def lbp_metrics(img, P=8, R=2):
    """Statistics of rotation-invariant uniform local binary patterns.

    Parameters
    ----------
    img : numpy.ndarray
        Single-band image in [0, 1].
    P : int
        Number of neighbours.
    R : float
        Radius (px).

    Returns
    -------
    dict
        ``lbp_entropy`` (bits), ``lbp_flat_fraction`` (flat bright pattern),
        ``lbp_edge_fraction`` (half-on patterns: edges and lines),
        ``lbp_nonuniform_fraction`` (irregular patterns) and ``lbp_histogram``.
    """
    lbp = local_binary_pattern((img * 255).astype(np.uint8), P, R, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(P + 3), density=True)
    nz = hist[hist > 0]
    return {
        "lbp_entropy": float(-(nz * np.log2(nz)).sum()),
        "lbp_flat_fraction": float(hist[P]),      # uniform pattern "all ones" (flat bright)
        "lbp_edge_fraction": float(hist[P // 2]),  # half-on patterns: edges / lines
        "lbp_nonuniform_fraction": float(hist[P + 1]),
        "lbp_histogram": hist.round(5).tolist(),
    }


# --------------------------------------------------------------------------- complexity
def box_counting_dimension(mask, sizes=None):
    """Box-counting fractal dimension of a mask.

    Parameters
    ----------
    mask : numpy.ndarray of bool
        Filament mask.
    sizes : sequence of int, optional
        Box sizes (px). Default: powers of two from 2 to about min(H, W) / 4.

    Returns
    -------
    float
        Slope of log(occupied boxes) against -log(box size); 0 if undefined.
    """
    H, W = mask.shape
    if sizes is None:
        kmax = max(2, int(np.log2(min(H, W))) - 2)
        sizes = 2 ** np.arange(1, kmax + 1)
    counts = []
    for s in sizes:
        h, w = H // s * s, W // s * s
        blocks = mask[:h, :w].reshape(h // s, s, w // s, s).any(axis=(1, 3))
        counts.append(max(int(blocks.sum()), 1))
    if len(set(counts)) == 1:
        return 0.0
    return float(-np.polyfit(np.log(sizes), np.log(counts), 1)[0])


def lacunarity(mask, box=16):
    """Gliding-box lacunarity of a mask.

    Parameters
    ----------
    mask : numpy.ndarray of bool
        Filament mask.
    box : int
        Box size (px).

    Returns
    -------
    float
        Second moment / squared first moment of the box mass: 1 = homogeneous,
        larger = gappier. NaN for an empty mask.
    """
    s = ndi.uniform_filter(mask.astype(float), size=box, mode="constant") * box * box
    s = s[box // 2: -box // 2 or None, box // 2: -box // 2 or None]
    m = s.mean()
    return float((s ** 2).mean() / (m * m)) if m > 0 else float("nan")


def power_spectrum_slope(img, kmin=4):
    """Log-log slope of the radially averaged power spectrum.

    Steeper (more negative) slopes mean smoother images dominated by large scales.

    Parameters
    ----------
    img : numpy.ndarray
        Single-band image.
    kmin : int
        Lowest wavenumber (in cycles per image) used in the fit.

    Returns
    -------
    float
        Slope, or NaN if fewer than 3 wavenumbers are available.
    """
    H, W = img.shape
    F = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))) ** 2
    fy, fx = np.indices(F.shape)
    r = np.hypot(fy - H // 2, fx - W // 2).astype(int)
    ps = np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    k = np.arange(len(ps))
    sel = (k >= kmin) & (k < min(H, W) // 2) & (ps > 0)
    if sel.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(k[sel]), np.log(ps[sel]), 1)[0])
