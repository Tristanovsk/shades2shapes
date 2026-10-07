"""Diagnosis of a single image: filaments, network, orientation, eddies, texture
and complexity, plus a diagnostic figure.

The entry point is :func:`diagnose`, which returns a :class:`Diagnosis`. Options are
grouped in :class:`DiagnosisConfig`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

from . import features as F
from .eddies import Eddy, detect_eddies
from .preprocessing import ArrayLike, prepare


@dataclass
class DiagnosisConfig:
    """Options of :func:`diagnose`.

    Any field can also be passed to :func:`diagnose` (and :func:`~shades2shapes.discriminate`)
    as a keyword argument.

    Attributes
    ----------
    channel : str
        Band to analyse: ``'auto'`` (band with the highest 5-95 % percentile spread),
        ``'r'``, ``'g'``, ``'b'``, ``'gray'`` or a band index given as a string.
    stretch : bool
        Percentile-normalise the band (1-99 %) to [0, 1], so that images are comparable.
    bright_ridges : bool
        Filaments are brighter than the background. Set to ``False`` for dark filaments.
    ridge_sigmas : tuple of float
        Scales (px) of the Sato ridge filter.
    mask_method : {'otsu', 'local'}
        Threshold of the ridge response: global Otsu, or local (adaptive), which keeps
        faint filaments better.
    min_object_size : int
        Connected filament objects of this size (px) or smaller are removed.
    tensor_sigma : float
        Smoothing scale (px) of the structure tensor, i.e. of the orientation field.
    glcm_distances : tuple of int
        Pixel distances of the grey-level co-occurrence matrices.
    glcm_levels : int
        Number of grey levels used for the GLCM.
    lbp_points : int
        Number of neighbours of the local binary patterns.
    lbp_radius : int
        Radius (px) of the local binary patterns.
    detect_eddies : bool
        Run eddy detection (the slowest step).
    eddy_radii : tuple of int, optional
        Candidate eddy radii (px). Default: 8 radii from 12 px to a third of the image size.
    eddy_min_alignment : float
        Minimum tangential alignment index of an eddy (0-1).
    eddy_min_significance : float
        Minimum significance, alignment x sqrt(n_eff).
    eddy_min_coverage : float
        Minimum fraction of the 8 angular sectors around the centre that contain filaments.
    eddy_min_relative_significance : float
        Secondary eddies must reach this fraction of the dominant eddy's significance.
    eddy_min_inside : float
        Minimum fraction of the eddy disk lying inside the image.
    pixel_size : float, optional
        Physical size of a pixel. When given, lengths are also reported in `unit`
        (metrics ending in ``_phys``).
    unit : str
        Unit of `pixel_size`, used in reports.
    """

    channel: str = "auto"            # 'auto', 'r', 'g', 'b', 'gray'
    stretch: bool = True             # 1-99 percentile normalisation
    bright_ridges: bool = True       # filaments brighter than background
    ridge_sigmas: tuple = (1, 2, 3)
    mask_method: str = "otsu"        # 'otsu' or 'local'
    min_object_size: int = 30
    tensor_sigma: float = 4.0
    glcm_distances: tuple = (1, 3, 5)
    glcm_levels: int = 64
    lbp_points: int = 8
    lbp_radius: int = 2
    detect_eddies: bool = True
    eddy_radii: Optional[tuple] = None
    eddy_min_alignment: float = 0.55
    eddy_min_significance: float = 1.5
    eddy_min_coverage: float = 0.75
    eddy_min_relative_significance: float = 0.5
    eddy_min_inside: float = 0.8
    pixel_size: Optional[float] = None   # physical size of a pixel
    unit: str = "km"


@dataclass
class Diagnosis:
    """Result of :func:`diagnose` for one image.

    Attributes
    ----------
    source : str
        File path of the image, or ``'<array>'``.
    channel : str
        Band actually analysed (``'R'``, ``'G'``, ``'B'``, ``'gray'``...).
    shape : tuple of int
        Image size ``(H, W)`` in pixels.
    metrics : dict
        All metrics, by name (see :doc:`/methods`). Lengths are in pixels unless the
        name ends in ``_phys``.
    eddies : list of Eddy
        Detected eddies, most significant first.
    config : DiagnosisConfig
        Options used.
    band : numpy.ndarray
        Analysed band, normalised to [0, 1] (inverted when ``bright_ridges=False``).
    rgb : numpy.ndarray or None
        Original colour image, if any.
    ridges : numpy.ndarray
        Sato ridge response.
    mask : numpy.ndarray of bool
        Filament mask.
    graph : dict
        Skeleton graph: boolean arrays ``skeleton``, ``junctions``, ``endpoints`` and the
        labelled ``branches`` (see :func:`~shades2shapes.features.skeleton_graph`).
    theta : numpy.ndarray
        Local filament orientation (radians).
    coherence : numpy.ndarray
        Local orientation coherence in [0, 1] (0 = isotropic, 1 = parallel filaments).
    """

    source: str
    channel: str
    shape: tuple
    metrics: dict
    eddies: list
    config: DiagnosisConfig
    # arrays kept for plotting / further analysis (not serialised)
    band: np.ndarray = field(repr=False, default=None)
    rgb: Optional[np.ndarray] = field(repr=False, default=None)
    ridges: np.ndarray = field(repr=False, default=None)
    mask: np.ndarray = field(repr=False, default=None)
    graph: dict = field(repr=False, default=None)
    theta: np.ndarray = field(repr=False, default=None)
    coherence: np.ndarray = field(repr=False, default=None)

    # ------------------------------------------------------------------ export
    def to_dict(self):
        """Serialisable summary: source, channel, shape, metrics, eddies and config.

        The arrays (band, mask, orientation field...) are not included.
        """
        cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(self.config).items()}
        return {"source": self.source, "channel": self.channel, "shape": list(self.shape),
                "metrics": self.metrics, "eddies": [e.to_dict() for e in self.eddies],
                "config": cfg}

    def to_json(self, path=None, indent=2):
        """JSON version of :meth:`to_dict`.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            If given, the JSON is also written to this file.
        indent : int
            JSON indentation.

        Returns
        -------
        str
            The JSON text.
        """
        txt = json.dumps(self.to_dict(), indent=indent, default=float)
        if path:
            Path(path).write_text(txt)
        return txt

    def scalar_metrics(self):
        """Flat dict of scalar metrics (no histograms), handy for tables."""
        return {k: v for k, v in self.metrics.items() if np.isscalar(v) or v is None}

    def summary(self):
        """Human-readable report of the main metrics, grouped by theme.

        Returns
        -------
        str
        """
        m = self.metrics
        H, W = self.shape
        L = []
        L.append(f"shades2shapes diagnosis: {self.source}")
        L.append(f"  image {W}x{H} px, channel {self.channel}")
        L.append("Filaments")
        L.append(f"  filament fraction       {m['filament_fraction']:.3f}")
        L.append(f"  skeleton density        {m['skeleton_density']:.1f} px / 1000 px^2")
        L.append("Network")
        L.append(f"  junctions / endpoints   {m['n_junctions']} / {m['n_endpoints']}"
                 f"  (ratio {m['junction_endpoint_ratio']:.2f})")
        L.append(f"  junctions per 100 px    {m['junctions_per_100px']:.2f}")
        L.append(f"  branches                {m['n_branches']}  (mean {m['branch_length_mean']:.1f},"
                 f" median {m['branch_length_median']:.1f} px)")
        L.append(f"  closed mesh cells       {m['n_cells']}  (median area {m['cell_area_median']:.0f} px^2)")
        L.append("Orientation")
        L.append(f"  coherence (all / fil.)  {m['coherence_mean']:.3f} / {m['coherence_filaments']:.3f}")
        L.append(f"  orientation order R     {m['orientation_order']:.3f}"
                 f"  (dominant {m['dominant_orientation_deg']:.0f} deg)")
        L.append("Eddies")
        if self.eddies:
            for i, e in enumerate(self.eddies, 1):
                L.append(f"  #{i}: centre ({e.x:.0f}, {e.y:.0f}) px, radius {e.radius:.0f} px,"
                         f" alignment {e.alignment:.2f}, significance {e.significance:.2f}")
        else:
            L.append("  none detected")
        L.append("Texture")
        d = self.config.glcm_distances
        L.append(f"  GLCM contrast d{d[0]}/d{d[-1]}     {m[f'glcm_contrast_d{d[0]}']:.2f} / {m[f'glcm_contrast_d{d[-1]}']:.2f}")
        L.append(f"  GLCM correlation d{d[0]}/d{d[-1]}  {m[f'glcm_correlation_d{d[0]}']:.3f} / {m[f'glcm_correlation_d{d[-1]}']:.3f}"
                 f"  (decay {m['glcm_correlation_decay']:.3f})")
        L.append(f"  GLCM homogeneity d{d[0]}    {m[f'glcm_homogeneity_d{d[0]}']:.3f}")
        L.append(f"  LBP entropy             {m['lbp_entropy']:.3f} bits"
                 f"  (non-uniform fraction {m['lbp_nonuniform_fraction']:.3f})")
        L.append("Complexity")
        L.append(f"  fractal dimension       {m['fractal_dimension']:.3f}")
        L.append(f"  lacunarity (16 px)      {m['lacunarity']:.3f}")
        L.append(f"  power spectrum slope    {m['spectral_slope']:.2f}")
        if self.config.pixel_size:
            u = self.config.unit
            L.append(f"Physical units (pixel = {self.config.pixel_size} {u})")
            L.append(f"  mean branch length      {m['branch_length_mean_phys']:.3g} {u}")
            L.append(f"  skeleton density        {m['skeleton_density_phys']:.3g} {u}/{u}^2")
            for i, e in enumerate(self.eddies, 1):
                L.append(f"  eddy #{i} radius          {e.radius * self.config.pixel_size:.3g} {u}")
        return "\n".join(L)

    # ------------------------------------------------------------------ figure
    def plot(self, path=None, dpi=110):
        """Diagnostic figure with six panels.

        The panels are: original image, ridge response, filament mask, skeleton with
        junctions, coherence, and orientation field. Eddies are drawn on the last two.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            If given, the figure is saved there and closed. Note that this switches
            matplotlib to the non-interactive ``Agg`` backend; in a notebook, call
            ``plot()`` without a path and save the returned figure instead.
        dpi : int
            Resolution of the saved figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib
        if path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        H, W = self.shape
        fig, ax = plt.subplots(2, 3, figsize=(15, 15 * H / W * 2 / 3 + 1.2))
        ax[0, 0].imshow(self.rgb if self.rgb is not None else self.band, cmap="gray")
        ax[0, 0].set_title(f"Original (analysed channel: {self.channel})")
        ax[0, 1].imshow(self.ridges, cmap="magma")
        ax[0, 1].set_title("Ridge response (Sato)")
        ax[0, 2].imshow(self.mask, cmap="gray")
        ax[0, 2].set_title(f"Filament mask ({self.metrics['filament_fraction']:.1%})")
        ax[1, 0].imshow(self.graph["skeleton"], cmap="gray_r")
        jy, jx = np.nonzero(self.graph["junctions"])
        ax[1, 0].plot(jx, jy, ".", color="#d62728", ms=2)
        ax[1, 0].set_title(f"Skeleton + junctions ({self.metrics['n_junctions']})")
        ax[1, 1].imshow(self.coherence, cmap="viridis", vmin=0, vmax=1)
        ax[1, 1].set_title(f"Coherence (mean {self.metrics['coherence_mean']:.2f}) + eddies")
        ax[1, 2].imshow(self.band, cmap="gray")
        step = max(8, int(min(H, W) / 25))
        Y, X = np.mgrid[step // 2:H:step, step // 2:W:step]
        t, c = self.theta[Y, X], self.coherence[Y, X]
        ax[1, 2].quiver(X, Y, np.cos(t) * c, -np.sin(t) * c, color="lime", pivot="mid",
                        headwidth=0, headlength=0, headaxislength=0, scale=W / step * 0.9,
                        width=0.003)
        ax[1, 2].set_title("Orientation field (length = coherence)")
        for i, e in enumerate(self.eddies, 1):
            for a in (ax[1, 1], ax[1, 2]):
                a.add_patch(plt.Circle((e.x, e.y), e.radius, fill=False, color="#ff7f0e", lw=1.5))
                a.plot(e.x, e.y, "+", color="#ff7f0e", ms=10, mew=2)
                a.text(e.x, e.y - e.radius - 3, f"#{i} ({e.alignment:.2f})", color="#ff7f0e",
                       ha="center", va="bottom", fontsize=8, weight="bold")
        for a in ax.ravel():
            a.set_xlim(0, W)
            a.set_ylim(H, 0)
            a.axis("off")
        fig.suptitle(Path(self.source).name, fontsize=11)
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
        return fig


def diagnose(source: ArrayLike, config: Optional[DiagnosisConfig] = None, **kwargs) -> Diagnosis:
    """Compute the spatial-pattern diagnosis of one image.

    The band is normalised, filaments are extracted with a ridge filter, then the
    network topology, orientation field, eddies, texture and complexity are measured.

    Parameters
    ----------
    source : str, pathlib.Path or numpy.ndarray
        Image file, or array of shape ``(H, W)`` or ``(H, W, 3)`` (an alpha channel is
        dropped). Integer arrays are scaled to [0, 1].
    config : DiagnosisConfig, optional
        Options. Defaults to ``DiagnosisConfig()``.
    **kwargs
        Override fields of `config`, e.g. ``channel="r"`` or ``pixel_size=0.3``.

    Returns
    -------
    Diagnosis

    Raises
    ------
    TypeError
        If a keyword is not a field of :class:`DiagnosisConfig`.

    Examples
    --------
    >>> import shades2shapes as s2s
    >>> d = s2s.diagnose("M_rubrum.png", channel="auto", mask_method="local")
    >>> print(d.summary())
    >>> d.metrics["filament_fraction"], len(d.eddies)
    """
    cfg = replace(config) if config is not None else DiagnosisConfig()
    for k, v in kwargs.items():
        if not hasattr(cfg, k):
            raise TypeError(f"Unknown option '{k}'")
        setattr(cfg, k, v)

    band, rgb, ch = prepare(source, cfg.channel, cfg.stretch)
    if not cfg.bright_ridges:
        band = 1.0 - band
    H, W = band.shape

    ridges = F.ridge_response(band, cfg.ridge_sigmas)
    mask = F.filament_mask(ridges, cfg.mask_method, cfg.min_object_size)
    graph = F.skeleton_graph(mask)
    theta, coh = F.orientation_field(band, cfg.tensor_sigma)

    m = {}
    m.update(F.network_metrics(mask))
    m["coherence_mean"] = float(coh.mean())
    m["coherence_filaments"] = float(coh[mask].mean()) if mask.any() else float("nan")
    m["orientation_order"] = F.orientation_order(theta, coh)
    m["dominant_orientation_deg"] = F.dominant_orientation(theta, coh)

    eddies: list[Eddy] = []
    if cfg.detect_eddies:
        eddies = detect_eddies(theta, coh, mask, ridges=ridges, radii=cfg.eddy_radii,
                               sigma=cfg.tensor_sigma, min_alignment=cfg.eddy_min_alignment,
                               min_coverage=cfg.eddy_min_coverage,
                               min_significance=cfg.eddy_min_significance,
                               min_relative_significance=cfg.eddy_min_relative_significance,
                               min_inside=cfg.eddy_min_inside)
    m["n_eddies"] = len(eddies)
    dom = eddies[0] if eddies else None
    m["eddy_max_significance"] = dom.significance if dom else 0.0
    m["eddy_dominant_radius"] = dom.radius if dom else 0.0
    m["eddy_dominant_alignment"] = dom.alignment if dom else 0.0
    m["eddy_dominant_radius_rel"] = dom.radius / min(H, W) if dom else 0.0

    m.update(F.glcm_metrics(band, cfg.glcm_distances, cfg.glcm_levels))
    m.update(F.lbp_metrics(band, cfg.lbp_points, cfg.lbp_radius))
    m["fractal_dimension"] = F.box_counting_dimension(mask)
    m["lacunarity"] = F.lacunarity(mask, 16)
    m["spectral_slope"] = F.power_spectrum_slope(band)

    if cfg.pixel_size:
        p = cfg.pixel_size
        m["branch_length_mean_phys"] = m["branch_length_mean"] * p
        m["branch_length_median_phys"] = m["branch_length_median"] * p
        m["cell_area_median_phys"] = m["cell_area_median"] * p * p
        m["skeleton_density_phys"] = m["skeleton_length_px"] * p / (H * W * p * p)
        m["eddy_dominant_radius_phys"] = m["eddy_dominant_radius"] * p

    name = str(source) if isinstance(source, (str, Path)) else "<array>"
    return Diagnosis(source=name, channel=ch, shape=(H, W), metrics=m, eddies=eddies,
                     config=cfg, band=band, rgb=rgb, ridges=ridges, mask=mask, graph=graph,
                     theta=theta, coherence=coh)
