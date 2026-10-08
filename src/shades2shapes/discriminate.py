"""Discrimination of two or more images (or groups of images) from their
spatial-pattern diagnoses.

Each image is diagnosed, then cut into tiles on which a set of scale-robust
features is computed. Features are ranked by how well they separate the
classes (Cohen's d and AUC for two classes; eta-squared, Kruskal-Wallis and
pairwise AUC for more), a compact non-redundant subset is selected, and a
linear discriminant classifier is cross-validated.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from skimage import measure

from . import features as F
from .diagnose import Diagnosis, DiagnosisConfig, diagnose

#: Features computed on each tile by :func:`~shades2shapes.tile_features` (name -> description).
TILE_FEATURES = {
    "filament_fraction": "Filament area fraction",
    "skeleton_density": "Skeleton density (px / 1000 px^2)",
    "junctions_per_100px": "Junctions per 100 skeleton px",
    "junction_endpoint_ratio": "Junction / endpoint ratio",
    "coherence": "Structure-tensor coherence",
    "orientation_order": "Orientation order R",
    "glcm_contrast": "GLCM contrast (d=1)",
    "glcm_homogeneity": "GLCM homogeneity (d=1)",
    "glcm_correlation_decay": "GLCM correlation decay (d4/d1)",
    "lbp_entropy": "LBP entropy",
    "lbp_nonuniform_fraction": "LBP non-uniform pattern fraction",
    "fractal_dimension": "Fractal dimension (box counting)",
    "lacunarity": "Lacunarity (8 px box)",
}


# --------------------------------------------------------------------------- tiles
def tile_features(diag: Diagnosis, tile: int = 64, overlap: float = 0.5,
                  min_std: float = 0.02) -> pd.DataFrame:
    """Compute :data:`~shades2shapes.discriminate.TILE_FEATURES` on overlapping tiles of a diagnosed image.

    The ridge response, mask and orientation field come from the whole image (no
    tile-edge artefacts); texture is computed on the tile itself.

    Parameters
    ----------
    diag : Diagnosis
        Diagnosed image.
    tile : int
        Tile size (px).
    overlap : float
        Overlap between neighbouring tiles, as a fraction of `tile` (0 to < 1).
    min_std : float
        Tiles whose normalised band has a lower standard deviation (e.g. uniform
        background or masked areas) are skipped.

    Returns
    -------
    pandas.DataFrame
        One row per tile: upper-left corner ``x``, ``y`` and one column per feature.
    """
    band, mask, theta, coh = diag.band, diag.mask, diag.theta, diag.coherence
    H, W = band.shape
    step = max(1, int(tile * (1 - overlap)))
    rows = []
    for y in range(0, H - tile + 1, step):
        for x in range(0, W - tile + 1, step):
            sl = np.s_[y:y + tile, x:x + tile]
            t = band[sl]
            if t.std() < min_std:
                continue
            m = mask[sl]
            g = F.skeleton_graph(m)
            n_sk = int(g["skeleton"].sum())
            n_j = int(measure.label(g["junctions"], connectivity=2).max())
            n_e = int(g["endpoints"].sum())
            gl = F.glcm_metrics(t, distances=(1, 4), levels=32)
            lb = F.lbp_metrics(t)
            c = coh[sl]
            rows.append({
                "x": x, "y": y,
                "filament_fraction": float(m.mean()),
                "skeleton_density": n_sk / tile ** 2 * 1000,
                "junctions_per_100px": 100 * n_j / max(n_sk, 1),
                "junction_endpoint_ratio": n_j / max(n_e, 1),
                "coherence": float(c.mean()),
                "orientation_order": F.orientation_order(theta[sl], c),
                "glcm_contrast": gl["glcm_contrast_d1"],
                "glcm_homogeneity": gl["glcm_homogeneity_d1"],
                "glcm_correlation_decay": gl["glcm_correlation_decay"],
                "lbp_entropy": lb["lbp_entropy"],
                "lbp_nonuniform_fraction": lb["lbp_nonuniform_fraction"],
                "fractal_dimension": F.box_counting_dimension(m, sizes=[2, 4, 8, 16]),
                "lacunarity": F.lacunarity(m, 8) if m.any() else np.nan,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- statistics
def _auc(a, b):
    """P(b > a) + 0.5 P(b == a), via Mann-Whitney U."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    u = stats.mannwhitneyu(b, a, alternative="two-sided").statistic
    return float(u / (len(a) * len(b)))


def _cohen_d(a, b):
    s = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(b) - np.mean(a)) / s) if s > 0 else 0.0


def rank_features(df: pd.DataFrame, label_col: str = "label", features=None) -> pd.DataFrame:
    """Separability of each feature between classes.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per sample (tile), with a class column and feature columns.
    label_col : str
        Name of the class column.
    features : list of str, optional
        Features to rank. Default: the
        :data:`~shades2shapes.discriminate.TILE_FEATURES` present in `df`.

    Returns
    -------
    pandas.DataFrame
        One row per feature, best first, with columns:

        - ``separability``: 2 x min over class pairs of ``abs(AUC - 0.5)`` (0 = none,
          1 = perfect). Being the worst pair, it rewards features that separate
          *every* class.
        - ``mean_pairwise_auc``: mean over class pairs of max(AUC, 1 - AUC).
        - ``eta2``: fraction of the variance explained by the classes.
        - ``kruskal_H``, ``kruskal_p``: Kruskal-Wallis test.
        - ``mean[<class>]``, ``sd[<class>]``: per-class statistics.
        - for two classes only, ``cohen_d`` and ``auc`` (oriented: > 0.5 means higher
          in the second class).

        Features that are constant, or have fewer than 2 values in a class, are left out.
    """
    features = features or [f for f in TILE_FEATURES if f in df.columns]
    classes = list(pd.unique(df[label_col]))
    out = []
    for f in features:
        groups = [df.loc[df[label_col] == c, f].dropna().to_numpy() for c in classes]
        if any(len(g) < 2 for g in groups) or np.nanstd(df[f]) == 0:
            continue
        pair_auc = {}
        for (i, a), (j, b) in combinations(enumerate(groups), 2):
            pair_auc[(classes[i], classes[j])] = _auc(a, b)
        sep = [abs(v - 0.5) * 2 for v in pair_auc.values()]
        allv = np.concatenate(groups)
        ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
        ss_t = ((allv - allv.mean()) ** 2).sum()
        try:
            H, p = stats.kruskal(*groups)
        except ValueError:
            H, p = np.nan, np.nan
        row = {"feature": f, "description": TILE_FEATURES.get(f, f),
               "separability": float(np.min(sep)),
               "mean_pairwise_auc": float(np.mean([max(v, 1 - v) for v in pair_auc.values()])),
               "eta2": float(ss_b / ss_t) if ss_t > 0 else 0.0,
               "kruskal_H": float(H), "kruskal_p": float(p)}
        for c, g in zip(classes, groups):
            row[f"mean[{c}]"] = float(g.mean())
            row[f"sd[{c}]"] = float(g.std(ddof=1))
        if len(classes) == 2:
            row["cohen_d"] = _cohen_d(*groups)
            row["auc"] = pair_auc[(classes[0], classes[1])]
        out.append(row)
    res = pd.DataFrame(out)
    if len(res):
        res = res.sort_values(["separability", "eta2"], ascending=False).reset_index(drop=True)
    return res


def select_features(df: pd.DataFrame, ranking: pd.DataFrame, k: int = 4,
                    max_corr: float = 0.8) -> list:
    """Greedy pick of a compact, non-redundant feature set.

    Features are taken in ranking order and kept if their absolute Spearman
    correlation with every feature already chosen is below `max_corr`.

    Parameters
    ----------
    df : pandas.DataFrame
        Samples (tiles) with feature columns.
    ranking : pandas.DataFrame
        Output of :func:`~shades2shapes.rank_features`.
    k : int
        Number of features to select.
    max_corr : float
        Maximum allowed absolute Spearman correlation between selected features.

    Returns
    -------
    list of str
    """
    chosen = []
    for f in ranking["feature"]:
        if all(abs(stats.spearmanr(df[f], df[c], nan_policy="omit")[0]) < max_corr
               for c in chosen):
            chosen.append(f)
        if len(chosen) == k:
            break
    return chosen


def cross_validate(df: pd.DataFrame, features: Sequence[str], label_col="label",
                   group_col="image"):
    """Cross-validated accuracy of a linear discriminant classifier.

    The model is standardisation followed by shrinkage LDA. If every class has at
    least 2 images, whole images are held out (leave-one-image-out, an honest
    estimate). Otherwise tiles are cross-validated with up to 5 stratified folds,
    which is optimistic because tiles of one image are correlated.

    Parameters
    ----------
    df : pandas.DataFrame
        Samples (tiles) with feature, class and image columns.
    features : sequence of str
        Features used by the classifier.
    label_col, group_col : str
        Names of the class and image columns.

    Returns
    -------
    dict
        ``scheme`` (str), ``accuracy`` (float), ``confusion`` (DataFrame, true classes
        in rows), ``model`` (scikit-learn pipeline refitted on all samples) and
        ``features`` (list).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = df.dropna(subset=list(features))
    X, y = d[list(features)].to_numpy(), d[label_col].astype(str).to_numpy()
    imgs_per_class = d.groupby(label_col)[group_col].nunique()
    model = make_pipeline(StandardScaler(),
                          LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    if (imgs_per_class >= 2).all():
        scheme = "leave-one-image-out"
        pred = cross_val_predict(model, X, y, groups=d[group_col], cv=LeaveOneGroupOut())
    else:
        n_splits = int(min(5, pd.Series(y).value_counts().min()))
        scheme = f"{n_splits}-fold over tiles (optimistic: single image per class)"
        pred = cross_val_predict(model, X, y, cv=StratifiedKFold(n_splits, shuffle=True,
                                                                 random_state=0))
    labels = sorted(set(y))
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels)
    model.fit(X, y)
    return {"scheme": scheme, "accuracy": float(accuracy_score(y, pred)),
            "confusion": cm, "model": model, "features": list(features)}


# --------------------------------------------------------------------------- result
@dataclass
class DiscriminationResult:
    """Result of :func:`discriminate`.

    Attributes
    ----------
    diagnoses : list of Diagnosis
        Diagnosis of each image.
    labels : list of str
        Class of each image.
    tiles : pandas.DataFrame
        Tile features (:func:`tile_features`) of all images, with ``image`` and
        ``label`` columns.
    ranking : pandas.DataFrame
        Features ranked by separability (:func:`rank_features`).
    image_table : pandas.DataFrame
        Main whole-image metrics, one column per image, plus a ``label`` row.
    selected : list of str
        Recommended non-redundant feature set
        (:func:`~shades2shapes.discriminate.select_features`).
    cv_all : dict
        Cross-validation with all features
        (:func:`~shades2shapes.discriminate.cross_validate`).
    cv_selected : dict
        Cross-validation with the selected set. ``cv_selected["model"]`` can classify
        new tiles.
    notes : list of str
        Warnings about the comparison (sample size, channels, image sizes).
    """

    diagnoses: list
    labels: list
    tiles: pd.DataFrame
    ranking: pd.DataFrame
    image_table: pd.DataFrame
    selected: list
    cv_all: dict
    cv_selected: dict
    notes: list = field(default_factory=list)

    def summary(self, top: int = 8) -> str:
        """Human-readable report: images, feature ranking, accuracy, whole-image table, notes.

        Parameters
        ----------
        top : int
            Number of features shown in the ranking.

        Returns
        -------
        str
        """
        L = ["shades2shapes discrimination",
             f"  {len(self.diagnoses)} image(s), {len(set(self.labels))} class(es), "
             f"{len(self.tiles)} tiles"]
        for d, lab, name in zip(self.diagnoses, self.labels, self.image_table.columns):
            n = int((self.tiles['image'] == name).sum())
            L.append(f"  - [{lab}] {name} (channel {d.channel}, {n} tiles)")
        L.append("\nFeature ranking (tile level; separability 0 = none, 1 = perfect)")
        cols = ["feature", "separability", "mean_pairwise_auc", "eta2"]
        if "cohen_d" in self.ranking:
            cols += ["cohen_d", "auc"]
        L.append(self.ranking[cols].head(top).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        L.append(f"\nRecommended feature set: {', '.join(self.selected)}")
        L.append(f"LDA accuracy, all features: {self.cv_all['accuracy']:.1%}  [{self.cv_all['scheme']}]")
        L.append(f"LDA accuracy, selected set: {self.cv_selected['accuracy']:.1%}")
        L.append("\nWhole-image comparison")
        fmt = self.image_table.apply(lambda col: col.map(
            lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)))
        L.append(fmt.to_string())
        if self.notes:
            L.append("\nNotes")
            L += [f"  * {n}" for n in self.notes]
        return "\n".join(L)

    def to_dict(self):
        """Serialisable summary (the fitted models are left out)."""
        return {"images": [{"source": d.source, "label": lab, "channel": d.channel}
                           for d, lab in zip(self.diagnoses, self.labels)],
                "n_tiles": int(len(self.tiles)),
                "ranking": self.ranking.to_dict(orient="records"),
                "selected_features": self.selected,
                "cv_all_features": {k: v for k, v in self.cv_all.items() if k in ("scheme", "accuracy")}
                | {"confusion": self.cv_all["confusion"].to_dict()},
                "cv_selected_features": {k: v for k, v in self.cv_selected.items() if k in ("scheme", "accuracy")}
                | {"confusion": self.cv_selected["confusion"].to_dict()},
                "whole_image": self.image_table.to_dict(),
                "notes": self.notes}

    def save(self, outdir, plots: bool = True):
        """Write the results to a directory.

        Files: ``summary.txt``, ``feature_ranking.csv``, ``tile_features.csv``,
        ``whole_image_metrics.csv``, ``discrimination.json`` and, with `plots`,
        ``feature_distributions.png`` and ``projection.png``.

        Parameters
        ----------
        outdir : str or pathlib.Path
            Output directory, created if needed.
        plots : bool
            Also write the figures. This switches matplotlib to the ``Agg`` backend;
            in a notebook, use ``plots=False`` and save the figures returned by
            :meth:`plot_features` and :meth:`plot_projection`.

        Returns
        -------
        pathlib.Path
            The output directory.
        """
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        self.tiles.to_csv(out / "tile_features.csv", index=False)
        self.ranking.to_csv(out / "feature_ranking.csv", index=False)
        self.image_table.to_csv(out / "whole_image_metrics.csv")
        (out / "discrimination.json").write_text(json.dumps(self.to_dict(), indent=2, default=str))
        (out / "summary.txt").write_text(self.summary())
        if plots:
            self.plot_features(out / "feature_distributions.png")
            self.plot_projection(out / "projection.png")
        return out

    # ------------------------------------------------------------------ figures
    def plot_features(self, path=None, top: int = 8):
        """Box plots of the best-ranked tile features, per class.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            If given, the figure is saved there and closed (``Agg`` backend).
        top : int
            Number of features shown.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib
        if path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        feats = list(self.ranking["feature"].head(top))
        classes = list(dict.fromkeys(self.tiles["label"]))
        ncol = 4
        nrow = int(np.ceil(len(feats) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for ax, f in zip(axes.ravel(), feats):
            data = [self.tiles.loc[self.tiles["label"] == c, f].dropna() for c in classes]
            bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.55)
            ax.set_xticks(range(1, len(classes) + 1), classes, rotation=20, fontsize=8)
            sep = float(self.ranking.set_index("feature").loc[f, "separability"])
            ax.set_title(f"{TILE_FEATURES.get(f, f)}\nseparability {sep:.2f}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        for ax in axes.ravel()[len(feats):]:
            ax.axis("off")
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=110)
            plt.close(fig)
        return fig

    def plot_projection(self, path=None):
        """Scatter plot of the tiles in discriminant space.

        Tiles are projected on the first two LDA axes, or, for two classes, on the LDA
        axis and the first PCA axis.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            If given, the figure is saved there and closed (``Agg`` backend).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib
        if path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler

        feats = self.cv_all["features"]
        d = self.tiles.dropna(subset=feats)
        X = StandardScaler().fit_transform(d[feats])
        y = d["label"].astype(str).to_numpy()
        classes = list(dict.fromkeys(y))
        lda = LinearDiscriminantAnalysis().fit(X, y)
        Z = lda.transform(X)
        if Z.shape[1] == 1:
            Z = np.c_[Z, PCA(1).fit_transform(X)]
            xl, yl = "LDA axis 1", "PCA axis 1"
        else:
            xl, yl = "LDA axis 1", "LDA axis 2"
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for c in classes:
            s = y == c
            ax.scatter(Z[s, 0], Z[s, 1], s=14, alpha=0.6, label=c)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"Tiles in discriminant space (CV accuracy {self.cv_all['accuracy']:.0%})")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=110)
            plt.close(fig)
        return fig


# --------------------------------------------------------------------------- main API
IMAGE_TABLE_METRICS = [
    "filament_fraction", "skeleton_density", "junction_endpoint_ratio", "junctions_per_100px",
    "branch_length_median", "n_cells", "coherence_mean", "orientation_order",
    "n_eddies", "eddy_max_significance", "eddy_dominant_radius_rel", "eddy_dominant_alignment",
    "glcm_correlation_decay", "lbp_entropy", "lbp_nonuniform_fraction",
    "fractal_dimension", "lacunarity", "spectral_slope",
]


def discriminate(images: Sequence, labels: Optional[Sequence[str]] = None,
                 channels: Optional[Sequence[str]] = None, tile: int = 64,
                 overlap: float = 0.5, n_select: int = 4,
                 config: Optional[DiagnosisConfig] = None, **diag_kwargs) -> DiscriminationResult:
    """Discriminate two or more images, or groups of images.

    Each image is diagnosed and cut into overlapping tiles, on which
    :data:`~shades2shapes.discriminate.TILE_FEATURES` are computed. Features are ranked by separability, a compact
    non-redundant set is selected, and an LDA classifier is cross-validated with all
    features and with the selected set.

    Parameters
    ----------
    images : sequence of str, pathlib.Path, numpy.ndarray, xarray.DataArray, xarray.Dataset or Diagnosis
        Images to compare (any input of :func:`~shades2shapes.diagnose`). Already
        computed :class:`Diagnosis` objects are reused.
    labels : sequence of str, optional
        Class of each image (default: one class per image, named after the file). Give
        the same label to several images to discriminate groups.
    channels : sequence of str, optional
        Channel of each image (``'auto'``, ``'r'``, ``'g'``, ``'b'``, ``'gray'``, or a
        band name for georeferenced images).
        Ignored for Diagnosis inputs. Default: `config` / `diag_kwargs`, else ``'auto'``.
    tile : int
        Tile size (px).
    overlap : float
        Tile overlap fraction.
    n_select : int
        Size of the recommended non-redundant feature set.
    config : DiagnosisConfig, optional
        Diagnosis options.
    **diag_kwargs
        Override fields of `config` (see :class:`DiagnosisConfig`).

    Returns
    -------
    DiscriminationResult

    Raises
    ------
    ValueError
        With fewer than two images or two classes, or if `labels` has the wrong length.

    Examples
    --------
    >>> import shades2shapes as s2s
    >>> r = s2s.discriminate(["a1.png", "a2.png", "b1.png", "b2.png"],
    ...                      labels=["A", "A", "B", "B"])
    >>> print(r.summary())
    >>> r.cv_selected["accuracy"]          # leave-one-image-out
    >>> r.save("comparison/")
    """
    if len(images) < 2:
        raise ValueError("Need at least two images")
    names = []
    diags = []
    for i, im in enumerate(images):
        if isinstance(im, Diagnosis):
            d = im
        else:
            kw = dict(diag_kwargs)
            if channels is not None:
                kw["channel"] = channels[i]
            d = diagnose(im, config=config, **kw)
        diags.append(d)
        names.append(Path(d.source).name if not d.source.startswith("<") else f"image{i + 1}")
    if len(set(names)) < len(names):  # disambiguate duplicate file names
        names = [f"{i + 1}:{n}" for i, n in enumerate(names)]
    labels = list(labels) if labels is not None else names
    if len(labels) != len(diags):
        raise ValueError("labels must have one entry per image")
    if len(set(labels)) < 2:
        raise ValueError("Need at least two distinct classes")

    notes = []
    frames = []
    for d, n, lab in zip(diags, names, labels):
        t = tile_features(d, tile=tile, overlap=overlap)
        if len(t) < 5:
            notes.append(f"{n}: only {len(t)} tiles of {tile} px; consider a smaller --tile.")
        t.insert(0, "label", lab)
        t.insert(0, "image", n)
        frames.append(t)
    tiles = pd.concat(frames, ignore_index=True)

    ranking = rank_features(tiles)
    selected = select_features(tiles, ranking, k=n_select)
    feats_all = list(ranking["feature"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_all = cross_validate(tiles, feats_all)
        cv_sel = cross_validate(tiles, selected)
    if "optimistic" in cv_all["scheme"]:
        notes.append("Each class has a single image: tile-level accuracy is optimistic because "
                     "tiles of one image are correlated. Provide >= 2 images per class (same "
                     "label) for a leave-one-image-out estimate.")
    chans = {d.channel for d in diags}
    if len(chans) > 1:
        notes.append(f"Images were analysed on different channels ({', '.join(sorted(chans))}); "
                     "intensities are percentile-normalised, but colour itself is not compared.")
    shapes = {d.shape for d in diags}
    if len(shapes) > 1 and not any(d.config.pixel_size for d in diags):
        notes.append("Images differ in size and no pixel size was given: length-based metrics "
                     "(branch length, eddy radius) are in pixels and may not be comparable.")

    image_table = pd.DataFrame(
        {n: {k: d.metrics.get(k) for k in IMAGE_TABLE_METRICS} for d, n in zip(diags, names)})
    image_table.loc["label"] = labels

    return DiscriminationResult(diagnoses=diags, labels=labels, tiles=tiles, ranking=ranking,
                                image_table=image_table, selected=selected, cv_all=cv_all,
                                cv_selected=cv_sel, notes=notes)
