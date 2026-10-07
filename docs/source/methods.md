# Methods

This page describes what {func}`~shades2shapes.diagnose` and
{func}`~shades2shapes.discriminate` measure. Metric names are the keys of
`Diagnosis.metrics`. Lengths and areas are in pixels; with `pixel_size`, the main ones
are also given in physical units (keys ending in `_phys`).

## Preprocessing

One band is analysed: the band with the highest 5–95 % percentile spread (`auto`) or
the one requested. It is percentile-normalised (1–99 %) to [0, 1], so that images from
different sensors or channels are comparable. With `bright_ridges=False` the band is
inverted, for dark filaments.

## Filaments

A Sato ridge filter (σ = 1, 2, 3 px) enhances filament-like structures. A threshold of
the ridge response (global Otsu, or local with `mask_method="local"`) gives the filament
mask; objects of 30 px or less are removed.

| Metric | Meaning |
|---|---|
| `filament_fraction` | Fraction of the image covered by filaments |

## Network topology

The mask is skeletonised. Skeleton pixels with three or more neighbours are junctions,
pixels with one neighbour are endpoints, and the skeleton without junctions splits into
branches. Holes enclosed by the skeleton are the mesh cells.

| Metric | Meaning |
|---|---|
| `skeleton_length_px`, `skeleton_density` | Skeleton length, and per 1000 px² |
| `n_components` | Connected pieces of the network |
| `n_junctions`, `n_endpoints`, `junction_endpoint_ratio` | High ratio = closed mesh, low = open, branching filaments |
| `junctions_per_100px` | Junctions per 100 skeleton pixels |
| `n_branches`, `branch_length_mean`, `branch_length_median` | Branches of 3 px or more |
| `n_cells`, `cell_area_median`, `cell_area_mean` | Closed mesh cells of 10 px or more, not touching the border |

## Orientation

The structure tensor, smoothed at `tensor_sigma` (4 px), gives the local filament
orientation θ and the *coherence* $(\lambda_1 - \lambda_2)/(\lambda_1 + \lambda_2)$,
from 0 (isotropic) to 1 (parallel filaments).

| Metric | Meaning |
|---|---|
| `coherence_mean`, `coherence_filaments` | Mean coherence over the image and over filaments |
| `orientation_order` | $R = \lvert \sum_i w_i e^{2i\theta_i} \rvert / \sum_i w_i$ with coherence weights: 0 no preferred direction, 1 all parallel |
| `dominant_orientation_deg` | Mean axial direction, degrees counter-clockwise from horizontal |

## Eddies

Eddies are spiral or circular arrangements of filaments. For every candidate centre
$c$ and radius $R$, the *tangential alignment index* is

$$
A(c, R) = \frac{\sum_{i \in \text{disk}} w_i \cos 2(\theta_i - \varphi_i)}{\sum_{i \in \text{disk}} w_i},
$$

where $\varphi_i$ is the direction tangent to the circle through pixel $i$ and the
weights $w_i$ are coherence × ridge strength. $A = +1$ for circular filaments, 0 for
random ones and −1 for radial ones. All centres are evaluated at once by FFT
convolution, at 8 radii from 12 px to a third of the image size.

A detection must satisfy all of the following:

- it is strongly aligned: $A \geq 0.55$ (`eddy_min_alignment`);
- filaments surround the centre in at least 6 of 8 angular sectors
  (`eddy_min_coverage`), so a single curved filament is not counted;
- at least 80 % of the disk lies inside the image (`eddy_min_inside`);
- its significance $A\sqrt{n_\text{eff}} \geq 1.5$ (`eddy_min_significance`), where
  $n_\text{eff}$ is the weighted filament area in units of the orientation correlation
  area $(2\sigma)^2$;
- secondary eddies are at least half as significant as the dominant one
  (`eddy_min_relative_significance`).

The radius is the one that maximises the significance. Each eddy also gets a radial
profile of the alignment, annulus by annulus.

| Metric | Meaning |
|---|---|
| `n_eddies` | Number of eddies |
| `eddy_max_significance` | Significance of the dominant eddy |
| `eddy_dominant_radius`, `eddy_dominant_radius_rel` | Its radius, in px and relative to the smaller image side |
| `eddy_dominant_alignment` | Its alignment index |

## Texture

| Metric | Meaning |
|---|---|
| `glcm_<prop>_d<d>` | Haralick contrast, homogeneity, energy, correlation, dissimilarity and ASM of the grey-level co-occurrence matrix at distances 1, 3 and 5 px, averaged over 4 directions |
| `glcm_contrast_anisotropy` | Max / min contrast over directions |
| `glcm_correlation_decay` | Correlation at 5 px / at 1 px: close to 1 for smooth, large structures |
| `lbp_entropy` | Entropy of the uniform local binary pattern histogram (bits) |
| `lbp_flat_fraction`, `lbp_edge_fraction`, `lbp_nonuniform_fraction` | Fractions of flat, edge/line and irregular patterns |

## Complexity

| Metric | Meaning |
|---|---|
| `fractal_dimension` | Box-counting dimension of the mask: 1 for lines, up to 2 for space-filling networks |
| `lacunarity` | Gliding-box lacunarity (16 px boxes): 1 = homogeneous, larger = gappier |
| `spectral_slope` | Log-log slope of the radially averaged power spectrum; steeper = smoother |

## Discrimination

Each image is diagnosed and cut into overlapping tiles (64 px, 50 % overlap). On each
tile, 13 scale-robust features are computed ({data}`~shades2shapes.discriminate.TILE_FEATURES`). The
ridge response, mask and orientation field come from the whole image, so tiles have no
edge artefacts.

Ranking
: Features are ranked by *separability* = 2 × min over class pairs of |AUC − 0.5|,
  which is 0 for no separation and 1 for perfect. Being the worst pair, it rewards
  features that separate every class. Also reported: η², Kruskal–Wallis H and p and,
  for two classes, Cohen's d.

Selected set
: A greedy pick of top-ranked features whose mutual |Spearman r| < 0.8.

Classifier
: Standardisation followed by shrinkage LDA, cross-validated by leave-one-image-out
  when every class has at least 2 images, otherwise by 5-fold over tiles (flagged as
  optimistic).

## Caveats

Scale
: Without `pixel_size`, lengths and radii are in pixels. Compare images at the same
  resolution, or give the pixel size.

Colour is not used
: Only one band is analysed, after normalisation. This is deliberate: colour mostly
  reflects pigments and sensor processing, not spatial pattern.

Eddies are inferred from filament geometry, not flow
: For dynamics, combine with velocity fields (e.g. Okubo–Weiss or vorticity from
  altimetry). The rotation sense cannot be recovered from axial orientations.

Sample size
: Validate discrimination on several images per class. Tile-level accuracy from a single
  image per class overstates real performance, because tiles of one image are
  correlated.

Image quality
: Texture features also respond to blur, compression and resolution. Differences
  between classes may partly reflect how the images were acquired or processed.
