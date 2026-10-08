# shades2shapes

Turn the **shades** of an image into **shapes**: quantitative descriptors of the spatial
patterns formed by filaments, networks and eddies. Designed for ocean-colour and true-colour
images of phytoplankton or cyanobacteria blooms, it works on any image of bright (or dark)
filamentous structures.

Two tools:

| Command | What it does |
|---|---|
| `diagnose` | Full spatial diagnosis of one image: filaments, network topology, orientation, eddies, texture, complexity. Writes a summary, a JSON file and a diagnostic figure. |
| `discriminate` | Compares two or more images, or groups of images. Ranks the features that separate them, recommends a compact non-redundant feature set and cross-validates a classifier. |

## Installation

```bash
pip install .            # from the unzipped folder
pip install -e .[test]   # editable, with pytest
pip install .[geo]       # with xarray/rioxarray/zarr, for satellite images (GeoTIFF, NetCDF, Zarr)
```

Requires Python ≥ 3.9 with numpy, scipy, scikit-image, scikit-learn, pandas and matplotlib.

## Command line

```bash
# Diagnose one or more images (each independently)
shades2shapes diagnose bloom.png -o results/
shades2shapes diagnose bloom.png --channel r --pixel-size 0.03 --unit km

# Discriminate images: one class per image
shades2shapes discriminate eddy.png mesh.png -o comparison/

# Discriminate groups: repeat labels; channels can differ per image
shades2shapes discriminate a1.png a2.png b1.png b2.png \
    --labels calm,calm,turbulent,turbulent --channel g,g,r,r --save-diagnoses

# Georeferenced scenes (geo extra): pixel size from the CRS, maps saved as NetCDF
shades2shapes diagnose scene.tif --channel B8 --unit m --save-maps

# Same area at every level of a Zarr pyramid (eddies matched across levels)
shades2shapes pyramid scene.zarr --variable Rrs --channel 665 --mask mask \
    --transform log10 --bbox 612000,4790600,678000,4800600
```

`s2s` is a short alias for `shades2shapes`. Run `shades2shapes diagnose -h` or
`shades2shapes discriminate -h` for all options.

**Main options**

| Option | Default | Meaning |
|---|---|---|
| `--channel` | `auto` | Band to analyse: `auto` (highest contrast), `r`, `g`, `b`, `gray`, or a band name of a GeoTIFF/NetCDF file |
| `--dark-ridges` | off | Filaments are darker than the background |
| `--mask-method` | `otsu` | `local` keeps faint filaments better |
| `--tensor-sigma` | 4 | Smoothing scale of the orientation field (px) |
| `--pixel-size`, `--unit` | none, km | Also report lengths in physical units (default for georeferenced files: from the CRS) |
| `--save-maps` | off | Also write the maps (mask, ridges, orientation...) as NetCDF |
| `--no-eddies` | off | Skip eddy detection |
| `--tile`, `--overlap` | 64, 0.5 | Tile size (px) and overlap for discrimination |
| `--labels` | file names | Class of each image (`discriminate`) |
| `--n-select` | 4 | Size of the recommended feature set |

## Python API

```python
import shades2shapes as s2s

d = s2s.diagnose("bloom.png", channel="auto", mask_method="otsu")
print(d.summary())
d.plot("bloom_diagnosis.png")
d.to_json("bloom_diagnosis.json")
d.metrics            # dict of all metrics
d.eddies             # list of Eddy(x, y, radius, alignment, significance, ...)
d.mask, d.theta, d.coherence  # arrays for further analysis

r = s2s.discriminate(["a1.png", "a2.png", "b1.png"], labels=["A", "A", "B"])
print(r.summary())
r.ranking            # pandas DataFrame, best features first
r.selected           # recommended non-redundant features
r.cv_selected["accuracy"], r.cv_selected["confusion"]
r.cv_selected["model"].predict(new_tiles[r.selected])  # classify new tiles
r.save("comparison/")
```

Inputs can be file paths, NumPy arrays (H×W or H×W×3) or existing `Diagnosis` objects.

### Satellite images (xarray)

With the `geo` extra, inputs can also be georeferenced: an `xarray.DataArray` (e.g. read
with `rioxarray.open_rasterio`, bands first as is usual), an `xarray.Dataset`, or a
GeoTIFF/NetCDF file. Any number of bands is accepted and `channel` can be a band name.

```python
import rioxarray

da = rioxarray.open_rasterio("S2_scene.tif", masked=True)    # (band, y, x), UTM
da = da.rio.clip_box(minx, miny, maxx, maxy)                  # area of interest
d = s2s.diagnose(da, channel="B8", unit="m")
d.config.pixel_size        # 10.0, from the CRS
d.eddies[0].lon, d.eddies[0].lat, d.eddies[0].x_map, d.eddies[0].y_map
ds = d.to_xarray()         # georeferenced maps: mask, ridges, coherence, orientation...
ds.to_netcdf("S2_maps.nc"); ds.mask.rio.to_raster("S2_mask.tif")
```

The image is analysed north-up. In geographic coordinates (degrees), the pixel size is
not set: reproject first (`da.rio.reproject(da.rio.estimate_utm_crs())`). No-data
pixels (NaN: land, clouds) are not supported yet: crop to a valid area or fill them
(`shades2shapes.geo.fill_nodata`).

### Pyramids (several resolutions)

`diagnose_pyramid` diagnoses the same area at every level of an image pyramid (a Zarr
store with groups `0, 1, 2...`, a list of levels, or one image coarsened by 2, 4...),
fills flagged/NaN pixels, skips levels that are too small, and matches the eddies found
at several levels:

```python
p = s2s.diagnose_pyramid("S2_GRS.zarr", variable="Rrs", channel="665", mask="mask",
                         transform="log10", bbox=(612000, 4790600, 678000, 4800600))
print(p.summary())
p.eddy_tracks        # structures found at several resolutions
p.metrics_table()    # metrics per level
p.save("pyramid/")
```

The notebook `examples/s2_pyramid_example.ipynb` runs this on a Sentinel-2 GRS scene of
the Rhône delta (tile 31TFJ, 23 May 2025), from 20 to 160 m. The scene is not included:
point `S2_ZARR` to your own Zarr product.

The notebook `examples/uroglena_planetscope_example.ipynb` analyses the bloom of the
golden alga *Uroglena* in Lake Geneva on 6 September 2021, from PlanetScope SuperDove
images (8 bands, 3 m): TOA reflectance, water mask, averaging to 12 m and pyramid
analysis (12–96 m) of the central and western Grand Lac. Among other structures, it
finds a gyre of 3.2 km radius in the western Grand Lac at three resolutions. The images
are not included (Planet licence): point `PLANET_DIR` to your own order.

## What is measured

**Filaments.** Sato ridge filter (σ = 1, 2, 3 px), then a threshold (Otsu or local) to
give the filament mask and its area fraction.

**Network topology.** Skeleton of the mask. Pixels with ≥ 3 neighbours are junctions and
pixels with 1 neighbour are endpoints. Reported: branches and their lengths, closed mesh
cells, skeleton density, junction/endpoint ratio and junctions per 100 skeleton pixels.

**Orientation.** The structure tensor gives local filament orientation and *coherence*
(0 = isotropic, 1 = parallel). Also reported: the global orientation order R and the
dominant direction.

**Eddies.** For every candidate centre and several radii, the *tangential alignment
index* is the coherence- and ridge-weighted mean of cos 2(θ_filament − θ_tangential):
+1 is circular, 0 random, −1 radial. All centres are evaluated at once by FFT
convolution. A detection must satisfy all of the following:

- it is strongly aligned (≥ 0.55);
- filaments surround the centre in ≥ 6 of 8 angular sectors, so a single curved
  filament is not counted;
- ≥ 80% of the disk lies inside the image;
- significance = alignment × √n_eff ≥ 1.5, where n_eff is the weighted filament area
  in units of the orientation correlation area;
- secondary eddies are at least half as significant as the dominant one.

Each eddy also gets a radial alignment profile.

**Texture.** GLCM (Haralick) contrast, homogeneity, energy, correlation and dissimilarity
at several distances, plus contrast anisotropy and correlation decay. Uniform LBP
histogram, entropy and pattern fractions.

**Complexity.** Box-counting fractal dimension, gliding-box lacunarity and the slope of
the radially averaged power spectrum.

**Discrimination.** Each image is percentile-normalised (1–99 %), so images analysed on
different channels are comparable. It is then cut into overlapping tiles, and 13
scale-robust features are computed per tile.

- *Ranking.* Features are ranked by *separability* = 2 × min over class pairs of
  |AUC − 0.5|, which is 0 for no separation and 1 for perfect. Also reported: η²,
  Kruskal–Wallis H and p and, for two classes, Cohen's d.
- *Selected set.* A greedy pick of top-ranked features whose mutual |Spearman r| < 0.8.
- *Classifier.* Shrinkage LDA is cross-validated by leave-one-image-out when every class
  has ≥ 2 images, otherwise by 5-fold over tiles (flagged as optimistic).

**Outputs of `discriminate`:** `summary.txt`, `feature_ranking.csv`, `tile_features.csv`,
`whole_image_metrics.csv`, `discrimination.json`, `feature_distributions.png` and
`projection.png`. With `--save-diagnoses`, each image's diagnosis JSON and figure are
written too.

## Example

`examples/` contains three bloom images, one per species, and their outputs
(`out_discrimination/`). The notebook `examples/shades2shapes_example.ipynb` runs the same
analysis step by step with the Python API.

```bash
cd examples
shades2shapes discriminate N_scintillans.png Aphanizomenon_sp.png M_rubrum.png \
    --labels N_scintillans,Aphanizomenon_sp,M_rubrum -o out_discrimination --save-diagnoses
```

| | N_scintillans.png | Aphanizomenon_sp.png | M_rubrum.png |
|---|---|---|---|
| Filament fraction | 0.08 | 0.27 | 0.12 |
| Junction/endpoint ratio | 0.81 | 1.28 | 0.71 |
| Coherence | 0.52 | 0.44 | 0.51 |
| Eddies (dominant radius) | 1 (98 px, alignment 0.72) | 3 (43 px, alignment 0.58) | 1 (57 px, alignment 0.67) |
| GLCM contrast (d = 1) | 26.9 | 62.5 | 2.1 |
| Fractal dimension | 1.41 | 1.75 | 1.56 |

*Aphanizomenon* sp. forms a dense mesh of filaments. *N. scintillans* and *M. rubrum*
have sparser networks of similar topology and are told apart mainly by texture. The best
tile features are GLCM contrast, LBP entropy and GLCM homogeneity (separability 0.56–0.66).
Network features such as filament fraction and skeleton density separate *Aphanizomenon*
sp. well but score low overall (≈ 0.16), because separability is the worst class pair.
Accuracy is 91% over tiles with all 13 features and 80% with the selected 4-feature set,
which is optimistic because each class has only one image. When each image is split in two
halves and halves are held out, accuracy drops to about 56% with all features and 58% with
the selected set (chance is 33%).

*M. rubrum* has a much lower GLCM contrast and a steeper power spectrum (slope −2.95,
against −2.3 to −2.4 for the others), which suggests a smoother or blurrier image. Part of
the texture-based separation may therefore reflect image resolution or processing rather
than the bloom pattern.

## Caveats

- **Scale.** Without `--pixel-size`, lengths and radii are in pixels. Compare images at
  the same resolution, or give the pixel size.
- **Colour is not used.** Only one band is analysed, after normalisation. This is
  deliberate: colour mostly reflects pigments and sensor processing, not spatial pattern.
- **Eddies are inferred from filament geometry, not flow.** For dynamics, combine with
  velocity fields (e.g. Okubo–Weiss or vorticity from altimetry). The rotation sense
  cannot be recovered from axial orientations.
- **No-data.** NaN pixels (land, clouds, scene borders) raise an error: crop the image to
  a valid area or fill them before the analysis.
- **Sample size.** Validate discrimination on several images per class. Tile-level
  accuracy from a single image per class overstates real performance.

## Documentation

The documentation (usage, methods, API reference and the example notebooks) is in `docs/`
and is set up for Read the Docs (`.readthedocs.yaml`). To build it locally:

```bash
pip install -e .[docs]
make -C docs html        # docs/build/html/index.html
```

## Tests

```bash
pytest tests/
```

The tests use synthetic spirals and meshes and check eddy localisation, metric ranges,
two-class, grouped and three-class discrimination, and the CLI.

## License

MIT
