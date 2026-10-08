# Usage

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

# Same area at every level of a Zarr pyramid
shades2shapes pyramid scene.zarr --variable Rrs --channel 665 --mask mask \
    --transform log10 --bbox 612000,4790600,678000,4800600
```

`s2s` is a short alias for `shades2shapes`, and `python -m shades2shapes` also works.
Run `shades2shapes diagnose -h` or `shades2shapes discriminate -h` for the full help.

### Options

These options are shared by both commands:

| Option | Default | Meaning |
|---|---|---|
| `--channel` | `auto` | Band to analyse: `auto` (highest contrast), `r`, `g`, `b`, `gray`, or a band or variable name of a GeoTIFF/NetCDF file. For `discriminate`, one value for all images or a comma-separated list, one per image |
| `--dark-ridges` | off | Filaments are darker than the background |
| `--mask-method` | `otsu` | Filament threshold: global `otsu`, or `local`, which keeps faint filaments better |
| `--tensor-sigma` | 4 | Smoothing scale of the orientation field (px) |
| `--no-eddies` | off | Skip eddy detection |
| `--eddy-min-alignment` | 0.55 | Minimum tangential alignment of an eddy |
| `--eddy-min-significance` | 1.5 | Minimum eddy significance |
| `--pixel-size`, `--unit` | none, km | Also report lengths in physical units. For georeferenced files the pixel size is read from the CRS |
| `-o`, `--out` | `s2s_diagnosis` / `s2s_discrimination` | Output directory |
| `--no-plot` | off | Do not write figures |
| `--save-maps` | off | Also write the maps (mask, ridges, orientation...) as `<name>_maps.nc`, georeferenced for GeoTIFF/NetCDF inputs (with `discriminate`, together with `--save-diagnoses`) |

These are specific to `discriminate`:

| Option | Default | Meaning |
|---|---|---|
| `--labels` | file names | Comma-separated class of each image; repeat a label to group images |
| `--tile`, `--overlap` | 64, 0.5 | Tile size (px) and overlap |
| `--n-select` | 4 | Size of the recommended feature set |
| `--save-diagnoses` | off | Also write each image's diagnosis JSON and figure |

### Outputs

`diagnose` writes, for each image, `<name>_summary.txt`, `<name>_diagnosis.json` and
`<name>_diagnosis.png`.

`discriminate` writes `summary.txt`, `feature_ranking.csv`, `tile_features.csv`,
`whole_image_metrics.csv`, `discrimination.json`, `feature_distributions.png` and
`projection.png`. With `--save-diagnoses`, each image's diagnosis JSON and figure are
written too.

## Python

### Diagnose an image

```python
import shades2shapes as s2s

d = s2s.diagnose("bloom.png", channel="auto", mask_method="otsu")
print(d.summary())
d.to_json("bloom_diagnosis.json")

d.metrics                      # dict of all metrics
d.eddies                       # list of Eddy(x, y, radius, alignment, significance, ...)
d.mask, d.theta, d.coherence   # arrays for further analysis
```

{func}`~shades2shapes.diagnose` returns a {class}`~shades2shapes.Diagnosis`. Every
option of {class}`~shades2shapes.DiagnosisConfig` can be passed as a keyword, or grouped
in a config object that is reused:

```python
cfg = s2s.DiagnosisConfig(mask_method="local", pixel_size=0.03, unit="km")
d = s2s.diagnose("bloom.png", config=cfg)
```

Inputs can be file paths, NumPy arrays (`H×W` or `H×W×3`), georeferenced images (see
{ref}`satellite-images`) or, for `discriminate`, existing `Diagnosis` objects.

### Discriminate images or groups

```python
r = s2s.discriminate(["a1.png", "a2.png", "b1.png", "b2.png"],
                     labels=["A", "A", "B", "B"])
print(r.summary())

r.ranking                       # pandas DataFrame, best features first
r.selected                      # recommended non-redundant features
r.cv_selected["accuracy"], r.cv_selected["confusion"]
r.save("comparison/")
```

{func}`~shades2shapes.discriminate` returns a
{class}`~shades2shapes.DiscriminationResult`. Use at least two images per class, so that
accuracy is estimated by leaving whole images out (see {doc}`methods`).

### Classify a new image

The classifier fitted on the selected features labels the tiles of any new image:

```python
new = s2s.diagnose("unknown.png")
tiles = s2s.tile_features(new).dropna(subset=r.selected)
tiles["pred"] = r.cv_selected["model"].predict(tiles[r.selected].to_numpy())
tiles["pred"].value_counts(normalize=True)
```

(satellite-images)=
### Satellite images (xarray, GeoTIFF, NetCDF)

With the `geo` extra (`pip install shades2shapes[geo]`), the input can be georeferenced:

- an {class}`xarray.DataArray` with two spatial dimensions (`x`/`y` or `lon`/`lat`) and
  at most one band dimension, in any order, e.g. `(band, y, x)` as read by
  {func}`rioxarray.open_rasterio`. Other dimensions of size 1 (`time`) are dropped;
- an {class}`xarray.Dataset`: `channel` names the variable (e.g. `"chlor_a"`); to
  analyse several variables as bands, stack them with
  `ds[["B4", "B3", "B2"]].to_array("band")`;
- the path of a GeoTIFF (`.tif`, `.jp2`, `.vrt`) or NetCDF (`.nc`) file
  ({func}`~shades2shapes.geo.open_geo`).

```python
import rioxarray

da = rioxarray.open_rasterio("S2_scene.tif", masked=True)    # (band, y, x), UTM
da = da.rio.clip_box(minx, miny, maxx, maxy)                  # area of interest
d = s2s.diagnose(da, channel="B8", unit="m")
print(d.summary())
```

Compared with a plain image:

- any number of bands is accepted; `channel` can be a band name (a value of the band
  coordinate, or the 1-based band number of a GeoTIFF without band descriptions) and
  `"auto"` picks the band with the highest contrast;
- the image is analysed north-up (it is flipped if the y coordinate increases);
- in a projected CRS, the pixel size is derived from the coordinates in the requested
  `unit` (`"m"` or `"km"`), unless `pixel_size` is given, so the `_phys` metrics are
  always computed;
- each {class}`~shades2shapes.Eddy` gets its centre in map coordinates (`x_map`,
  `y_map`) and, when the CRS is known, in longitude and latitude (`lon`, `lat`);
- `Diagnosis.geo` holds the georeferencing
  ({class}`~shades2shapes.geo.GeoInfo`), and {meth}`~shades2shapes.Diagnosis.to_xarray`
  returns the maps on the image grid, with the CRS attached:

```python
d.eddies[0].lon, d.eddies[0].lat
ds = d.to_xarray()        # band, ridges, mask, skeleton, junctions, coherence, orientation
ds.to_netcdf("S2_maps.nc")
ds.mask.rio.to_raster("S2_mask.tif")
```

{func}`~shades2shapes.discriminate` accepts the same inputs, and `channels` can be
band names.

Limitations:

- **geographic coordinates** (degrees): the pixel size is not set (with a warning),
  because it varies with latitude. Reproject first:
  `da = da.rio.reproject(da.rio.estimate_utm_crs())`, or pass `pixel_size`;
- **no-data** (NaN: land, clouds, scene borders) is not supported yet and raises an
  error: crop the image to a valid area, or fill the gaps;
- scenes are loaded in memory: crop large scenes to the area of interest.

(pyramids)=
### Pyramids: several resolutions

{func}`~shades2shapes.diagnose_pyramid` diagnoses the same area at every level of an
image pyramid, e.g. a Zarr store whose groups `0, 1, 2...` are 2, 4... times coarser
(read from its `multiscales` attribute), and returns a
{class}`~shades2shapes.PyramidDiagnosis`:

```python
p = s2s.diagnose_pyramid("S2_GRS.zarr", variable="Rrs", channel="665", mask="mask",
                         transform="log10", bbox=(612000, 4790600, 678000, 4800600))
print(p.summary())
p.levels            # pixel size, shape, no-data fraction and status of each level
p.metrics_table()   # all metrics, one row per level
p.eddy_tracks       # eddies matched across levels (same place, similar radius)
p.diagnoses["1"]    # the Diagnosis of one level
p.save("pyramid/")  # tables, JSON and figures (pyramid.png, metrics_by_level.png)
```

- The source can also be a dict or list of levels (xarray objects or arrays), or a
  single image, which is then block-averaged by `factors` (default 1, 2, 4, 8, 16).
- `bbox` cuts the same area from every level, in map coordinates or in `bbox_crs`
  (e.g. `"EPSG:4326"` for longitudes and latitudes).
- No-data: `mask` names a variable flagging invalid pixels (clouds, land), in addition
  to NaN. Invalid pixels are filled from their neighbours
  ({func}`~shades2shapes.geo.fill_nodata`); levels with more than `max_nodata` (25 %)
  invalid pixels are skipped, so crop out land with `bbox`.
- Levels smaller than `min_size` (64 px) are skipped: eddies need room.
- `transform="log10"` helps with reflectances or concentrations spanning orders of
  magnitude, where a bright coastal band would otherwise saturate the stretch.

An eddy found at several levels, at the same place and with a similar radius
(`max_shift`, `max_ratio`), is more robust than one seen at a single level. Most
metrics depend on the pixel size: finer levels resolve smaller filaments, but also more
noise. Compare images at the same level, and use the trends across levels
({meth}`~shades2shapes.PyramidDiagnosis.plot_metrics`) to see at which scales the
structures appear. The {doc}`pyramid example <examples/s2_pyramid_example>` analyses a
Sentinel-2 scene this way, and the {doc}`Uroglena example
<examples/uroglena_planetscope_example>` the 2021 bloom in Lake Geneva on PlanetScope
images.

### In a notebook

`Diagnosis.plot(path)` and `DiscriminationResult.save(..., plots=True)` switch
matplotlib to the non-interactive `Agg` backend, after which figures no longer show
inline. In a notebook, call the plotting methods without a path and save the returned
figures:

```python
fig = d.plot()
fig.savefig("bloom_diagnosis.png", dpi=110)

r.save("comparison/", plots=False)
r.plot_features().savefig("comparison/feature_distributions.png")
```

The {doc}`example notebook <examples/shades2shapes_example>` shows a complete analysis.
