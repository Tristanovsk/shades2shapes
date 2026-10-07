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
```

`s2s` is a short alias for `shades2shapes`, and `python -m shades2shapes` also works.
Run `shades2shapes diagnose -h` or `shades2shapes discriminate -h` for the full help.

### Options

These options are shared by both commands:

| Option | Default | Meaning |
|---|---|---|
| `--channel` | `auto` | Band to analyse: `auto` (highest contrast), `r`, `g`, `b`, `gray`. For `discriminate`, one value for all images or a comma-separated list, one per image |
| `--dark-ridges` | off | Filaments are darker than the background |
| `--mask-method` | `otsu` | Filament threshold: global `otsu`, or `local`, which keeps faint filaments better |
| `--tensor-sigma` | 4 | Smoothing scale of the orientation field (px) |
| `--no-eddies` | off | Skip eddy detection |
| `--eddy-min-alignment` | 0.55 | Minimum tangential alignment of an eddy |
| `--eddy-min-significance` | 1.5 | Minimum eddy significance |
| `--pixel-size`, `--unit` | none, km | Also report lengths in physical units |
| `-o`, `--out` | `s2s_diagnosis` / `s2s_discrimination` | Output directory |
| `--no-plot` | off | Do not write figures |

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

Inputs can be file paths, NumPy arrays (`H×W` or `H×W×3`) or, for `discriminate`,
existing `Diagnosis` objects.

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
