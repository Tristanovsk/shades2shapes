# shades2shapes

```{image} _static/shades2shapes.png
:alt: shades2shapes
:width: 560px
:align: center
:class: only-light
```

```{image} _static/shades2shapes_dark.png
:alt: shades2shapes
:width: 560px
:align: center
:class: only-dark
```

**shades2shapes** turns the *shades* of an image into *shapes*: quantitative descriptors
of the spatial patterns formed by filaments, networks and eddies. It is designed for
ocean-colour and true-colour images of phytoplankton or cyanobacteria blooms, and works
on any image of bright (or dark) filamentous structures.

It provides two tools, available from the command line and from Python:

`diagnose`
: Full spatial diagnosis of one image: filaments, network topology, orientation,
  eddies, texture and complexity. Writes a summary, a JSON file and a diagnostic figure.

`discriminate`
: Compares two or more images, or groups of images. Ranks the features that separate
  them, recommends a compact non-redundant feature set and cross-validates a classifier.

```python
import shades2shapes as s2s

d = s2s.diagnose("bloom.png")
print(d.summary())
d.plot()

r = s2s.discriminate(["N_scintillans.png", "Aphanizomenon_sp.png", "M_rubrum.png"])
print(r.summary())
```

```{toctree}
:maxdepth: 2
:caption: User guide

installation
usage
methods
examples/shades2shapes_example
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
