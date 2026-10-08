"""Pyramid (multi-level) analysis; skipped without the geo extra."""
import warnings

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("rioxarray")

import shades2shapes as s2s  # noqa: E402
from shades2shapes.cli import main  # noqa: E402
from shades2shapes.multiscale import coarsen_pyramid  # noqa: E402

from test_geo import utm_image  # noqa: E402
from test_shades2shapes import spiral_image  # noqa: E402


def big_spiral():
    return spiral_image(n=320, cx=160, cy=160)


def test_pyramid_from_single_image_tracks_eddy():
    da = utm_image(big_spiral())
    p = s2s.diagnose_pyramid(da, factors=(1, 2, 4), channel="B4", unit="m")
    assert list(p.diagnoses) == ["x1", "x2", "x4"]
    assert list(p.metrics_table()["pixel_size"]) == [20.0, 40.0, 80.0]
    best = p.eddy_tracks.iloc[0]
    assert best["n_levels"] >= 2
    # centre of the spiral: pixel 160 at 20 m from the corner
    assert best["x_map"] == pytest.approx(500000 + 20 * 160.5, abs=200)
    assert best["y_map"] == pytest.approx(4800000 - 20 * 160.5, abs=200)
    assert set(p.eddies["track"]) >= {1}
    assert "Eddy tracks" in p.summary()


def test_pyramid_bbox_mask_fill_and_skip(tmp_path):
    da = utm_image(big_spiral())
    ds = da.to_dataset("band")
    ds["B4"][100:104, 100:104] = np.nan                 # small gap: filled
    ds["mask"] = (("y", "x"), np.zeros(da.shape[1:], np.uint8))
    ds["mask"][:5, :] = 1                               # flagged: filled
    levels = {str(k): v for k, v in enumerate(coarsen_pyramid(ds, (1, 2, 4, 8)).values())}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = s2s.diagnose_pyramid(levels, variable="B4", mask="mask",
                                 bbox=(500000, 4793600, 506400, 4800000), detect_eddies=False)
    lv = p.levels
    assert lv.loc["0", "shape"] == "320x320"
    assert 0 < lv.loc["0", "nodata_fraction"] < 0.05
    assert lv.loc["3", "status"].startswith("skipped")        # 40 px < min_size
    assert list(p.diagnoses) == ["0", "1", "2"]
    out = p.save(tmp_path / "pyr", maps=True)
    for f in ("summary.txt", "levels.csv", "eddy_tracks.csv", "pyramid.png",
              "metrics_by_level.png", "level0_maps.nc"):
        assert (out / f).exists()
    with pytest.raises(ValueError, match="No level"):
        s2s.diagnose_pyramid(levels, variable="B4", min_size=1000)


def test_pyramid_bbox_lonlat_and_log():
    da = utm_image(big_spiral())
    lon, lat = s2s.diagnose(da, detect_eddies=False).geo.lonlat([0, 319], [319, 0])
    p = s2s.diagnose_pyramid(da, factors=(1, 2), bbox=(lon[0], lat[0], lon[1], lat[1]),
                             bbox_crs="EPSG:4326", transform="log10", detect_eddies=False)
    assert p.levels.loc["x1", "shape"] in ("320x320", "319x319", "320x319", "319x320")


def test_pyramid_zarr_and_cli(tmp_path):
    pytest.importorskip("zarr")
    ds = utm_image(big_spiral()).to_dataset("band")
    store = tmp_path / "pyr.zarr"
    ds.attrs["multiscales"] = [{"layout": [{"group": "0", "scale": 1},
                                           {"group": "1", "scale": 2}]}]
    xr.Dataset(attrs=ds.attrs).to_zarr(store, mode="w")
    for k, lev in zip("01", coarsen_pyramid(ds, (1, 2)).values()):
        lev.attrs = {}
        lev.to_zarr(store, group=k, mode="a")
    assert s2s.multiscale.pyramid_levels(store) == ["0", "1"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(["pyramid", str(store), "--variable", "B4", "--no-plot", "-o", str(tmp_path / "o")])
    assert (tmp_path / "o" / "eddy_tracks.csv").exists()
