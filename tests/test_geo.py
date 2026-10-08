"""Georeferenced inputs (xarray, GeoTIFF, NetCDF); skipped without the geo extra."""
import json
import warnings

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("rioxarray")

import shades2shapes as s2s  # noqa: E402
from shades2shapes.cli import main  # noqa: E402

from test_shades2shapes import mesh_image, spiral_image  # noqa: E402

RES = 20.0                      # m
X0, Y0 = 500000.0, 4800000.0    # upper-left corner, UTM 31N


def utm_image(img, bands=("B4", "B3", "B2"), crs="EPSG:32631"):
    """(band, y, x) DataArray of reflectances, north-up, as read by rioxarray."""
    H, W = img.shape
    data = np.stack([img * (0.06 + 0.01 * i) for i in range(len(bands))])
    da = xr.DataArray(data, dims=("band", "y", "x"),
                      coords={"band": list(bands),
                              "y": Y0 - RES * (np.arange(H) + 0.5),
                              "x": X0 + RES * (np.arange(W) + 0.5)})
    return da.rio.write_crs(crs)


def test_band_first_dataarray_matches_array():
    img = spiral_image()
    d = s2s.diagnose(utm_image(img), channel="B4")
    ref = s2s.diagnose(img)
    assert d.shape == img.shape and d.channel == "B4"
    assert d.metrics["n_junctions"] == ref.metrics["n_junctions"]
    assert d.source == "<xarray>"


def test_pixel_size_and_eddy_coordinates():
    d = s2s.diagnose(utm_image(spiral_image()), unit="km")
    assert d.config.pixel_size == pytest.approx(0.02)
    assert d.metrics["eddy_dominant_radius_phys"] == pytest.approx(
        d.metrics["eddy_dominant_radius"] * 0.02)
    e = d.eddies[0]
    assert e.x_map == pytest.approx(X0 + RES * (e.x + 0.5))
    assert e.y_map == pytest.approx(Y0 - RES * (e.y + 0.5))
    assert 2.9 < e.lon < 3.1 and 43 < e.lat < 43.4      # UTM 31N, around 3 E, 43.3 N
    assert "lon" in d.summary() and "EPSG:32631" in d.summary()
    assert json.loads(d.to_json())["geo"]["crs"] == "EPSG:32631"
    assert d.geo.transform[0] == RES and d.geo.transform[2] == X0


def test_explicit_pixel_size_is_kept():
    d = s2s.diagnose(utm_image(mesh_image()), pixel_size=1.0, unit="km", detect_eddies=False)
    assert d.config.pixel_size == 1.0


def test_south_up_and_dataset_variable():
    img = spiral_image(cx=60, cy=80)
    da = utm_image(img).isel(y=slice(None, None, -1))          # y increasing: south-up
    ds = da.to_dataset("band")
    d = s2s.diagnose(ds, channel="B3")
    assert d.channel == "B3"
    assert np.allclose(d.geo.y, np.sort(d.geo.y)[::-1])        # flipped back to north-up
    e = d.eddies[0]
    assert abs(e.x - 60) <= 4 and abs(e.y - 80) <= 4


def test_geographic_warns_and_keeps_degrees():
    img = mesh_image()
    H, W = img.shape
    da = xr.DataArray(img, dims=("lat", "lon"),
                      coords={"lat": np.linspace(44, 43, H), "lon": np.linspace(3, 4, W)})
    with pytest.warns(UserWarning, match="geographic"):
        d = s2s.diagnose(da, detect_eddies=False)
    assert d.config.pixel_size is None and d.geo.geographic


def test_nan_is_rejected():
    da = utm_image(mesh_image())
    da[:, :10, :10] = np.nan
    with pytest.raises(ValueError, match="no-data"):
        s2s.diagnose(da)


def test_to_xarray(tmp_path):
    d = s2s.diagnose(utm_image(spiral_image()))
    ds = d.to_xarray()
    assert set(ds.data_vars) >= {"mask", "ridges", "coherence", "orientation"}
    assert ds.rio.crs.to_epsg() == 32631
    assert float(ds.x[0]) == X0 + RES / 2
    assert len(json.loads(ds.attrs["eddies"])) == len(d.eddies)
    ds.to_netcdf(tmp_path / "maps.nc")
    ds.mask.rio.to_raster(tmp_path / "mask.tif")
    # plain arrays: pixel coordinates
    assert s2s.diagnose(mesh_image(), detect_eddies=False).to_xarray().x[-1] == 199


def test_geotiff_and_cli(tmp_path):
    tif = tmp_path / "scene.tif"
    utm_image(spiral_image()).rio.to_raster(tif)
    d = s2s.diagnose(tif, channel="1")                  # GeoTIFF bands are numbered from 1
    assert d.channel == "1" and d.geo.crs == "EPSG:32631" and d.eddies
    nc = tmp_path / "other.nc"
    utm_image(mesh_image()).to_dataset("band").to_netcdf(nc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(["diagnose", str(tif), "--save-maps", "--no-plot", "-o", str(tmp_path / "d")])
        main(["discriminate", str(tif), str(nc), "--channel", "1,B4", "--no-plot",
              "-o", str(tmp_path / "c")])
    assert (tmp_path / "d" / "scene_maps.nc").exists()
    assert (tmp_path / "c" / "feature_ranking.csv").exists()


def test_dataset_grid_mapping_variable():
    """CF products (e.g. GRS Zarr) keep the CRS in a data variable named by grid_mapping."""
    ds = utm_image(mesh_image()).to_dataset("band")
    ds = ds.reset_coords("spatial_ref")                 # spatial_ref as a data variable
    for v in ("B4", "B3", "B2"):
        ds[v].attrs["grid_mapping"] = "spatial_ref"
    d = s2s.diagnose(ds, channel="B4", detect_eddies=False)
    assert d.geo.crs == "EPSG:32631"
