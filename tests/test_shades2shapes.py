import json

import numpy as np
import pytest
from scipy import ndimage as ndi

import shades2shapes as s2s
from shades2shapes.cli import main


def spiral_image(n=200, cx=100, cy=100, arms=3, seed=0):
    """Bright spiral filaments around (cx, cy) on a dark background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - cx, yy - cy) + 1e-6
    phi = np.arctan2(yy - cy, xx - cx)
    # logarithmic spiral with small pitch: almost circular filaments
    v = np.cos(arms * phi + 0.6 * r)
    img = np.exp(-((1 - v) / 0.08)) * (r > 4) * (r < 0.45 * n)
    return np.clip(img + 0.05 * rng.standard_normal(img.shape), 0, 1)


def mesh_image(n=200, seed=0, density=0.08):
    """Random isotropic mesh of filaments."""
    rng = np.random.default_rng(seed)
    noise = ndi.gaussian_filter(rng.standard_normal((n, n)), 3)
    lap = -ndi.gaussian_laplace(noise, 2)
    img = np.clip(lap / np.percentile(lap, 99), 0, 1)
    return np.clip(img + 0.05 * rng.standard_normal(img.shape), 0, 1)


def test_diagnose_returns_metrics():
    d = s2s.diagnose(mesh_image())
    m = d.metrics
    for k in ("filament_fraction", "skeleton_density", "n_junctions", "coherence_mean",
              "fractal_dimension", "glcm_contrast_d1", "lbp_entropy", "spectral_slope"):
        assert k in m
    assert 0 < m["filament_fraction"] < 1
    assert 1.0 < m["fractal_dimension"] < 2.0
    json.loads(d.to_json())
    assert "Filaments" in d.summary()


def test_eddy_detected_at_spiral_centre():
    d = s2s.diagnose(spiral_image(cx=110, cy=95))
    assert d.eddies, "no eddy found on a synthetic spiral"
    e = d.eddies[0]
    assert abs(e.x - 110) < 15 and abs(e.y - 95) < 15
    assert e.alignment > 0.7


def test_spiral_more_coherent_than_mesh():
    a = s2s.diagnose(spiral_image()).metrics
    b = s2s.diagnose(mesh_image()).metrics
    assert a["coherence_mean"] > b["coherence_mean"]
    assert a["eddy_max_significance"] > b["eddy_max_significance"]


def test_options_and_pixel_size():
    d = s2s.diagnose(mesh_image(), mask_method="local", pixel_size=0.3, unit="km",
                     detect_eddies=False)
    assert d.metrics["n_eddies"] == 0
    assert d.metrics["branch_length_mean_phys"] == pytest.approx(
        d.metrics["branch_length_mean"] * 0.3)
    with pytest.raises(TypeError):
        s2s.diagnose(mesh_image(), not_an_option=1)


def test_discriminate_two_images(tmp_path):
    r = s2s.discriminate([spiral_image(), mesh_image()], labels=["spiral", "mesh"])
    assert len(r.ranking) > 5
    assert r.ranking["separability"].iloc[0] > 0.5
    assert len(r.selected) == 4
    assert r.cv_all["accuracy"] > 0.75
    assert "optimistic" in r.cv_all["scheme"]
    out = r.save(tmp_path / "res")
    for f in ("tile_features.csv", "feature_ranking.csv", "discrimination.json",
              "summary.txt", "feature_distributions.png", "projection.png"):
        assert (out / f).exists()


def test_discriminate_groups_leave_one_image_out():
    imgs = [spiral_image(seed=i, cx=100 + 5 * i) for i in range(2)] + \
           [mesh_image(seed=i) for i in range(2)]
    r = s2s.discriminate(imgs, labels=["spiral", "spiral", "mesh", "mesh"], tile=48)
    assert r.cv_all["scheme"] == "leave-one-image-out"
    assert r.cv_selected["accuracy"] > 0.7


def test_discriminate_three_classes():
    noise = np.random.default_rng(3).random((200, 200))
    r = s2s.discriminate([spiral_image(), mesh_image(), noise])
    assert {"eta2", "kruskal_p", "separability"} <= set(r.ranking.columns)
    assert r.cv_all["confusion"].shape == (3, 3)


def test_discriminate_needs_two_classes():
    with pytest.raises(ValueError):
        s2s.discriminate([mesh_image()])
    with pytest.raises(ValueError):
        s2s.discriminate([mesh_image(), mesh_image(seed=1)], labels=["a", "a"])


def test_cli(tmp_path):
    from skimage import io
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    io.imsave(a, (spiral_image() * 255).astype(np.uint8), check_contrast=False)
    io.imsave(b, (mesh_image() * 255).astype(np.uint8), check_contrast=False)
    main(["diagnose", str(a), "-o", str(tmp_path / "d")])
    assert (tmp_path / "d" / "a_diagnosis.json").exists()
    assert (tmp_path / "d" / "a_diagnosis.png").exists()
    main(["discriminate", str(a), str(b), "--labels", "spiral,mesh", "-o", str(tmp_path / "c"),
          "--no-plot"])
    assert (tmp_path / "c" / "feature_ranking.csv").exists()
