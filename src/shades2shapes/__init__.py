"""shades2shapes: diagnose and discriminate spatial patterns in images.

Quick start
-----------
>>> import shades2shapes as s2s
>>> d = s2s.diagnose("bloom.png")              # one image
>>> print(d.summary()); d.plot("bloom_diag.png")
>>> r = s2s.discriminate(["a.png", "b.png"])    # two or more images / groups
>>> print(r.summary()); r.save("results/")
>>> p = s2s.diagnose_pyramid("scene.zarr", variable="Rrs", channel="665")  # all levels
>>> print(p.summary()); p.save("pyramid/")
"""
__version__ = "0.1.0"

from .diagnose import Diagnosis, DiagnosisConfig, diagnose  # noqa: E402
from .discriminate import (DiscriminationResult, TILE_FEATURES, discriminate,  # noqa: E402
                           rank_features, tile_features)
from .eddies import Eddy, detect_eddies  # noqa: E402
from .multiscale import PyramidDiagnosis, diagnose_pyramid  # noqa: E402

__all__ = ["diagnose", "Diagnosis", "DiagnosisConfig", "discriminate",
           "DiscriminationResult", "tile_features", "rank_features", "TILE_FEATURES",
           "detect_eddies", "Eddy", "diagnose_pyramid", "PyramidDiagnosis", "__version__"]
