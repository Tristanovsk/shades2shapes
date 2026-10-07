API reference
=============

.. currentmodule:: shades2shapes

Everything in the first three sections is importable from the package itself:

.. code-block:: python

   import shades2shapes as s2s

   d = s2s.diagnose("bloom.png")
   r = s2s.discriminate(["a.png", "b.png"])

Diagnose and discriminate
-------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   diagnose
   discriminate

Results and options
-------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Diagnosis
   DiagnosisConfig
   DiscriminationResult
   Eddy

Building blocks
---------------

Use these to compute tile features, rank features or detect eddies on your own
data, without the full pipeline.

.. autosummary::
   :toctree: generated
   :nosignatures:

   tile_features
   rank_features
   detect_eddies

The tile features are listed in
:data:`shades2shapes.discriminate.TILE_FEATURES`.

Lower-level modules
-------------------

.. toctree::
   :maxdepth: 1

   api/features
   api/eddies
   api/statistics
   api/preprocessing
