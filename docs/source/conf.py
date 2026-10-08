import os
import shutil
import sys
from pathlib import Path

# Make the package importable without installation (local builds);
# on Read the Docs the package is also pip-installed (see .readthedocs.yaml).
DOCS_SOURCE = Path(__file__).resolve().parent
REPO_ROOT = DOCS_SOURCE.parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

import shades2shapes

on_rtd = os.environ.get('READTHEDOCS') == 'True'

# The example notebook and its images live in examples/; copy them next to the
# docs so that the notebook runs from its own folder (docs/source/examples/,
# git-ignored).
_examples_src = REPO_ROOT / 'examples'
_examples_dst = DOCS_SOURCE / 'examples'
_examples_dst.mkdir(exist_ok=True)
for _f in list(_examples_src.glob('*.ipynb')) + list(_examples_src.glob('*.png')):
    _dst = _examples_dst / _f.name
    _dst.unlink(missing_ok=True)       # sources may be read-only: copy content only
    shutil.copyfile(_f, _dst)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'shades2shapes'
copyright = '2026, Tristan Harmel'
author = 'Tristan Harmel'
version = shades2shapes.__version__
release = shades2shapes.__version__
today_fmt = '%Y-%m-%d'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'myst_nb',
]

templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store']

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autoclass_content = 'class'
autodoc_typehints = 'none'          # types are given in the NumPy docstrings
autodoc_member_order = 'bysource'
# 'members' is set in the autosummary templates (_templates/autosummary/) to
# avoid documenting objects twice
add_module_names = False

# NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    name: f'~shades2shapes.{name}'
    for name in ('Diagnosis', 'DiagnosisConfig', 'DiscriminationResult', 'Eddy')
}
napoleon_type_aliases['GeoInfo'] = '~shades2shapes.geo.GeoInfo'
napoleon_type_aliases['sequence'] = ':term:`sequence`'
napoleon_type_aliases['callable'] = ':term:`callable`'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'skimage': ('https://scikit-image.org/docs/stable', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
    'rioxarray': ('https://corteva.github.io/rioxarray/stable', None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'
pygments_style = 'sphinx'

html_theme_options = {
    'repository_url': 'https://github.com/Tristanovsk/shades2shapes',
    'repository_branch': 'main',
    'path_to_docs': 'docs/source',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_edit_page_button': True,
    'use_download_button': True,
    'navigation_with_keys': True,
    'show_toc_level': 2,
    'logo': {
        'image_light': '_static/logo.svg',
        'image_dark': '_static/logo_dark.svg',
    },
}

html_title = 'shades2shapes'
html_logo = '_static/logo.svg'
html_favicon = '_static/icon_dark.svg'

html_static_path = ['_static']
html_show_sourcelink = False
html_last_updated_fmt = today_fmt

htmlhelp_basename = 'shades2shapes_doc'

# -- MyST / notebook rendering -----------------------------------------------

myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'linkify',
    'smartquotes',
]
myst_heading_anchors = 3

# The example notebook runs in ~30 s, so it is executed at build time (also on
# Read the Docs). Outputs are cached in docs/build/.jupyter_cache and reused
# until the notebook changes. A failing notebook fails the build.
nb_execution_mode = 'cache'
# The satellite examples need images that are not distributed (Sentinel-2 scene,
# PlanetScope images under licence): they are rendered with the outputs stored in
# the notebooks.
nb_execution_excludepatterns = ['*s2_pyramid_example*', '*uroglena_planetscope_example*']
nb_execution_cache_path = str(DOCS_SOURCE.parent / 'build' / '.jupyter_cache')
nb_execution_timeout = 600
nb_execution_raise_on_error = True
nb_merge_streams = True
