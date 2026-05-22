from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

try:
    # First try to run sphinx_build against installed dist
    # This is primarily included for nox-based doc builds
    import psi_io
except ImportError:
    # Fallback: add project root to sys.path
    # This is included for local dev builds without install
    sys.path.insert(0, Path(__file__).resolve().parents[2].as_posix())
    import psi_io

try:
    from pthree import build_node_tree, node_tree_to_dict
except ImportError:
    raise ImportError(
        "The 'pthree' package is required to build the documentation. "
        "Please install it via 'pip install pthree' and try again."
    )

# ------------------------------------------------------------------------------
# Project Information
# ------------------------------------------------------------------------------
project = "psi-io"
author = "Predictive Science Inc"
copyright = f"{datetime.now():%Y}, {author}"
version = psi_io.__version__
release = psi_io.__version__

# ------------------------------------------------------------------------------
# General Configuration
# ------------------------------------------------------------------------------
extensions = []

# --- HTML Theme
_logo = "https://predsci.com/doc/assets/static/psi_logo.png"
html_favicon = _logo
html_logo = _logo
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "show_prev_next": False,
    "navigation_with_keys": False,
    "show_nav_level": 3,
    "navigation_depth": 5,
    "logo": {
        "text": f"{project} v{version}",
        "image_light": _logo,
        "image_dark": _logo,
    },
    'icon_links': [
        {
            'name': 'PSI Home',
            'url': 'https://www.predsci.com/',
            'icon': 'fa fa-home fa-fw',
            "type": "fontawesome",
        },
        {
            'name': 'Repository',
            'url': 'https://github.com/predsci/psi-io',
            "icon": "fa-brands fa-github fa-fw",
            "type": "fontawesome",
        },
        {
            'name': 'Documentation',
            'url': 'https://predsci.com/doc/psi-io',
            "icon": "fa fa-file fa-fw",
            "type": "fontawesome",
        },
        {
            'name': 'Contact',
            'url': 'https://www.predsci.com/portal/contact.php',
            'icon': 'fa fa-envelope fa-fw',
            "type": "fontawesome",
        },
    ],
}

# --- Python Syntax
add_module_names = False
python_maximum_signature_line_length = 80

# --- Templating
templates_path = ['_templates', ]

# ------------------------------------------------------------------------------
# Viewcode Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx.ext.viewcode")

viewcode_line_numbers = True

# ------------------------------------------------------------------------------
# Autosummary Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx.ext.autosummary")

root_package = 'psi_io'
exclude_private = False
exclude_tests = True
exclude_dunder = True
sort_members = False
exclusions = [
    "psi_io.psi_io._",

    "psi_io._models._MAS_QUANTITY_PROPS_MAPPING",
    "psi_io._models._POT3D_QUANTITY_PROPS_MAPPING",
    "psi_io._models._PSI_SCALE_PROPS_MAPPING",
    "psi_io._models.MATCH_QUANTITIES",
    "psi_io._models.FILEPATH_SCHEMA",
    "psi_io._models.MODEL_TYPE",
    "psi_io._models._PROP_GETTER_MAPPING",
    "psi_io._models.Props._mesh",

    "psi_io._mesh._MESH_CODE_REVERSE_MAPPING",
    "psi_io._mesh._normalize_mesh_code",
    "psi_io._mesh._average_adjacent",
    "psi_io._mesh._remesh_array",
    "psi_io._mesh._parse_remesh",
    "psi_io._mesh.Mesh.HALF",
    "psi_io._mesh.Mesh.MAIN",

    r"psi_io\.mhd_io\.(?!PsiData\b)",
]

node_tree = build_node_tree(root_package,
                            sort_members,
                            exclude_private,
                            exclude_tests,
                            exclude_dunder,
                            exclusions)

autosummary_context = dict(pkgtree=node_tree_to_dict(node_tree))

# ------------------------------------------------------------------------------
# Autodoc Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx.ext.autodoc")

autodoc_typehints = "none"
autodoc_member_order = 'bysource'
autodoc_default_options = {
    "show-inheritance": True,
}

# ------------------------------------------------------------------------------
# Numpydoc Configuration
# ------------------------------------------------------------------------------
extensions.append("numpydoc")

numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"optional", "default", "of", "or"}
numpydoc_xref_aliases = {
    # --- Python standard library ---
    "Path": "pathlib.Path",
    "Callable": "collections.abc.Callable",
    "Sequence": "collections.abc.Sequence",
    "Any": "typing.Any",
    "Literal": "typing.Literal",
    # --- NumPy ---
    "np.ndarray": "numpy.ndarray",
    "np.dtype": "numpy.dtype",
    # --- SciPy ---
    "RegularGridInterpolator": "scipy.interpolate.RegularGridInterpolator",
    # --- Astropy ---
    "u.Quantity": "astropy.units.Quantity",
    "u.Unit": "astropy.units.UnitBase",
    "QuantityLike": "astropy.units.typing.QuantityLike",
    "UnitLike": "astropy.units.typing.UnitLike",
    "astropy.units.Quantity": "astropy.units.Quantity",
    "astropy.units.Unit": "astropy.units.UnitBase",
    # --- psi_io.psi_io ---
    "PathLike": "psi_io.psi_io.PathLike",
    "HdfScaleMeta": "psi_io.psi_io.HdfScaleMeta",
    "HdfDataMeta": "psi_io.psi_io.HdfDataMeta",
    "HdfExtType": "psi_io.psi_io.HdfExtType",
    # --- psi_io._mesh ---
    "Mesh": "psi_io._mesh.Mesh",
    "MeshCodeType": "psi_io._mesh.MeshCodeType",
    # --- psi_io_models ---
    "Props": "psi_io._models.Props",
    "MasQuantities": "psi_io._models.MasQuantities",
    "Pot3dQuantities": "psi_io._models.Pot3dQuantities",
    "PsiScales": "psi_io._models.PsiScales",
    "ModelType": "psi_io._models.ModelType",
    # --- psi_io.mhd_io ---
    "HdfVersionType": "psi_io.mhd_io.HdfVersionType",
    "Scales": "psi_io.mhd_io.Scales",
    "H5Scale": "psi_io.mhd_io.H5Scale",
    "H4Scale": "psi_io.mhd_io.H4Scale",
    "H5Data": "psi_io.mhd_io.H5Data",
    "H4Data": "psi_io.mhd_io.H4Data",
}

# ------------------------------------------------------------------------------
# Intersphinx Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx.ext.intersphinx")

DOCS = Path(__file__).resolve().parents[1]
INV = DOCS / "_intersphinx"
intersphinx_cache_limit = 30
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        # (INV / "python-objects.inv").as_posix(),
        None
    ),
    "numpy": (
        "https://numpy.org/doc/stable/",
        # (INV / "numpy-objects.inv").as_posix(),
        None
    ),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/reference/",
        # (INV / "scipy-objects.inv").as_posix(),
        None
    ),
    "matplotlib": (
        "https://matplotlib.org/stable/",
        # (INV / "matplotlib-objects.inv").as_posix(),
        None
    ),
    "pooch": (
        "https://www.fatiando.org/pooch/latest/",
        # (INV / "pooch-objects.inv").as_posix(),
        None
    ),
    "h5py": (
        "https://docs.h5py.org/en/stable/",
        # (INV / "h5py-objects.inv").as_posix(),
        None
    ),
    "pyhdf": (
        "https://fhs.github.io/pyhdf/",
        # (INV / "pyhdf-objects.inv").as_posix(),
        None
    ),
    "astropy": (
        "https://docs.astropy.org/en/stable/",
        # (INV / "astropy-objects.inv").as_posix(),
        None
    ),
}

# ------------------------------------------------------------------------------
# Sphinx-Gallery Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx_gallery.gen_gallery")

import matplotlib
matplotlib.use("Agg")
os.environ.setdefault('SPHINX_GALLERY_BUILD', '1')

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["gallery"],
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
    "remove_config_comments": True,
    "filename_pattern": r"\.py$",
    "plot_gallery": True,
    "run_stale_examples": True,
    "matplotlib_animations": True,
    "default_thumb_file": (Path(__file__).parent / "_static/assets/psi_logo.png").as_posix(),
}

# ------------------------------------------------------------------------------
# Sphinx Copy Button Configuration
# ------------------------------------------------------------------------------
extensions.append("sphinx_copybutton")

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

