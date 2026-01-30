"""
PSI I/O – a Python package of readers and writers for PSI data.

This package is designed to standardize the reading and writing of HDF files within the PSI
(Predictive Science Inc.) data ecosystem. It provides a consistent interface for handling
both HDF4 and HDF5 file formats (although the aim is to deprecate the latter in the future),
and includes utilities for managing PSI-specific conventions.
"""
from . import psi_io as _psi_io
from .psi_io import *

__all__ = list(_psi_io.__all__)

try:
    from importlib.metadata import version as _pkg_version
    from importlib.metadata import PackageNotFoundError
    from pathlib import Path
    __version__ = _pkg_version("psi-io")  # type: ignore[assignment]
except PackageNotFoundError as e:  # dev/editable without metadata
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # pip install tomli

    pyproject = Path(__file__).parents[1].resolve() / 'pyproject.toml'
    data = tomllib.loads(pyproject.read_text())

    project_version = data.get("project", {}).get("version", "0+unknown")
    project_version = project_version.replace('"', '').replace("'", '')
    __version__ = project_version
