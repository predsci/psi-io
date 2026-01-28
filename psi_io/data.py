"""
Module for fetching HDF5 assets used through examples.

This module uses the ``pooch`` library to manage the downloading and caching of
HDF4 and HDF5 files that adhere to PSI data conventions. It defines functions to
fetch specific example datasets, including 1D radial scale data, 2D coronal hole
maps, 3D radial magnetic field data, magnetic fieldline data, and synchronic maps
used in coronal and heliospheric magnetic field modeling.

Currently, these files are hosted on the PredSci documentation website:
at https://www.predsci.com/doc/assets/ and are primarily intended for use in
building examples in the PSI I/O and mapflpy packages.
"""


from __future__ import annotations

import inspect
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from psi_io.psi_io import HdfExtType, HDFEXT

try:
    import pooch
except ImportError as e:
    raise ImportError(
        "Missing the optional 'pooch' dependency required for data fetching. "
        "Please install it via pip or conda to access the necessary datasets."
    ) from e


REGISTRY = {
	"h4h5-files/rscale.h5": "sha256:60a0cbcd4dc69f7d250cbbdddd6fc3680f09d87c1e4cee6a79d8ec3731533718",
    "h4h5-files/chmap.h5": "sha256:668b5fe7e86903e6af4effdf65e3d2dd499a1217e93ca60d8b54b68941b6f1f7",
    "h4h5-files/fieldline.h5": "sha256:a5b2a1cc0c458d0d9510d8eacc93d3b4a2cc7e99e0a3f86cd3d6b164e74f370d",
    "h4h5-files/br.h5": "sha256:2038dc8e67303cf0b31414d532352b40e8c75ebd8917bc8b68614cf4e7b24055",
    "h4h5-files/rscale.hdf": "sha256:1c15bd669fc5a92dfdda7dc23703294c23f0a09440599fd5c30cf7a0e1a6f3c4",
    "h4h5-files/chmap.hdf": "sha256:fa2f1134aa4f1c9c0dd729b4e8f23f480bea5cb178e44e8da01bdffad09a2225",
    "h4h5-files/fieldline.hdf": "sha256:a4149783780e1ce44a8fe76a83c674e0a3082cd78c6a635b6c8e860e0fdd3891",
    "h4h5-files/br.hdf": "sha256:3a4b3174e5d6f45244bd25826890486b5659196b8fe093541c542375a88cdf52",
    "h4h5-files/synchronic_map.h5": "sha256:170794a5a19684246339ca9782a2b89066b89661400ec48bb6fc0a082e0a2450"
}
"""Registry of available magnetic field files with their SHA256 hashes. 

This registry is used by the pooch fetcher to verify the integrity of
downloaded files, and is primarily intended for building sphinx-gallery
examples that require MHD data files.
"""


BASE_URL = "https://www.predsci.com/doc/assets/"
"""Base URL hosting magnetic field file assets.
"""


FETCHER = pooch.create(
    path=pooch.os_cache("psi"),
    base_url=BASE_URL,
    registry=REGISTRY,
    env="PSI_IO_CACHE",
)
"""Pooch fetcher for downloading and caching magnetic field files.

.. note::
    The cache directory can be overridden by setting the ``PSI_IO_CACHE``
    environment variable to a desired path. Otherwise, the default cache
    directory is platform-dependent, as determined by :func:`pooch.os_cache`.
    
.. note::
    The default (os-dependent) cache directory stores assets under a
    subdirectory named ``psi``. The reason for this naming choice – as opposed
    to ``psi_io`` – is to maintain consistency with other PredSci packages
    that utilize the same asset hosting and caching mechanism.
"""

_P = ParamSpec("_P")
_R = TypeVar("_R")

def check_hdf_type(func: Callable[_P, _R]) -> Callable[_P, _R]:
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        hdf = bound.arguments["hdf"]  # assumes the param is literally named "hdf"
        if hdf not in HDFEXT:
            raise ValueError(f"Invalid HDF type {hdf!r}. Must be in {sorted(HDFEXT)}.")

        return func(*bound.args, **bound.kwargs)

    return wrapper


def file_ids() -> list[str]:
    """List all available magnetic field files in the registry.

    Returns
    -------
    list[str]
        List of available magnetic field file names.
    """
    return list(FETCHER.registry.keys())


@check_hdf_type
def get_1d_data(hdf: HdfExtType = ".h5") -> str:
    """Fetch the radial scale (1D) data file.

    Parameters
    ----------
    hdf : HdfExtType, optional
        The HDF file format to fetch, by default "h5".
        Accepted values are "h5" for HDF5 and "hdf" for HDF4.

    Returns
    -------
    str
        Path to the downloaded radial scale data file.
    """
    filename = f"h4h5-files/rscale{hdf}"
    return FETCHER.fetch(filename)


@check_hdf_type
def get_2d_data(hdf: HdfExtType = ".h5") -> str:
    """Fetch the coronal hole map (2D) data file.

    Parameters
    ----------
    hdf : HdfExtType, optional
        The HDF file format to fetch, by default "h5".
        Accepted values are "h5" for HDF5 and "hdf" for HDF4.

    Returns
    -------
    str
        Path to the downloaded coronal hole map data file.
    """
    filename = f"h4h5-files/chmap{hdf}"
    return FETCHER.fetch(filename)


@check_hdf_type
def get_3d_data(hdf: HdfExtType = ".h5") -> str:
    """Fetch the radial magnetic field (3D) data file.

    Parameters
    ----------
    hdf : HdfExtType, optional
        The HDF file format to fetch, by default "h5".
        Accepted values are "h5" for HDF5 and "hdf" for HDF4.

    Returns
    -------
    str
        Path to the downloaded radial magnetic field data file.
    """
    filename = f"h4h5-files/br{hdf}"
    return FETCHER.fetch(filename)


@check_hdf_type
def get_fieldline_data(hdf: HdfExtType = ".h5") -> str:
    """Fetch the magnetic fieldline (2D) data file.

    .. warning::
        Unlike the other example data files, fieldline data files do not
        contain scale datasets.

    Parameters
    ----------
    hdf : HdfExtType, optional
        The HDF file format to fetch, by default "h5".
        Accepted values are "h5" for HDF5 and "hdf" for HDF4.

    Returns
    -------
    str
        Path to the downloaded magnetic fieldline data file.
    """
    filename = f"h4h5-files/fieldline{hdf}"
    return FETCHER.fetch(filename)


@check_hdf_type
def get_synchronic_map_data(hdf: HdfExtType = ".h5") -> str:
    """Fetch the synchronic map data file.

    .. warning::
        Synchronic map data is only available in HDF5 format. Furthermore,
        unlike the other example data files, synchronic map data files contain
        additional datasets beyond the primary data and scales.

    Parameters
    ----------
    hdf : HdfExtType, optional
        The HDF file format to fetch, by default "h5".
        Accepted values are "h5" for HDF5 and "hdf" for HDF4.

    Returns
    -------
    str
        Path to the downloaded synchronic map data file.
    """
    if hdf == ".hdf":
        raise NotImplemented("Synchronic map data is only available in HDF5 format.")
    filename = f"h4h5-files/synchronic_map{hdf}"
    return FETCHER.fetch(filename)