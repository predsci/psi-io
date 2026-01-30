"""
Routines for reading/writing PSI style HDF5 and HDF4 data files.

Written by Ronald M. Caplan, Ryder Davidson, & Cooper Downs.

2023/09: Start with SVN version r454, 2023/09/12 by RC, Predictive Science Inc.

2024/06: CD: add the get_scales subroutines.

2024/11: RD: Major Update: Add several generic data loading capabilites for faster IO.
         - Read only the portions of data required (`read_hdf_by_index`, `read_hdf_by_value`).
         - Interpolate to slices along a given axes (`np_interpolate_slice_from_hdf`) or
           generic positions (`interpolate_positions_from_hdf`).

2025/06: CD: Prep for integration into psi-io package, HDF4 is now optional.

2026/01: RD: Refactor legacy routines to use new generic routines where possible.
"""

from __future__ import annotations

__all__ = [
    "read_hdf_meta",
    "read_rtp_meta",

    "get_scales_1d",
    "get_scales_2d",
    "get_scales_3d",

    "read_hdf_by_index",
    "read_hdf_by_value",
    "read_hdf_by_ivalue",

    "np_interpolate_slice_from_hdf",
    "sp_interpolate_slice_from_hdf",
    "interpolate_positions_from_hdf",

    "instantiate_linear_interpolator",
    "interpolate_point_from_1d_slice",
    "interpolate_point_from_2d_slice",

    "read_hdf_data",
    "rdhdf_1d",
    "rdhdf_2d",
    "rdhdf_3d",

    "write_hdf_data",
    "wrhdf_1d",
    "wrhdf_2d",
    "wrhdf_3d",
]

import math
from collections import namedtuple
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Literal, Tuple, Iterable, List, Dict, Union, Callable, Any

import numpy as np
import h5py as h5

# -----------------------------------------------------------------------------
# Optional Imports and Import Checking
# -----------------------------------------------------------------------------
# These packages are needed by several functions and must be imported in the
# module namespace.
try:
    import pyhdf.SD as h4
    H4_AVAILABLE = True
    NPTYPES_TO_SDCTYPES = MappingProxyType({
        "int8": h4.SDC.INT8,
        "uint8": h4.SDC.UINT8,
        "int16": h4.SDC.INT16,
        "uint16": h4.SDC.UINT16,
        "int32": h4.SDC.INT32,
        "uint32": h4.SDC.UINT32,
        "float16": h4.SDC.FLOAT32,
        "float32": h4.SDC.FLOAT32,
        "float64": h4.SDC.FLOAT64,
    })
except ImportError:
    H4_AVAILABLE = False
    NPTYPES_TO_SDCTYPES = {}

try:
    from scipy.interpolate import RegularGridInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# Functions to stop execution if a package doesn't exist.
def _except_no_pyhdf():
    if not H4_AVAILABLE:
        raise ImportError('The pyhdf package is required to read/write HDF4 .hdf files!')
    return


def _except_no_scipy():
    if not SCIPY_AVAILABLE:
        raise ImportError('The scipy package is required for the interpolation routines!')
    return


SDC_TYPE_CONVERSIONS = MappingProxyType({
    3: np.dtype("ubyte"),
    4: np.dtype("byte"),
    5: np.dtype("float32"),
    6: np.dtype("float64"),
    20: np.dtype("int8"),
    21: np.dtype("uint8"),
    22: np.dtype("int16"),
    23: np.dtype("uint16"),
    24: np.dtype("int32"),
    25: np.dtype("uint32")
})
"""Helper dictionary for mapping HDF4 types to numpy dtypes"""


PSI_DATA_ID = MappingProxyType({
    'h4': 'Data-Set-2',
    'h5': 'Data'
})
"""Mapping of PSI standard dataset names for HDF4 and HDF5 files"""


PSI_SCALE_ID = MappingProxyType({
    'h4': ('fakeDim0', 'fakeDim1', 'fakeDim2'),
    'h5': ('dim1', 'dim2', 'dim3')
})
"""Mapping of PSI standard scale names for HDF4 and HDF5 files"""


HDFEXT = {'.hdf', '.h5'}
"""Set of possible HDF file extensions"""


HdfExtType = Literal['.hdf', '.h5']
"""Type alias for possible HDF file extensions"""


HdfScaleMeta = namedtuple('HdfScaleMeta', ['name', 'type', 'shape', 'imin', 'imax'])
"""
    Named tuples for HDF metadata

    Parameters
    ----------
    name : str
        The name of the scale.
    type : str
        The data type of the scale.
    shape : Tuple[int, ...]
        The shape of the scale.
    imin : float
        The minimum value of the scale.
        This assumes the scale is monotonically increasing.
    imax : float
        The maximum value of the scale.
        This assumes the scale is monotonically increasing.
"""


HdfDataMeta = namedtuple('HdfDataMeta', ['name', 'type', 'shape', 'scales'])
"""
    Named tuple for HDF dataset metadata

    Parameters
    ----------
    name : str
        The name of the dataset.
    type : str
        The data type of the dataset.
    shape : tuple of int
        The shape of the dataset.
    scales : list of HdfScaleMeta
        A list of scale metadata objects corresponding to each dimension of the dataset.
        If the dataset has no scales, this list will be empty.
"""


def _dispatch_by_ext(ifile: Union[Path, str],
                     hdf4_func: Callable,
                     hdf5_func: Callable,
                     *args: Any, **kwargs: Any
                     ):
    """
    Dispatch function to call HDF4 or HDF5 specific functions based on file extension.

    Parameters
    ----------
    ifile : Path | str
        The path to the HDF file.
    hdf4_func : Callable
        The function to call for HDF4 files.
    hdf5_func : Callable
        The function to call for HDF5 files.
    *args : Any
        Positional arguments to pass to the selected function.

    Raises
    ------
    ValueError
        If the file does not have a `.hdf` or `.h5` extension.
    ImportError
        If the file is HDF4 and the `pyhdf` package is not available
    """
    ipath = Path(ifile)
    if ipath.suffix == '.h5':
        return hdf5_func(ifile, *args, **kwargs)
    if ipath.suffix == '.hdf':
        _except_no_pyhdf()
        return hdf4_func(ifile, *args, **kwargs)
    raise ValueError("File must be HDF4 (.hdf) or HDF5 (.h5)")


# -----------------------------------------------------------------------------
# "Classic" HDF reading and writing routines adapted from psihdf.py or psi_io.py.
# -----------------------------------------------------------------------------


def rdhdf_1d(hdf_filename: str
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Read a 1D PSI-style HDF5 or HDF4 file.

    Parameters
    ----------
    hdf_filename : str
        The path to the 1D HDF5 (.h5) or HDF4 (.hdf) file to read.

    Returns
    -------
    x : np.ndarray
        1D array of scales.
    f : np.ndarray
        1D array of data.
    """
    return _rdhdf_nd(hdf_filename, dimensionality=1)


def rdhdf_2d(hdf_filename: str
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a 2D PSI-style HDF5 or HDF4 file.

    The data in the HDF file is assumed to be ordered X,Y in Fortran order.

    Each dimension is assumed to have a 1D "scale" associated with it that
    describes the rectilinear grid coordinates in each dimension.

    Parameters
    ----------
    hdf_filename : str
        The path to the 1D HDF5 (.h5) or HDF4 (.hdf) file to read.

    Returns
    -------
    x : np.ndarray
        1D array of scales in the X dimension.
    y : np.ndarray
        1D array of scales in the Y dimension.
    f : np.ndarray
        2D array of data, C-ordered as shape(ny,nx) for Python (see note 1).
    """
    return _rdhdf_nd(hdf_filename, dimensionality=2)


def rdhdf_3d(hdf_filename: str
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a 3D PSI-style HDF5 or HDF4 file.

    The data in the HDF file is assumed to be ordered X,Y,Z in Fortran order.

    Each dimension is assumed to have a 1D "scale" associated with it that
    describes the rectilinear grid coordinates in each dimension.

    Parameters
    ----------
    hdf_filename : str
        The path to the 1D HDF5 (.h5) or HDF4 (.hdf) file to read.

    Returns
    -------
    x : np.ndarray
        1D array of scales in the X dimension.
    y : np.ndarray
        1D array of scales in the Y dimension.
    z : np.ndarray
        1D array of scales in the Z dimension.
    f : np.ndarray
        3D array of data, C-ordered as shape(nz,ny,nx) for Python (see note 1).
    """
    return _rdhdf_nd(hdf_filename, dimensionality=3)


def wrhdf_1d(hdf_filename: str,
             x: np.ndarray,
             f: np.ndarray) -> None:
    """Write a 1D PSI-style HDF5 or HDF4 file.

    Parameters
    ----------
    hdf_filename : str
        The path to the 1D HDF5 (.h5) or HDF4 (.hdf) file to write.
    x : np.ndarray
        1D array of scales.
    f : np.ndarray
        1D array of data.
    """
    x = np.asarray(x)
    y = np.array([])
    z = np.array([])
    f = np.asarray(f)
    _wrhdf(hdf_filename, x, y, z, f)


def wrhdf_2d(hdf_filename: str,
             x: np.ndarray,
             y: np.ndarray,
             f: np.ndarray) -> None:
    """Write a 2D PSI-style HDF5 or HDF4 file.

    The data in the HDF file will appear as X,Y in Fortran order.

    Each dimension requires a 1D "scale" associated with it that
    describes the rectilinear grid coordinates in each dimension.

    Parameters
    ----------
    hdf_filename : str
        The path to the 2D HDF5 (.h5) or HDF4 (.hdf) file to write.
    x : np.ndarray
        1D array of scales in the X dimension.
    y : np.ndarray
        1D array of scales in the Y dimension.
    f : np.ndarray
        2D array of data, C-ordered as shape(ny,nx) for Python (see note 1).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.array([])
    f = np.asarray(f)
    if hdf_filename.endswith('h5'):
        _wrhdf(hdf_filename, x, y, z, f)
    else:
        _wrhdf(hdf_filename, y, x, z, f)


def wrhdf_3d(hdf_filename: str,
             x: np.ndarray,
             y: np.ndarray,
             z: np.ndarray,
             f: np.ndarray) -> None:
    """Write a 3D PSI-style HDF5 or HDF4 file.

    The data in the HDF file will appear as X,Y,Z in Fortran order.

    Each dimension requires a 1D "scale" associated with it that
    describes the rectilinear grid coordinates in each dimension.

    Parameters
    ----------
    hdf_filename : str
        The path to the 3D HDF5 (.h5) or HDF4 (.hdf) file to write.
    x : np.ndarray
        1D array of scales in the X dimension.
    y : np.ndarray
        1D array of scales in the Y dimension.
    z : np.ndarray
        1D array of scales in the Z dimension.
    f : np.ndarray
        3D array of data, C-ordered as shape(nz,ny,nx) for Python (see note 1).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    f = np.asarray(f)
    if hdf_filename.endswith('h5'):
        _wrhdf(hdf_filename, x, y, z, f)
    else:
        _wrhdf(hdf_filename, z, y, x, f)


def get_scales_1d(filename: str
                  ) -> np.ndarray:
    """Wrapper to return the scales of a 1D PSI style HDF5 or HDF4 dataset.

    This routine does not load the data array so it can be quite fast for large files.

    Parameters
    ----------
    filename : str
        The path to the 1D HDF5 (.h5) or HDF4 (.hdf) file to read.

    Returns
    -------
    x : np.ndarray
        1D array of scales in the X dimension.
    """
    return _dispatch_by_ext(filename, _get_scales_nd_h4, _get_scales_nd_h5,
                            dimensionality=1)


def get_scales_2d(filename: str
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper to return the scales of a 2D PSI style HDF5 or HDF4 dataset.

    This routine does not load the data array so it can be quite fast for large files.

    The data in the HDF file is assumed to be ordered X,Y in Fortran order.

    Parameters
    ----------
    filename : str
        The path to the 1D HDF5 (.h5) or HDF4 (.hdf) file to read.

    Returns
    -------
    x : np.ndarray
        1D array of scales in the X dimension.
    y : np.ndarray
        1D array of scales in the Y dimension.
    """
    return _dispatch_by_ext(filename, _get_scales_nd_h4, _get_scales_nd_h5,
                            dimensionality=2)


def get_scales_3d(filename: str
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper to return the scales of a 3D PSI style HDF5 or HDF4 dataset.

    This routine does not load the data array so it can be quite fast for large files.

    The data in the HDF file is assumed to be ordered X,Y,Z in Fortran order.

    Parameters
    ----------
    filename : str
        The path to the 1D HDF5 (.h5) or HDF4 (.hdf) file to read.

    Returns
    -------
    x : np.ndarray
        1D array of scales in the X dimension.
    y : np.ndarray
        1D array of scales in the Y dimension.
    z : np.ndarray
        1D array of scales in the Z dimension.
    """
    return _dispatch_by_ext(filename, _get_scales_nd_h4, _get_scales_nd_h5,
                            dimensionality=3)


# -----------------------------------------------------------------------------
# "Updated" HDF reading and slicing routines for Hdf4 and Hdf5 datasets.
# -----------------------------------------------------------------------------


def read_hdf_meta(ifile: Union[Path, str], /,
                  dataset_id: Optional[str] = None
                  ) -> List[HdfDataMeta]:
    """
    Read metadata from an HDF4 (.hdf) or HDF5 (.h5) file.

    This function provides a unified interface to read metadata from both HDF4 and HDF5 files.

    .. warning::
       Unlike elsewhere in this module, the scales and datasets are read **as is**, *i.e.* without
       reordering scales to match PSI's Fortran data ecosystem.

    .. warning::
       Unlike elsewhere in this module, when ``None`` is passed to ``dataset_id``, all (non-scale)
       datasets are returned (instead of the default psi datasets *e.g.* 'Data-Set-2' or 'Data').
       This will, effectively, return the standard PSI datasets when reading PSI-style HDF files.

    Parameters
    ----------
    ifile : Path | str
        The path to the HDF file to read.
    dataset_id : str, optional
        The identifier of the dataset for which to read metadata.
        If ``None``, metadata for **all** datasets is returned.

    Returns
    -------
    out : list[HdfDataMeta]
        A list of metadata objects corresponding to the specified datasets.

    Raises
    ------
    ValueError
        If the file does not have a `.hdf` or `.h5` extension.

    Notes
    -----
    This function delegates to :func:`_read_h5_meta` for HDF5 files and :func:`_read_h4_meta`
    for HDF4 files based on the file extension.

    Although this function is designed to read metadata for dataset objects, it is possible to
    read metadata for coordinate variables (scales) by passing their names to ``dataset_id``,
    *e.g.* 'dim1', 'dim2', etc. However, this is not the intended use case.
    """

    return _dispatch_by_ext(ifile, _read_h4_meta, _read_h5_meta,
                            dataset_id=dataset_id)


def read_rtp_meta(ifile: Union[Path, str], /) -> Dict:
    """
    Read the scale metadata for PSI's 3D cubes.

    Parameters
    ----------
    ifile : Path | str
        The path to the HDF file to read.

    Returns
    -------
    out : dict
        A dictionary containing the RTP metadata.
        The value for each key ('r', 't', and 'p') is a tuple containing:

        1. The scale length
        2. The scale's value at the first index
        3. The scale's value at the last index

    Raises
    ------
    ValueError
        If the file does not have a `.hdf` or `.h5` extension.

    Notes
    -----
    This function delegates to :func:`_read_h5_rtp` for HDF5 files and :func:`_read_h4_rtp`
    for HDF4 files based on the file extension.

    """
    return _dispatch_by_ext(ifile, _read_h4_rtp, _read_h5_rtp)


def read_hdf_data(ifile: Union[Path, str], /,
                  dataset_id: Optional[str] = None,
                  return_scales: bool = True,
                  ) -> Tuple[np.ndarray]:
    """
    Read data from an HDF4 (.hdf) or HDF5 (.h5) file.

    Parameters
    ----------
    ifile : Path | str
         The path to the HDF file to read.
    dataset_id : str | None
        The identifier of the dataset to read.
        If None, a default dataset is used ('Data-Set-2' for HDF4 and 'Data' for HDF5).
    return_scales : bool
        If True, the scales (coordinate arrays) for each dimension are also returned.

    Returns
    -------
    out : np.ndarray | tuple[np.ndarray]
        The data array.
        If ``return_scales`` is True, returns a tuple containing the data array
        and the scales for each dimension.

    Raises
    ------
    ValueError
        If the file does not have a `.hdf` or `.h5` extension.

    See Also
    --------
    read_hdf_by_index
        Read HDF datasets by index.
    read_hdf_by_value
        Read HDF datasets by value ranges.
    read_hdf_by_ivalue
        Read HDF datasets by subindex values.

    Notes
    -----
    This function delegates to :func:`_read_h5_data` for HDF5 files
    and :func:`_read_h4_data` for HDF4 files based on the file extension.
    """
    return _dispatch_by_ext(ifile, _read_h4_data, _read_h5_data,
                            dataset_id=dataset_id, return_scales=return_scales)


def read_hdf_by_index(ifile: Union[Path, str], /,
                      *xi: Union[int, Tuple[Union[int, None], Union[int, None]], None],
                      dataset_id: Optional[str] = None,
                      return_scales: bool = True,
                      ) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """
    Read data from an HDF4 (.hdf) or HDF5 (.h5) file by index.

    .. attention::
       For each dimension, the *minimum* number of elements returned is 1 *e.g.*
       if 3 ints are passed (as positional `*xi` arguments) for a 3D dataset,
       the resulting subset will have a shape of (1, 1, 1,) with scales of length 1.

    Parameters
    ----------
    ifile : Path | str
       The path to the HDF file to read.
    *xi : int | tuple[int | None, int | None] | None
       Indices or ranges for each dimension of the `n`-dimensional dataset.
       Use None for a dimension to select all indices. If no arguments are passed,
       the entire dataset (and its scales) will be returned – see
       :func:`~psi_io.psi_io.read_hdf_data`.
    dataset_id : str | None
       The identifier of the dataset to read.
       If None, a default dataset is used ('Data-Set-2' for HDF4 and 'Data' for HDF5).
    return_scales : bool
       If True, the scales (coordinate arrays) for each dimension are also returned.

    Returns
    -------
    out : np.ndarray | tuple[np.ndarray]
       The selected data array.
       If ``return_scales`` is True, returns a tuple containing the data array
       and the scales for each dimension.

    Raises
    ------
    ValueError
       If the file does not have a `.hdf` or `.h5` extension.

    See Also
    --------
    read_hdf_by_value
        Read HDF datasets by value ranges.
    read_hdf_by_ivalue
        Read HDF datasets by subindex values.
    read_hdf_data
        Read entire HDF datasets.

    Notes
    -----
    This function delegates to :func:`_read_h5_by_index` for HDF5 files and
    :func:`_read_h4_by_index` for HDF4 files based on the file extension.

    This function assumes that the dataset is Fortran (or column-major) ordered *viz.* for
    compatibility with PSI's data ecosystem; as such, a given :math:`n`-dimensional array,
    of shape :math:`(D_0, D_1, ..., D_{n-1})`, has scales :math:`(x_0, x_1, ..., x_{n-1})`,
    such that :math:`| x_i | = | D_{(n-1)-i} |`. For example, a 3D dataset with shape
    :math:`(D_p, D_t, D_r)` has scales :math:`r, t, p` corresponding to the radial, theta,
    and phi dimensions respectively.

    This function extracts a subset of the given dataset/scales without reading the
    entire data into memory. For a given scale :math:`x_j`, an index, index range, or ``None``
    can be provided; these inputs are forwarded through to Python's builtin :class:`slice` to
    extract the desired subset of the scale(s) / dataset.

    Examples
    --------
    Import a 3D HDF5 cube.

    >>> from psi_io.data import get_3d_data
    >>> from psi_io import read_hdf_by_index
    >>> filepath = get_3d_data()

    Extract a radial slice (at the first radial-scale index) from a 3D cube:

    >>> f, r, t, p = read_hdf_by_value(filepath, 0, None, None)
    >>> f.shape, r.shape, t.shape, p.shape
    ((181, 100, 1), (1,), (100,), (181,))

    Extract a phi slice at the 90th index from a 3D cube:

    >>> f, r, t, p = read_hdf_by_value(filepath, None, None, 90)
    >>> f.shape, r.shape, t.shape, p.shape
    ((1, 100, 151), (151,), (100,), (1,))

    Extract the values up to the 20th index (in the radial dimension) and with
    phi indices from 10 to 25:

    >>> f, r, t, p = read_hdf_by_value(filepath, (None, 20), None, (10, 25))
    >>> f.shape, r.shape, t.shape, p.shape
    ((15, 100, 20), (20,), (100,), (15,))
    """
    if not xi:
        return read_hdf_data(ifile, dataset_id=dataset_id, return_scales=return_scales)
    return _dispatch_by_ext(ifile, _read_h4_by_index, _read_h5_by_index,
                            *xi, dataset_id=dataset_id, return_scales=return_scales)


def read_hdf_by_value(ifile: Union[Path, str], /,
                      *xi: Union[float, Tuple[float, float], None],
                      dataset_id: Optional[str] = None,
                      return_scales: bool = True,
                      ) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """
    Read data from an HDF4 (.hdf) or HDF5 (.h5) file by value.

    .. note::
       For each dimension, the minimum number of elements returned is 2 *e.g.*
       if 3 floats are passed (as positional `*xi` arguments) for a 3D dataset,
       the resulting subset will have a shape of (2, 2, 2,) with scales of length 2.

    Parameters
    ----------
    ifile : Path | str
        The path to the HDF file to read.
    *xi : float | tuple[float, float] | None
        Values or value ranges corresponding to each dimension of the `n`-dimensional
        dataset specified by the ``dataset_id``. If no arguments are passed, the
        entire dataset (and its scales) will be returned.
    dataset_id : str | None
        The identifier of the dataset to read.
        If None, a default dataset is used ('Data-Set-2' for HDF4 and 'Data' for HDF5).
    return_scales : bool
        If True, the scales for the specified dataset are also returned.

    Returns
    -------
    out : np.ndarray | tuple[np.ndarray]
        The selected data array.
        If ``return_scales`` is True, returns a tuple containing the data array
        and the scales for each dimension.

    Raises
    ------
    ValueError
        If the file does not have a `.hdf` or `.h5` extension.

    See Also
    --------
    read_hdf_by_index
        Read HDF datasets by index.
    read_hdf_by_ivalue
        Read HDF datasets by subindex values.
    read_hdf_data
        Read entire HDF datasets.
    sp_interpolate_slice_from_hdf
        Interpolate slices from HDF datasets using SciPy's
        :class:`~scipy.interpolate.RegularGridInterpolator`
    np_interpolate_slice_from_hdf
        Perform linear, bilinear, or trilinear interpolation using vectorized numpy-based
        routines.

    Notes
    -----
    This function delegates to :func:`_read_h5_by_value` for HDF5 files and
    :func:`_read_h4_by_value` for HDF4 files based on the file extension.

    This function assumes that the dataset is Fortran (or column-major) ordered *viz.* for
    compatibility with PSI's data ecosystem; as such, a given :math:`n`-dimensional array,
    of shape :math:`(D_0, D_1, ..., D_{n-1})`, has scales :math:`(x_0, x_1, ..., x_{n-1})`,
    such that :math:`| x_i | = | D_{(n-1)-i} |`. For example, a 3D dataset with shape
    :math:`(D_p, D_t, D_r)` has scales :math:`r, t, p` corresponding to the radial, theta,
    and phi dimensions respectively.

    This function extracts a subset of the given dataset/scales without reading the
    entire data into memory. For a given scale :math:`x_j`, if:

    - *i)* a single float is provided (:math:`a`), the function will return a 2-element
      subset of the scale (:math:`xʹ_j`) such that :math:`xʹ_j[0] <= a < xʹ_j[1]`.
    - *ii)* a (float, float) tuple is provided (:math:`a_0, a_1`), the function will return an
      *m*-element subset of the scale (:math:`xʹ_j`) where
      :math:`xʹ_j[0] <= a_0` and :math:`xʹ_j[m-1] > a_1`.
    - *iii)* a **None** value is provided, the function will return the entire scale :math:`x_j`

    The returned subset can then be passed to a linear interpolation routine to extract the
    "slice" at the desired fixed dimensions.

    Examples
    --------
    Import a 3D HDF5 cube.

    >>> from psi_io.data import get_3d_data
    >>> from psi_io import read_hdf_by_value
    >>> filepath = get_3d_data()

    Extract a radial slice at r=15 from a 3D cube:

    >>> f, r, t, p = read_hdf_by_value(filepath, 15, None, None)
    >>> f.shape, r.shape, t.shape, p.shape
    ((181, 100, 2), (2,), (100,), (181,))

    Extract a phi slice at p=1.57 from a 3D cube:

    >>> f, r, t, p = read_hdf_by_value(filepath, None, None, 1.57)
    >>> f.shape, r.shape, t.shape, p.shape
    ((2, 100, 151), (151,), (100,), (2,))

    Extract the values between 3.2 and 6.4 (in the radial dimension) and with
    phi equal to 4.5

    >>> f, r, t, p = read_hdf_by_value(filepath, (3.2, 6.4), None, 4.5)
    >>> f.shape, r.shape, t.shape, p.shape
    ((2, 100, 15), (15,), (100,), (2,))
    """
    if not xi:
        return read_hdf_data(ifile, dataset_id=dataset_id, return_scales=return_scales)
    return _dispatch_by_ext(ifile, _read_h4_by_value, _read_h5_by_value,
                            *xi, dataset_id=dataset_id, return_scales=return_scales)


def read_hdf_by_ivalue(ifile: Union[Path, str], /,
                      *xi: Union[float, Tuple[float, float], None],
                      dataset_id: Optional[str] = None,
                      return_scales: bool = True,
                      ) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """
    Read data from an HDF4 (.hdf) or HDF5 (.h5) file by value.

    .. note::
       For each dimension, the minimum number of elements returned is 2 *e.g.*
       if 3 floats are passed (as positional `*xi` arguments) for a 3D dataset,
       the resulting subset will have a shape of (2, 2, 2,) with scales of length 2.

    Parameters
    ----------
    ifile : Path | str
        The path to the HDF file to read.
    *xi : float | tuple[float, float] | None
        Values or value ranges corresponding to each dimension of the `n`-dimensional
        dataset specified by the ``dataset_id``. If no arguments are passed, the
        entire dataset (and its scales) will be returned.
    dataset_id : str | None
        The identifier of the dataset to read.
        If None, a default dataset is used ('Data-Set-2' for HDF4 and 'Data' for HDF5).
    return_scales : bool
        If True, arrays of indices for the specified dataset are also returned.
        Note, regardless of whether the provided dataset has coordinate variables
        (scales), the returned index arrays are always 0-based indices generated
        through :func:`~numpy.arange`.

    Returns
    -------
    out : np.ndarray | tuple[np.ndarray]
        The selected data array.
        If ``return_scales`` is True, returns a tuple containing the data array
        and the scales for each dimension.

    Raises
    ------
    ValueError
        If the file does not have a `.hdf` or `.h5` extension.

    See Also
    --------
    read_hdf_by_index
        Read HDF datasets by index.
    read_hdf_data
        Read entire HDF datasets.
    sp_interpolate_slice_from_hdf
        Interpolate slices from HDF datasets using SciPy's
        :class:`~scipy.interpolate.RegularGridInterpolator`
    np_interpolate_slice_from_hdf
        Perform linear, bilinear, or trilinear interpolation using vectorized numpy-based
        routines.

    Notes
    -----
    This function delegates to :func:`_read_h5_by_ivalue` for HDF5 files and
    :func:`_read_h4_by_ivalue` for HDF4 files based on the file extension.

    This function assumes that the dataset is Fortran (or column-major) ordered *viz.* for
    compatibility with PSI's data ecosystem; as such, a given :math:`n`-dimensional array,
    of shape :math:`(D_0, D_1, ..., D_{n-1})`, has scales :math:`(x_0, x_1, ..., x_{n-1})`,
    such that :math:`| x_i | = | D_{(n-1)-i} |`. For example, a 3D dataset with shape
    :math:`(D_p, D_t, D_r)` has scales :math:`r, t, p` corresponding to the radial, theta,
    and phi dimensions respectively.

    This function extracts a subset of the given dataset/scales without reading the
    entire data into memory. For a given scale :math:`x_j`, if:

    - *i)* a single float is provided (:math:`a`), the function will return a 2-element
      subset of the scale (:math:`xʹ_j`): :math:`xʹ_j[floor(a)], xʹ_j[ceil(a)]`.
    - *ii)* a (float, float) tuple is provided (:math:`a_0, a_1`), the function will return an
      *m*-element subset of the scale (:math:`xʹ_j`): :math:`xʹ_j[floor(a_0)], xʹ_j[ceil(a_1)]`
    - *iii)* a **None** value is provided, the function will return the entire scale :math:`x_j`

    The returned subset can then be passed to a linear interpolation routine to extract the
    "slice" at the desired fixed dimensions.
    """
    if not xi:
        return read_hdf_data(ifile, dataset_id=dataset_id, return_scales=return_scales)
    return _dispatch_by_ext(ifile, _read_h4_by_ivalue, _read_h5_by_ivalue,
                            *xi, dataset_id=dataset_id, return_scales=return_scales)


def write_hdf_data(ifile: Union[Path, str], /,
                   data: np.ndarray,
                   *scales: Iterable[np.ndarray],
                   dataset_id: Optional[str] = None
                   ) -> Path:
    """
    Write data to an HDF4 (.hdf) or HDF5 (.h5) file.

    Following PSI conventions, the data array is assumed to be Fortran-ordered,
    with the scales provided in the order corresponding to each dimension *e.g.* a
    3D dataset with shape (Dp, Dt, Dr) has scales r, t, p corresponding to the
    radial, theta, and phi dimensions respectively.

    Parameters
    ----------
    ifile : Path | str
        The path to the HDF file to write.
    data : np.ndarray
        The data array to write.
    *scales : Iterable[np.ndarray]
        The scales (coordinate arrays) for each dimension.
    dataset_id : str | None
        The identifier of the dataset to write.
        If None, a default dataset is used ('Data-Set-2' for HDF
        and 'Data' for HDF5).

    Returns
    -------
    out : Path
        The path to the written HDF file.

    Raises
    ------
    ValueError
        If the file does not have a `.hdf` or `.h5` extension.

    Notes
    -----
    This function delegates to :func:`_write_h5_data` for HDF5 files
    and :func:`_write_h4_data` for HDF4 files based on the file extension.

    See Also
    --------
    wrhdf_1d
        Write 1D HDF files.
    wrhdf_2d
        Write 2D HDF files.
    wrhdf_3d
        Write 3D HDF files.
    """
    return _dispatch_by_ext(ifile, _write_h4_data, _write_h5_data, data,
                            *scales, dataset_id=dataset_id)


def instantiate_linear_interpolator(*args, **kwargs):
    """
    Instantiate a linear interpolator using the provided data and scales.

    Parameters
    ----------
    *args : sequence[array_like]
        The first argument is the data array.
        Subsequent arguments are the scales (coordinate arrays) for each dimension.
    **kwargs : dict
        Additional keyword arguments to pass to
        :class:`~scipy.interpolate.RegularGridInterpolator`.

    Returns
    -------
    out : RegularGridInterpolator
        An instance of RegularGridInterpolator initialized
        with the provided data and scales.

    Notes
    -----
    This function transposes the data array and passes it along with the scales
    to RegularGridInterpolator. Given a PSI-style Fortran-ordered 3D dataset,
    the resulting interpolator can be queried using (r, theta, phi) coordinates.

    Examples
    --------
    Import a 3D HDF5 cube.

    >>> from psi_io.data import get_3d_data
    >>> from psi_io import read_hdf_by_value
    >>> from numpy import pi
    >>> filepath = get_3d_data()

    Read the dataset by value (at 15 R_sun in the radial dimension).

    >>> data_and_scales = read_hdf_by_value(filepath, 15, None, None)
    >>> interpolator = instantiate_linear_interpolator(*data_and_scales)

    Interpolate at a specific position.

    >>> interpolator((15, pi/2, pi))
    0.0012864485109423877
    """
    _except_no_scipy()
    return RegularGridInterpolator(
        values=args[0].T,
        points=args[1:],
        **kwargs)


def sp_interpolate_slice_from_hdf(*xi, **kwargs):
    """
    Interpolate a slice from HDF data using SciPy's `RegularGridInterpolator`.

    .. note::
       Slicing routines result in a dimensional reduction. The dimensions
       that are fixed (i.e. provided as float values in `*xi`) are removed
       from the output slice, while the dimensions that are not fixed
       (*i.e.* provided as None in `*xi`) are retained.

    Parameters
    ----------
    *xi : sequence
        Positional arguments passed-through to :func:`read_hdf_by_value`.
    **kwargs : dict
        Keyword arguments passed-through to :func:`read_hdf_by_value`.
        **NOTE: Instantiating a linear interpolator requires the** ``return_scales``
        **keyword argument to be set to True; this function overrides
        any provided value for** ``return_scales`` **to ensure this behavior.**

    Returns
    -------
    slice : np.ndarray
        The interpolated data slice.
    scales : list
        A list of scales for the dimensions that were not fixed.

    Notes
    -----
    This function reads data from an HDF file, creates a linear interpolator,
    and interpolates a slice based on the provided values.

    .. note::
       The returned slice is Fortran-ordered *e.g.* radial slices will have shape
       (phi, theta), phi slices will have shape (r, theta), etc.

    .. note::
       SciPy's `RegularGridInterpolator` casts all input data to `float64` internally.
       Therefore, PSI HDF datasets with single-precision (`float32`) data will be upcast
       during interpolation.

    Examples
    --------
    >>> from psi_io.data import get_3d_data
    >>> from psi_io import sp_interpolate_slice_from_hdf
    >>> from numpy import pi
    >>> filepath = get_3d_data()

    Fetch a 2D slice at r=15 from 3D map

    >>> slice_, theta_scale, phi_scale = sp_interpolate_slice_from_hdf(filepath, 15, None, None)
    >>> slice_.shape, theta_scale.shape, phi_scale.shape
    ((181, 100), (100,), (181,))

    Fetch a single point from 3D map

    >>> point_value, *_ = sp_interpolate_slice_from_hdf(filepath, 1, pi/2, pi)
    >>> point_value
    6.084495480971823
    """
    filepath, *args = xi
    kwargs.pop('return_scales', None)
    result = read_hdf_by_value(filepath, *args, **kwargs)
    interpolator = instantiate_linear_interpolator(*result)
    grid = [yi[0] if yi[0] is not None else yi[1] for yi in zip(args, result[1:])]
    slice_ = interpolator(tuple(np.meshgrid(*grid, indexing='ij')))
    indices = [0 if yi is not None else slice(None, None) for yi in args]
    return slice_[tuple(indices)].T, *[yi[1] for yi in zip(args, result[1:]) if yi[0] is None]


def np_interpolate_slice_from_hdf(ifile: Union[Path, str], /,
                       *xi: Union[float, Tuple[float, float], None],
                       dataset_id: Optional[str] = None,
                       by_index: bool = False,
                       ):
    """
    Interpolate a slice from HDF data using linear interpolation.

    .. note::
       Slicing routines result in a dimensional reduction. The dimensions
       that are fixed (i.e. provided as float values in `*xi`) are removed
       from the output slice, while the dimensions that are not fixed
       (*i.e.* provided as `None` in `*xi`) are retained.

    Parameters
    ----------
    ifile : Path | str
        The path to the HDF file to read.
    *xi : sequence
        Positional arguments passed-through to reader function.
    dataset_id : str | None
        The identifier of the dataset to read.
        If None, a default dataset is used ('Data-Set-2' for HDF
        and 'Data' for HDF5).
    by_index : bool
        If True, use :func:`read_hdf_by_ivalue` to read data by subindex values.
        If False, use :func:`read_hdf_by_value` to read data by value ranges.

    Returns
    -------
    slice : np.ndarray
        The interpolated data slice.
    scales : list
        A list of scales for the dimensions that were not fixed.

    Raises
    ------
    ValueError
        If the number of dimensions to interpolate over is not supported.

    Notes
    -----
    This function supports linear, bilinear, and trilinear interpolation
    depending on the number of dimensions fixed in `xi`.

    Examples
    --------
    >>> from psi_io.data import get_3d_data
    >>> from psi_io import np_interpolate_slice_from_hdf
    >>> from numpy import pi
    >>> filepath = get_3d_data()

    Fetch a 2D slice at r=15 from 3D map

    >>> slice_, theta_scale, phi_scale = np_interpolate_slice_from_hdf(filepath, 15, None, None)
    >>> slice_.shape, theta_scale.shape, phi_scale.shape
    ((181, 100), (100,), (181,))

    Fetch a single point from 3D map

    >>> point_value, *_ = np_interpolate_slice_from_hdf(filepath, 1, pi/2, pi)
    >>> point_value
    6.084496

    """
    reader = read_hdf_by_value if not by_index else read_hdf_by_ivalue
    data, *scales = reader(ifile, *xi, dataset_id=dataset_id, return_scales=True)
    f_ = np.transpose(data)
    slice_type = sum([yi is not None for yi in xi])
    if slice_type == 1:
        return _np_linear_interpolation(xi, scales, f_).T, *[yi[1] for yi in zip(xi, scales) if yi[0] is None]
    elif slice_type == 2:
        return _np_bilinear_interpolation(xi, scales, f_).T, *[yi[1] for yi in zip(xi, scales) if yi[0] is None]
    elif slice_type == 3:
        return _np_trilinear_interpolation(xi, scales, f_).T, *[yi[1] for yi in zip(xi, scales) if yi[0] is None]
    else:
        raise ValueError("Not a valid number of dimensions for supported linear interpolation methods")


def interpolate_positions_from_hdf(ifile, *xi, **kwargs):
    """
    Interpolate at a list of scale positions using SciPy's `RegularGridInterpolator`.

    Parameters
    ----------
    ifile : Path | str
        The path to the HDF file to read.
    *xi : sequence[np.ndarray]
       Iterable scale values for each dimension of the `n`-dimensional dataset.
       Each array should have the same shape *i.e.* :math:`(m, )` – the function composes
       these array into a :math:`m x n` column stack for interpolation.
    **kwargs : dict
        Keyword arguments to pass to :func:`read_hdf_by_value`.

    Returns
    -------
    out : np.ndarray
        The interpolated values at the provided positions.

    Notes
    -----
    This function reads data from an HDF file, creates a linear interpolator,
    and interpolates at the provided scale values. For each dimension, the
    minimum and maximum values from the provided arrays are used to read
    the necessary subset of data from the HDF file *viz.* to avoid loading
    the entire dataset into memory.

    Examples
    --------
    Import a 3D HDF5 cube.

    >>> from psi_io.data import get_3d_data
    >>> from psi_io import interpolate_positions_from_hdf
    >>> import numpy as np
    >>> filepath = get_3d_data()

    Set up positions to interpolate.

    >>> r_vals = np.array([15, 20, 25])
    >>> theta_vals = np.array([np.pi/4, np.pi/2, 3*np.pi/4])
    >>> phi_vals = np.array([0, np.pi, 2*np.pi])

    Interpolate at the specified positions.

    >>> interpolate_positions_from_hdf(filepath, r_vals, theta_vals, phi_vals)
    [0.0008402743657585175, 0.000723875405654482, -0.00041033233811179216]
    """
    xi_ = [(np.nanmin(i), np.nanmax(i)) for i in xi]
    f, *scales = read_hdf_by_value(ifile, *xi_, **kwargs)
    interpolator = instantiate_linear_interpolator(f, *scales, bounds_error=False)
    return interpolator(np.stack(xi, axis=len(xi[0].shape)))


def interpolate_point_from_1d_slice(xi, scalex, values):
    """
    Interpolate a point from a 1D slice using linear interpolation.

    Parameters
    ----------
    xi : float
        The scale value at which to interpolate.
    scalex : np.ndarray
        The scale (coordinate array) for the dimension.
    values : np.ndarray
        The data array to interpolate.

    Returns
    -------
    out : np.ndarray
        The interpolated data point.
    """
    if scalex[0] > scalex[-1]:
        scalex, values = scalex[::-1], values[::-1]
    xi_ = int(np.searchsorted(scalex, xi))
    sx_ = slice(*_check_index_ranges(len(scalex), xi_, xi_))
    return _np_linear_interpolation([xi], [scalex[sx_]], values[sx_])


def interpolate_point_from_2d_slice(xi, yi, scalex, scaley, values):
    """
    Interpolate a point from a 2D slice using bilinear interpolation.

    Parameters
    ----------
    xi : float
        The scale value for the first dimension.
    yi : float
        The scale value for the second dimension.
    scalex : np.ndarray
        The scale (coordinate array) for the first dimension.
    scaley : np.ndarray
        The scale (coordinate array) for the second dimension.
    values : np.ndarray
        The data array to interpolate.

    Returns
    -------
    out : np.ndarray
        The interpolated data point.
    """
    values = np.transpose(values)
    if scalex[0] > scalex[-1]:
        scalex, values = scalex[::-1], values[::-1, :]
    if scaley[0] > scaley[-1]:
        scaley, values = scaley[::-1], values[:, ::-1]
    xi_, yi_ = int(np.searchsorted(scalex, xi)), int(np.searchsorted(scaley, yi))
    sx_, sy_ = slice(*_check_index_ranges(len(scalex), xi_, xi_)), slice(*_check_index_ranges(len(scaley), yi_, yi_))
    return _np_bilinear_interpolation([xi, yi], [scalex[sx_], scaley[sy_]], values[(sx_, sy_)])


def _rdhdf_nd(hdf_filename: str, dimensionality: int) -> Tuple[np.ndarray, ...]:
    f, *scales = read_hdf_data(hdf_filename)
    if f.ndim != dimensionality:
        err = f'Expected {dimensionality}D data, got {f.ndim}D data instead.'
        raise ValueError(err)
    scales = scales or (np.empty(0) for _ in f.shape)
    return *scales, f


def _wrh5(h5_filename: str,
          x: np.ndarray,
          y: np.ndarray,
          z: np.ndarray,
          f: np.ndarray) -> None:
    """Base writer for 1D, 2D, and 3D HDF5 files.

    See Also
    --------
    :func:`wrhdf_1d`, :func:`wrhdf_2d`, :func:`wrhdf_3d`
    """
    h5file = h5.File(h5_filename, 'w')

    # Create the dataset (Data is the name used by the psi data)).
    h5file.create_dataset("Data", data=f)

    # Make sure the scales are desired by checking x type, which can
    # be None or None converted by np.asarray (have to trap seperately)
    if x is None:
        x = np.array([], dtype=f.dtype)
        y = np.array([], dtype=f.dtype)
        z = np.array([], dtype=f.dtype)
    if x.any() == None:
        x = np.array([], dtype=f.dtype)
        y = np.array([], dtype=f.dtype)
        z = np.array([], dtype=f.dtype)

    # Make sure scales are the same precision as data.
    x = x.astype(f.dtype)
    y = y.astype(f.dtype)
    z = z.astype(f.dtype)

    # Get number of dimensions:
    ndims = np.ndim(f)

    # Set the scales:
    for i in range(0, ndims):
        if i == 0 and len(x) != 0:
            dim = h5file.create_dataset("dim1", data=x)
            #            h5file['Data'].dims.create_scale(dim,'dim1')
            dim.make_scale('dim1')
            h5file['Data'].dims[0].attach_scale(dim)
            h5file['Data'].dims[0].label = 'dim1'
        if i == 1 and len(y) != 0:
            dim = h5file.create_dataset("dim2", data=y)
            #            h5file['Data'].dims.create_scale(dim,'dim2')
            dim.make_scale('dim2')
            h5file['Data'].dims[1].attach_scale(dim)
            h5file['Data'].dims[1].label = 'dim2'
        elif i == 2 and len(z) != 0:
            dim = h5file.create_dataset("dim3", data=z)
            #            h5file['Data'].dims.create_scale(dim,'dim3')
            dim.make_scale('dim3')
            h5file['Data'].dims[2].attach_scale(dim)
            h5file['Data'].dims[2].label = 'dim3'

    # Close the file:
    h5file.close()


def _wrhdf(hdf_filename: str,
          x: np.ndarray,
          y: np.ndarray,
          z: np.ndarray,
          f: np.ndarray) -> None:
    """Base writer for 1D, 2D, and 3D HDF4 files.

    See Also
    --------
    :func:`wrhdf_1d`, :func:`wrhdf_2d`, :func:`wrhdf_3d`
    """
    if hdf_filename.endswith('h5'):
        return _wrh5(hdf_filename, x, y, z, f)

    # Check for HDF4
    _except_no_pyhdf()

    # Create an HDF file
    sd_id = h4.SD(hdf_filename, h4.SDC.WRITE | h4.SDC.CREATE | h4.SDC.TRUNC)

    # Due to bug, need to only write 64-bit.
    f = f.astype(np.float64)
    ftype = h4.SDC.FLOAT64

    #    if f.dtype == np.float32:
    #        ftype = h4.SDC.FLOAT32
    #    elif f.dtype == np.float64:
    #        ftype = h4.SDC.FLOAT64

    # Create the dataset (Data-Set-2 is the name used by the psi data)).
    sds_id = sd_id.create("Data-Set-2", ftype, f.shape)

    # Get number of dimensions:
    ndims = np.ndim(f)

    # Make sure the scales are desired by checking x type, which can
    # be None or None converted by np.asarray (have to trap seperately)
    if x is None:
        x = np.array([], dtype=f.dtype)
        y = np.array([], dtype=f.dtype)
        z = np.array([], dtype=f.dtype)
    if x.any() == None:
        x = np.array([], dtype=f.dtype)
        y = np.array([], dtype=f.dtype)
        z = np.array([], dtype=f.dtype)

    # Due to python hdf4 bug, need to use double scales only.

    # NOTE: this bug is due to an inability of SWIG (which is used to generate
    # the C-to-python pyhdf bindings) to handle numpy types in the overloaded functions.
    # The only reason that np.float64 works is that it is equivalent to
    # the native python float type. This issue has been remedied below in the
    # write_hdf functions.

    x = x.astype(np.float64)
    y = y.astype(np.float64)
    z = z.astype(np.float64)

    # Set the scales (or don't if x is none or length zero)
    for i in range(0, ndims):
        dim = sds_id.dim(i)
        if i == 0 and len(x) != 0:
            if x.dtype == np.float32:
                stype = h4.SDC.FLOAT32
            elif x.dtype == np.float64:
                stype = h4.SDC.FLOAT64
            dim.setscale(stype, x)
        elif i == 1 and len(y) != 0:
            if y.dtype == np.float32:
                stype = h4.SDC.FLOAT32
            elif y.dtype == np.float64:
                stype = h4.SDC.FLOAT64
            dim.setscale(stype, y)
        elif i == 2 and len(z) != 0:
            if z.dtype == np.float32:
                stype = h4.SDC.FLOAT32
            elif z.dtype == np.float64:
                stype = h4.SDC.FLOAT64
            dim.setscale(stype, z)

    # Write the data:
    sds_id.set(f)

    # Close the dataset:
    sds_id.endaccess()

    # Flush and close the HDF file:
    sd_id.end()


def _get_scales_nd_h5(ifile: Union[ Path, str], /,
                      dimensionality: int,
                      dataset_id: Optional[str] = None,
                      ):
    with h5.File(ifile, 'r') as hdf:
        data = hdf[dataset_id or PSI_DATA_ID['h5']]
        ndim = data.ndim
        if ndim != dimensionality:
            err = f'Expected {dimensionality}D data, got {ndim}D data instead.'
            raise ValueError(err)
        scales = []
        for dim in data.dims:
            if dim:
                scales.append(dim[0][:])
            else:
                raise ValueError(f'Dimension has no scale associated with it.')
    return tuple(scales)


def _get_scales_nd_h4(ifile: Union[ Path, str], /,
                      dimensionality: int,
                      dataset_id: Optional[str] = None,
                      ):
    hdf = h4.SD(str(ifile))
    data = hdf.select(dataset_id or PSI_DATA_ID['h4'])
    ndim = data.info()[1]
    if ndim != dimensionality:
        err = f'Expected {dimensionality}D data, got {ndim}D data instead.'
        raise ValueError(err)
    scales = []
    for k_, v_ in reversed(data.dimensions(full=1).items()):
        if v_[3]:
            scales.append(hdf.select(k_)[:])
        else:
            raise ValueError('Dimension has no scale associated with it.')
    return tuple(scales)


def _read_h5_meta(ifile: Union[Path, str], /,
                  dataset_id: Optional[str] = None
                  ):
    """HDF5 (.h5) version of :func:`read_hdf_meta`."""
    with h5.File(ifile, 'r') as hdf:
        # Raises KeyError if ``dataset_id`` not found
        # If ``dataset_id`` is None, get all non-scale :class:`h5.Dataset`s
        if dataset_id:
            datasets = (dataset_id, hdf[dataset_id]),
        else:
            datasets = ((k, v) for k, v in hdf.items() if not v.is_scale)

        # One should avoid multiple calls to ``dimproxy[0]`` – *e.g.* ``dimproxy[0].dtype`` and
        # ``dimproxy[0].shape`` – because the __getitem__ method creates and returns a new
        # :class:`~h5.DimensionProxy` object each time it is called. [Does this matter? Probably not.]
        return [HdfDataMeta(name=k,
                            type=v.dtype,
                            shape=v.shape,
                            scales=[HdfScaleMeta(name=dimproxy.label,
                                                 type=dim.dtype,
                                                 shape=dim.shape,
                                                 imin=dim[0],
                                                 imax=dim[-1])
                                    for dimproxy in v.dims if dimproxy and (dim := dimproxy[0])])
                for k, v in datasets]


def _read_h4_meta(ifile: Union[Path, str], /,
                  dataset_id: Optional[str] = None
                  ):
    """HDF4 (.hdf) version of :func:`read_hdf_meta`."""
    hdf = h4.SD(str(ifile))
    # Raises HDF4Error if ``dataset_id`` not found
    # If ``dataset_id`` is None, get all non-scale :class:`pyhdf.SD.SDS`s
    if dataset_id:
        datasets = (dataset_id, hdf.select(dataset_id)),
    else:
        datasets = ((k, hdf.select(k)) for k in hdf.datasets().keys() if not hdf.select(k).iscoordvar())

    # The inner list comprehension differs in approach from the HDF5 version because calling
    # ``dimensions(full=1)`` on an :class:`~pyhdf.SD.SDS` returns a dictionary of dimension
    # dataset identifiers (keys) and tuples containing dimension metadata (values). Even if no
    # coordinate-variable datasets are defined, this dictionary is still returned; the only
    # indication that the datasets returned do not exist is that the "type" field (within the
    # tuple of dimension metadata) is set to 0.

    # Also, one cannot avoid multiple calls to ``hdf.select(k_)`` within the inner list comprehension
    # because :class:`~pyhdf.SD.SDS` objects do not define a ``__bool__`` method, and the fallback
    # behavior of Python is to assess if the __len__ method returns a non-zero value (which, in
    # this case, always returns 0).
    return [HdfDataMeta(name=k,
                        type=SDC_TYPE_CONVERSIONS[v.info()[3]],
                        shape=_cast_shape_tuple(v.info()[2]),
                        scales=[HdfScaleMeta(name=k_,
                                             type=SDC_TYPE_CONVERSIONS[v_[3]],
                                             shape=_cast_shape_tuple(v_[0]),
                                             imin=hdf.select(k_)[0],
                                             imax=hdf.select(k_)[-1])
                                for k_, v_ in v.dimensions(full=1).items() if v_[3]])
            for k, v in datasets]


def _read_h5_rtp(ifile: Union[ Path, str], /):
    """HDF5 (.h5) version of :func:`read_rtp_meta`."""
    with h5.File(ifile, 'r') as hdf:
        return {k: (hdf[v].size, hdf[v][0], hdf[v][-1])
                for k, v in zip('rtp', PSI_SCALE_ID['h5'])}


def _read_h4_rtp(ifile: Union[ Path, str], /):
    """HDF4 (.hdf) version of :func:`read_rtp_meta`."""
    hdf = h4.SD(str(ifile))
    return {k: (hdf.select(v).info()[2], hdf.select(v)[0], hdf.select(v)[-1])
            for k, v in zip('ptr', PSI_SCALE_ID['h4'])}


def _read_h5_data(ifile: Union[Path, str], /,
                  dataset_id: Optional[str] = None,
                  return_scales: bool = True,
                  ) -> Union[np.ndarray, Tuple[np.ndarray]]:
    with h5.File(ifile, 'r') as hdf:
        data = hdf[dataset_id or PSI_DATA_ID['h5']]
        dataset = data[:]
        if return_scales:
            scales = [dim[0][:] for dim in data.dims if dim]
            return dataset, *scales
        return dataset


def _read_h4_data(ifile: Union[Path, str], /,
                  dataset_id: Optional[str] = None,
                  return_scales: bool = True,
                  ) -> Union[np.ndarray, Tuple[np.ndarray]]:
    hdf = h4.SD(str(ifile))
    data = hdf.select(dataset_id or PSI_DATA_ID['h4'])
    if return_scales:
        out = (data[:],
               *[hdf.select(k_)[:] for k_, v_ in reversed(data.dimensions(full=1).items()) if v_[3]])
    else:
        out = data[:]
    return out


def _read_h5_by_index(ifile: Union[Path, str], /,
                      *xi: Union[int, Tuple[Union[int, None], Union[int, None]], None],
                      dataset_id: Optional[str] = None,
                      return_scales: bool = True,
                      ) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """ HDF5(.h5) version of :func:`read_hdf_by_index`."""
    with h5.File(ifile, 'r') as hdf:
        data = hdf[dataset_id or PSI_DATA_ID['h5']]
        if len(xi) != data.ndim:
            raise ValueError(f"len(xi) must equal the number of scales for {dataset_id}")
        slices = [_parse_index_inputs(slice_input) for slice_input in xi]
        dataset = data[tuple(reversed(slices))]
        if return_scales:
            scales = [dim[0][si] for si, dim in zip(slices, data.dims) if dim]
            return dataset, *scales
        return dataset

def _read_h4_by_index(ifile: Union[Path, str], /,
                      *xi: Union[int, Tuple[Union[int, None], Union[int, None]], None],
                      dataset_id: Optional[str] = None,
                      return_scales: bool = True,
                      ) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """HDF4(.hdf) version of :func:`read_hdf_by_index`."""
    hdf = h4.SD(str(ifile))
    data = hdf.select(dataset_id or PSI_DATA_ID['h4'])
    ndim = data.info()[1]
    if len(xi) != ndim:
        raise ValueError(f"len(xi) must equal the number of scales for {dataset_id}")
    slices = [_parse_index_inputs(slice_input) for slice_input in xi]
    dataset = data[tuple(reversed(slices))]
    if return_scales:
        scales = [hdf.select(k_)[si] for si, (k_, v_) in zip(slices, reversed(data.dimensions(full=1).items())) if v_[3]]
        return dataset, *scales
    return dataset


def _read_h5_by_value(ifile: Union[Path, str], /,
                      *xi: Union[float, Tuple[float, float], None],
                      dataset_id: Optional[str] = None,
                      return_scales: bool = True,
                      ) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """HDF5 (.h5) version of :func:`read_hdf_by_value`."""
    with h5.File(ifile, 'r') as hdf:
        data = hdf[dataset_id or PSI_DATA_ID['h5']]
        if len(xi) != data.ndim:
            raise ValueError(f"len(xi) must equal the number of scales for {dataset_id}")
        slices = []
        for dimproxy, value in zip(data.dims, xi):
            if dimproxy:
                slices.append(_parse_value_inputs(dimproxy[0], value))
            elif value is None:
                slices.append(slice(None))
            else:
                raise ValueError("Cannot slice by value on dimension without scales")
        dataset = data[tuple(reversed(slices))]
        if return_scales:
            scales = [dim[0][si] for si, dim in zip(slices, data.dims) if dim]
            return dataset, *scales
        return dataset


def _read_h4_by_value(ifile: Union[Path, str], /,
                      *xi: Union[float, Tuple[float, float], None],
                      dataset_id: Optional[str] = None,
                      return_scales: bool = True,
                      ) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """HDF4 (.hdf) version of :func:`read_hdf_by_value`."""
    hdf = h4.SD(str(ifile))
    data = hdf.select(dataset_id or PSI_DATA_ID['h4'])
    ndim = data.info()[1]
    if len(xi) != ndim:
        raise ValueError(f"len(xi) must equal the number of scales for {dataset_id}")
    slices = []
    for (k_, v_), value in zip(reversed(data.dimensions(full=1).items()), xi):
        if v_[3] != 0:
            slices.append(_parse_value_inputs(hdf.select(k_), value))
        elif value is None:
            slices.append(slice(None))
        else:
            raise ValueError("Cannot slice by value on dimension without scales")
    dataset = data[tuple(reversed(slices))]
    if return_scales:
        scales = [hdf.select(k_)[si] for si, (k_, v_) in zip(slices, reversed(data.dimensions(full=1).items())) if v_[3]]
        return dataset, *scales
    return dataset


def _read_h5_by_ivalue(ifile: Union[Path, str], /,
                       *xi: Union[float, Tuple[float, float], None],
                       dataset_id: Optional[str] = None,
                       return_scales: bool = True,
                       ) -> Union[np.ndarray, Tuple[np.ndarray]]:
    with h5.File(ifile, 'r') as hdf:
        data = hdf[dataset_id or PSI_DATA_ID['h5']]
        if len(xi) != data.ndim:
            raise ValueError(f"len(xi) must equal the number of scales for {dataset_id}")
        slices = [_parse_ivalue_inputs(*args) for args in zip(reversed(data.shape), xi)]
        dataset = data[tuple(reversed(slices))]
        if return_scales:
            scales = [np.arange(si.start or 0, si.stop or size) for si, size in zip(slices, reversed(data.shape))]
            return dataset, *scales
        return dataset


def _read_h4_by_ivalue(ifile: Union[Path, str], /,
                       *xi: Union[float, Tuple[float, float], None],
                       dataset_id: Optional[str] = None,
                       return_scales: bool = True,
                       ) -> Union[np.ndarray, Tuple[np.ndarray]]:
    hdf = h4.SD(str(ifile))
    data = hdf.select(dataset_id or PSI_DATA_ID['h4'])
    ndim, shape = data.info()[1], _cast_shape_tuple(data.info()[2])
    if len(xi) != ndim:
        raise ValueError(f"len(xi) must equal the number of scales for {dataset_id}")
    slices = [_parse_ivalue_inputs(*args) for args in zip(reversed(shape), xi)]
    dataset = data[tuple(reversed(slices))]
    if return_scales:
        scales = [np.arange(si.start or 0, si.stop or size) for si, size in zip(slices, reversed(shape))]
        return dataset, *scales
    return dataset


def _write_h4_data(ifile: Union[Path, str], /,
                   data: np.ndarray,
                   *scales: Iterable[np.ndarray],
                   dataset_id: Optional[str] = None
                   ) -> Path:
    dataid = dataset_id or PSI_DATA_ID['h4']
    h4file = h4.SD(str(ifile), h4.SDC.WRITE | h4.SDC.CREATE | h4.SDC.TRUNC)
    sds_id = h4file.create(dataid, NPTYPES_TO_SDCTYPES[data.dtype.name], data.shape)

    if scales:
        for i, scale in enumerate(reversed(scales)):
            sds_id.dim(i).setscale(NPTYPES_TO_SDCTYPES[scale.dtype.name], scale.tolist())

    sds_id.set(data)
    sds_id.endaccess()
    h4file.end()

    return ifile


def _write_h5_data(ifile: Union[Path, str], /,
                   data: np.ndarray,
                   *scales: Iterable[np.ndarray],
                   dataset_id: Optional[str] = None
                   ) -> Path:
    dataid = dataset_id or PSI_DATA_ID['h5']
    with h5.File(ifile, "w") as h5file:
        h5file.create_dataset(dataid, data=data, dtype=data.dtype, shape=data.shape)

        if scales:
            for i, scale in enumerate(scales):
                h5file.create_dataset(f"dim{i+1}", data=scale, dtype=scale.dtype, shape=scale.shape)
                h5file[dataid].dims[i].attach_scale(h5file[f"dim{i+1}"])
                h5file[dataid].dims[i].label = f"dim{i+1}"

    return ifile


def _np_linear_interpolation(xi: Iterable, scales: Iterable, values: np.ndarray):
    """
    Perform linear interpolation over one dimension.

    Parameters
    ----------
    xi : list
        List of values or None for each dimension.
    scales : list
        List of scales (coordinate arrays) for each dimension.
    values : np.ndarray
        The data array to interpolate.

    Returns
    -------
    np.ndarray
        The interpolated data.

    """
    index0 = next((i for i, v in enumerate(xi) if v is not None), None)
    t = (xi[index0] - scales[index0][0])/(scales[index0][1] - scales[index0][0])
    f0 = [slice(None, None)]*values.ndim
    f1 = [slice(None, None)]*values.ndim
    f0[index0] = 0
    f1[index0] = 1

    return (1 - t)*values[tuple(f0)] + t*values[tuple(f1)]


def _np_bilinear_interpolation(xi, scales, values):
    """
    Perform bilinear interpolation over two dimensions.

    Parameters
    ----------
    xi : Iterable
        List of values or None for each dimension.
    scales : Iterable
        List of scales (coordinate arrays) for each dimension.
    values : np.ndarray
        The data array to interpolate.

    Returns
    -------
    np.ndarray
        The interpolated data.

    """
    index0, index1 = [i for i, v in enumerate(xi) if v is not None]
    t, u = [(xi[i] - scales[i][0])/(scales[i][1] - scales[i][0]) for i in (index0, index1)]

    f00 = [slice(None, None)]*values.ndim
    f10 = [slice(None, None)]*values.ndim
    f01 = [slice(None, None)]*values.ndim
    f11 = [slice(None, None)]*values.ndim
    f00[index0], f00[index1] = 0, 0
    f10[index0], f10[index1] = 1, 0
    f01[index0], f01[index1] = 0, 1
    f11[index0], f11[index1] = 1, 1

    return (
          (1 - t)*(1 - u)*values[tuple(f00)] +
          t*(1 - u)*values[tuple(f10)] +
          (1 - t)*u*values[tuple(f01)] +
          t*u*values[tuple(f11)]
    )


def _np_trilinear_interpolation(xi, scales, values):
    """
    Perform trilinear interpolation over three dimensions.

    Parameters
    ----------
    xi : list
        List of values or None for each dimension.
    scales : list
        List of scales (coordinate arrays) for each dimension.
    values : np.ndarray
        The data array to interpolate.

    Returns
    -------
    np.ndarray
        The interpolated data.

    """
    index0, index1, index2 = [i for i, v in enumerate(xi) if v is not None]
    t, u, v = [(xi[i] - scales[i][0])/(scales[i][1] - scales[i][0]) for i in (index0, index1, index2)]

    f000 = [slice(None, None)]*values.ndim
    f100 = [slice(None, None)]*values.ndim
    f010 = [slice(None, None)]*values.ndim
    f110 = [slice(None, None)]*values.ndim
    f001 = [slice(None, None)]*values.ndim
    f101 = [slice(None, None)]*values.ndim
    f011 = [slice(None, None)]*values.ndim
    f111 = [slice(None, None)]*values.ndim

    f000[index0], f000[index1], f000[index2] = 0, 0, 0
    f100[index0], f100[index1], f100[index2] = 1, 0, 0
    f010[index0], f010[index1], f010[index2] = 0, 1, 0
    f110[index0], f110[index1], f110[index2] = 1, 1, 0
    f001[index0], f001[index1], f001[index2] = 0, 0, 1
    f101[index0], f101[index1], f101[index2] = 1, 0, 1
    f011[index0], f011[index1], f011[index2] = 0, 1, 1
    f111[index0], f111[index1], f111[index2] = 1, 1, 1

    c00 = values[tuple(f000)]*(1 - t) + values[tuple(f100)]*t
    c10 = values[tuple(f010)]*(1 - t) + values[tuple(f110)]*t
    c01 = values[tuple(f001)]*(1 - t) + values[tuple(f101)]*t
    c11 = values[tuple(f011)]*(1 - t) + values[tuple(f111)]*t

    c0 = c00*(1 - u) + c10*u
    c1 = c01*(1 - u) + c11*u

    return c0*(1 - v) + c1*v


def _check_index_ranges(arr_size: int,
                        i0: Union[int, np.integer],
                        i1: Union[int, np.integer]
                        ) -> Tuple[int, int]:
    """
    Adjust index ranges to ensure they cover at least two indices.

    Parameters
    ----------
    arr_size : int
        The size of the array along the dimension.
    i0 : int
        The starting index.
    i1 : int
        The ending index.

    Returns
    -------
    Tuple[int, int]
        Adjusted starting and ending indices.

    Notes
    -----
    This function ensures that the range between `i0` and `i1` includes at least
    two indices for interpolation purposes.

    """
    i0, i1 = int(i0), int(i1)
    if i0 == 0:
        return (i0, i1 + 2) if i1 == 0 else (i0, i1 + 1)
    elif i0 == arr_size:
        return i0 - 2, i1
    else:
        return i0 - 1, i1 + 1


def _cast_shape_tuple(input: Union[int, Iterable[int]]
                      ) -> tuple[int, ...]:
    """
    Cast an input to a tuple of integers.

    Parameters
    ----------
    input : int | Iterable[int]
        The input to cast.

    Returns
    -------
    tuple[int]
        The input cast as a tuple of integers.

    Raises
    ------
    TypeError
        If the input is neither an integer nor an iterable of integers.
    """
    if isinstance(input, int):
        return (input,)
    elif isinstance(input, Iterable):
        return tuple(int(i) for i in input)
    else:
        raise TypeError("Input must be an integer or an iterable of integers.")


def _parse_index_inputs(input: Union[int, slice, Iterable[Union[int, None]], None]
                        ) -> slice:
    """
    Parse various slice input formats into a standard slice object.

    Parameters
    ----------
    input : int | slice | Iterable[Union[int, None]] | None
        The input to parse.
    arr_size : int
        The size of the array along the dimension.

    Returns
    -------
    slice
        The parsed slice object.

    Raises
    ------
    TypeError
        If the input type is unsupported.
    ValueError
        If the input iterable does not have exactly two elements.
    """
    if isinstance(input, int):
        return slice(input, input + 1)
    elif isinstance(input, Iterable):
        return slice(*input)
    elif input is None:
        return slice(None)
    else:
        raise TypeError("Unsupported input type for slicing.")


def _parse_value_inputs(dimproxy,
                        value,
                        scale_exists: bool = True
                        ) -> slice:
    if value is None:
        return slice(None)
    if not scale_exists:
        raise ValueError("Cannot parse value inputs when scale does not exist.")
    dim = dimproxy[:]
    if not isinstance(value, Iterable):
        insert_index = np.searchsorted(dim, value)
        return slice(*_check_index_ranges(dim.size, insert_index, insert_index))
    else:
        temp_range = list(value)
        if temp_range[0] is None:
            temp_range[0] = -np.inf
        if temp_range[-1] is None:
            temp_range[-1] = np.inf
        insert_indices = np.searchsorted(dim, temp_range)
        return slice(*_check_index_ranges(dim.size, *insert_indices))


def _parse_ivalue_inputs(dimsize,
                         input: Union[Union[int, float], slice, Iterable[Union[Union[int, float], None]], None]
                         ) -> slice:
    """
    Parse various slice input formats into a standard slice object.

    Parameters
    ----------
    input : int | slice | Iterable[Union[int, None]] | None
        The input to parse.
    arr_size : int
        The size of the array along the dimension.

    Returns
    -------
    slice
        The parsed slice object.

    Raises
    ------
    TypeError
        If the input type is unsupported.
    ValueError
        If the input iterable does not have exactly two elements.
    """
    if input is None:
        return slice(None)
    elif isinstance(input, (int, float)):
        i0, i1 = math.floor(input), math.ceil(input)
    elif isinstance(input, Iterable):
        i0, i1 = math.floor(input[0]), math.ceil(input[1])
    else:
        raise TypeError("Unsupported input type for slicing.")

    if i0 > i1:
        i0, i1 = i1, i0
    i0, i1 = max(0, i0), min(dimsize - 1, i1)
    if i0 > i1:
        i0, i1 = i1, i0
    i0, i1 = max(0, i0), min(dimsize - 1, i1)
    if (i1 - i0) < 2:
        return slice(i0, i1 + 2 - (i1-i0))
    else:
        return slice(i0, i1)
