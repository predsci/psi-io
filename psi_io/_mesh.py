r"""Mesh management utilities for PSI staggered grid data.

MHD model output from MAS and POT3D is computed on *staggered grids*: different
physical quantities are defined on different sets of grid nodes.  For a
three-dimensional spherical domain :math:`(r, \theta, \phi)`, each axis is
independently classified as either a *main-mesh* axis (quantity defined at cell
centres) or a *half-mesh* axis (quantity defined at cell faces, offset by half a
grid spacing in that direction).

A *mesh code* encodes the staggering of every axis in a single compact
representation.  The most common form is a non-negative integer whose binary
digits indicate, per axis, whether the data lives on the half mesh (1) or the
main mesh (0).

This module provides the following objects:

- :class:`Mesh` — enum distinguishing the two mesh positions.
- :data:`MeshCodeType` — type alias for the accepted mesh code forms.
- :data:`_MESH_CODE_REVERSE_MAPPING` — mapping from string tokens to integer
  mesh codes.
- :func:`_normalize_mesh_code` — converts any :data:`MeshCodeType` to a
  canonical ``tuple[Mesh, ...]``.
- :func:`_average_adjacent` — averages neighbouring element pairs along one
  array axis.
- :func:`remesh_arr` — shifts an array between meshes along selected axes.
- :func:`main_mesh` — converts a staggered array fully to the main mesh.

Examples
--------
Convert a radial magnetic-field array (half mesh on the last numpy axis) to the
main mesh:

>>> import numpy as np
>>> from psi_io._mesh import main_mesh
>>> br = np.ones((128, 64, 57))   # half-mesh size along last axis
>>> br_main = main_mesh(br, mesh_code=0b100)
>>> br_main.shape
(128, 64, 56)
"""

from __future__ import annotations

import enum
from types import MappingProxyType
from typing import Sequence, Any, Union, Literal

import numpy as np

MeshCodeType = Union[int, Literal['main', 'half'], Sequence[Any]]
"""Type alias for mesh stagger specifications accepted by :func:`_normalize_mesh_code`.

A mesh code may be expressed in any of three forms:

- :class:`int` — each binary digit encodes the stagger for one axis (1 = half
  mesh, 0 = main mesh).  The most-significant digit maps to the last numpy axis.
- ``'main'`` or ``'half'`` — shorthand for applying the same stagger to every
  axis.
- :class:`~typing.Sequence` — one element per array dimension; each element must
  be a key recognised by :data:`_MESH_CODE_REVERSE_MAPPING`.
"""


class Mesh(enum.Enum):
    """Enum identifying the mesh position of one array axis.

    MAS and POT3D solve their equations on staggered grids.  Each axis of a
    multi-dimensional output array is independently classified as :attr:`MAIN`
    (cell-centre position) or :attr:`HALF` (cell-face position, offset by half
    a grid spacing).

    Attributes
    ----------
    MAIN : int
        Main-mesh (cell-centre) position; encoded as ``0``.
    HALF : int
        Half-mesh (cell-face) position, offset by half a grid spacing; encoded
        as ``1``.

    Examples
    --------
    >>> from psi_io._mesh import Mesh
    >>> Mesh.MAIN.value
    0
    >>> Mesh.HALF.value
    1
    >>> str(Mesh.HALF)
    'HALF'
    """

    HALF = 1
    MAIN = 0

    def __str__(self) -> str:
        """Return the name of the mesh position as a string.

        Returns
        -------
        out : str
            The enum member name — either ``'MAIN'`` or ``'HALF'``.

        Examples
        --------
        >>> from psi_io._mesh import Mesh
        >>> str(Mesh.MAIN)
        'MAIN'
        >>> str(Mesh.HALF)
        'HALF'
        """
        return str(self.name)


_MESH_CODE_REVERSE_MAPPING = MappingProxyType({
    '1': 1, 'h': 1, 'half': 1, 'true': 1,
    '0': 0, 'm': 0, 'main': 0, 'false': 0
})
"""Read-only mapping from string tokens to integer mesh-position codes.

Maps every accepted string representation of a mesh position to its integer
equivalent: ``0`` for main mesh and ``1`` for half mesh.  Used by
:func:`_normalize_mesh_code` to validate and convert per-axis entries when a
:class:`~typing.Sequence` mesh code is provided.
"""


def _normalize_mesh_code(mesh_code: MeshCodeType, ndim: int) -> tuple[Mesh, ...]:
    r"""Normalize a mesh stagger specification to a tuple of :class:`Mesh` members.

    Accepts any of the three forms described by :data:`MeshCodeType` and converts
    them to a fixed-length tuple with one :class:`Mesh` entry per array axis.

    Parameters
    ----------
    mesh_code : MeshCodeType
        Stagger specification in any supported form:

        - :class:`int` — binary-encoded stagger.  The integer is formatted as an
          *ndim*-bit binary string; a ``1`` bit means half mesh for that axis.
          For example, ``0b010`` (decimal 2) applied to a 3-D array sets the
          middle axis to half mesh and leaves the others on the main mesh.
        - ``'main'`` — all axes on the main mesh; equivalent to the integer ``0``.
        - ``'half'`` — all axes on the half mesh; equivalent to the integer
          :math:`2^{ndim} - 1`.
        - :class:`~typing.Sequence` — one element per axis; each element must be
          a key in :data:`_MESH_CODE_REVERSE_MAPPING` (accepted values: ``0``,
          ``1``, ``'m'``, ``'h'``, ``'main'``, ``'half'``, ``'true'``,
          ``'false'``).
    ndim : int
        Number of array dimensions.  Used to zero-pad integer codes to the
        correct binary width and to validate the length of sequence codes.

    Returns
    -------
    out : tuple[Mesh, ...]
        Length-*ndim* tuple of :class:`Mesh` members, one per axis.

    Raises
    ------
    ValueError
        If *mesh_code* is a sequence whose length does not equal *ndim*.
    ValueError
        If any element of *mesh_code* is not a recognized mesh-code token.

    See Also
    --------
    Mesh : The enum returned in each position of the output tuple.
    remesh_arr : Remeshes an array using an explicit boolean mask per axis.
    main_mesh : Uses the normalized code to shift a staggered array to the main mesh.

    Examples
    --------
    >>> from psi_io._mesh import _normalize_mesh_code, Mesh
    >>> _normalize_mesh_code('main', ndim=3)
    (Mesh.MAIN, Mesh.MAIN, Mesh.MAIN)
    >>> _normalize_mesh_code('half', ndim=2)
    (Mesh.HALF, Mesh.HALF)
    >>> _normalize_mesh_code(0b010, ndim=3)
    (Mesh.MAIN, Mesh.HALF, Mesh.MAIN)
    >>> _normalize_mesh_code([1, 0, 1], ndim=3)
    (Mesh.HALF, Mesh.MAIN, Mesh.HALF)
    """
    if isinstance(mesh_code, int):
        mesh_code = format(mesh_code, f'0{ndim}b')
    elif mesh_code == 'main':
        mesh_code = '0' * ndim
    elif mesh_code == 'half':
        mesh_code = '1' * ndim
    elif len(mesh_code) != ndim:
        raise ValueError(f'Mesh code length {len(mesh_code)} does not match data ndim {ndim}.')
    try:
        return tuple(Mesh(_MESH_CODE_REVERSE_MAPPING[str(c).lower()]) for c in mesh_code)
    except KeyError as e:
        raise ValueError(f"Invalid mesh code character '{e.args[0]}'. "
                         f"Valid characters are: {', '.join(_MESH_CODE_REVERSE_MAPPING.keys())}") from None


def _average_adjacent(arr: np.ndarray, axis: int) -> np.ndarray:
    """Average adjacent pairs of elements along one array axis.

    Each output element is the arithmetic mean of two consecutive input elements,
    so the output has one fewer element than the input along *axis*.  This
    operation shifts data from a half-mesh position to the corresponding
    main-mesh position along that axis.

    Parameters
    ----------
    arr : np.ndarray
        Input array of any shape and numeric dtype.
    axis : int
        Axis along which to average.  Supports negative indexing.

    Returns
    -------
    out : np.ndarray
        Array with the same shape as *arr* except along *axis*, where the size
        is ``arr.shape[axis] - 1``.

    See Also
    --------
    remesh_arr : Applies adjacent averaging over multiple axes simultaneously.
    main_mesh : Applies adjacent averaging on all half-mesh axes of an array.

    Examples
    --------
    >>> import numpy as np
    >>> from psi_io._mesh import _average_adjacent
    >>> a = np.array([1.0, 3.0, 5.0])
    >>> _average_adjacent(a, axis=0)
    array([2., 4.])
    >>> b = np.ones((4, 3))
    >>> _average_adjacent(b, axis=0).shape
    (3, 3)
    >>> _average_adjacent(b, axis=-1).shape
    (4, 2)
    """
    slc_lo = [slice(None)] * arr.ndim
    slc_hi = [slice(None)] * arr.ndim
    slc_lo[axis] = slice(None, -1)
    slc_hi[axis] = slice(1, None)
    return 0.5 * (arr[tuple(slc_lo)] + arr[tuple(slc_hi)])


def remesh_arr(data: np.ndarray, remesh: Sequence[bool] | bool) -> np.ndarray:
    """Shift an array between meshes by averaging adjacent elements along selected axes.

    Each axis flagged for remeshing is reduced by one element via adjacent
    averaging (see :func:`_average_adjacent`).  Axes are processed sequentially
    from axis 0 to the last axis.

    Parameters
    ----------
    data : np.ndarray
        Input data array to remesh.
    remesh : Sequence[bool] | bool
        Remesh flags.  A single :class:`bool` applies the same flag to every
        axis.  A sequence must have one entry per array dimension; ``True``
        triggers adjacent averaging on that axis and ``False`` leaves it
        unchanged.

    Returns
    -------
    out : np.ndarray
        Remeshed array.  Its size along each flagged axis is reduced by one.

    See Also
    --------
    main_mesh : Higher-level wrapper that derives the remesh flags from a
        :data:`MeshCodeType`.
    _average_adjacent : The per-axis averaging operation used internally.

    Examples
    --------
    >>> import numpy as np
    >>> from psi_io._mesh import remesh_arr
    >>> a = np.arange(6.0).reshape(2, 3)
    >>> remesh_arr(a, remesh=[False, True]).shape
    (2, 2)
    >>> remesh_arr(a, remesh=True).shape
    (1, 2)
    >>> remesh_arr(a, remesh=False).shape
    (2, 3)
    """
    if isinstance(remesh, bool):
        remesh = [remesh] * data.ndim
    for i, shift in enumerate(remesh):
        if shift:
            data = _average_adjacent(data, i)
    return data


def main_mesh(data: np.ndarray,
              mesh_code: int | Sequence) -> np.ndarray:
    """Convert a staggered array to the main mesh in all half-mesh dimensions.

    For each axis whose mesh code is :attr:`Mesh.HALF`, adjacent elements are
    averaged (via :func:`_average_adjacent`), reducing that axis by one element.
    Axes already on the main mesh (:attr:`Mesh.MAIN`) are left unchanged.

    Parameters
    ----------
    data : np.ndarray
        Input array on a (possibly mixed) staggered grid.
    mesh_code : int | Sequence
        Mesh stagger specification in any form accepted by
        :func:`_normalize_mesh_code`.  Identifies which axes are on the half
        mesh and therefore require adjacent averaging.

    Returns
    -------
    out : np.ndarray
        Array fully on the main mesh.  Its size along each half-mesh axis is
        reduced by one.

    Notes
    -----
    The mapping from mesh-code bits to numpy axes reverses the standard
    binary digit ordering to match PSI's HDF storage convention.  Given an
    *ndim*-bit integer mesh code:

    - The **most-significant bit** maps to the **last** numpy axis (``axis=-1``).
    - The **least-significant bit** maps to the **first** numpy axis (``axis=0``).

    This reversal is consistent with HDF4 files written in Fortran column-major
    order, where the first physical coordinate varies fastest.  Reading such
    files into numpy (C row-major order) inverts the axis ordering, and the
    mesh code's bit convention accounts for this inversion.

    See Also
    --------
    _normalize_mesh_code : Converts *mesh_code* to a ``tuple[Mesh, ...]``.
    remesh_arr : Lower-level remesh that accepts an explicit boolean flag per axis.
    Mesh : Enum defining the :attr:`~Mesh.MAIN` and :attr:`~Mesh.HALF` states.

    Examples
    --------
    >>> import numpy as np
    >>> from psi_io._mesh import main_mesh
    >>> data = np.ones((4, 5, 6))

    Half mesh on the last numpy axis only (most-significant bit set):

    >>> main_mesh(data, mesh_code=0b100).shape
    (4, 5, 5)

    Half mesh on the first numpy axis only (least-significant bit set):

    >>> main_mesh(data, mesh_code=0b001).shape
    (3, 5, 6)

    All axes already on the main mesh — array is returned unchanged:

    >>> main_mesh(data, mesh_code=0).shape
    (4, 5, 6)
    """
    mesh_code = _normalize_mesh_code(mesh_code, data.ndim)
    for i, code in enumerate(mesh_code):
        if code.value:
            data = _average_adjacent(data, -i-1)
    return data