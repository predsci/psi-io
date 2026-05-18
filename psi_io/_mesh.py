r"""Mesh management utilities for PSI staggered grid data.

MHD model output from MAS and POT3D is computed on *staggered grids*: different
physical quantities are defined on different sets of grid nodes.  For a
three-dimensional spherical domain :math:`(r, \theta, \phi)`, each axis is
independently classified as either a *main-mesh* axis (quantity defined at cell
centers) or a *half-mesh* axis (quantity defined at cell faces, offset by half a
grid spacing in that direction).

A *mesh code* encodes the staggering of every axis in a single compact
representation.  The most common form is a non-negative integer whose binary
digits indicate, per axis, whether the data lives on the half mesh (1) or the
main mesh (0).

This module provides the following objects:

- :class:`Mesh` — enum distinguishing the two mesh positions.
- :data:`MeshCodeType` — type alias for the accepted mesh code forms.
- :data:`ArrayOrdering` — type alias for array memory-order strings.
- :data:`_MESH_CODE_REVERSE_MAPPING` — mapping from string tokens to integer
  mesh codes.
- :func:`_normalize_mesh_code` — converts any :data:`MeshCodeType` to a
  canonical ``tuple[Mesh, ...]``.
- :func:`_average_adjacent` — averages neighboring element pairs along one
  array axis.
- :func:`_remesh_array` — applies per-axis adjacent averaging from a boolean
  flag sequence.
- :func:`remesh_array` — shifts an array from one mesh stagger to another,
  given source and target :data:`MeshCodeType` specifications.
- :func:`_parse_remesh` — derives per-axis remesh flags from source and
  target mesh tuples.

Examples
--------
Convert a radial magnetic-field array (half mesh in *r*, the last numpy axis)
to the main mesh:

>>> import numpy as np
>>> from psi_io._mesh import remesh_array
>>> br = np.ones((128, 64, 57))   # half-mesh size along last axis
>>> br_main = remesh_array(br, imesh=0b100, omesh='main')
>>> br_main.shape
(128, 64, 56)
"""

from __future__ import annotations

import enum
from types import MappingProxyType
from typing import Sequence, Any, Union, Literal, Generator, Optional, Iterable

import numpy as np


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


MeshCodeType = Union[int, Literal['main', 'half'], Sequence[Any]]
"""Type alias for mesh stagger specifications accepted by :func:`_normalize_mesh_code`.

A mesh code may be expressed in any of three forms:

- :class:`int` — each binary digit encodes the stagger for one axis (1 = half
  mesh, 0 = main mesh).  The most-significant digit maps to the last numpy axis.
- ``'main'`` or ``'half'`` — shorthand for applying the same stagger to every
  axis.
- :class:`~typing.Sequence` — one element per array dimension; each element must
  be a key recognized by :data:`_MESH_CODE_REVERSE_MAPPING`.
"""

ArrayOrdering = Literal['F', 'C']
"""Type alias for accepted array memory-order strings used by :func:`remesh_array`.

``'F'``
    Fortran (column-major) order.  PSI HDF files are written by Fortran code, so
    the physical ``(r, θ, φ)`` axis ordering is **reversed** relative to numpy's
    C-major storage.  This is the default and should be used when reading data
    directly from PSI HDF files.
``'C'``
    C (row-major) order.  Use when the array is already in numpy-native axis order
    (first axis = first physical coordinate).
"""


class Mesh(enum.Enum):
    """Enum identifying the mesh position of one array axis.

    MAS and POT3D solve their equations on staggered grids.  Each axis of a
    multi-dimensional output array is independently classified as :attr:`MAIN`
    (cell-center position) or :attr:`HALF` (cell-face position, offset by half
    a grid spacing).

    Attributes
    ----------
    MAIN : int
        Main-mesh (cell-center) position; encoded as ``0``.
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

    @classmethod
    def _missing_(cls, key: Any) -> Mesh:
        """Return the :class:`Mesh` member corresponding to *key*, or ``None``.

        Called automatically by :class:`~enum.Enum` when a direct value lookup
        (``Mesh(value)``) fails.  Converts *key* to a string, lower-cases it,
        and looks it up in :data:`_MESH_CODE_REVERSE_MAPPING`.  Accepted tokens
        are the same as for per-axis sequence codes in
        :func:`_normalize_mesh_code`: ``'0'``, ``'1'``, ``'m'``, ``'h'``,
        ``'main'``, ``'half'``, ``'true'``, ``'false'``.

        Parameters
        ----------
        key : Any
            Value that was not found by the normal enum lookup.  Converted to
            ``str`` before comparison, so integers, booleans, and strings all
            work.

        Returns
        -------
        out : Mesh or None
            The matching :class:`Mesh` member, or ``None`` if *key* is not
            recognized (which causes :class:`~enum.Enum` to raise
            :class:`ValueError`).

        Examples
        --------
        >>> from psi_io._mesh import Mesh
        >>> Mesh('half')
        <Mesh.HALF: 1>
        >>> Mesh('m')
        <Mesh.MAIN: 0>
        >>> Mesh('true')
        <Mesh.HALF: 1>
        """
        code_ = _MESH_CODE_REVERSE_MAPPING.get(str(key).lower())
        return cls(code_) if code_ is not None else None  # type: ignore


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


def _average_adjacent(arr: np.ndarray,
                      axis: int
                      ) -> np.ndarray:
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
    _remesh_array : Applies adjacent averaging over multiple axes simultaneously.
    remesh_array : Higher-level wrapper accepting mesh-code specifications.

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


def _remesh_array(data: np.ndarray,
                  remesh: Iterable[bool] | bool
                  ) -> np.ndarray:
    """Shift an array between meshes by averaging adjacent elements along selected axes.

    Each axis flagged for remeshing is reduced by one element via adjacent
    averaging (see :func:`_average_adjacent`).  Axes are processed sequentially
    from axis 0 to the last axis.

    Parameters
    ----------
    data : np.ndarray
        Input data array to remesh.
    remesh : Iterable[bool] | bool
        Remesh flags.  A single :class:`bool` applies the same flag to every
        axis.  An iterable must yield one entry per array dimension; ``True``
        triggers adjacent averaging on that axis and ``False`` leaves it
        unchanged.

    Returns
    -------
    out : np.ndarray
        Remeshed array.  Its size along each flagged axis is reduced by one.

    See Also
    --------
    remesh_array : Higher-level wrapper that derives the remesh flags from
        source and target :data:`MeshCodeType` specifications.
    _average_adjacent : The per-axis averaging operation used internally.

    Examples
    --------
    >>> import numpy as np
    >>> from psi_io._mesh import _remesh_array
    >>> a = np.arange(6.0).reshape(2, 3)
    >>> _remesh_array(a, remesh=[False, True]).shape
    (2, 2)
    >>> _remesh_array(a, remesh=True).shape
    (1, 2)
    >>> _remesh_array(a, remesh=False).shape
    (2, 3)
    """
    if isinstance(remesh, bool):
        remesh = [remesh] * data.ndim
    for i, shift in enumerate(remesh):
        if shift:
            data = _average_adjacent(data, i)
    return data


def remesh_array(data: np.ndarray,
                 imesh: MeshCodeType,
                 omesh: Optional[MeshCodeType] = None,
                 order: ArrayOrdering = 'F') -> np.ndarray:
    """Shift an array from one mesh stagger to another.

    Derives per-axis remesh flags by comparing *imesh* against *omesh*, then
    applies adjacent averaging (via :func:`_remesh_array`) on every axis that
    needs to move from half mesh to main mesh.  Only the half → main direction
    is supported; requesting main → half raises :class:`ValueError`.

    If *omesh* is ``None`` the array is returned unchanged.

    Parameters
    ----------
    data : np.ndarray
        Input array on the stagger described by *imesh*.
    imesh : MeshCodeType
        Source mesh stagger in any form accepted by :func:`_normalize_mesh_code`.
    omesh : MeshCodeType or None, optional
        Target mesh stagger.  ``None`` (default) is a no-op — the array is
        returned as-is.  Pass ``0`` or ``'main'`` to move all half-mesh axes
        to the main mesh.
    order : {'F', 'C'}, optional
        Memory-order convention that governs how mesh-code bits map to numpy
        axes.  ``'F'`` (default) reverses the bit–axis mapping to match PSI
        HDF files written in Fortran column-major order (last numpy axis = r).
        Use ``'C'`` when the array is already in C row-major order.

    Returns
    -------
    out : np.ndarray
        Array on the target mesh.  Each remeshed axis is reduced by one element
        (adjacent averaging).

    Raises
    ------
    ValueError
        If any axis in *omesh* requests half mesh where *imesh* is already on
        the main mesh (upsampling is not supported).

    See Also
    --------
    _remesh_array : Low-level per-axis averaging from an explicit boolean sequence.
    _parse_remesh : Derives the per-axis boolean flags from normalized mesh tuples.
    _normalize_mesh_code : Converts *imesh* / *omesh* to ``tuple[Mesh, ...]``.

    Examples
    --------
    Convert a radial magnetic-field array from half mesh in *r* to all-main:

    >>> import numpy as np
    >>> from psi_io._mesh import remesh_array
    >>> br = np.ones((128, 64, 57))   # (Nφ, Nθ, Nr); Nr is half-mesh size
    >>> br_main = remesh_array(br, imesh=0b100, omesh='main')
    >>> br_main.shape
    (128, 64, 56)

    ``omesh=None`` is a no-op:

    >>> remesh_array(br, imesh=0b100).shape
    (128, 64, 57)
    """
    if omesh is None:
        return data
    imesh_norm = _normalize_mesh_code(imesh, data.ndim)
    omesh_norm = _normalize_mesh_code(omesh, data.ndim)
    remesh_flags = _parse_remesh(imesh_norm, omesh_norm, order == 'F')
    return _remesh_array(data, remesh_flags)


def _parse_remesh(imesh: tuple[Mesh, ...],
                  omesh: tuple[Mesh, ...],
                  reverse: bool = False
                  ) -> Generator[bool]:
    """Compute per-axis remesh flags from source and target mesh tuples.

    Compares the stored mesh stagger *imesh* against the requested target *omesh*
    and yields a boolean flag for each axis:

    - ``False`` — meshes match; no averaging needed.
    - ``True``  — source is half mesh, target is main mesh; averaging will be applied.
    - :class:`ValueError` — source is main mesh but target requests half mesh (not
      supported; averaging can only go from half → main).

    Parameters
    ----------
    imesh : tuple[Mesh, ...]
        Current mesh stagger of the stored data, one :class:`Mesh` per axis.
    omesh : tuple[Mesh, ...]
        Target mesh stagger requested by the caller, same length as *imesh*.
    reverse : bool, optional
        If ``True``, iterate the axis pairs in reverse order before yielding
        flags.  Used when the array was loaded from a Fortran column-major
        file (``order='F'`` in :func:`remesh_array`) so that the MSB of the
        mesh code — which maps to the **last** numpy axis — is processed first.
        Defaults to ``False``.

    Yields
    ------
    flag : bool
        ``True`` if that axis should be remeshed (half → main), ``False`` if
        the meshes already match.

    Raises
    ------
    ValueError
        If any axis requests upsampling from main mesh to half mesh.
    ValueError
        If any axis pair contains an unrecognized :class:`Mesh` combination.
    """
    if reverse:
        imesh, omesh = reversed(imesh), reversed(omesh)
    for im, om in zip(imesh, omesh, strict=True):
        if im == om:
            yield False
        elif im == Mesh.HALF and om == Mesh.MAIN:
            yield True
        elif im == Mesh.MAIN and om == Mesh.HALF:
            raise ValueError(f"Cannot remesh from MAIN mesh to HALF mesh.")
        else:
            raise ValueError(f"Invalid mesh combination: {im} → {om}.")
