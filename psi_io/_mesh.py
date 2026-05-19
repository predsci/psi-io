r"""Mesh management utilities for PSI staggered grid data.

MAS and POT3D solve their equations on *staggered* (Yee-type) spherical grids
:math:`(r, \theta, \varphi)`.  Different physical quantities are located at
different positions within each grid cell so that discrete differential operators
(curl, divergence) are exactly satisfied at the discrete level.  Each axis of a
multi-dimensional output array is independently classified as either:

- **Main mesh** — quantity sampled at the cell-center nodes.
- **Half mesh** — quantity sampled at the face or edge midpoint, displaced by half a
  grid spacing along that axis.

Mesh codes
----------
A *mesh code* encodes the staggering of every axis in a single compact integer.
Each binary bit indicates, per axis, whether the data lives on the half mesh
(``1``) or the main mesh (``0``).  With PSI's Fortran column-major HDF convention
the **most-significant bit maps to the last numpy axis** (the radial :math:`r`
direction), so a three-bit code reads :math:`(r, \theta, \varphi)` MSB → LSB:

.. list-table::
   :header-rows: 1

   * - Code
     - :math:`r`
     - :math:`\theta`
     - :math:`\varphi`
     - Typical quantities
   * - ``0b100``
     - half
     - main
     - main
     - :math:`B_r` (MAS)
   * - ``0b010``
     - main
     - half
     - main
     - :math:`B_\theta` (MAS)
   * - ``0b001``
     - main
     - main
     - half
     - :math:`B_\varphi` (MAS)
   * - ``0b011``
     - main
     - half
     - half
     - :math:`v_r`, :math:`J_r` (MAS); :math:`B_r` (POT3D)
   * - ``0b101``
     - half
     - main
     - half
     - :math:`v_\theta`, :math:`J_\theta` (MAS); :math:`B_\theta` (POT3D)
   * - ``0b110``
     - half
     - half
     - main
     - :math:`v_\varphi`, :math:`J_\varphi` (MAS); :math:`B_\varphi` (POT3D)
   * - ``0b111``
     - half
     - half
     - half
     - scalars: :math:`T`, :math:`\rho`, :math:`p`, …
   * - ``0b000``
     - main
     - main
     - main
     - all-main; result of remeshing every axis

Accepted input forms for a mesh code are described by :data:`MeshCodeType`
(integer, string shorthand ``'main'``/``'half'``, or per-axis sequence).
The memory-order convention is described by :data:`ArrayOrdering`.

Public API
----------
:class:`Mesh`
    Enum with two members — :attr:`~Mesh.MAIN` and :attr:`~Mesh.HALF` — representing
    the two mesh positions.
:data:`MeshCodeType`
    Type alias for the three accepted forms of a mesh stagger specification.
:data:`ArrayOrdering`
    Type alias for the memory-order string (``'F'`` or ``'C'``) accepted by
    :func:`remesh_array`.
:func:`remesh_array`
    Shift an array from one mesh stagger to another by averaging adjacent elements
    along each axis that needs to move from half mesh to main mesh.

Examples
--------
Convert a radial magnetic-field array (half-mesh in :math:`r`, the last numpy
axis) to the all-main mesh:

>>> import numpy as np
>>> from psi_io._mesh import remesh_array
>>> br = np.ones((128, 64, 57))   # shape (Nφ, Nθ, Nr); Nr is half-mesh size
>>> br_main = remesh_array(br, imesh=0b100, omesh='main')
>>> br_main.shape
(128, 64, 56)

Remesh a scalar quantity (all-half, ``0b111``) to all-main:

>>> rho = np.ones((128, 64, 57))
>>> remesh_array(rho, imesh=0b111, omesh='main').shape
(127, 63, 56)
"""

from __future__ import annotations

__all__ = [
    "Mesh",
    "remesh_array",
]

import enum
from types import MappingProxyType
from typing import Sequence, Any, Union, Literal, Generator, Optional, Iterable

import numpy as np


_MESH_CODE_REVERSE_MAPPING = MappingProxyType({
    '1': 1, 'h': 1, 'half': 1, 'true': 1,
    '0': 0, 'm': 0, 'main': 0, 'false': 0
})
"""String-token → integer (0/1) lookup used to validate per-axis sequence mesh codes."""


MeshCodeType = Union[int, Literal['main', 'half'], Sequence[Any]]
"""Type alias for mesh stagger specifications accepted by :func:`remesh_array`.

A mesh stagger may be expressed in any of three equivalent forms:

- :class:`int` — binary-encoded stagger, one bit per axis (``1`` = half mesh,
  ``0`` = main mesh).  With PSI's Fortran HDF convention the most-significant
  bit maps to the last numpy axis (:math:`r`).  For example, ``0b100`` places
  the array on the half mesh only along :math:`r` (the last axis).
- ``'main'`` or ``'half'`` — string shorthand that applies the same stagger to
  every axis uniformly.
- :class:`~typing.Sequence` — one element per array dimension; each element may
  be ``0``, ``1``, ``'m'``, ``'h'``, ``'main'``, ``'half'``, ``'true'``, or
  ``'false'``.
"""

ArrayOrdering = Literal['F', 'C']
"""Type alias for the memory-order convention accepted by :func:`remesh_array`.

Controls how the bits of a :data:`MeshCodeType` integer map to numpy array axes.

``'F'``
    Fortran (column-major) order — the default for PSI data.  Because PSI HDF
    files are written by Fortran code, the physical ``(r, θ, φ)`` axis ordering
    is **reversed** in numpy storage: the **last** numpy axis corresponds to
    :math:`r`, the middle to :math:`\\theta`, and the **first** to :math:`\\varphi`.
    The most-significant bit of the mesh code therefore maps to the last numpy axis.
    Use this setting whenever the array was loaded directly from a PSI HDF file.
``'C'``
    C (row-major) order.  Use when the array has been transposed to numpy-native
    axis order (first axis = first physical coordinate, e.g. shape ``(Nr, Nθ, Nφ)``),
    so that the most-significant bit maps to the first numpy axis.
"""


class Mesh(enum.Enum):
    """Enum identifying the stagger position of one array axis.

    MAS and POT3D solve their equations on Yee-type staggered spherical grids.
    Each axis of a multi-dimensional output array is independently classified as
    :attr:`MAIN` (cell-center position) or :attr:`HALF` (cell-face/edge position,
    displaced by half a grid spacing along that axis).

    The stagger arrangement is physically motivated:

    - Magnetic field components (:math:`B_r`, :math:`B_\\theta`, :math:`B_\\varphi`)
      are face-centred — each component lives on the face through which it is the
      outward normal — so that :math:`\\nabla \\cdot \\mathbf{B} = 0` is satisfied
      exactly at the discrete level.
    - Current density components follow from
      :math:`\\mathbf{J} = \\nabla \\times \\mathbf{B}` and are therefore
      edge-centred (half mesh on the two transverse axes).
    - Scalar quantities (temperature, density, pressure) occupy the cell corners,
      which correspond to the half-mesh position on all three axes (``0b111``).

    :class:`Mesh` members appear as elements of the normalized mesh tuple returned
    by :attr:`~psi_io._models.Props.mesh` and accepted by :func:`remesh_array`.

    Attributes
    ----------
    MAIN : int
        Cell-center mesh position; encoded as ``0``.
    HALF : int
        Cell-face/edge mesh position, offset by half a grid spacing; encoded as ``1``.

    Examples
    --------
    >>> from psi_io._mesh import Mesh
    >>> Mesh.MAIN.value
    0
    >>> Mesh.HALF.value
    1
    >>> Mesh('half')
    <Mesh.HALF: 1>
    >>> Mesh('m')
    <Mesh.MAIN: 0>
    >>> str(Mesh.HALF)
    'HALF'
    """

    HALF = 1
    MAIN = 0

    @classmethod
    def _missing_(cls, key: Any) -> Mesh:
        """Look up *key* in :data:`_MESH_CODE_REVERSE_MAPPING`; return ``None`` if unrecognized."""
        code_ = _MESH_CODE_REVERSE_MAPPING.get(str(key).lower())
        return cls(code_) if code_ is not None else None  # type: ignore


    def __str__(self) -> str:
        """Return the enum member name (``'MAIN'`` or ``'HALF'``)."""
        return str(self.name)


def _normalize_mesh_code(mesh_code: MeshCodeType, ndim: int) -> tuple[Mesh, ...]:
    """Convert *mesh_code* to a length-*ndim* tuple of :class:`Mesh` members.

    Parameters
    ----------
    mesh_code : MeshCodeType
        Integer, ``'main'``/``'half'`` shorthand, or per-axis sequence.
    ndim : int
        Number of array dimensions.

    Returns
    -------
    out : tuple[Mesh, ...]
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
    """Return the mean of adjacent element pairs along *axis*, reducing that dimension by one."""
    slc_lo = [slice(None)] * arr.ndim
    slc_hi = [slice(None)] * arr.ndim
    slc_lo[axis] = slice(None, -1)
    slc_hi[axis] = slice(1, None)
    return 0.5 * (arr[tuple(slc_lo)] + arr[tuple(slc_hi)])


def _remesh_array(data: np.ndarray,
                  remesh: Iterable[bool] | bool
                  ) -> np.ndarray:
    """Apply :func:`_average_adjacent` on each axis where *remesh* is ``True``."""
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

    Compares the source mesh *imesh* against the target mesh *omesh* axis by axis
    and applies adjacent-element averaging on every axis that needs to move from
    half mesh to main mesh.  Only the half → main direction is supported;
    requesting main → half raises :class:`ValueError`.

    This is commonly needed before performing interpolation or arithmetic between
    quantities on different mesh positions.  For example, computing the magnitude of the
    magnetic requires :math:`B_r`, :math:`B_\\theta`, and
    :math:`B_\\varphi` on the same mesh: each must be remeshed from its native
    stagger (``0b100``, ``0b010``, ``0b001``) to a common target before squaring
    and summing.

    If *omesh* is ``None``, the array is returned unchanged.

    Parameters
    ----------
    data : np.ndarray
        Input array on the stagger described by *imesh*.
    imesh : MeshCodeType
        Source mesh stagger in any form accepted by :data:`MeshCodeType`.
    omesh : MeshCodeType or None, optional
        Target mesh stagger.  ``None`` (default) is a no-op.  Pass ``0`` or
        ``'main'`` to move every half-mesh axis to the main mesh.
    order : ArrayOrdering, optional
        Memory-order convention controlling how mesh-code bits map to numpy
        axes; see :data:`ArrayOrdering`.  Defaults to ``'F'`` (Fortran /
        PSI HDF column-major: last numpy axis = :math:`r`).

    Returns
    -------
    out : np.ndarray
        Array on the target mesh.  Each remeshed axis is reduced by one element
        via adjacent averaging; axes that already match are left unchanged.

    Raises
    ------
    ValueError
        If any axis in *omesh* requests half mesh where *imesh* is already main
        (upsampling is not supported).

    See Also
    --------
    Mesh : Enum representing the two mesh positions.
    MeshCodeType : Accepted forms for mesh stagger specifications.
    ArrayOrdering : Memory-order convention for bit–axis mapping.

    Examples
    --------
    Convert a radial magnetic-field array (half-mesh in :math:`r`, the last
    numpy axis) to the all-main mesh:

    >>> import numpy as np
    >>> from psi_io._mesh import remesh_array
    >>> br = np.ones((128, 64, 57))   # shape (Nφ, Nθ, Nr); Nr is half-mesh size
    >>> br_main = remesh_array(br, imesh=0b100, omesh='main')
    >>> br_main.shape
    (128, 64, 56)

    Remesh a scalar quantity (all-half, ``0b111``) to all-main:

    >>> rho = np.ones((128, 64, 57))
    >>> remesh_array(rho, imesh=0b111, omesh='main').shape
    (127, 63, 56)

    ``omesh=None`` is a no-op:

    >>> remesh_array(br, imesh=0b100).shape
    (128, 64, 57)

    No-op when source and target stagger already match:

    >>> remesh_array(br, imesh=0b100, omesh=0b100).shape
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
    """Yield per-axis remesh flags (``True`` = half→main) by comparing *imesh* to *omesh*."""
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
