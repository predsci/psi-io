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
import functools
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Sequence, Any, Union, Literal, Generator, Optional, Iterable

import numpy as np


_MESH_CODE_REVERSE_MAPPING = MappingProxyType({
    '1': 1, 'h': 1, 'half': 1, 'true': 1,
    '0': 0, 'm': 0, 'main': 0, 'false': 0
})
"""String-token → integer (0/1) lookup used to validate per-axis sequence mesh codes."""


MeshCodeType = Union[bool, int, Literal['main', 'half'], Sequence[Any]]
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

MeshLike = Union['Mesh', MeshCodeType]
"""Type alias for any accepted mesh specification, including an existing :class:`Mesh`."""

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


def _coerce_mesh_target(method: Callable) -> Callable:
    """Coerce the *target* argument of a :class:`Mesh` method to a :class:`Mesh`.

    If *target* is already a :class:`Mesh`, its ``ndim`` is verified against
    ``self.ndim``.  Otherwise :meth:`Mesh.parse` is called with
    ``ndim=self.ndim`` to normalize it.
    """
    @functools.wraps(method)
    def wrapper(self: 'Mesh', target: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(target, Mesh):
            if self.ndim != target.ndim:
                raise ValueError(f"ndim mismatch: {self.ndim} vs {target.ndim}.")
        elif target is None:
            target = self
        else:
            target = Mesh.parse(target, ndim=self.ndim)
        return method(self, target, *args, **kwargs)
    return wrapper


@functools.total_ordering
@dataclass(frozen=True)
class Mesh:
    """Compact, immutable representation of a multi-axis mesh stagger code.

    Wraps a binary integer *code* whose bits indicate, per axis, whether the
    data lives on the half mesh (``1``) or the main mesh (``0``).  The
    most-significant bit maps to the first logical axis (physical :math:`r`
    direction in PSI convention).

    Prefer constructing via :meth:`parse` rather than directly, as it
    accepts all forms described by :data:`MeshCodeType`.

    Parameters
    ----------
    code : int
        Binary-encoded stagger integer, must fit within *ndim* bits.
    ndim : int
        Number of array dimensions (bits) represented by this code.

    Raises
    ------
    ValueError
        If *code* has bits set above position *ndim* - 1.

    Examples
    --------
    >>> Mesh.parse(0b100, ndim=3)
    Mesh(code=4, ndim=3)
    >>> str(Mesh.parse(0b100, ndim=3))
    'HALF, MAIN, MAIN'
    >>> Mesh.parse('MMH', ndim=3)
    Mesh(code=1, ndim=3)
    >>> Mesh.parse([True, False, True], ndim=3)
    Mesh(code=5, ndim=3)
    """

    code: int
    ndim: int

    def __post_init__(self) -> None:
        mask = (1 << self.ndim) - 1
        if self.code & ~mask:
            raise ValueError(f"Code 0b{self.code:b} exceeds {self.ndim} bits.")

    def __repr__(self) -> str:
        return f"Mesh({self})"

    def __len__(self) -> int:
        return self.ndim

    def __bool__(self) -> bool:
        return self.code != 0

    def __index__(self) -> int:
        return self.code

    def __lt__(self, other: Mesh | int) -> bool:
        if isinstance(other, Mesh):
            if self.ndim != other.ndim:
                raise ValueError(f"Cannot compare MeshCodes with different ndim: {self.ndim} vs {other.ndim}.")
            return self.code < other.code
        if isinstance(other, int):
            return self.code < other
        return NotImplemented

    def __getitem__(self, item: int | slice) -> Mesh:
        if isinstance(item, int):
            if item < 0:
                item += self.ndim
            if not 0 <= item < self.ndim:
                raise IndexError(f"Index {item} out of range for Mesh with ndim={self.ndim}.")
            return Mesh((self.code >> (self.ndim - 1 - item)) & 1, 1)
        if isinstance(item, slice):
            indices = range(*item.indices(self.ndim))
            code = 0
            for i in indices:
                code = (code << 1) | ((self.code >> (self.ndim - 1 - i)) & 1)
            return Mesh(code, len(indices))
        raise TypeError(f"Indices must be integers or slices, not {type(item).__name__!r}.")

    def __iter__(self) -> Generator[bool, None, None]:
        """Yield per-axis half-mesh flags MSB-first (logical axis order).

        Yields ``True`` for each axis on the half mesh, ``False`` for main.
        Iterating a :meth:`needs_remesh` result gives the remesh flags directly.
        """
        for i in range(self.ndim):
            yield bool((self.code >> (self.ndim - 1 - i)) & 1)

    def __reversed__(self) -> Generator[bool, None, None]:
        return iter(self.reverse())

    def __str__(self) -> str:
        """Return a human-readable per-axis stagger string, e.g. ``'HALF, MAIN, MAIN'``."""
        return ', '.join(
            'HALF' if (self.code >> (self.ndim - 1 - i)) & 1 else 'MAIN'
            for i in range(self.ndim)
        )

    def __rshift__(self, other: Optional[MeshLike]) -> tuple[bool]:
        """Return remesh flags for ``self`` → ``other`` (axes that need averaging)."""
        return self.remesh(other)

    def reverse(self) -> Mesh:
        """Return a new :class:`Mesh` with the axis order reversed (MSB ↔ LSB).

        Useful for converting between C-order and F-order axis conventions.
        """
        result, code = 0, self.code
        for _ in range(self.ndim):
            result = (result << 1) | (code & 1)
            code >>= 1
        return Mesh(result, self.ndim)

    @functools.singledispatchmethod
    @classmethod
    def parse(cls, mesh_code: MeshLike, *args: Any):
        """Normalize *mesh_code* into a :class:`Mesh`.

        Parameters
        ----------
        mesh_code : MeshCodeType or Mesh
            Stagger specification in any accepted form: an integer, the
            ``'main'``/``'half'`` shorthands, a per-axis sequence of tokens
            (``0``/``1``, ``'m'``/``'h'``, ``True``/``False``, etc.), or an
            existing :class:`Mesh` (returned as-is).
        ndim : int, optional
            Number of dimensions.  Required when *mesh_code* is an integer or
            the ``'main'``/``'half'`` shorthand; inferred from sequence length
            otherwise.  If both are provided they must agree.

        Returns
        -------
        Mesh

        Raises
        ------
        ValueError
            If *ndim* is required but not supplied, if *ndim* conflicts with
            the sequence length, or if any token is unrecognized.
        """
        if isinstance(mesh_code, Mesh):
            return mesh_code
        raise TypeError(f"Cannot convert {type(mesh_code).__name__!r} to Mesh.")

    @parse.register(bool)
    @classmethod
    def _(cls, mesh_code, ndim: int):
        return cls(0 if not mesh_code else (1 << ndim) - 1, ndim)

    @parse.register(int)
    @classmethod
    def _(cls, mesh_code, ndim: int):
        return cls(mesh_code, ndim)

    @parse.register(str)
    @classmethod
    def _(cls, mesh_code, ndim: Optional[int] = None):
        if mesh_code.lower() in ('main', 'half'):
            if ndim is None:
                raise ValueError("ndim is required for 'main'/'half' shorthands.")
            return cls(0 if mesh_code == 'main' else (1 << ndim) - 1, ndim)
        seq = list(mesh_code)
        return cls.parse(seq, ndim)

    @parse.register(ABCSequence)
    @classmethod
    def _(cls, mesh_code, ndim: Optional[int] = None):
        if ndim is not None and ndim != len(mesh_code):
            raise ValueError(f"ndim={ndim} conflicts with sequence length {len(mesh_code)}.")
        code = 0
        for c in mesh_code:
            bit = _MESH_CODE_REVERSE_MAPPING.get(str(c).lower())
            if bit is None:
                raise ValueError(f"Invalid mesh code token {c!r}.")
            code = (code << 1) | bit
        return cls(code, ndim or len(mesh_code))

    @_coerce_mesh_target
    def remesh(self, target: Optional[MeshLike], strict: bool = True) -> tuple[bool]:
        """Return a :class:`Mesh` whose set bits are axes that require averaging.

        An axis needs remeshing when the source is on the half mesh (``1``) and
        the target is on the main mesh (``0``).

        Parameters
        ----------
        target : MeshCodeType or Mesh
            Desired output stagger; coerced to :class:`Mesh` if necessary.

        Returns
        -------
        Mesh
            Bitmask of axes to average; ``0`` means no remeshing is needed.
        """
        mask = (1 << self.ndim) - 1
        if strict and (~self.code & mask) & target.code:
            raise ValueError(f"Cannot remesh from {self} to {target}: main → half transitions are not supported.")
        return tuple(Mesh(self.code & (~target.code & mask), self.ndim))


def _average_adjacent(arr: np.ndarray,
                      axis: int
                      ) -> np.ndarray:
    """Return the mean of adjacent element pairs along *axis*, reducing that dimension by one."""
    if arr.shape[axis] < 2:
        raise ValueError(f"Cannot remesh axis {axis} with size {arr.shape[axis]}."
                         f" Need at least 2 elements to average adjacent pairs.")
    slc_lo = [slice(None)] * arr.ndim
    slc_hi = [slice(None)] * arr.ndim
    slc_lo[axis] = slice(None, -1)
    slc_hi[axis] = slice(1, None)
    return 0.5 * (arr[tuple(slc_lo)] + arr[tuple(slc_hi)])


def _remesh_array(data: np.ndarray,
                  remesh: Iterable[bool] | bool,
                  order: ArrayOrdering = 'F') -> np.ndarray:
    """Apply :func:`_average_adjacent` on each axis where *remesh* is ``True``."""
    if isinstance(remesh, bool):
        remesh = [remesh] * data.ndim
    if order == 'F':
        remesh = reversed(remesh)
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
    remesh = Mesh.parse(imesh, data.ndim).remesh(omesh)
    return _remesh_array(data, remesh, order=order)

