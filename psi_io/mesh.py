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
    Immutable (frozen) dataclass wrapping a binary stagger :attr:`~Mesh.code` and
    its axis count :attr:`~Mesh.ndim`.  Each bit of the code classifies one axis as
    half mesh (``1``) or main mesh (``0``).
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
>>> from psi_io.mesh import remesh_array
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
"""String-token to integer (0/1) lookup used to validate per-axis sequence mesh codes.

Each key is a recognized string representation of a single-axis stagger token.
The value ``1`` means half mesh; the value ``0`` means main mesh.

Keys
----
``'1'``, ``'h'``, ``'half'``, ``'true'``
    Map to ``1`` (half mesh).
``'0'``, ``'m'``, ``'main'``, ``'false'``
    Map to ``0`` (main mesh).

Notes
-----
Token matching is performed case-insensitively via :func:`str.lower` before
lookup.  An unrecognized token returns ``None``, which callers treat as an
error condition.

Examples
--------
>>> _MESH_CODE_REVERSE_MAPPING['half']
1
>>> _MESH_CODE_REVERSE_MAPPING['m']
0
>>> _MESH_CODE_REVERSE_MAPPING.get('x') is None
True
"""


MeshCodeType = Union[bool, int, Literal['main', 'half'], Sequence[Any]]
r"""Type alias for mesh stagger specifications accepted by :func:`remesh_array`.

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

Examples
--------
All three forms below encode the same 3-D stagger (half only along :math:`r`):

>>> from psi_io.mesh import Mesh
>>> str(Mesh.parse(0b100, ndim=3))
'HALF, MAIN, MAIN'
>>> str(Mesh.parse([True, False, False], ndim=3))
'HALF, MAIN, MAIN'
"""

MeshLike = Union['Mesh', MeshCodeType]
r"""Type alias for any accepted mesh specification, including an existing :class:`Mesh`.

This is a superset of :data:`MeshCodeType`; it additionally accepts an already-
constructed :class:`Mesh` instance, which is passed through unchanged.

See Also
--------
MeshCodeType : Accepted forms that do not include :class:`Mesh` itself.
Mesh.parse : Normalizes any :data:`MeshLike` value into a :class:`Mesh`.

Examples
--------
>>> from psi_io.mesh import Mesh
>>> m = Mesh.parse(0b101, ndim=3)
>>> Mesh.parse(m, ndim=3) is m   # Mesh passthrough
True
>>> str(Mesh.parse('half', ndim=2))
'HALF, HALF'
"""

ArrayOrdering = Literal['F', 'C']
r"""Type alias for the memory-order convention accepted by :func:`remesh_array`.

Controls how the bits of a :data:`MeshCodeType` integer map to numpy array axes.

``'F'``
    Fortran (column-major) order — the default for PSI data.  Because PSI HDF
    files are written by Fortran code, the physical ``(r, θ, φ)`` axis ordering
    is **reversed** in numpy storage: the **last** numpy axis corresponds to
    :math:`r`, the middle to :math:`\theta`, and the **first** to :math:`\varphi`.
    The most-significant bit of the mesh code therefore maps to the last numpy axis.
    Use this setting whenever the array was loaded directly from a PSI HDF file.
``'C'``
    C (row-major) order.  Use when the array has been transposed to numpy-native
    axis order (first axis = first physical coordinate, e.g. shape ``(Nr, Nθ, Nφ)``),
    so that the most-significant bit maps to the first numpy axis.

Examples
--------
>>> import numpy as np
>>> from psi_io.mesh import remesh_array
>>> arr = np.ones((57, 64, 128))   # C-order: shape (Nr, Nθ, Nφ); Nr is half-mesh
>>> out = remesh_array(arr, imesh=0b100, omesh='main', order='C')
>>> out.shape
(56, 64, 128)
"""


def _coerce_mesh_target(method: Callable) -> Callable:
    """Coerce the *target* argument of a :class:`Mesh` method to a :class:`Mesh`.

    Decorator that normalizes the second positional argument (*target*) of a
    bound :class:`Mesh` method.  If *target* is already a :class:`Mesh` its
    ``ndim`` is verified to match ``self.ndim``.  If *target* is ``None`` it
    is replaced by ``self`` (a no-op target).  Otherwise :meth:`Mesh.parse` is
    called with ``ndim=self.ndim`` to produce the normalized :class:`Mesh`.

    Parameters
    ----------
    method : Callable
        The bound :class:`Mesh` method whose second positional argument should
        be coerced.  The wrapper preserves the method's ``__name__``,
        ``__doc__``, and other attributes via :func:`functools.wraps`.

    Returns
    -------
    wrapper : Callable
        Wrapped version of *method* with automatic coercion of the *target*
        argument.

    Raises
    ------
    ValueError
        If *target* is a :class:`Mesh` whose ``ndim`` differs from
        ``self.ndim``.

    Examples
    --------
    The decorator is applied internally; here is the observable behavior it
    enables on :meth:`Mesh.remesh`:

    >>> from psi_io.mesh import Mesh
    >>> m = Mesh.parse(0b111, ndim=3)
    >>> m.remesh('main')         # string target is coerced automatically
    (True, True, True)
    >>> m.remesh(None)           # None is replaced by self → no-op
    (False, False, False)
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
    r"""Compact, immutable representation of a multi-axis mesh stagger code.

    A :class:`Mesh` is a frozen dataclass that bundles two integers: a binary
    stagger *code* and the axis count *ndim*.  Together they describe, for a
    *ndim*-dimensional array, which axes are sampled on the **half mesh** (face or
    edge midpoints, bit ``1``) versus the **main mesh** (cell-center nodes, bit
    ``0``).  Because it replaces the legacy two-member ``Mesh`` enum, a single
    instance now encodes the stagger of *every* axis at once rather than one axis
    per enum member.

    The bit-to-axis mapping follows PSI's Fortran column-major HDF convention: the
    **most-significant bit maps to the first logical axis** (physical :math:`r`),
    descending to the least-significant bit at the last axis (:math:`\varphi`).
    For a 3-bit code the axes therefore read :math:`(r, \theta, \varphi)` from
    MSB to LSB.  For example, ``code=0b100, ndim=3`` means :math:`r` is on the
    half mesh while :math:`\theta` and :math:`\varphi` are on the main mesh
    (the MAS :math:`B_r` stagger).

    Once built, a :class:`Mesh` supports rich operations: it iterates as a
    sequence of per-axis booleans (:meth:`__iter__`), indexes to a single-axis
    :class:`Mesh` (:meth:`__getitem__`), reports its remesh requirements against a
    target via :meth:`remesh` / the ``>>`` operator, reverses axis order with
    :meth:`reverse`, and acts as a plain integer code in index contexts
    (:meth:`__index__`).

    Prefer constructing via :meth:`parse` rather than calling the constructor
    directly: :meth:`parse` accepts every form described by :data:`MeshCodeType`
    (integers, the shorthand strings ``'main'``/``'half'``, per-axis token
    strings such as ``'MMH'``, and per-axis boolean/int sequences) and infers or
    validates *ndim* for you.  The direct constructor accepts only an explicit
    integer *code* and *ndim*.

    Parameters
    ----------
    code : int
        Binary-encoded stagger integer.  Bit ``i`` (counting from the LSB) sets
        the stagger of logical axis ``ndim - 1 - i``: ``1`` for half mesh, ``0``
        for main mesh.  Must fit within *ndim* bits (i.e. ``0 <= code < 2**ndim``).
    ndim : int
        Number of array dimensions (and therefore significant bits) represented
        by this code.  Fixes the width of the stagger field and the length of the
        iterated/indexed sequence.

    Raises
    ------
    ValueError
        If *code* has any bit set at or above position *ndim* (i.e.
        ``code >= 2**ndim``), since that bit could not correspond to a real axis.

    See Also
    --------
    Mesh.parse : Build a :class:`Mesh` from any :data:`MeshCodeType` form.
    Mesh.remesh : Per-axis flags for moving from this stagger to a target.
    remesh_array : Apply an actual half-to-main mesh shift to a NumPy array.

    Examples
    --------
    >>> from psi_io.mesh import Mesh
    >>> str(Mesh.parse(0b100, ndim=3))
    'HALF, MAIN, MAIN'
    >>> str(Mesh.parse('MMH', ndim=3))
    'MAIN, MAIN, HALF'
    >>> str(Mesh.parse([True, False, True], ndim=3))
    'HALF, MAIN, HALF'

    The direct constructor takes an explicit code and dimension count:

    >>> Mesh(code=0b100, ndim=3)
    Mesh(HALF, MAIN, MAIN)
    """

    code: int
    ndim: int

    def __post_init__(self) -> None:
        """Validate that *code* fits within *ndim* bits.

        Runs automatically after the dataclass constructor assigns :attr:`code`
        and :attr:`ndim`.  Guards the class invariant that every set bit of
        *code* corresponds to a real axis, so an over-wide code is rejected at
        construction time rather than producing a silently truncated stagger.

        Raises
        ------
        ValueError
            If *code* has any bits set at position *ndim* or higher (i.e.
            ``code >= 2**ndim``).

        Examples
        --------
        >>> from psi_io.mesh import Mesh
        >>> Mesh(code=0b100, ndim=3).code   # valid: 3 bits, MSB is bit 2
        4
        >>> import pytest
        >>> with pytest.raises(ValueError):
        ...     Mesh(code=0b1000, ndim=3)   # 4-bit value in 3-bit field
        """
        mask = (1 << self.ndim) - 1
        if self.code & ~mask:
            raise ValueError(f"Code 0b{self.code:b} exceeds {self.ndim} bits.")

    def __repr__(self) -> str:
        """Return an unambiguous string representation of this :class:`Mesh`.

        The representation uses the human-readable stagger labels from
        :meth:`__str__` rather than the raw integer code.

        Returns
        -------
        out : str
            String of the form ``'Mesh(<stagger>)'``, e.g.
            ``'Mesh(HALF, MAIN, MAIN)'``.

        Examples
        --------
        >>> from psi_io.mesh import Mesh
        >>> repr(Mesh.parse(0b101, ndim=3))
        'Mesh(HALF, MAIN, HALF)'
        """
        return f"Mesh({self})"

    def __len__(self) -> int:
        """Return the number of axes encoded by this :class:`Mesh`.

        Returns
        -------
        out : int
            Equal to :attr:`ndim`.

        Examples
        --------
        >>> from psi_io.mesh import Mesh
        >>> len(Mesh.parse(0b101, ndim=3))
        3
        """
        return self.ndim

    def __bool__(self) -> bool:
        """Return ``True`` if any axis is on the half mesh.

        Returns
        -------
        out : bool
            ``True`` when :attr:`code` is non-zero; ``False`` when all axes
            are on the main mesh (code ``0``).

        Examples
        --------
        >>> from psi_io.mesh import Mesh
        >>> bool(Mesh.parse('half', ndim=3))
        True
        >>> bool(Mesh.parse('main', ndim=3))
        False
        """
        return self.code != 0

    def __index__(self) -> int:
        """Return the raw integer mesh code for use in index contexts.

        Allows a :class:`Mesh` to be used wherever a plain integer code is
        expected (e.g., passed directly to :func:`remesh_array` as *imesh*).

        Returns
        -------
        out : int
            Equal to :attr:`code`.

        Examples
        --------
        >>> from psi_io.mesh import Mesh
        >>> m = Mesh.parse(0b110, ndim=3)
        >>> int(m)
        6
        >>> hex(m)
        '0x6'
        """
        return self.code

    def __lt__(self, other: Mesh | int) -> bool:
        """Compare this :class:`Mesh` to *other* by code value.

        Parameters
        ----------
        other : Mesh | int
            The object to compare against.  If a :class:`Mesh`, its
            ``ndim`` must equal ``self.ndim``.  If an :class:`int`, the
            comparison is against the raw code.

        Returns
        -------
        out : bool
            ``True`` when ``self.code < other`` (or ``other.code``).

        Raises
        ------
        ValueError
            If *other* is a :class:`Mesh` with a different ``ndim``.

        Examples
        --------
        >>> from psi_io.mesh import Mesh
        >>> Mesh.parse(0b001, ndim=3) < Mesh.parse(0b100, ndim=3)
        True
        >>> Mesh.parse(0b100, ndim=3) < 3
        False
        """
        if isinstance(other, Mesh):
            if self.ndim != other.ndim:
                raise ValueError(f"Cannot compare MeshCodes with different ndim: {self.ndim} vs {other.ndim}.")
            return self.code < other.code
        if isinstance(other, int):
            return self.code < other
        return NotImplemented

    def __getitem__(self, item: int | slice) -> Mesh:
        """Return a sub-:class:`Mesh` for the axis or slice specified by *item*.

        Parameters
        ----------
        item : int | slice
            Axis index or slice.  Negative integer indices are supported.
            Slicing follows the same semantics as Python sequences.

        Returns
        -------
        out : Mesh
            A new :class:`Mesh` containing only the selected axis or axes.
            The ``ndim`` of the result equals ``1`` for integer indexing and
            ``len(range(*item.indices(self.ndim)))`` for slice indexing.

        Raises
        ------
        IndexError
            If an integer *item* is out of range for this :class:`Mesh`.
        TypeError
            If *item* is neither an :class:`int` nor a :class:`slice`.

        Examples
        --------
        >>> from psi_io.mesh import Mesh
        >>> m = Mesh.parse(0b101, ndim=3)   # HALF, MAIN, HALF
        >>> str(m[0])
        'HALF'
        >>> str(m[1])
        'MAIN'
        >>> str(m[:2])
        'HALF, MAIN'
        """
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
        Iterating the result of :meth:`remesh` gives the remesh flags directly.

        Yields
        ------
        flag : bool
            ``True`` if the current axis is on the half mesh; ``False`` if on
            the main mesh.  Axes are yielded in logical order (MSB first).

        Examples
        --------
        >>> from psi_io.mesh import Mesh
        >>> list(Mesh.parse(0b101, ndim=3))
        [True, False, True]
        >>> list(Mesh.parse('main', ndim=3))
        [False, False, False]
        """
        for i in range(self.ndim):
            yield bool((self.code >> (self.ndim - 1 - i)) & 1)

    def __reversed__(self) -> Generator[bool, None, None]:
        """Yield per-axis half-mesh flags in reverse (LSB-first) order.

        Equivalent to iterating over :meth:`reverse`.

        Yields
        ------
        flag : bool
            ``True`` if the current axis is on the half mesh; ``False`` if on
            the main mesh.  Axes are yielded in reverse logical order (LSB
            first).

        Examples
        --------
        >>> from psi_io.mesh import Mesh
        >>> list(reversed(Mesh.parse(0b101, ndim=3)))
        [True, False, True]
        >>> list(reversed(Mesh.parse(0b100, ndim=3)))
        [False, False, True]
        """
        return iter(self.reverse())

    def __str__(self) -> str:
        """Return a human-readable per-axis stagger string.

        Returns
        -------
        out : str
            Comma-separated ``'HALF'``/``'MAIN'`` labels, one per axis in
            logical order (MSB first).  For example ``'HALF, MAIN, MAIN'``.

        Examples
        --------
        >>> from psi_io.mesh import Mesh
        >>> str(Mesh.parse(0b100, ndim=3))
        'HALF, MAIN, MAIN'
        >>> str(Mesh.parse('main', ndim=2))
        'MAIN, MAIN'
        """
        return ', '.join(
            'HALF' if (self.code >> (self.ndim - 1 - i)) & 1 else 'MAIN'
            for i in range(self.ndim)
        )

    def __rshift__(self, other: Optional[MeshLike]) -> tuple[bool]:
        """Return remesh flags for the transition ``self`` → ``other``.

        Syntactic sugar for :meth:`remesh`.  Each ``True`` in the result
        indicates an axis that must be averaged (half → main) when
        transforming data from this mesh to *other*.

        Parameters
        ----------
        other : MeshLike | None
            Target mesh stagger in any form accepted by :data:`MeshLike`.
            ``None`` is a no-op (returns all ``False``).

        Returns
        -------
        out : tuple[bool, ...]
            Per-axis remesh flags; ``True`` means the axis needs averaging.

        Examples
        --------
        >>> from psi_io.mesh import Mesh
        >>> src = Mesh.parse(0b111, ndim=3)
        >>> src >> 'main'
        (True, True, True)
        >>> src >> None
        (False, False, False)
        """
        return self.remesh(other)

    def reverse(self) -> Mesh:
        """Return a new :class:`Mesh` with the axis order reversed (MSB to LSB).

        Useful for converting between C-order and F-order axis conventions,
        where the first physical axis becomes the last numpy axis or vice
        versa.

        Returns
        -------
        out : Mesh
            A new :class:`Mesh` with the same ``ndim`` but with the bit order
            of :attr:`code` reversed so that what was the MSB becomes the LSB.

        Examples
        --------
        >>> from psi_io.mesh import Mesh
        >>> m = Mesh.parse(0b100, ndim=3)   # HALF, MAIN, MAIN
        >>> str(m.reverse())
        'MAIN, MAIN, HALF'
        >>> str(Mesh.parse(0b110, ndim=3).reverse())
        'MAIN, HALF, HALF'
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
        mesh_code : MeshLike
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
        out : Mesh
            Normalized :class:`Mesh` with the specified stagger and
            dimensionality.

        Raises
        ------
        ValueError
            If *ndim* is required but not supplied, if *ndim* conflicts with
            the sequence length, or if any token is unrecognized.
        TypeError
            If *mesh_code* is of an unsupported type.

        Examples
        --------
        >>> from psi_io.mesh import Mesh
        >>> str(Mesh.parse(0b100, ndim=3))
        'HALF, MAIN, MAIN'
        >>> str(Mesh.parse('half', ndim=2))
        'HALF, HALF'
        >>> str(Mesh.parse([1, 0, 1], ndim=3))
        'HALF, MAIN, HALF'
        >>> m = Mesh.parse(0b010, ndim=3)
        >>> Mesh.parse(m) is m
        True
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
        """Return per-axis flags indicating which axes require averaging.

        An axis needs remeshing when the source is on the half mesh (``1``) and
        the target is on the main mesh (``0``).  By default, requesting a
        main-to-half transition (upsampling) raises a :exc:`ValueError`.

        Parameters
        ----------
        target : MeshLike | None
            Desired output stagger; coerced to :class:`Mesh` via
            :func:`_coerce_mesh_target` if necessary.  ``None`` is treated
            as ``self`` (no-op: returns all ``False``).
        strict : bool, optional
            If ``True`` (default), raise :exc:`ValueError` when any axis in
            *target* is on the half mesh but the corresponding axis in
            ``self`` is already on the main mesh (main → half is not
            supported).  Set to ``False`` to silently ignore such axes.

        Returns
        -------
        out : tuple[bool, ...]
            Tuple of per-axis boolean flags in logical axis order (MSB first).
            ``True`` at position *i* means axis *i* must be averaged (half →
            main); ``False`` means no averaging is needed on that axis.

        Raises
        ------
        ValueError
            If *strict* is ``True`` and *target* requests a half-mesh axis
            where ``self`` is already on the main mesh.

        Examples
        --------
        >>> from psi_io.mesh import Mesh
        >>> src = Mesh.parse(0b111, ndim=3)   # all-half
        >>> src.remesh('main')
        (True, True, True)
        >>> src.remesh(None)                   # no-op: target == self
        (False, False, False)
        >>> src.remesh(0b101)                  # only theta needs averaging
        (False, True, False)
        """
        mask = (1 << self.ndim) - 1
        if strict and (~self.code & mask) & target.code:
            raise ValueError(f"Cannot remesh from {self} to {target}: main → half transitions are not supported.")
        return tuple(Mesh(self.code & (~target.code & mask), self.ndim))


def _average_adjacent(arr: np.ndarray,
                      axis: int
                      ) -> np.ndarray:
    """Return the mean of adjacent element pairs along *axis*, reducing its size by one.

    Computes ``0.5 * (arr[..., :-1, ...] + arr[..., 1:, ...])`` where the
    ellipsis notation represents all other axes.  The result has the same
    shape as *arr* on every axis except *axis*, which shrinks by one element.

    Parameters
    ----------
    arr : np.ndarray
        Input array of any dtype and number of dimensions.
    axis : int
        Axis along which to average adjacent pairs.  Must satisfy
        ``0 <= axis < arr.ndim``.

    Returns
    -------
    out : np.ndarray
        Array with the same shape as *arr* except ``out.shape[axis] ==
        arr.shape[axis] - 1``.  The dtype is determined by NumPy's
        float promotion rules (typically ``float64`` for integer input).

    Raises
    ------
    ValueError
        If ``arr.shape[axis] < 2``, because at least two elements are needed
        to form one averaged pair.

    Examples
    --------
    >>> import numpy as np
    >>> from psi_io.mesh import _average_adjacent
    >>> arr = np.array([1.0, 3.0, 5.0, 7.0])
    >>> _average_adjacent(arr, axis=0)
    array([2., 4., 6.])
    >>> arr2d = np.array([[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]])
    >>> _average_adjacent(arr2d, axis=0).shape
    (2, 2)
    """
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
    """Apply adjacent-element averaging on each axis where *remesh* is ``True``.

    Iterates over axes in numpy index order (0, 1, 2, …) and calls
    :func:`_average_adjacent` on each axis flagged for remeshing.  When
    *order* is ``'F'`` the *remesh* iterable is reversed before the loop so
    that the logical MSB-first ordering of :class:`Mesh` maps correctly to
    numpy's last-axis-first Fortran convention.

    Parameters
    ----------
    data : np.ndarray
        Input array on the source mesh stagger.
    remesh : Iterable[bool] | bool
        Per-axis remesh flags in logical axis order (MSB first), as returned
        by :meth:`Mesh.remesh`.  A single :class:`bool` is broadcast to all
        axes.
    order : ArrayOrdering, optional
        Memory-order convention; ``'F'`` (default) reverses *remesh* before
        iterating so that MSB maps to the last numpy axis.  ``'C'`` uses the
        flags as-is so that MSB maps to the first numpy axis.

    Returns
    -------
    out : np.ndarray
        Array with averaged values along every flagged axis.  Shape is reduced
        by one on each remeshed axis.

    Examples
    --------
    >>> import numpy as np
    >>> from psi_io.mesh import _remesh_array
    >>> arr = np.ones((4, 4, 4))
    >>> out = _remesh_array(arr, remesh=[False, False, True], order='F')
    >>> out.shape   # F-order: LSB flag (True) maps to numpy axis 0 (phi)
    (3, 4, 4)
    >>> out2 = _remesh_array(arr, remesh=True)
    >>> out2.shape  # all axes averaged
    (3, 3, 3)
    """
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
    r"""Shift an array from one mesh stagger to another.

    Compares the source mesh *imesh* against the target mesh *omesh* axis by axis
    and applies adjacent-element averaging on every axis that needs to move from
    half mesh to main mesh.  Only the half → main direction is supported;
    requesting main → half raises :exc:`ValueError`.

    This is commonly needed before performing interpolation or arithmetic between
    quantities on different mesh positions.  For example, computing the magnitude of the
    magnetic field requires :math:`B_r`, :math:`B_\theta`, and
    :math:`B_\varphi` on the same mesh: each must be remeshed from its native
    stagger (``0b100``, ``0b010``, ``0b001``) to a common target before squaring
    and summing.

    If *omesh* is ``None``, the array is returned unchanged.

    Parameters
    ----------
    data : np.ndarray
        Input array on the stagger described by *imesh*.
    imesh : MeshCodeType
        Source mesh stagger in any form accepted by :data:`MeshCodeType`.
    omesh : MeshCodeType | None, optional
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
    Mesh : Compact representation of a multi-axis mesh stagger code.
    MeshCodeType : Accepted forms for mesh stagger specifications.
    ArrayOrdering : Memory-order convention for bit-axis mapping.

    Examples
    --------
    Convert a radial magnetic-field array (half-mesh in :math:`r`, the last
    numpy axis) to the all-main mesh:

    >>> import numpy as np
    >>> from psi_io.mesh import remesh_array
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
