r"""Lazy HDF readers for PSI MAS and POT3D magnetohydrodynamic model output.

This module provides a unified, unit-aware interface for reading three-dimensional
field variables from Predictive Science Inc.'s MAS and POT3D solvers.  Both HDF4
(``.hdf``) and HDF5 (``.h5``) files are supported through a common API.  The sole
public symbol exported by this module is :func:`PsiData`.

.. rubric:: Entry Point

:func:`PsiData` is a factory function that opens a PSI HDF file and returns a
lazy reader object.  The file extension determines the I/O backend — ``.h5``
files are read via `h5py <https://www.h5py.org/>`_ and ``.hdf`` files via
`pyhdf <https://fhs.github.io/pyhdf/>`_.  The *model* argument (``'mas'`` or
``'pot3d'``) selects the quantity-property and mesh-stagger metadata tables
applied during metadata resolution:

.. code-block:: python

    from psi_io.mhd_io import PsiData

    reader = PsiData('br001001.h5')                    # MAS HDF5 (default)
    reader = PsiData('br001001.hdf', model='mas')      # MAS HDF4
    reader = PsiData('br001.h5', model='pot3d')        # POT3D HDF5

.. rubric:: Lazy Loading and Caching

No data are read from disk at construction time.  Metadata — quantity name,
sequence number, physical unit, and mesh stagger — are parsed from the filename
stem and/or HDF file attributes.  Data are transferred from disk only when
:meth:`read` or :meth:`vslice` is called.

When a read or vslice call covers the *entire* dataset with no spatial
restrictions, the result is cached on the reader object.  Subsequent full-array
calls return the cached copy without a second disk read.  Partial reads — any
call with at least one non-full-axis argument — are never cached.  The cache
state is exposed via the ``is_cached`` property.

.. rubric:: Attributes

The following attributes are available on every object returned by
:func:`PsiData`:

``quantity`` : :class:`str`
    Canonical lower-case quantity identifier (e.g. ``'br'``, ``'vr'``,
    ``'t'``).  Resolved from the filename stem, HDF file attributes, or the
    ``quantity`` keyword argument passed to :func:`PsiData`.

``sequence`` : :class:`int`
    Time-step sequence number extracted from the filename
    (e.g. ``br001001.h5`` → ``1001``) or from HDF file attributes.

``unit`` : :class:`~astropy.units.Unit`
    Conversion factor from one code unit to physical units.  For MAS
    quantities these are the custom normalization units defined in
    :mod:`psi_io._units` (e.g. :data:`~psi_io._units.MAS_b` ≈ 2.2 G for
    magnetic field, :data:`~psi_io._units.MAS_v` ≈ 481 km s⁻¹ for velocity).
    For POT3D the default is dimensionless — see the :func:`PsiData` warning.

``mesh`` : :class:`tuple` of :class:`~psi_io._mesh.Mesh`
    Yee-grid stagger position for each spatial axis in physical ``(r, θ, φ)``
    order.  Each element is either :attr:`~psi_io._mesh.Mesh.HALF` (offset by
    half a cell spacing) or :attr:`~psi_io._mesh.Mesh.MAIN` (cell-centred).
    The integer encoding and per-quantity stagger codes are defined in
    :mod:`psi_io._mesh`; the canonical default for each quantity is stored in
    its :class:`~psi_io._models.Props` descriptor (see :mod:`psi_io._models`).

``props`` : :class:`~psi_io._models.Props`
    Complete property descriptor for the quantity, bundling its name,
    description, unit, and mesh code.  Looked up from the appropriate
    quantity-properties mapping in :mod:`psi_io._models`.

``description`` : :class:`str`
    Human-readable description of the physical quantity
    (e.g. ``'MAS Magnetic Field (Radial Component)'``).  Derived from
    :attr:`~psi_io._models.Props.desc`.

``scales`` : ``Scales(r, t, p)``
    Named tuple of coordinate scale readers.  Each element wraps the
    one-dimensional coordinate array stored in the HDF file.  The radial
    coordinate uses :data:`~psi_io._units.PSI_rsun` (solar radii); the
    co-latitude and longitude coordinates use
    :data:`~psi_io._units.PSI_angle` (radians).  Each scale reader exposes
    the same :meth:`read` interface as the main data reader.

``shape`` : :class:`tuple` of :class:`int`
    Array dimensions in HDF storage order ``(Nφ, Nθ, Nr)`` — the radial
    axis is *last* due to Fortran column-major convention.  All slicing APIs
    accept arguments in physical ``(r, θ, φ)`` order and reverse the indexing
    internally.

``ndim`` : :class:`int`
    Number of spatial dimensions; always ``3`` for MAS and POT3D field
    variables.

``size`` : :class:`int`
    Total element count (``Nφ × Nθ × Nr``).

``nbytes`` : :class:`int`
    Dataset size in bytes.

``dtype`` : :class:`numpy.dtype`
    Element type of the stored dataset (typically ``float32``).

``attrs`` : :class:`dict`
    HDF file-level attributes as a plain Python dictionary.

``is_cached`` : :class:`bool`
    ``True`` once a full-array read has populated the in-memory cache;
    ``False`` otherwise.

.. rubric:: Reading Data — ``read``

.. code-block:: python

    odata[, r, t, p] = reader.read(*args, unit=None, mesh=None, scales=True)

Each positional argument restricts one spatial axis in physical ``(r, θ, φ)``
order:

- Omitted / ``None`` — full axis.
- ``int`` — single index; axis retained as a length-1 dimension.
- ``slice`` — standard Python slice.
- ``(start, stop)`` or ``(start, stop, step)`` — converted to a slice.
- ``Ellipsis`` — expands to ``None`` for all remaining axes.

**unit**
    Output unit.  String aliases ``'native'`` / ``'code'`` / ``'model'`` /
    ``'psi'`` return values in MAS code units (the units defined in
    :mod:`psi_io._units`).  Aliases ``'real'`` / ``'phys'`` / ``'physical'``
    / ``'cgs'`` call :func:`~psi_io._units.decompose_mas_units` to express
    values in CGS base units.  Any other string or
    :class:`~astropy.units.Unit` instance is forwarded to
    :meth:`astropy.units.Quantity.to`.

**mesh**
    Target mesh stagger (:data:`~psi_io._mesh.MeshCodeType`).  Axes that are
    on the half mesh in the stored data but on the main mesh in *mesh* are
    averaged via :func:`~psi_io._mesh.remesh_array`.  Up-sampling
    (main → half) raises :exc:`ValueError`.

**scales**
    If ``True`` (default), return the corresponding coordinate slice for each
    axis as additional :class:`~astropy.units.Quantity` values ``(r, t, p)``
    after the data array.

.. note::
    PSI HDF files are written in Fortran column-major order so that numpy
    reads them with shape ``(Nφ, Nθ, Nr)`` — the radial axis is *last*.  All
    positional arguments to :meth:`read` and the returned coordinate scales
    are in physical ``(r, θ, φ)`` order regardless of on-disk layout.

.. rubric:: Value-Space Slicing — ``vslice``

.. code-block:: python

    odata[, r, t, p] = reader.vslice(*args, unit=None, mesh=None, scales=True,
                                     bounds_error=True, fill_value=None)

``vslice`` is a superset of :meth:`read` that additionally accepts physical
coordinate values as positional arguments.  Passing a
:class:`~astropy.units.Quantity` or a bare scalar for an axis locates the two
nearest grid points, loads a 2-element window, and linearly interpolates to the
target value; the resulting axis has size 1.  Index-space arguments (``slice``,
``int``, ``None``, ``Ellipsis``) are handled identically to :meth:`read`.

**bounds_error**
    If ``True`` (default), raise :exc:`ValueError` when a value is outside the
    coordinate range.  Set to ``False`` and supply a **fill_value** to replace
    out-of-bounds points; pass ``fill_value=None`` to extrapolate silently.

When ``scales=True``, value-interpolated axes return the target coordinate as
a length-1 :class:`~astropy.units.Quantity`; index-space axes return the
corresponding coordinate slice as usual.

.. rubric:: File Lifecycle

Readers hold an open HDF file handle.  Use the context-manager protocol to
guarantee cleanup:

.. code-block:: python

    with PsiData('br001001.h5') as reader:
        data, r, t, p = reader.read(unit='Gauss')

Alternatively, call ``reader.close()`` and ``reader.open()`` to manage the
handle explicitly.  The handle is also released automatically when the reader
is garbage-collected.

.. rubric:: Supported Quantities

**MAS** (19 quantities):

.. list-table::
   :widths: 12 32 24 32
   :header-rows: 1

   * - Key(s)
     - Description
     - Code unit
     - Stagger ``(r, θ, φ)``
   * - ``br``, ``bt``, ``bp``
     - Magnetic field components
     - :data:`~psi_io._units.MAS_b` (≈ 2.2 G)
     - ``HALF/MAIN/MAIN``, ``MAIN/HALF/MAIN``, ``MAIN/MAIN/HALF``
   * - ``vr``, ``vt``, ``vp``
     - Velocity components
     - :data:`~psi_io._units.MAS_v` (≈ 481 km s⁻¹)
     - ``MAIN/HALF/HALF``, ``HALF/MAIN/HALF``, ``HALF/HALF/MAIN``
   * - ``jr``, ``jt``, ``jp``
     - Current density components
     - :data:`~psi_io._units.MAS_j`
     - ``MAIN/HALF/HALF``, ``HALF/MAIN/HALF``, ``HALF/HALF/MAIN``
   * - ``t``, ``te``, ``tp``
     - Temperature (single / electron / proton)
     - :data:`~psi_io._units.MAS_t` (≈ 28 MK)
     - ``HALF/HALF/HALF``
   * - ``rho``
     - Mass density
     - :data:`~psi_io._units.MAS_n` (10⁸ cm⁻³)
     - ``HALF/HALF/HALF``
   * - ``p``
     - Thermal pressure
     - :data:`~psi_io._units.MAS_p`
     - ``HALF/HALF/HALF``
   * - ``ep``, ``em``
     - Alfvén wave energy density (outward / inward)
     - :data:`~psi_io._units.MAS_p`
     - ``HALF/HALF/HALF``
   * - ``zp``, ``zm``
     - Elsässer wave amplitudes (outward / inward)
     - :data:`~psi_io._units.MAS_v`
     - ``HALF/HALF/HALF``
   * - ``heat``
     - Volumetric coronal heating rate
     - :data:`~psi_io._units.MAS_heat`
     - ``HALF/HALF/HALF``

**POT3D** (3 quantities): ``br``, ``bt``, ``bp`` — magnetic field components,
unit :data:`~psi_io._units.POT3D_b` (dimensionless by default; see the
:func:`PsiData` warning regarding unit declaration).

.. rubric:: Examples

Full MAS radial field read with coordinate scales:

.. code-block:: python

    from psi_io.mhd_io import PsiData

    reader = PsiData('br001001.h5')
    data, r, t, p = reader.read()              # code units (MAS_b)
    data, r, t, p = reader.read(unit='Gauss')  # convert to Gauss on the fly

Partial read — inner radial shell in CGS base units, coordinates suppressed:

.. code-block:: python

    data = reader.read(slice(0, 10), unit='physical', scales=False)

Value-space slice — extract the r = 2.5 R☉ surface:

.. code-block:: python

    data, r, t, p = reader.vslice(2.5, unit='Gauss')  # bare scalar → native coord unit

POT3D file — unit must be declared at construction because it is not encoded in
the file:

.. code-block:: python

    reader = PsiData('br001.hdf', model='pot3d', unit='Gauss')
    data, r, t, p = reader.read()

Inspect metadata without loading data:

.. code-block:: python

    reader = PsiData('rho001001.h5')
    reader.quantity      # 'rho'
    reader.description   # 'MAS Density'
    reader.unit          # MAS_n
    reader.mesh          # (Mesh.HALF, Mesh.HALF, Mesh.HALF)
    reader.is_cached     # False
    reader.shape         # (Nφ, Nθ, Nr) — numpy storage order
"""

from __future__ import annotations

__all__ = ['PsiData',]

import re
import warnings
from abc import abstractmethod, ABC
from collections import namedtuple, UserDict
from collections.abc import Sequence, Iterable, Collection
from functools import partial
from itertools import repeat, chain
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Optional, Literal, ClassVar, Mapping
import numpy as np
import h5py as h5
import astropy.units as u
from astropy.table import QTable
from numpy.lib.recfunctions import structured_to_unstructured
try:
    from scipy.interpolate import RegularGridInterpolator
except ImportError:
    RegularGridInterpolator = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from astropy.units.typing import UnitLike, QuantityLike

try:
    import pyhdf.SD as h4
except ImportError:
    h4 = None

from psi_io._mesh import (MeshCodeType,
                          _remesh_array,
                          ArrayOrdering, Mesh, MeshLike,
                          )
from psi_io._models import (ModelType,
                            extract_quantity_from_filepath,
                            extract_sequence_from_filepath,
                            get_model_prop_caller,
                            get_psi_scale_properties,
                            _PROP_GETTER_MAPPING,
                            _PSI_SCALE_PROPS_MAPPING, )
from psi_io._units import decompose_mas_units
from psi_io.psi_io import (PathLike,
                           PSI_DATA_ID,
                           SDC_TYPE_CONVERSIONS,
                           _dispatch_by_ext,
                           _except_no_scipy, )

class MetaDataWarning(UserWarning):
    ...

class CacheWarning(UserWarning):
    ...

HdfVersionType = Literal[4, 5]
"""Literal type alias for the two supported HDF file format versions.

``4`` — HDF4, accessed via pyhdf (optional dependency).
``5`` — HDF5, accessed via h5py.
"""


_HDF_EXT_MAPPING = MappingProxyType({'h4': '.hdf', 'h5': '.h5', })
"""Mapping from HDF version string to file extension.

Used by :class:`_HdfData.__init__` to validate that the supplied file has an
extension consistent with the concrete class's format mixin.

``'h5'`` → ``'.h5'`` (HDF5 files, read via h5py)
``'h4'`` → ``'.hdf'`` (HDF4 files, read via pyhdf)
"""


_CODE_UNIT_ALIASES = {'native', 'code', 'model', 'psi'}
"""Set of strings that request code-unit output from :meth:`_HdfInterface.read`.

When the ``units`` argument to ``read()`` is one of these strings, the data are
returned in MAS code units (dimensionless ratios) without any physical conversion.
"""


_REAL_UNIT_ALIASES = {'real', 'phys', 'physical', 'cgs'}
"""Set of strings that request decomposed CGS output from :meth:`_HdfInterface.read`.

When the ``units`` argument to ``read()`` is one of these strings, the data are
converted to physical CGS units via :func:`~psi_io._units.decompose_mas_units`.
"""

_BASE_SLOTS = ('_ref', '_id', '_cache', '_name', '_desc', '_unit', '_scalar', '_mesh', '_order', '_vcache',)
_SCALE_SLOTS = _BASE_SLOTS
_DATA_SLOTS = _BASE_SLOTS + ('_filepath', '_sequence', '_model', '_scales', '_icache')


METADATA_SCHEMA = dict.fromkeys(['name', 'desc', 'unit', 'scalar', 'mesh', 'order', 'sequence', 'model', 'scales'])
# TODO: docstring

SCALES_SCHEMA = dict.fromkeys(['name', 'desc', 'unit',])
# TODO: docstring

CacheType = Optional[Literal['lazy', 'eager']]


def _interpolate_dim(arr: QuantityLike,
                     axis: int,
                     value: QuantityLike,
                     scale: QuantityLike,
                     ) -> QuantityLike:
    """Linearly interpolate *arr* along *axis* to *value*, collapsing that axis to size 1.

    Both *value* and *scale* must carry the same unit (or both be dimensionless).
    The result is a fill array when *value* is outside the 2-element *scale* window
    and *fill_value* is not ``None``; otherwise a linear interpolation is performed.
    Extrapolation occurs silently when *fill_value* is ``None``.
    """
    if arr.shape[axis] != 2 or len(scale) != 2:
        raise ValueError("Interpolation is only supported for 2-element arrays and scales.")
    t = (value - scale[0]) / (scale[1] - scale[0])
    slc_lo = [slice(None)] * arr.ndim
    slc_hi = [slice(None)] * arr.ndim
    slc_lo[axis] = slice(None, -1)
    slc_hi[axis] = slice(1, None)
    return (1.0 - t) * arr[tuple(slc_lo)] +  t * arr[tuple(slc_hi)]


def _slice_array(data: QuantityLike,
                 scales: Sequence[Optional[QuantityLike]],
                 values: Sequence[Optional[QuantityLike]],
                 order: ArrayOrdering = 'F') -> QuantityLike:
    """Interpolate *data* to physical values along each axis that has a non-``None`` entry.

    Iterates over axes in storage order (reversed from physical order when
    ``order='F'``) and calls :func:`_interpolate_dim` for each axis whose
    entry in *values* is not ``None``.  Axes with ``None`` entries are left
    unchanged.  *scales* must be in the same order as *values* and provide the
    two-element coordinate window for each interpolated axis.
    """
    if order == 'F':
        values, scales = reversed(values), reversed(scales)
    for i, (v, s) in enumerate(zip(values, scales)):
        if v is not None:
            data = _interpolate_dim(data, i, v, s)
    return data


def _expand_args(*args, ndim: int) -> tuple:
    """Expand *args* to a length-*ndim* tuple, replacing ``Ellipsis`` and padding with ``None``.

    Parameters
    ----------
    *args : object
        User-supplied dimension arguments.  An ``Ellipsis`` is replaced by the
        appropriate number of ``None`` values so that the total length equals *ndim*.
        If fewer than *ndim* arguments are provided, trailing ``None``\\ s are appended.
    ndim : int
        Target length of the returned tuple.

    Returns
    -------
    out : tuple
        Length-*ndim* tuple of dimension arguments.
    """
    if Ellipsis in args:
        n_missing = ndim - (len(args) - 1)
        idx = args.index(Ellipsis)
        args = args[:idx] + (None,) * n_missing + args[idx + 1:]
    if len(args) < ndim:
        args += (None,) * (ndim - len(args))
    elif len(args) > ndim:
        raise ValueError(f"Too many dimension arguments: expected at most {ndim}, got {len(args)}.")
    return args


def _expand_quantity_filter(quantities: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for q in quantities:
        q = q.lower()
        if q in {'b', 'j', 'v'}:
            out.update(f"{q}{c}" for c in 'rtp')
        else:
            out.add(q)
    return out


def _parse_islice_args(*args,
                       shape: tuple[int, ...],
                       remesh: tuple[bool, ...],):
    for arg, size, rmesh in zip(args, shape, remesh):
        slice_ = _cast_to_slice(arg)
        if rmesh and slice_.stop is not None:
            slice_ = slice(slice_.start, slice_.stop + 1, slice_.step)
        start, stop, step = slice_.indices(size - bool(rmesh))
        if stop <= start:
            raise ValueError(f"Slice argument {arg!r} yields an empty dimension.")
        if step != 1:
            raise ValueError(f"Slice argument {arg!r} has a step size of {step}, but only step=1 is supported.")
        yield slice_


def _parse_vslice_args(*args,
                       scales: tuple[int, ...],
                       remesh: tuple[bool, ...]):
    for arg, scale, rmesh in zip(args, scales, remesh):
        if arg is None:
            yield (None, None), slice(None)
            continue
        arg = u.Quantity(arg, unit=scale.unit, ndmin=1)
        if arg.size not in {1, 2}:
            raise ValueError(f"Invalid argument {arg!r}: expected a scalar or 2-element sequence.")
        nan_mask = np.isnan(arg)
        if nan_mask[0]:
            arg[0] = -np.inf
        if nan_mask[-1]:
            arg[-1] = np.inf
        if np.all(np.isinf(arg)):
            yield (None, None), slice(None)
            continue
        offset = int(rmesh)
        n = scale.size
        raw = np.clip(np.searchsorted(scale[:], arg), 1 + offset, n - 1 - offset).tolist()
        start, stop = raw[0] - 1 - offset, raw[-1] + 1 + offset
        if stop <= start:
            raise ValueError(f"Slice argument {arg!r} yields an empty dimension.")
        yield arg, slice(start, stop)


def _apply_units(data: u.Quantity,
                 unit: Optional[UnitLike]) -> u.Quantity:
    """Apply a unit conversion to *data*, returning a :class:`~u.Quantity`.

    Parameters
    ----------
    data : u.Quantity
        Data in code units.
    unit : str, u.Unit, or None
        Requested output unit.  ``None`` is a no-op.  Special string aliases:
        ``'native'`` / ``'code'`` / ``'model'`` / ``'psi'`` — return *data*
        unchanged; ``'real'`` / ``'phys'`` / ``'physical'`` / ``'cgs'`` —
        decompose to CGS base unit via
        :func:`~psi_io._units.decompose_mas_units`.  Any other value is
        forwarded to :meth:`~u.Quantity.to`.

    Returns
    -------
    out : u.Quantity
        *data* in the requested unit.
    """
    if unit is None:
        return data
    ounit = str(unit).lower()
    if ounit in _CODE_UNIT_ALIASES:
        return data
    if ounit in _REAL_UNIT_ALIASES:
        return decompose_mas_units(data)
    return data.to(unit)


def _cast_to_slice(input: None | int | slice | Sequence) -> slice:
    """Convert a dimension index argument to a :class:`slice` object.

    Parameters
    ----------
    input : None, int, slice, list, or tuple
        - ``None`` → ``slice(None)`` (full axis).
        - :class:`int` → ``slice(input, input + 1)`` (single element, axis retained).
        - :class:`slice` → returned unchanged.
        - :class:`list` or :class:`tuple` → ``slice(*input)`` (unpacked as
          ``(start, stop)`` or ``(start, stop, step)``).

    Returns
    -------
    out : slice

    Raises
    ------
    TypeError
        If *input* is not one of the recognized types.
    """
    if input is None:
        return slice(None)
    elif isinstance(input, int):
        return slice(input, input + 1) if input >= 0 else slice(input - 1, input)
    elif isinstance(input, slice):
        return input
    elif isinstance(input, Collection):
        return slice(*input)
    else:
        raise TypeError(f"Invalid slice argument: {input!r}. Expected int or 2-element sequence.")

# =============================================================================
# Abstract interface
# =============================================================================

class _HdfArray(ABC):

    __slots__ = ()

    _HDFN: ClassVar[HdfVersionType]

    def __init__(self,
                 *args,
                 cache: CacheType = 'lazy',
                 **kwargs):
        self._vcache = None
        self._cache = cache and cache.lower()
        if self._cache not in {'lazy', 'eager', None}:
            raise ValueError(f"Invalid cache method: {cache!r}. "
                             f"Expected 'lazy', 'eager', or None.")

        try:
            self._set_metadata(**self._parse_inputs(**kwargs))
        except (TypeError, ValueError) as e:
            raise ValueError("Missing or incompatible metadata") from e

        if self._cache == 'eager':
            self.load(recursive=False)


    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"name={self.name!r} [{self.desc}], "
                f"order={self.order!r}, "
                f"shape={self.shape!r}, "
                f"unit={self.unit!r}, "
                f"mesh={self.mesh!r}, "
                f"cached={self.cached!r})")

    def __getitem__(self, args: str | int | slice | tuple):
        if isinstance(args, str):
            return self._dataset(args)

        if not isinstance(args, tuple):
            args = (args,)
        if self._reverse:
            args = args[::-1]
        if self._vcache is not None:
            return self._vcache[args]
        else:
            odata = self.dataset[args]
            if self._cache and odata.shape == self._shape:
                self._vcache = odata
            return odata

    def select(self, id_: str) -> Sequence:
        return self._dataset(id_)

    @property
    @abstractmethod
    def _shape(self) -> tuple[int, ...]:
        ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        ...

    @property
    @abstractmethod
    def nbytes(self) -> int:
        ...

    @property
    @abstractmethod
    def ndim(self) -> int:
        ...

    @property
    @abstractmethod
    def attrs(self) -> dict:
        ...

    @property
    def name(self) -> str:
        return self._name

    @property
    def desc(self) -> str:
        return self._desc

    @desc.setter
    def desc(self, value: str):
        self._desc = str(value)

    @property
    def unit(self) -> u.Unit:
        return self._unit

    @unit.setter
    def unit(self, value: UnitLike):
        self._unit = u.Unit(str(value))

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape[::-1] if self._reverse else self._shape

    @property
    def order(self) -> ArrayOrdering:
        return self._order

    @property
    def data_cached(self) -> bool:
        return self._vcache is not None

    @property
    def cached(self) -> bool:
        return self.data_cached

    @property
    def cache(self) -> str:
        return self._cache

    @cache.setter
    def cache(self, method: CacheType):
        self._cache = method and method.lower()
        if self._cache not in {'lazy', 'eager', None}:
            raise ValueError(f"Invalid cache method: {method!r}. "
                             f"Expected 'lazy', 'eager', or None.")
        if self._cache == 'eager':
            self.load()
        elif self._cache is None:
            self.clear()


    @property
    def _reverse(self) -> bool:
        return self.order == 'F'

    @property
    def dataset(self):
        return self._dataset(self._id)

    @abstractmethod
    def _dataset(self, id_: str):
        ...

    @abstractmethod
    def _parse_inputs(self, **kwargs) -> dict:
        ...

    @abstractmethod
    def _set_metadata(self, **kwargs) -> None:
        ...

    @abstractmethod
    def validate_metadata(self) -> None:
        if self.unit == u.dimensionless_unscaled:
            warnings.warn(f"{self.__class__.__name__}({self}) has a dimensionless unit. "
                          f"Ensure the correct unit is declared at instantiation or written to "
                          f"the HDF dataset's attribute mapping.", MetaDataWarning, stacklevel=3)


    def read(self,
             *args: None | int | tuple[int | None, ...] | slice,
             unit: Optional[str | UnitLike] = None,
             mesh: Optional[MeshLike] = None,
             order: Optional[ArrayOrdering] = None,
             scales: bool = False) -> u.Quantity | tuple[u.Quantity, ...]:
        remesh = self.mesh >> mesh
        args = _expand_args(*args, ndim=self.ndim)
        sargs = tuple(_parse_islice_args(*args, shape=self.shape, remesh=remesh))
        odata = _apply_units(self._read(*sargs, remesh=remesh), unit)
        if not scales:
            return odata
        return (odata,)

    def slice(self, *args, **kwargs) -> u.Quantity | tuple[u.Quantity, ...]:
        return self.read(*args, **kwargs)

    def _read(self, *args, remesh: tuple[bool,...]) -> u.Quantity:
        return _remesh_array(self[args], remesh=remesh, order=self.order) * self.unit

    def load(self, **kwargs):
        if self._cache is None:
            warnings.warn(f"{self.__class__.__name__}({self}) has caching disabled; load() has no effect.", CacheWarning, stacklevel=3)
            return
        self._vcache = self.dataset[:]

    def clear(self, **kwargs):
        if self._cache == 'eager':
            warnings.warn(f"{self.__class__.__name__}({self}) has eager caching enabled; clear() was called explicitly.", CacheWarning, stacklevel=3)
        self._vcache = None


class _HdfScale(_HdfArray, ABC):
    def __init__(self,
                 parent: '_HdfData',
                 dataset_id: Optional[str],
                 **kwargs):
        self._ref = parent
        self._id = dataset_id
        super().__init__(**kwargs)

    def validate_metadata(self) -> None:
        super().validate_metadata()
        if 1 != self.ndim != len(self.mesh) != len(self.shape):
            warnings.warn(f'Scale {self} has {self.ndim} dimensions; expected 1.', MetaDataWarning, stacklevel=3)
        if self.name not in _PSI_SCALE_PROPS_MAPPING:
                warnings.warn(f"{self.__class__.__name__}({self}) has an unrecognized scale name {self.name!r}. "
                            f"Check that the correct name is declared at instantiation or written to the HDF dataset's attribute mapping.", MetaDataWarning, stacklevel=3)
        elif self.name[0] == 't' and not self.mesh and not np.isclose(self.read(0)[0], 0 * self.unit, rtol=0.0, atol=1e-12):
                warnings.warn(f"{self.__class__.__name__}({self}) has a main-mesh stagger but non-zero values at the inner boundary. "
                              f"Check that the correct mesh code is declared at instantiation or written to the HDF dataset's attribute mapping.", MetaDataWarning, stacklevel=3)
        elif self.name[0] == 'p' and not self.mesh and not np.isclose(self.read(0)[0], 0 * self.unit, rtol=0.0, atol=1e-12):
                warnings.warn(f"{self.__class__.__name__}({self}) has a main-mesh stagger but non-zero values at the inner boundary. "
                              f"Check that the correct mesh code is declared at instantiation or written to the HDF dataset's attribute mapping.", MetaDataWarning, stacklevel=3)

    def _parse_inputs(self, **kwargs):
        input_attrs = {k: v for k, v in kwargs.items()}
        file_attrs = {k: v for k, v in self.attrs.items() if k in SCALES_SCHEMA}
        combined_attrs = {**file_attrs, **input_attrs}

        if (name := combined_attrs.get('name')) in _PSI_SCALE_PROPS_MAPPING:
            native_attrs = get_psi_scale_properties(name)._asdict()
        else:
            native_attrs = {}

        return {**native_attrs, **combined_attrs}

    def _set_metadata(self,
                      *,
                      name: str,
                      unit: str = '',
                      desc: str = '',
                      validate: bool = False) -> None:
        self._name: str = str(name)
        self._desc: str = str(desc)
        self._unit: u.Unit = u.Unit(str(unit))
        self._scalar: bool = True
        self._mesh: Mesh = self._ref.mesh[self._ref._scales.index(self._name)]
        self._order: str = 'C'
        if validate:
            self.validate_metadata()

class _HdfData(_HdfArray, ABC):
    def __init__(self,
                 ifile: PathLike,
                 dataset_id: Optional[str] = None,
                 **kwargs):
        ifile = Path(ifile)
        hdfv = f'h{self._HDFN}'
        if not ifile.is_file():
            raise FileNotFoundError(f"File '{ifile}' does not exist.")
        if ifile.suffix != _HDF_EXT_MAPPING[hdfv]:
            raise ValueError(f"File '{ifile}' does not have the correct extension for "
                             f"{self._HDFN} files (expected '{_HDF_EXT_MAPPING[hdfv]}' extension).")

        self._filepath: Path = ifile
        self._ref = self.read_file(ifile)
        self._id = dataset_id or PSI_DATA_ID[hdfv]
        self._icache = None
        super().__init__(**kwargs)

    def __enter__(self):
        """Open (or re-open) the file and return ``self`` for use as a context manager."""
        self.open()
        return self

    def __exit__(self, *args):
        """Close the file handle when exiting the context manager."""
        self.close()

    def __del__(self):
        """Close the file handle when the object is garbage-collected."""
        self.delete()

    @classmethod
    @abstractmethod
    def read_file(cls, ifile: PathLike):
        """Open the HDF file at *ifile* and return the format-specific file handle."""
        ...

    @property
    def sequence(self) -> int:
        return self._sequence

    @sequence.setter
    def sequence(self, value: int):
        self._sequence = int(value)

    @property
    def model(self) -> str:
        return self._model

    @property
    def scales(self) -> tuple:
        return self._scales

    @property
    def interp_cached(self) -> bool:
        return self._icache is not None

    @abstractmethod
    def open(self):
        """Re-open the file handle if it was previously closed."""
        ...

    @abstractmethod
    def close(self):
        """Close the open file handle and set the internal reference to ``None``."""
        ...

    @abstractmethod
    def delete(self):
        """Release the file handle during garbage collection (called by ``__del__``)."""
        ...

    @abstractmethod
    def _get_dims(self) -> Sequence:
        ...

    @abstractmethod
    def _set_scales(self, scales: Sequence) -> type[tuple]:
        Scales = namedtuple('Scales', scales)
        self._scales = Scales._fields
        return Scales

    def validate_metadata(self) -> None:
        super().validate_metadata()
        if self._model not in _PROP_GETTER_MAPPING:
            warnings.warn(f"{self.__class__.__name__}({self}) has an unrecognized model {self._model!r}. "
                          f"Ensure metadata is declared at instantiation or written to "
                          f"the HDF dataset's attribute mapping.", MetaDataWarning, stacklevel=3)
        if any(self.ndim != len(length) for length in (self.scales, self.mesh, self.shape)):
            attributes = ('scales', 'mesh', 'shape')
            lengths = map(len, map(partial(getattr, self), attributes))
            msg = (f"{self.__class__.__name__}({self}) has inconsistent *data* dimensionality: "
                   f"{', '.join(f'len(self.{attr}) = {size}' for attr, size in zip(attributes, lengths))}")
            warnings.warn(msg, MetaDataWarning, stacklevel=3)
        if any(datashape != scaleshape.size for datashape, scaleshape in zip(self.shape, self.scales)):
            msg = (f"{self.__class__.__name__}({self}) has inconsistent *scale* dimensionality: "
                   f"shape = {self.shape!r}, order = {self.order!r}, scales = "
                   f"({', '.join(tuple(f'{scale}: {scale.size}' for scale in self.scales))})")
            warnings.warn(msg, MetaDataWarning, stacklevel=3)

    def _parse_inputs(self, **kwargs) -> dict:
        input_attrs = {k: v for k, v in kwargs.items()}
        file_attrs = {k: v for k, v in self.attrs.items() if k in METADATA_SCHEMA}
        combined_attrs = {**file_attrs, **input_attrs}

        combined_attrs.setdefault('model', 'custom')
        if combined_attrs['model'] in _PROP_GETTER_MAPPING:
            combined_attrs.setdefault('name', extract_quantity_from_filepath(self._filepath, ''))
            combined_attrs.setdefault('sequence', extract_sequence_from_filepath(self._filepath, 0))
            prop_getter = get_model_prop_caller(combined_attrs['model'])
            native_attrs = prop_getter(combined_attrs['name'])._asdict()
        else:
            native_attrs = {}

        return {**native_attrs, **combined_attrs}

    def _set_metadata(self,
                      *,
                      model: Optional[ModelType | str],
                      name: str,
                      mesh: MeshLike,
                      scalar: bool,
                      order: ArrayOrdering,
                      scales: Sequence,
                      unit: str = '',
                      sequence: int = 0,
                      desc: str = '',
                      validate: bool = True) -> None:
        self._model: str = str(model)
        self._name: str = str(name)
        self._desc: str = str(desc)
        self._unit: u.Unit = u.Unit(str(unit))
        self._scalar: bool = bool(scalar)
        self._mesh: Mesh = Mesh.parse(mesh, self.ndim)
        self._order: str = str(order).upper()
        self._sequence: int = int(sequence)
        self._set_scales(scales)
        if validate:
            self.validate_metadata()
            for scale in self.scales:
                scale.validate_metadata()

    def read(self,
             *args: None | int | tuple[int | None, ...] | slice,
             unit: Optional[str | UnitLike] = None,
             mesh: Optional[MeshLike] = None,
             order: Optional[ArrayOrdering] = None,
             scales: bool = True) -> u.Quantity | tuple[u.Quantity, ...]:
        remesh = self.mesh >> mesh
        args = _expand_args(*args, ndim=self.ndim)
        sargs = tuple(_parse_islice_args(*args, shape=self.shape, remesh=remesh))
        odata = _apply_units(self._read(*sargs, remesh=remesh), unit=unit)
        if order is not None and order.upper() != self.order:
            odata = odata.T
        if not scales:
            return odata
        oscales = (scale._read(sarg, remesh=rmesh) for scale, sarg, rmesh in zip(self.scales, sargs, remesh))
        return odata, *oscales

    def interp(self,
               data,
               unit: Optional[str | UnitLike] = None,
               **kwargs
               ) -> u.Quantity:
        _except_no_scipy()
        positions = QTable(data, names=self.scales._fields, units=[scale.unit for scale in self.scales])
        positions = structured_to_unstructured(positions.as_array())

        bounds_error = kwargs.get('bounds_error', True)
        vslice_args = [(np.min(positions[:, i]), np.max(positions[:, i]))
                       for i in range(positions.shape[-1])]

        if self._cache is None:
            data, *scales = self.vslice(*vslice_args, bounds_error=bounds_error, order='C')
            return _apply_units(
                RegularGridInterpolator(scales, data, **kwargs)(positions) << self.unit,
                unit=unit,
            )

        needs_build = (
            self._icache is None
            or any(lo < g[0] or hi > g[-1]
                   for g, (lo, hi) in zip(self._icache.grid, vslice_args))
        )

        if needs_build:
            if self.data_cached:
                arr = self._vcache[:].T if self._reverse else self._vcache[:]
                self._icache = RegularGridInterpolator(
                    [scale[:] for scale in self.scales], arr, **kwargs
                )
            else:
                data, *scales = self.vslice(*vslice_args, bounds_error=bounds_error, order='C')
                self._icache = RegularGridInterpolator(scales, data, **kwargs)

        return _apply_units(self._icache(positions) << self.unit, unit=unit)


    def vslice(self,
               *args,
               unit: Optional[str | UnitLike] = None,
               mesh: Optional[MeshCodeType] = None,
               order: Optional[ArrayOrdering] = None,
               scales: bool = True,
               bounds_error: bool = True,
               ) -> u.Quantity | tuple[u.Quantity, ...]:
        remesh = self.mesh >> mesh
        args = _expand_args(*args, ndim=self.ndim)
        varg_pairs = _parse_vslice_args(*args, scales=self.scales, remesh=remesh)
        slice_values, slice_args = map(list, zip(*varg_pairs))
        slice_mask = [len(sv) == 1 for sv in slice_values]

        remeshed_scales = [scale._read(sarg, remesh=rmesh)
              for scale, sarg, rmesh in zip(self.scales, slice_args, remesh)]
        for i, (svalue, rmesh) in enumerate(zip(slice_values, remesh)):
            if rmesh:
                if svalue[0] is not None and not np.isinf(svalue[0]) and svalue[0] > remeshed_scales[i][1]:
                    slice_args[i] = slice(slice_args[i].start + 1, slice_args[i].stop, slice_args[i].step)
                    remeshed_scales[i] = remeshed_scales[i][1:]
                if svalue[-1] is not None and not np.isinf(svalue[-1]) and svalue[-1] < remeshed_scales[i][-2]:
                    slice_args[i] = slice(slice_args[i].start, slice_args[i].stop - 1, slice_args[i].step)
                    remeshed_scales[i] = remeshed_scales[i][:-1]
            if bounds_error:
                if svalue[0] is not None and not np.isinf(svalue[0]) and svalue[0] < remeshed_scales[i][0]:
                    raise ValueError(f"Value {svalue[0]} is below the interpolation range {remeshed_scales[i][0]}.")
                if svalue[-1] is not None and not np.isinf(svalue[-1]) and svalue[-1] > remeshed_scales[i][-1]:
                    raise ValueError(f"Value {svalue[-1]} is above the interpolation range {remeshed_scales[i][-1]}.")

        pre_slice_data = _apply_units(self._read(*slice_args, remesh=remesh), unit)
        if all(not sm for sm in slice_mask):
            if order is not None and order.upper() != self.order:
                pre_slice_data = pre_slice_data.T
            if not scales:
                return pre_slice_data
            return pre_slice_data, *remeshed_scales

        pre_slice_scales = [sscale if sm else None for sscale, sm in zip(remeshed_scales, slice_mask)]
        pre_slice_values = [sv if sm else None for sv, sm in zip(slice_values, slice_mask)]

        sliced_data = _slice_array(pre_slice_data, pre_slice_scales, pre_slice_values, self.order)
        if order is not None and order.upper() != self.order:
            sliced_data = sliced_data.T
        if not scales:
            return sliced_data
        sliced_scales = (psvalue if psvalue is not None else sscale for psvalue, sscale in zip(pre_slice_values, remeshed_scales))
        return sliced_data, *sliced_scales

    def load(self, interp: bool = False, recursive: bool = True):
        if self._cache is None:
            warnings.warn(f"{self.__class__.__name__}({self}) has caching disabled; load() has no effect.", CacheWarning, stacklevel=3)
            return
        super().load()
        if recursive:
            for scale in self.scales:
                scale.load()
        if interp:
            _except_no_scipy()
            arr = self._vcache[:].T if self._reverse else self._vcache[:]
            self._icache = RegularGridInterpolator(
                [scale[:] for scale in self.scales], arr
            )

    def clear(self, data: bool = True, interp: bool = True, recursive: bool = True):
        if self._cache == 'eager':
            warnings.warn(f"{self.__class__.__name__}({self}) has eager caching enabled; clear() was called explicitly.", CacheWarning, stacklevel=3)
        if data:
            super().clear()
            if recursive:
                for scale in self.scales:
                    scale.clear()
        if interp:
            self._icache = None


class _H5ArrayMixin:
    __slots__ = ()
    _HDFN = 5

    @property
    def _shape(self) -> tuple[int, ...]:
        return self.dataset.shape

    @property
    def dtype(self) -> np.dtype:
        return self.dataset.dtype

    @property
    def size(self) -> int:
        return self.dataset.size

    @property
    def nbytes(self) -> int:
        return self.dataset.nbytes

    @property
    def ndim(self) -> int:
        return self.dataset.ndim

    @property
    def attrs(self) -> dict:
        return dict(self.dataset.attrs)

    def _dataset(self, id_: str):
        return self._ref[id_]


class _H4ArrayMixin:
    __slots__ = ()
    _HDFN = 4

    @property
    def _shape(self) -> tuple[int, ...]:
        shape_ = self.dataset.info()[2]
        return (shape_,) if not isinstance(shape_, Iterable) else tuple(shape_)

    @property
    def dtype(self) -> np.dtype:
        return SDC_TYPE_CONVERSIONS[self.dataset.info()[3]]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    @property
    def ndim(self) -> int:
        return self.dataset.info()[1]

    @property
    def attrs(self) -> dict:
        return self.dataset.attributes()

    def _dataset(self, id_: str):
        return self._ref.select(id_)


class H4Scale(_H4ArrayMixin, _HdfScale):
    ...


class H5Scale(_H5ArrayMixin, _HdfScale):
    ...


class H4Data(_H4ArrayMixin, _HdfData):
    @classmethod
    def read_file(cls, ifile: PathLike):
        """Open an HDF4 file for reading and return the pyhdf ``SD`` object."""
        return h4.SD(str(ifile), h4.SDC.READ)

    def open(self):
        """Re-open the HDF4 file if it was previously closed.  Returns ``self``."""
        if not self._ref:
            self._ref = self.read_file(self._filepath)
        return self

    def close(self):
        """Close the HDF4 file handle via ``end()``."""
        if self._ref is not None:
            self._ref.end()
            self._ref = None

    def delete(self):
        """Close the HDF4 file handle during garbage collection."""
        _ref = getattr(self, '_ref', None)
        if _ref is not None:
            _ref.end()
            self._ref = None

    def _get_dims(self) -> Sequence:
        sds = self.dataset
        dims = tuple(sds.dimensions(full=1).items())
        return dims[::-1] if self._reverse else dims

    def _set_scales(self,
                    scales: Sequence,) -> None:
        Scales = super()._set_scales(scales)
        dims = self._get_dims()
        self._scales: Scales = Scales(*(H4Scale(self, dim_label, cache=self._cache, name=scale)
                                        for (dim_label, dim_proxy), scale in zip(dims, Scales._fields)))


class H5Data(_H5ArrayMixin, _HdfData):

    @classmethod
    def read_file(cls, ifile: PathLike):
        """Open an HDF5 file for reading and return the :class:`h5py.File` handle."""
        return h5.File(ifile, 'r')

    def open(self):
        """Re-open the HDF5 file if it was previously closed.  Returns ``self``."""
        if not self._ref:
            self._ref = self.read_file(self._filepath)
        return self

    def close(self):
        """Close the HDF5 file handle.  Returns ``self``."""
        if self._ref is not None:
            self._ref.close()
            self._ref = None

    def delete(self):
        """Close the file handle during garbage collection (called by ``__del__``)."""
        _ref = getattr(self, '_ref', None)
        if _ref is not None:
            _ref.close()
            self._ref = None

    def _get_dims(self) -> Sequence:
        return self.dataset.dims

    def _set_scales(self,
                    scales: Sequence,) -> None:
        Scales = super()._set_scales(scales)
        dims = self._get_dims()
        self._scales: Scales = Scales(*(H5Scale(self, dim.label, cache=self._cache, name=scale) for
                                        dim, scale in zip(dims, Scales._fields)))


def PsiData(ifile: PathLike, /,
            *args,
            **kwargs):
    """Open a PSI MAS or POT3D HDF file and return the appropriate data reader.

    Inspects the file extension and *model* argument, selects the correct
    concrete reader (HDF5 or HDF4 backend), and returns it.  No data are read
    from disk at construction time — metadata is resolved from the filename and
    HDF file attributes, and data transfer happens only inside :meth:`read` or
    :meth:`vslice`.  Full-array reads are cached automatically; partial reads
    are not.

    The returned reader exposes the following attributes:

    - ``quantity`` — canonical lower-case quantity identifier (e.g. ``'br'``).
    - ``sequence`` — integer time-step sequence number.
    - ``unit`` — :class:`~astropy.units.Unit` for code → physical conversion;
      normalization constants are defined in :mod:`psi_io._units`.
    - ``mesh`` — per-axis Yee-grid stagger as a tuple of
      :class:`~psi_io._mesh.Mesh` members; see :mod:`psi_io._mesh`.
    - ``props`` — full :class:`~psi_io._models.Props` descriptor (name,
      description, unit, mesh code); see :mod:`psi_io._models`.
    - ``description`` — human-readable quantity description.
    - ``scales`` — ``Scales(r, t, p)`` named tuple of coordinate scale readers,
      each supporting the same :meth:`read` interface as the main reader.
    - ``shape``, ``ndim``, ``size``, ``nbytes``, ``dtype``, ``attrs`` — array
      metadata; shape is in HDF storage order ``(Nφ, Nθ, Nr)``.
    - ``is_cached`` — ``True`` after a full-array read has been cached.

    Use :meth:`read` to load a slice by index and :meth:`vslice` to slice by
    physical coordinate value with linear interpolation.  Both return data as
    :class:`~astropy.units.Quantity` objects in physical ``(r, θ, φ)`` order.
    The object also supports the context-manager protocol.

    .. warning:: **POT3D unit convention**

        POT3D applies no normalization to its output.  The stored values are in
        whatever physical unit the input photospheric magnetogram used — most
        commonly Gauss, but this is not encoded in the file.  The default
        ``unit`` for POT3D is ``dimensionless_unscaled`` (scale factor 1), so
        ``read(unit='physical')`` will not perform a meaningful conversion
        unless the correct unit is supplied at construction:

        .. code-block:: python

            reader = PsiData('br001.h5', model='pot3d', unit='Gauss')
            data, r, t, p = reader.read()

    Parameters
    ----------
    ifile : PathLike
        Path to the HDF4 (``.hdf``) or HDF5 (``.h5``) file.
    model : {'mas', 'pot3d'}, optional
        PSI model type.  Defaults to ``'mas'``.
    dataset_id : str, optional
        Dataset name within the HDF file.  Defaults to the PSI standard
        identifier for the given format.
    quantity : str, optional
        Override the quantity name inferred from the filename or file attributes.
    sequence : int, optional
        Override the time-step sequence number.
    unit : str or u.Unit, optional
        Override the code-to-physical unit from the quantity's
        :class:`~psi_io._models.Props` entry.  Accepts any string parseable by
        :class:`~astropy.units.Unit` or a :class:`~astropy.units.Unit` instance.
    mesh : MeshCodeType, optional
        Override the mesh stagger from the quantity's
        :class:`~psi_io._models.Props` entry.

    Returns
    -------
    out : H5Data or H4Data
        Open reader implementing the full ``_HdfInterface`` API.  Concrete type
        depends on the file extension.

    Raises
    ------
    ValueError
        If the file extension / model combination is unsupported or required
        metadata cannot be resolved.
    FileNotFoundError
        If *ifile* does not exist.

    See Also
    --------
    astropy.units.Unit : Unit constructor — accepts strings, compound
        expressions, and :class:`~astropy.units.Unit` instances.
    astropy.units.Quantity.to : Unit conversion used internally when
        a ``unit`` string is supplied to :meth:`read`.

    Examples
    --------
    Read a MAS radial field — full array with coordinate scales, then convert:

    >>> from psi_io.mhd_io import PsiData                  # doctest: +SKIP
    >>> reader = PsiData('br001001.h5')
    >>> data, r, t, p = reader.read()                      # code units (MAS_b)
    >>> data, r, t, p = reader.read(unit='Gauss')          # convert to Gauss

    Use as a context manager:

    >>> with PsiData('vr001001.h5') as reader:              # doctest: +SKIP
    ...     data, r, t, p = reader.read(unit='km/s')

    Inspect metadata without loading data:

    >>> reader = PsiData('rho001001.h5')                    # doctest: +SKIP
    >>> reader.quantity    # 'rho'
    >>> reader.unit        # MAS_n
    >>> reader.mesh        # (Mesh.HALF, Mesh.HALF, Mesh.HALF)
    >>> reader.is_cached   # False
    """
    return _dispatch_by_ext(ifile, H4Data, H5Data, *args, **kwargs)
