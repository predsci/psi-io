r"""Lazy, unit-aware HDF readers for PSI MAS and POT3D magnetohydrodynamic output.

This module reads three-dimensional field variables from Predictive Science Inc.'s
MAS and POT3D solvers through a single interface that spans both HDF4 (``.hdf``) and
HDF5 (``.h5``) files.  Readers are *lazy*: metadata is resolved at construction from
the filename and HDF attributes, while array data is transferred from disk only on
access, and full-dataset reads are cached on the reader.

.. rubric:: Entry point

:func:`PsiData` is the sole public symbol and the intended way to use this module.
It is a factory that inspects the file extension and ``model`` argument and returns
the matching concrete reader; the underlying ``_Hdf*`` classes should never be
instantiated directly.

.. code-block:: python

    from psi_io.mhd_io import PsiData

    PsiData('br001001.hdf', model='mas')                                    # MAS HDF4
    PsiData('br001.h5', model='pot3d', unit='Gauss')                        # POT3D HDF5 (unit declared)
    PsiData('emission.h5', mesh='MMM', scales='X,Y,Z', order='C', ...)      # Custom reader

.. rubric:: Reader interface

The object returned by :func:`PsiData` is an :class:`_HdfData` instance.  Refer to
the following classes for the complete reference:

- :class:`_HdfData` — the main field reader.  :meth:`_HdfData.read` slices by index
  and :meth:`_HdfData.vslice` slices by physical coordinate value with linear
  interpolation.  Both return :class:`~astropy.units.Quantity` data in physical
  ``(r, θ, φ)`` order, with optional unit conversion and mesh remapping.
- :class:`_HdfArray` — the array interface inherited by every reader, defining the
  metadata properties (``name``, ``desc``, ``unit``, ``mesh``, ``shape``, ``dtype``,
  ``data_cached``, ``interp_cached`` …) and the base :meth:`_HdfArray.read`.
- ``reader.scales`` — a named tuple of per-axis coordinate readers ``(r, t, p)``
  that expose the same :class:`_HdfArray` interface as the main reader.

.. note::
    PSI HDF files are stored in Fortran column-major order, so ``reader.shape`` and
    the on-disk layout are ``(Nφ, Nθ, Nr)`` with the radial axis last.  All
    slicing arguments and returned coordinate scales are nevertheless given in
    physical ``(r, θ, φ)`` order.

.. rubric:: Supported quantities

MAS provides 19 field variables — magnetic field, velocity, and current-density
components, temperatures, density, pressure, Alfvén/Elsässer wave energy, and
coronal heating — while POT3D provides the three magnetic-field components.  Each
quantity's canonical unit and mesh stagger are defined in :mod:`psi_io.models`, and
the corresponding normalization constants in :mod:`psi_io.units`.  POT3D output is
unnormalized; see the :func:`PsiData` warning on declaring its unit.
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

from psi_io.mesh import (MeshCodeType,
                          _remesh_array,
                          ArrayOrdering, Mesh, MeshLike,
                          )
from psi_io.models import (ModelType,
                            extract_quantity_from_filepath,
                            extract_sequence_from_filepath,
                            get_model_prop_caller,
                            get_psi_scale_properties,
                            _PROP_GETTER_MAPPING,
                            _PSI_SCALE_PROPS_MAPPING, )
from psi_io.units import decompose_mas_units
from psi_io.psi_io import (PathLike,
                           PSI_DATA_ID,
                           SDC_TYPE_CONVERSIONS,
                           _dispatch_by_ext,
                           _except_no_scipy, )

class MetaDataWarning(UserWarning):
    """Warning raised when HDF metadata is missing, ambiguous, or inconsistent.

    Emitted by :meth:`_HdfArray.validate_metadata` and its overrides when a
    required attribute (quantity name, unit, mesh code) cannot be resolved
    unambiguously from the file and keyword arguments.

    Examples
    --------
    >>> import warnings
    >>> from psi_io.mhd_io import MetaDataWarning
    >>> issubclass(MetaDataWarning, UserWarning)
    True
    """


class CacheWarning(UserWarning):
    """Warning raised when a cache operation is ignored or conflicts with the cache mode.

    Emitted by :meth:`_HdfArray.load` when caching is disabled (``cache=None``),
    and by :meth:`_HdfArray.clear` when the cache mode is ``'eager'`` and
    ``clear()`` is called explicitly.

    Examples
    --------
    >>> import warnings
    >>> from psi_io.mhd_io import CacheWarning
    >>> issubclass(CacheWarning, UserWarning)
    True
    """


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
converted to physical CGS units via :func:`~psi_io.units.decompose_mas_units`.
"""

_BASE_SLOTS = ('_ref', '_id', '_cache', '_name', '_desc', '_unit', '_scalar', '_mesh', '_order', '_vcache',)
"""Slot names shared by all :class:`_HdfArray` subclasses."""

_SCALE_SLOTS = _BASE_SLOTS
"""Slot names for :class:`_HdfScale` subclasses (identical to :data:`_BASE_SLOTS`)."""

_DATA_SLOTS = _BASE_SLOTS + ('_filepath', '_sequence', '_model', '_scales', '_icache')
"""Slot names for :class:`_HdfData` subclasses; extends :data:`_BASE_SLOTS` with data-reader fields."""


METADATA_SCHEMA = dict.fromkeys(['name', 'desc', 'unit', 'scalar', 'mesh', 'order', 'sequence', 'model', 'scales'])
"""Template dictionary of recognized HDF dataset-level metadata keys.

Keys that appear in an HDF dataset's attribute mapping and also appear here are
extracted and merged with keyword arguments during :meth:`_HdfData._parse_inputs`.
The value for each key is always ``None`` in this template; it is used only for
membership testing.

Keys
----
name : str
    Canonical lower-case quantity identifier (e.g. ``'br'``).
desc : str
    Human-readable quantity description.
unit : str
    String representation of the code-to-physical unit.
scalar : bool
    Whether the quantity is a scalar (``True``) or vector component (``False``).
mesh : int
    Integer mesh stagger code.
order : str
    Array memory layout (``'F'`` or ``'C'``).
sequence : int
    Time-step sequence number.
model : str
    PSI model type (``'mas'`` or ``'pot3d'``).
scales : sequence of str
    Names of the coordinate scale axes.
"""

SCALES_SCHEMA = dict.fromkeys(['name', 'desc', 'unit',])
"""Template dictionary of recognized HDF scale-dataset metadata keys.

Used analogously to :data:`METADATA_SCHEMA` for the one-dimensional coordinate
scale arrays.

Keys
----
name : str
    Coordinate axis name (``'r'``, ``'t'``, or ``'p'``).
desc : str
    Human-readable axis description.
unit : str
    String representation of the coordinate unit.
"""

CacheType = Optional[Literal['lazy', 'eager']]
"""Type alias for the three valid cache modes.

``'lazy'``
    Cache the full data array on the first full-array read.
``'eager'``
    Cache immediately at construction time via :meth:`_HdfArray.load`.
``None``
    Never cache; every read goes to disk.
"""


def _interpolate_dim(arr: QuantityLike,
                     axis: int,
                     value: QuantityLike,
                     scale: QuantityLike,
                     ) -> QuantityLike:
    """Linearly interpolate *arr* along *axis* to *value*, collapsing that axis to size 1.

    Both *value* and *scale* must carry compatible units.  The interpolated
    result retains all other axes unchanged; the axis dimension is reduced from
    2 to 1.

    Parameters
    ----------
    arr : QuantityLike
        Data array of shape ``(..., 2, ...)`` where ``2`` is along *axis*.
    axis : int
        Index of the axis to interpolate and collapse.
    value : QuantityLike
        Target coordinate value.  Must have the same unit as *scale*.
    scale : QuantityLike
        Two-element coordinate array bounding *value*.

    Returns
    -------
    out : QuantityLike
        Interpolated array with the *axis* dimension reduced to 1.

    Raises
    ------
    ValueError
        If ``arr.shape[axis] != 2`` or ``len(scale) != 2``.

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> arr = np.array([[1.0, 2.0]]) * u.Gauss
    >>> scale = np.array([0.0, 1.0]) * u.R_sun
    >>> _interpolate_dim(arr, axis=1, value=0.5 * u.R_sun, scale=scale)
    <Quantity [[1.5]] G>
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
    """Interpolate *data* to physical coordinate values along each non-``None`` axis.

    Iterates over axes in storage order (reversed from physical order when
    ``order='F'``) and calls :func:`_interpolate_dim` for each axis whose
    entry in *values* is not ``None``.  Axes with ``None`` entries are left
    unchanged.

    Parameters
    ----------
    data : QuantityLike
        Data array to interpolate.
    scales : sequence of QuantityLike or None
        Two-element coordinate windows for each axis, in the same order as
        *values*.  ``None`` entries correspond to axes that are not interpolated.
    values : sequence of QuantityLike or None
        Target coordinate values for each axis.  ``None`` means skip that axis.
    order : {'F', 'C'}, optional
        Array memory layout.  ``'F'`` reverses the axis iteration order so that
        axis 0 corresponds to the last physical axis.  Default is ``'F'``.

    Returns
    -------
    out : QuantityLike
        *data* with each non-``None`` axis collapsed to size 1 by interpolation.

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> data = np.ones((2, 3)) * u.Gauss
    >>> scale = np.array([0.0, 1.0]) * u.R_sun
    >>> result = _slice_array(data, [scale, None], [0.5 * u.R_sun, None], order='C')
    >>> result.shape
    (1, 3)
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

    Raises
    ------
    ValueError
        If the number of explicit arguments (excluding ``Ellipsis``) exceeds *ndim*.

    Examples
    --------
    >>> _expand_args(0, 1, ndim=3)
    (0, 1, None)
    >>> _expand_args(..., 5, ndim=3)
    (None, None, 5)
    >>> _expand_args(ndim=2)
    (None, None)
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
    """Expand shorthand quantity group names to a flat set of canonical identifiers.

    The single-letter codes ``'b'``, ``'j'``, and ``'v'`` each expand to the
    three spherical-component variants (e.g. ``'b'`` → ``{'br', 'bt', 'bp'}``).
    All other strings are passed through unchanged after lowercasing.

    Parameters
    ----------
    quantities : iterable of str
        Quantity identifiers or group codes to expand.

    Returns
    -------
    out : set[str]
        Flat set of lower-case canonical quantity names.

    Examples
    --------
    >>> sorted(_expand_quantity_filter(['b', 'rho']))
    ['bp', 'br', 'bt', 'rho']
    >>> sorted(_expand_quantity_filter(['vr', 'V']))
    ['vp', 'vr', 'vt']
    """
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
    """Validate and normalize index-space slice arguments for each array axis.

    Converts each element of *args* to a :class:`slice` and adjusts the stop
    index when the axis needs remeshing (to include the extra element required
    for averaging).

    Parameters
    ----------
    *args : None | int | slice | tuple
        One argument per axis in physical ``(r, t, p)`` order.  Passed through
        :func:`_cast_to_slice`.
    shape : tuple[int, ...]
        Array shape in physical order.
    remesh : tuple[bool, ...]
        Per-axis remesh flags from :meth:`Mesh.__rshift__`.

    Yields
    ------
    out : slice
        Validated slice for each axis.

    Raises
    ------
    ValueError
        If a slice yields an empty dimension or uses a non-unit step.

    Examples
    --------
    >>> list(_parse_islice_args(None, 1, shape=(10, 5), remesh=(False, False)))
    [slice(None, None, None), slice(1, 2, None)]
    """
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
    """Parse value-space slice arguments, returning coordinate values and index slices.

    For each axis argument, determines whether it is an index-space argument
    (``None`` or ``slice``) or a physical-coordinate argument
    (:class:`~astropy.units.Quantity`, a bare scalar including :class:`int`, or
    a 2-element sequence of coordinate bounds).  Physical values are converted
    to the scale unit, ``NaN`` is treated as an open bound, and the surrounding
    index window is computed via :func:`numpy.searchsorted`.

    .. note::
       Unlike :meth:`PsiData.read`, an :class:`int` argument here is **not** an
       array index — it is a physical coordinate value (in the scale's native
       unit) that triggers interpolation.  Only ``None`` and ``slice`` are
       index-space.

    Parameters
    ----------
    *args : None | slice | QuantityLike
        One argument per axis in physical ``(r, t, p)`` order.
    scales : tuple
        Scale reader objects for each axis; used for unit conversion and
        coordinate lookup.
    remesh : tuple[bool, ...]
        Per-axis remesh flags from :meth:`Mesh.__rshift__`.

    Yields
    ------
    value : tuple[QuantityLike | None, QuantityLike | None]
        Target coordinate value(s) for interpolation, or ``(None, None)`` for
        index-space axes.
    slice_ : slice
        Index window into the axis that brackets the target coordinate.

    Raises
    ------
    ValueError
        If a physical-coordinate argument has more than 2 elements, or if it
        yields an empty index window.

    Examples
    --------
    >>> # Index-space arguments pass through as (None, None) / full slice
    >>> list(_parse_vslice_args(None, scales=[None], remesh=[False]))
    [((None, None), slice(None, None, None))]
    """
    for arg, scale, rmesh in zip(args, scales, remesh):
        if arg is None or isinstance(arg, slice):
            yield (None, None), _cast_to_slice(arg)
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
    data : Quantity
        Data in code units.
    unit : str | Unit | None
        Requested output unit.  ``None`` is a no-op.  Special string aliases:
        ``'native'`` / ``'code'`` / ``'model'`` / ``'psi'`` — return *data*
        unchanged; ``'real'`` / ``'phys'`` / ``'physical'`` / ``'cgs'`` —
        decompose to CGS base unit via
        :func:`~psi_io.units.decompose_mas_units`.  Any other value is
        forwarded to :meth:`~u.Quantity.to`.

    Returns
    -------
    out : Quantity
        *data* in the requested unit.

    Raises
    ------
    astropy.units.UnitConversionError
        If *unit* is not compatible with the unit of *data*.

    Examples
    --------
    >>> import astropy.units as u
    >>> data = 1.0 * u.Gauss
    >>> _apply_units(data, None) is data
    True
    >>> _apply_units(data, 'native').unit
    Unit("G")
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

    Examples
    --------
    >>> _cast_to_slice(None)
    slice(None, None, None)
    >>> _cast_to_slice(3)
    slice(3, 4, None)
    >>> _cast_to_slice((2, 8))
    slice(2, 8, None)
    >>> _cast_to_slice(slice(1, 5, 2))
    slice(1, 5, 2)
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
    """Abstract base class for a single HDF dataset with optional caching.

    Provides the common interface for both data arrays (:class:`_HdfData`) and
    coordinate scale arrays (:class:`_HdfScale`).  Concrete subclasses supply
    the HDF-version-specific implementations of the abstract properties
    (:attr:`_shape`, :attr:`dtype`, :attr:`size`, :attr:`nbytes`, :attr:`ndim`,
    :attr:`attrs`) and the :meth:`_dataset` factory.

    Subclasses must not be instantiated directly.  Use :func:`PsiData` or the
    concrete classes :class:`H4Data`, :class:`H5Data`, :class:`H4Scale`, and
    :class:`H5Scale`.

    Attributes
    ----------
    _vcache : np.ndarray | None
        In-memory copy of the full dataset array, or ``None`` when not cached.
    _cache : CacheType
        Active cache mode: ``'lazy'``, ``'eager'``, or ``None``.

    See Also
    --------
    _HdfData : Subclass that adds interpolation and scale management.
    _HdfScale : Subclass for one-dimensional coordinate arrays.
    PsiData : Public factory function.
    """

    __slots__ = ()

    _HDFN: ClassVar[HdfVersionType]

    def __init__(self,
                 *args,
                 cache: CacheType = 'lazy',
                 **kwargs):
        """Initialize the array with metadata and optionally load data into cache.

        Parameters
        ----------
        *args : object
            Passed to :meth:`_parse_inputs` by subclass constructors.
        cache : CacheType, optional
            Cache mode.  ``'lazy'`` caches on first full read, ``'eager'`` loads
            immediately, ``None`` disables caching entirely.  Default is
            ``'lazy'``.
        **kwargs : object
            Metadata keyword arguments forwarded to :meth:`_parse_inputs`.

        Raises
        ------
        ValueError
            If *cache* is not one of ``'lazy'``, ``'eager'``, or ``None``, or if
            metadata cannot be resolved from *kwargs*.
        """
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
        """Return the quantity name as a string.

        Returns
        -------
        out : str
            The :attr:`name` of the dataset (e.g. ``'br'``).

        Examples
        --------
        >>> str(reader)  # doctest: +SKIP
        'br'
        """
        return f"{self.name}"

    def __repr__(self):
        """Return a detailed string representation including key metadata.

        Returns
        -------
        out : str
            String of the form
            ``ClassName(name=... [...], order=..., shape=..., unit=..., mesh=..., cached=...)``.

        Examples
        --------
        >>> repr(reader)  # doctest: +SKIP
        "H5Data(name='br' [...], order='F', shape=(...), unit=..., mesh=..., cached=False)"
        """
        return (f"{self.__class__.__name__}("
                f"name={self.name!r} [{self.desc}], "
                f"order={self.order!r}, "
                f"shape={self.shape!r}, "
                f"unit={self.unit!r}, "
                f"mesh={self.mesh!r}, "
                f"cached={self.cached!r})")

    def __getitem__(self, args: str | int | slice | tuple):
        """Index into the dataset, returning from cache when available.

        When *args* is a string, delegates to :meth:`_dataset` (attribute
        lookup on the HDF file object).  Otherwise applies the index tuple to
        the cached array if available, or reads directly from the HDF file and
        caches the result for full-array reads.

        Parameters
        ----------
        args : str | int | slice | tuple
            Index expression.  String values select HDF sub-datasets by name.

        Returns
        -------
        out : np.ndarray
            Indexed data.

        Examples
        --------
        >>> reader[0]        # first element along axis 0  # doctest: +SKIP
        >>> reader[:]        # full array (may populate cache)  # doctest: +SKIP
        """
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
        """Return the HDF dataset or sub-dataset identified by *id_*.

        Parameters
        ----------
        id_ : str
            Dataset key within the open HDF file.

        Returns
        -------
        out : Sequence
            The format-specific dataset object.

        Examples
        --------
        >>> reader.select('Data-Set-2')  # doctest: +SKIP
        """
        return self._dataset(id_)

    @property
    @abstractmethod
    def _shape(self) -> tuple[int, ...]:
        """Raw dataset shape in HDF storage order (not reversed for Fortran arrays)."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Element data type of the HDF dataset.

        Returns
        -------
        out : np.dtype
            NumPy dtype (typically ``float32``).
        """
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Total number of elements in the dataset.

        Returns
        -------
        out : int
            Product of all dimension sizes.
        """
        ...

    @property
    @abstractmethod
    def nbytes(self) -> int:
        """Dataset size in bytes.

        Returns
        -------
        out : int
            ``size * dtype.itemsize``.
        """
        ...

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of spatial dimensions in the dataset.

        Returns
        -------
        out : int
            Always ``3`` for MAS/POT3D field variables; ``1`` for scale arrays.
        """
        ...

    @property
    @abstractmethod
    def attrs(self) -> dict:
        """HDF dataset-level attributes as a plain Python dictionary.

        Returns
        -------
        out : dict
            Mapping of attribute name strings to their values.
        """
        ...

    @property
    def name(self) -> str:
        """Canonical lower-case quantity identifier.

        Returns
        -------
        out : str
            E.g. ``'br'``, ``'vr'``, ``'t'``, ``'r'``.
        """
        return self._name

    @property
    def desc(self) -> str:
        """Human-readable description of the physical quantity.

        Returns
        -------
        out : str
            E.g. ``'MAS Magnetic Field (Radial Component)'``.
        """
        return self._desc

    @desc.setter
    def desc(self, value: str):
        """Set the human-readable description."""
        self._desc = str(value)

    @property
    def unit(self) -> u.Unit:
        """Astropy unit for converting from code units to physical units.

        Returns
        -------
        out : Unit
            E.g. :data:`~psi_io.units.MAS_b` for MAS magnetic field.
        """
        return self._unit

    @unit.setter
    def unit(self, value: UnitLike):
        """Set the physical unit from a unit-like value."""
        self._unit = u.Unit(str(value))

    @property
    def mesh(self) -> Mesh:
        """Yee-grid stagger code for this dataset.

        Returns
        -------
        out : Mesh
            :class:`~psi_io.mesh.Mesh` instance encoding per-axis stagger.
        """
        return self._mesh

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape in physical ``(r, t, p)`` order.

        For Fortran-order (``order='F'``) arrays the raw HDF shape is reversed.

        Returns
        -------
        out : tuple[int, ...]
            Dimension sizes in physical coordinate order.
        """
        return self._shape[::-1] if self._reverse else self._shape

    @property
    def order(self) -> ArrayOrdering:
        """Memory layout of the stored array.

        Returns
        -------
        out : str
            ``'F'`` for Fortran (column-major, PSI default) or ``'C'`` for C
            (row-major).
        """
        return self._order

    @property
    def data_cached(self) -> bool:
        """Whether the full data array is currently held in memory.

        Returns
        -------
        out : bool
            ``True`` if :attr:`_vcache` is not ``None``.
        """
        return self._vcache is not None

    @property
    def cached(self) -> bool:
        """Alias for :attr:`data_cached`.

        Returns
        -------
        out : bool
            ``True`` if the data array is cached.
        """
        return self.data_cached

    @property
    def cache(self) -> str:
        """Active cache mode.

        Returns
        -------
        out : str | None
            ``'lazy'``, ``'eager'``, or ``None``.
        """
        return self._cache

    @cache.setter
    def cache(self, method: CacheType):
        """Set the cache mode and trigger load or clear as appropriate.

        Parameters
        ----------
        method : CacheType
            New cache mode.  Setting ``'eager'`` calls :meth:`load`; setting
            ``None`` calls :meth:`clear`.

        Raises
        ------
        ValueError
            If *method* is not ``'lazy'``, ``'eager'``, or ``None``.
        """
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
        """Whether axis order should be reversed when indexing (Fortran-order arrays)."""
        return self.order == 'F'

    @property
    def dataset(self):
        """The primary HDF dataset object for this reader.

        Returns
        -------
        out : object
            Format-specific dataset handle for the main data array.
        """
        return self._dataset(self._id)

    @abstractmethod
    def _dataset(self, id_: str):
        """Return the HDF dataset identified by *id_*."""
        ...

    @abstractmethod
    def _parse_inputs(self, **kwargs) -> dict:
        """Parse and merge file attributes with keyword overrides into a metadata dict."""
        ...

    @abstractmethod
    def _set_metadata(self, **kwargs) -> None:
        """Apply the merged metadata dictionary to instance attributes."""
        ...

    @abstractmethod
    def validate_metadata(self) -> None:
        """Validate resolved metadata and emit :exc:`MetaDataWarning` for any issues.

        The base implementation warns when :attr:`unit` is dimensionless.
        Subclasses should call ``super().validate_metadata()`` before applying
        additional checks.

        Raises
        ------
        MetaDataWarning
            If the unit is dimensionless or other metadata is inconsistent.

        Examples
        --------
        >>> reader.validate_metadata()  # doctest: +SKIP
        """
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
        """Read data by index with optional unit conversion.

        .. attention::

           When reading/slicing data, the ``*args`` are **always** supplied in physical (*e.g.*
           :math:`(r, \\theta, \\phi)`) order. However, unless specified with the ``order``
           argument, the sliced data array will be returned in storage order.

        .. attention::

           The ``scales`` argument is only functional on the main data array reader
           (:class:`_HdfData`). It is accepted by scale readers (:class:`_HdfScale`)
           for pass-through compatibility but has no effect there.

        Parameters
        ----------
        *args : int | tuple | slice | None
            Index-space axis arguments in physical ``(r, t, p)`` order.
        unit : UnitLike | None, optional
            Output unit.  Default is ``None`` (code units).
        mesh : MeshLike | None, optional
            Target stagger mesh.  Default is ``None`` (no remeshing).
        order : ArrayOrdering | None, optional
            Transpose the output if it differs from storage order.
            Default is ``None``.
        scales : bool, optional
            If ``True`` (default), return coordinate slices alongside data.

        Returns
        -------
        data : Quantity
            Sliced data array with the specified mesh staggering, units, and ordering
            applied.
        *scales : Quantity
            Sliced scales (only returned when *scales* is ``True``).

        Examples
        --------
        >>> data, = reader.read(scales=True)  # doctest: +SKIP
        """
        remesh = self.mesh >> mesh
        args = _expand_args(*args, ndim=self.ndim)
        sargs = tuple(_parse_islice_args(*args, shape=self.shape, remesh=remesh))
        odata = _apply_units(self._read(*sargs, remesh=remesh), unit)
        if not scales:
            return odata
        return (odata,)

    def slice(self, *args, **kwargs) -> u.Quantity | tuple[u.Quantity, ...]:
        """Alias for :meth:`read`.

        Parameters
        ----------
        *args : object
            Forwarded to :meth:`read`.
        **kwargs : object
            Forwarded to :meth:`read`.

        Returns
        -------
        out : Quantity | tuple[Quantity, ...]
            Same as :meth:`read`.
        """
        return self.read(*args, **kwargs)

    def _read(self, *args, remesh: tuple[bool,...]) -> u.Quantity:
        """Read and remesh the dataset slice, applying the physical unit.

        Parameters
        ----------
        *args : slice
            Per-axis slice objects in storage order.
        remesh : tuple[bool, ...]
            Per-axis remesh flags.

        Returns
        -------
        out : Quantity
            Sliced and remeshed data multiplied by :attr:`unit`.
        """
        return _remesh_array(self[args], remesh=remesh, order=self.order) * self.unit

    def load(self, **kwargs):
        """Load the full dataset into the in-memory cache.

        Has no effect (emits :exc:`CacheWarning`) when ``cache=None``.

        Parameters
        ----------
        **kwargs : object
            Accepted but ignored; present for subclass override compatibility.

        Examples
        --------
        >>> reader.load()  # doctest: +SKIP
        >>> reader.data_cached  # doctest: +SKIP
        True
        """
        if self._cache is None:
            warnings.warn(f"{self.__class__.__name__}({self}) has caching disabled; load() has no effect.", CacheWarning, stacklevel=3)
            return
        self._vcache = self.dataset[:]

    def clear(self, **kwargs):
        """Release the in-memory data cache.

        Emits :exc:`CacheWarning` when ``cache='eager'`` to flag an explicit
        clear that conflicts with the cache mode.

        Parameters
        ----------
        **kwargs : object
            Accepted but ignored; present for subclass override compatibility.

        Examples
        --------
        >>> reader.clear()  # doctest: +SKIP
        >>> reader.data_cached  # doctest: +SKIP
        False
        """
        if self._cache == 'eager':
            warnings.warn(f"{self.__class__.__name__}({self}) has eager caching enabled; clear() was called explicitly.", CacheWarning, stacklevel=3)
        self._vcache = None


class _HdfScale(_HdfArray, ABC):
    """Abstract base class for a one-dimensional HDF coordinate scale array.

    Wraps a single 1-D coordinate dataset stored alongside a PSI data file
    (radial, co-latitude, or longitude scale).  Concrete implementations
    are :class:`H4Scale` and :class:`H5Scale`.

    Attributes
    ----------
    _ref : _HdfData
        Parent data reader that owns this scale.
    _id : str | None
        Dataset key within the parent's HDF file.

    See Also
    --------
    H4Scale : HDF4 concrete implementation.
    H5Scale : HDF5 concrete implementation.
    """

    def __init__(self,
                 parent: '_HdfData',
                 dataset_id: Optional[str],
                 **kwargs):
        """Initialize a scale reader from the parent data reader.

        Parameters
        ----------
        parent : _HdfData
            The data reader that owns this coordinate scale.
        dataset_id : str | None
            HDF dataset key for the scale array; ``None`` uses a default key.
        **kwargs : object
            Metadata keyword arguments forwarded to :meth:`_HdfArray.__init__`.
        """
        self._ref = parent
        self._id = dataset_id
        super().__init__(**kwargs)

    def validate_metadata(self) -> None:
        """Validate scale-specific metadata, warning on dimensionality or name issues."""
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
        """Merge file attributes with keyword overrides, resolving PSI defaults by name."""
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
        """Apply resolved metadata to instance attributes for a scale array."""
        self._name: str = str(name)
        self._desc: str = str(desc)
        self._unit: u.Unit = u.Unit(str(unit))
        self._scalar: bool = True
        self._mesh: Mesh = self._ref.mesh[self._ref._scales.index(self._name)]
        self._order: str = 'C'
        if validate:
            self.validate_metadata()

class _HdfData(_HdfArray, ABC):
    """Abstract base class for a PSI MAS or POT3D HDF data reader.

    Extends :class:`_HdfArray` with file lifecycle management, coordinate scale
    readers, value-space slicing, and spatial interpolation.  Concrete
    implementations are :class:`H4Data` (HDF4 backend) and :class:`H5Data`
    (HDF5 backend).

    Instances should be obtained via :func:`PsiData`, not constructed directly.

    Attributes
    ----------
    _filepath : pathlib.Path
        Absolute path to the open HDF file.
    _icache : RegularGridInterpolator | None
        Cached scipy interpolator, or ``None`` if not yet built.

    See Also
    --------
    H4Data : HDF4 concrete implementation.
    H5Data : HDF5 concrete implementation.
    PsiData : Public factory function.
    """

    def __init__(self,
                 ifile: PathLike,
                 dataset_id: Optional[str] = None,
                 **kwargs):
        """Open an HDF file and initialize the reader with resolved metadata.

        Parameters
        ----------
        ifile : PathLike
            Path to the HDF file.  Must exist and have the correct extension for
            the concrete subclass (``'.h5'`` for :class:`H5Data`, ``'.hdf'`` for
            :class:`H4Data`).
        dataset_id : str | None, optional
            Dataset key within the HDF file.  Defaults to the PSI standard
            identifier for the given format.
        **kwargs : object
            Metadata keyword arguments (``model``, ``name``, ``unit``, ``mesh``,
            etc.) forwarded to :meth:`_parse_inputs` and :meth:`_set_metadata`.

        Raises
        ------
        FileNotFoundError
            If *ifile* does not exist.
        ValueError
            If the file extension does not match the expected format, or if
            metadata cannot be resolved.
        """
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
        """Time-step sequence number extracted from the filename or file attributes.

        Returns
        -------
        out : int
            E.g. ``1001`` for a file named ``br001001.h5``.
        """
        return self._sequence

    @sequence.setter
    def sequence(self, value: int):
        """Set the sequence number."""
        self._sequence = int(value)

    @property
    def model(self) -> str:
        """PSI model type string.

        Returns
        -------
        out : str
            ``'mas'``, ``'pot3d'``, or ``'custom'``.
        """
        return self._model

    @property
    def scales(self) -> tuple:
        """Named tuple of coordinate scale readers ``(r, t, p)``.

        Each element is a :class:`_HdfScale` instance that wraps the
        corresponding one-dimensional coordinate array.

        Returns
        -------
        out : tuple
            Named tuple with fields matching the scale names (``r``, ``t``,
            ``p`` by default).
        """
        return self._scales

    @property
    def interp_cached(self) -> bool:
        """Whether a :class:`~scipy.interpolate.RegularGridInterpolator` is cached.

        Returns
        -------
        out : bool
            ``True`` if :attr:`_icache` is not ``None``.
        """
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
        """Return the dimension labels from the open HDF file in physical order."""
        ...

    @abstractmethod
    def _set_scales(self, scales: Sequence) -> type[tuple]:
        """Build scale readers from dimension labels and attach them to ``_scales``."""
        Scales = namedtuple('Scales', scales)
        self._scales = Scales._fields
        return Scales

    def validate_metadata(self) -> None:
        """Validate data-reader metadata including model, scale, and shape consistency."""
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
        """Merge file attributes with keyword overrides, resolving model defaults."""
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
        """Apply resolved metadata to instance attributes for a data reader."""
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
        """Read data by index with optional unit conversion and coordinate scales.

        .. attention::

           When reading/slicing data, the ``*args`` are **always** supplied in physical (*e.g.*
           :math:`(r, \\theta, \\phi)`) order. However, unless specified with the ``order``
           argument, the sliced data array will be returned in storage order.

        Parameters
        ----------
        *args : int | tuple | slice | None
            Index-space axis arguments in physical ``(r, t, p)`` order.
        unit : UnitLike | None, optional
            Output unit.  Default is ``None`` (code units).
        mesh : MeshLike | None, optional
            Target stagger mesh.  Default is ``None`` (no remeshing).
        order : ArrayOrdering | None, optional
            Transpose the output if it differs from storage order.
            Default is ``None``.
        scales : bool, optional
            If ``True`` (default), return coordinate slices alongside data.

        Returns
        -------
        data : Quantity
            Sliced data array with the specified mesh staggering, units, and ordering
            applied.
        *scales : Quantity
            Sliced scales (only returned when *scales* is ``True``).

        Examples
        --------
        >>> data, r, t, p = reader.read()  # doctest: +SKIP
        >>> data_gauss = reader.read(scales=False, unit='Gauss')  # doctest: +SKIP
        """
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
        """Interpolate the dataset at arbitrary spatial positions.

        Builds or reuses a :class:`~scipy.interpolate.RegularGridInterpolator`
        and evaluates it at the positions given by *data*.  When caching is
        disabled (``cache=None``), a minimal bounding-box slice is read on each
        call.  When caching is enabled, the interpolator is cached and reused for
        subsequent calls that fall within the same grid extent.

        Parameters
        ----------
        data : ArrayLike | Table
            Positions to interpolate, as a :class:`~astropy.table.Table`-like :math:`N \\times S` array,
            where :math:`\\lvert N \\rvert` is the number of positions, and :math:`\\lvert S \\rvert`
            is the number of scales. When columns do not possess a unit, they are cast to the
            corresponding scale's units.
        unit : UnitLike | None, optional
            Output unit.  Default is ``None`` (code units).
        **kwargs : object
            Forwarded to :class:`~scipy.interpolate.RegularGridInterpolator`.
            Notable keywords: ``bounds_error`` (default ``True``),
            ``fill_value`` (default ``None``).

        Returns
        -------
        out : Quantity
            Interpolated values of shape ``(N,)``.

        Raises
        ------
        ImportError
            If scipy is not installed.

        Examples
        --------
        >>> import numpy as np
        >>> import astropy.units as u
        >>> positions = np.column_stack([[1.5, 2.0], [1.57, 1.57], [0.1, 0.2]])
        >>> result = reader.interp(positions)  # doctest: +SKIP
        """
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
        """Read data by physical coordinate value with linear interpolation.

        Extends :meth:`read` to accept physical coordinate values as positional
        arguments.  A scalar, :class:`int`, or :class:`~astropy.units.Quantity`
        argument for an axis locates the two nearest grid points and linearly
        interpolates to the target value.  Only ``None`` and ``slice`` arguments
        are index-space and handled identically to :meth:`read`.

        .. attention::

           When reading/slicing data, the ``*args`` are **always** supplied in physical (*e.g.*
           :math:`(r, \\theta, \\phi)`) order. However, unless specified with the ``order``
           argument, the sliced data array will be returned in storage order.

        .. note::
           Unlike :meth:`read`, an :class:`int` argument is treated as a physical
           coordinate **value** (in the axis's native unit), not an array index.
           Use a ``slice`` to select by index.

        Parameters
        ----------
        *args : QuantityLike | slice | None
            One argument per axis in physical ``(r, t, p)`` order.  A
            :class:`~astropy.units.Quantity`, bare scalar, or :class:`int`
            triggers interpolation to that coordinate value; ``None`` and
            ``slice`` are index-space and do not interpolate.
        unit : UnitLike | None, optional
            Output unit.  Default is ``None`` (code units).
        mesh : MeshCodeType | None, optional
            Target stagger mesh.  Default is ``None``.
        order : ArrayOrdering | None, optional
            Transpose output if it differs from storage order.  Default is ``None``.
        scales : bool, optional
            If ``True`` (default), return coordinate slices alongside the data.
        bounds_error : bool, optional
            If ``True`` (default), raise :exc:`ValueError` when a physical value
            is outside the coordinate range.

        Returns
        -------
        data : Quantity
            Interpolated or sliced data array.
        *scales : Quantity
            Sliced scales (only returned when ``scales`` is ``True``).

        Raises
        ------
        ValueError
            If *bounds_error* is ``True`` and a physical value falls outside the
            coordinate range.

        Examples
        --------
        >>> # Extract the r = 2.5 solar radii surface
        >>> data, r, t, p = reader.vslice(2.5 * u.R_sun)  # doctest: +SKIP
        """
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
        """Load the data array and optionally build the interpolator into memory.

        Parameters
        ----------
        interp : bool, optional
            If ``True``, also build and cache the
            :class:`~scipy.interpolate.RegularGridInterpolator` after loading
            the data.  Requires scipy.  Default is ``False``.
        recursive : bool, optional
            If ``True`` (default), also call :meth:`load` on each coordinate
            scale reader.

        Examples
        --------
        >>> reader.load()  # doctest: +SKIP
        >>> reader.data_cached  # doctest: +SKIP
        True
        >>> reader.load(interp=True)  # doctest: +SKIP
        >>> reader.interp_cached  # doctest: +SKIP
        True
        """
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
        """Release cached data and/or the cached interpolator.

        Parameters
        ----------
        data : bool, optional
            If ``True`` (default), release the in-memory data array cache.
        interp : bool, optional
            If ``True`` (default), release the cached interpolator.
        recursive : bool, optional
            If ``True`` (default) and *data* is ``True``, also call
            :meth:`clear` on each coordinate scale reader.

        Examples
        --------
        >>> reader.clear()  # doctest: +SKIP
        >>> reader.data_cached, reader.interp_cached  # doctest: +SKIP
        (False, False)
        """
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
    """Mixin that provides HDF5 (h5py) property implementations for :class:`_HdfArray`.

    Sets ``_HDFN = 5`` and implements :attr:`_shape`, :attr:`dtype`, :attr:`size`,
    :attr:`nbytes`, :attr:`ndim`, :attr:`attrs`, and :meth:`_dataset` using the
    :class:`h5py.Dataset` interface.

    See Also
    --------
    _H4ArrayMixin : Analogous mixin for HDF4 files.
    """

    __slots__ = ()
    _HDFN = 5

    @property
    def _shape(self) -> tuple[int, ...]:
        """Raw dataset shape from the h5py Dataset object."""
        return self.dataset.shape

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype from the h5py Dataset object."""
        return self.dataset.dtype

    @property
    def size(self) -> int:
        """Total element count from the h5py Dataset object."""
        return self.dataset.size

    @property
    def nbytes(self) -> int:
        """Byte size from the h5py Dataset object."""
        return self.dataset.nbytes

    @property
    def ndim(self) -> int:
        """Number of dimensions from the h5py Dataset object."""
        return self.dataset.ndim

    @property
    def attrs(self) -> dict:
        """HDF5 dataset attributes as a plain dict."""
        return dict(self.dataset.attrs)

    def _dataset(self, id_: str):
        """Return the h5py Dataset at key *id_* from the open file."""
        return self._ref[id_]


class _H4ArrayMixin:
    """Mixin that provides HDF4 (pyhdf) property implementations for :class:`_HdfArray`.

    Sets ``_HDFN = 4`` and implements :attr:`_shape`, :attr:`dtype`, :attr:`size`,
    :attr:`nbytes`, :attr:`ndim`, :attr:`attrs`, and :meth:`_dataset` using the
    ``pyhdf.SD`` interface.

    See Also
    --------
    _H5ArrayMixin : Analogous mixin for HDF5 files.
    """

    __slots__ = ()
    _HDFN = 4

    @property
    def _shape(self) -> tuple[int, ...]:
        """Raw dataset shape from pyhdf SDS ``info()``."""
        shape_ = self.dataset.info()[2]
        return (shape_,) if not isinstance(shape_, Iterable) else tuple(shape_)

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype mapped from the pyhdf SDC type code."""
        return SDC_TYPE_CONVERSIONS[self.dataset.info()[3]]

    @property
    def size(self) -> int:
        """Total element count (product of all dimension sizes)."""
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        """Dataset size in bytes (``size * dtype.itemsize``)."""
        return self.size * self.dtype.itemsize

    @property
    def ndim(self) -> int:
        """Number of dimensions from pyhdf SDS ``info()``."""
        return self.dataset.info()[1]

    @property
    def attrs(self) -> dict:
        """HDF4 dataset attributes as a plain dict."""
        return self.dataset.attributes()

    def _dataset(self, id_: str):
        """Return the pyhdf SDS object at key *id_* from the open SD file."""
        return self._ref.select(id_)


class H4Scale(_H4ArrayMixin, _HdfScale):
    """HDF4 coordinate scale reader.

    Combines :class:`_H4ArrayMixin` (pyhdf property implementations) with
    :class:`_HdfScale` (PSI scale metadata and validation).

    See Also
    --------
    H5Scale : HDF5 equivalent.
    H4Data : The data reader that owns instances of this class.
    """


class H5Scale(_H5ArrayMixin, _HdfScale):
    """HDF5 coordinate scale reader.

    Combines :class:`_H5ArrayMixin` (h5py property implementations) with
    :class:`_HdfScale` (PSI scale metadata and validation).

    See Also
    --------
    H4Scale : HDF4 equivalent.
    H5Data : The data reader that owns instances of this class.
    """


class H4Data(_H4ArrayMixin, _HdfData):
    """HDF4 PSI data reader.

    Combines :class:`_H4ArrayMixin` (pyhdf property implementations) with
    :class:`_HdfData` (PSI data metadata, slicing, and interpolation).  Uses
    ``pyhdf.SD`` to open ``.hdf`` files.

    Instances are normally obtained via :func:`PsiData`.

    See Also
    --------
    H5Data : HDF5 equivalent.
    PsiData : Public factory function.
    """
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
    """HDF5 PSI data reader.

    Combines :class:`_H5ArrayMixin` (h5py property implementations) with
    :class:`_HdfData` (PSI data metadata, slicing, and interpolation).  Uses
    ``h5py.File`` to open ``.h5`` files.

    Instances are normally obtained via :func:`PsiData`.

    See Also
    --------
    H4Data : HDF4 equivalent.
    PsiData : Public factory function.
    """

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

    The returned object is an :class:`_HdfData` instance; see :class:`_HdfData`
    and the inherited :class:`_HdfArray` interface for the full set of metadata
    properties (``name``, ``desc``, ``unit``, ``mesh``, ``scales``, ``shape``,
    ``dtype`` …) and reader methods.  In brief, use :meth:`~_HdfData.read` to load
    a slice by index and :meth:`~_HdfData.vslice` to slice by physical coordinate
    value with linear interpolation; both return :class:`~astropy.units.Quantity`
    data in physical ``(r, θ, φ)`` order, and the object supports the
    context-manager protocol.

    By default the reader is **lazy** (``cache='lazy'``): array data is
    transferred from disk only on access, and a full-array read is then cached on
    the reader.  Pass ``cache='eager'`` to load the data immediately at
    construction, or ``cache=None`` to disable caching entirely.

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

    .. rubric:: Metadata resolution

    Every metadata field is drawn from up to three sources, in *decreasing*
    priority:

    1. **Keyword arguments** passed to this function, named after the
       :class:`~psi_io.models.ModelProps` fields (``name``, ``desc``, ``unit``,
       ``scalar``, ``mesh``, ``order``, ``sequence``, ``scales``).
    2. The **HDF dataset attribute dictionary** (``reader.attrs``), for any of
       those same field names stored on the file.
    3. The **model-mapping defaults** in :mod:`psi_io.models`, consulted only
       when *model* names a recognized PSI model.

    A value supplied at a higher level overrides the levels beneath it: an
    explicit keyword argument always wins over a file attribute, which in turn
    overrides the model default.  This is how the keyword arguments below can
    *correct* or *complete* metadata that is missing, wrong, or absent from the
    file.

    When *model* is ``'mas'`` or ``'pot3d'``, the quantity ``name`` and
    ``sequence`` additionally fall back to the values parsed from the filename
    stem (e.g. ``br001001`` → ``name='br'``, ``sequence=1001``), and the
    resolved ``name`` selects the :class:`~psi_io.models.ModelProps` entry that
    provides the default ``unit``, ``mesh``, ``scalar``, ``order``, ``scales``,
    and ``desc``.

    When *model* is the default ``'custom'``, **no defaults are inferred** —
    every field must be given as a keyword argument or be present in the file
    attribute dictionary.  ``name``, ``mesh``, ``scalar``, ``order``, and
    ``scales`` are required; ``unit`` (defaults to dimensionless), ``sequence``
    (defaults to ``0``), and ``desc`` (defaults to ``''``) are optional.  If a
    required field cannot be resolved, :exc:`ValueError` (*"Missing or
    incompatible metadata"*) is raised.

    Parameters
    ----------
    ifile : PathLike
        Path to the HDF4 (``.hdf``) or HDF5 (``.h5``) file.
    model : ModelType, optional
        PSI model type.  Defaults to ``'custom'``.  When ``'mas'`` or ``'pot3d'``
        is given, the reader resolves the quantity name, unit, mesh stagger, and
        other metadata from the corresponding mapping in :mod:`psi_io.models`.
        With the default ``'custom'``, no metadata is inferred and the required
        fields (``name``, ``mesh``, ``scalar``, ``order``, ``scales``) must be
        supplied as keyword arguments.
    dataset_id : str, optional
        Dataset name within the HDF file.  Defaults to the PSI standard
        identifier for the given format.
    name : str, optional
        Override the quantity name inferred from the filename or file attributes.
    sequence : int, optional
        Override the time-step sequence number.
    unit : UnitLike, optional
        Override the code-to-physical unit from the quantity's
        :class:`~psi_io.models.ModelProps` entry.  Accepts any string parseable by
        :class:`~astropy.units.Unit` or a :class:`~astropy.units.Unit` instance.
    mesh : MeshCodeType, optional
        Override the mesh stagger from the quantity's
        :class:`~psi_io.models.ModelProps` entry.  Required when *model* is
        ``'custom'`` and no ``mesh`` attribute is stored on the file.
    scalar : bool, optional
        Whether the quantity is a scalar (rather than a component of a vector)
        field.  Required when *model* is ``'custom'``.
    order : ArrayOrdering, optional
        Memory layout of the stored array — ``'F'`` (Fortran / column-major, the
        PSI default) or ``'C'`` (row-major).  Determines whether :attr:`shape`
        and the slicing axes are reversed relative to the on-disk layout.
        Required when *model* is ``'custom'``.
    scales : Sequence, optional
        Names of the coordinate axes, in physical order (e.g.
        ``('r', 't', 'p')``).  The argument does three things:

        - **Names** the axes — the strings become the fields of the
          ``reader.scales`` named tuple and the ``name`` of each coordinate
          scale reader.
        - **Orders** them — the names are zipped positionally with the file's
          stored dimension arrays, so the *i*-th name labels the *i*-th
          dimension; the ordering must therefore match the physical axis order.
        - **Fixes the dimensionality** — ``len(scales)`` sets the expected
          dataset rank; :meth:`~_HdfData.validate_metadata` emits a
          :exc:`MetaDataWarning` if it disagrees with ``ndim`` (or with the
          length of *mesh* / :attr:`shape`), or if any named axis does not match
          the size of its corresponding data dimension.

        Required when *model* is ``'custom'``.
    desc : str, optional
        Override the human-readable description.  Optional; defaults to ``''``.
    cache : CacheType, optional
        Cache mode.  ``'lazy'`` (default) caches the array on the first full
        read, ``'eager'`` loads it immediately, and ``None`` disables caching.

    Returns
    -------
    out : H5Data | H4Data
        Open reader implementing the full :class:`_HdfData` API.  Concrete type
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
    _HdfData : Base data reader interface.
    _HdfArray : Base data array interface.

    Examples
    --------
    Read a MAS radial field — full array with coordinate scales, then convert:

    >>> from psi_io.mhd_io import PsiData                  # doctest: +SKIP
    >>> reader = PsiData('br001001.h5')  # doctest: +SKIP
    >>> data, r, t, p = reader.read()                      # code units (MAS_b)  # doctest: +SKIP
    >>> data, r, t, p = reader.read(unit='Gauss')          # convert to Gauss  # doctest: +SKIP

    Use as a context manager:

    >>> with PsiData('vr001001.h5') as reader:              # doctest: +SKIP
    ...     data, r, t, p = reader.read(unit='km/s')

    Inspect metadata without loading data:

    >>> reader = PsiData('rho001001.h5')                    # doctest: +SKIP
    >>> reader.name          # 'rho'  # doctest: +SKIP
    >>> reader.unit          # MAS_n  # doctest: +SKIP
    >>> reader.mesh          # Mesh(HALF, HALF, HALF)  # doctest: +SKIP
    >>> reader.data_cached   # False  # doctest: +SKIP
    """
    return _dispatch_by_ext(ifile, H4Data, H5Data, *args, **kwargs)
