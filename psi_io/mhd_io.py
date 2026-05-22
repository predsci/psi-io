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

from abc import abstractmethod, ABC
from collections import namedtuple
from collections.abc import Sequence
from itertools import repeat
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Literal, ClassVar
import numpy as np
import h5py as h5
import astropy.units as u
from astropy.units.typing import UnitLike, QuantityLike

try:
    import pyhdf.SD as h4
except ImportError:
    h4 = None

from psi_io._mesh import (Mesh,
                          MeshCodeType,
                          _normalize_mesh_code,
                          _remesh_array,
                          _parse_remesh, ArrayOrdering,
                          )
from psi_io._models import (Props,
                            PsiScales,
                            ModelType,
                            extract_quantity_from_filepath,
                            extract_sequence_from_filepath, get_model_prop_caller, MODEL_TYPE)
from psi_io._units import decompose_mas_units
from psi_io.psi_io import (PathLike,
                           PSI_DATA_ID,
                           SDC_TYPE_CONVERSIONS,
                           _except_no_pyhdf,)


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


_DATA_SLOTS = ('_model', '_fileref', '_filepath', '_datalabel', '_quantity', '_sequence', '_unit', '_mesh', '_scales', '_values')
"""Slot names shared by all concrete :class:`_HdfData` subclasses.

Stored as a module-level tuple so it can be referenced in ``__slots__`` declarations
of both the abstract base and the concrete classes without repetition.
"""

_SCALE_SLOTS = ('_model', '_dataref', '_datalabel', '_quantity', '_unit', '_mesh', '_values')
"""Slot names shared by all concrete :class:`_HdfScale` subclasses.

Stored as a module-level tuple so it can be referenced in ``__slots__`` declarations
of both :class:`H5Scale` and :class:`H4Scale` without repetition.
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


METADATA_SCHEMA = dict.fromkeys(['quantity', 'sequence', 'unit', 'scalar', 'mesh'])
"""Template dict defining the five recognised metadata fields for PSI HDF datasets.

Keys
----
``'quantity'``
    Canonical lower-case quantity identifier extracted from the filename or file
    attributes (e.g. ``'br'``, ``'vr'``).
``'sequence'``
    Integer time-step sequence number extracted from the filename or file attributes.
``'unit'``
    Code-to-physical unit for this quantity, as an :class:`~astropy.units.Unit`
    or a string parseable by it.
``'scalar'``
    ``True`` if the quantity is a scalar field; ``False`` for vector components.
``'mesh'``
    Staggered-grid mesh code (:data:`~psi_io._mesh.MeshCodeType`) describing the
    Yee-grid position of the quantity on each spatial axis.

Used by :meth:`_HdfData._parse_properties` to filter keyword overrides and HDF file
attributes against the set of known metadata keys.
"""


Scales = namedtuple("Scales", ['r', 't', 'p'])
"""Named tuple holding the three coordinate scale readers for a data object.

Fields
------
r : H5Scale or H4Scale
    Radial coordinate scale in solar radii.
t : H5Scale or H4Scale
    Co-latitude scale in radians.
p : H5Scale or H4Scale
    Longitude scale in radians.

Created by :meth:`_HdfData._set_scales` during instantiation and exposed as
:attr:`_HdfData.scales`.
"""


def _interpolate_dim(arr: QuantityLike,
                     axis: int,
                     value: QuantityLike,
                     scale: QuantityLike,
                     fill_value: Optional[QuantityLike]
                     ) -> QuantityLike:
    """Linearly interpolate *arr* along *axis* to *value*, collapsing that axis to size 1.

    Both *value* and *scale* must carry the same unit (or both be dimensionless).
    The result is a fill array when *value* is outside the 2-element *scale* window
    and *fill_value* is not ``None``; otherwise a linear interpolation is performed.
    Extrapolation occurs silently when *fill_value* is ``None``.
    """
    if arr.shape[axis] != 2 or len(scale) != 2:
        raise ValueError("Interpolation is only supported for 2-element arrays and scales.")
    if (value < scale[0] or value > scale[1]) and fill_value is not None:

        return np.full(arr.shape[:axis] + (1,) + arr.shape[axis + 1:], fill_value, dtype=arr.dtype)
    t = (value - scale[0]) / (scale[1] - scale[0])
    slc_lo = [slice(None)] * arr.ndim
    slc_hi = [slice(None)] * arr.ndim
    slc_lo[axis] = slice(None, -1)
    slc_hi[axis] = slice(1, None)
    return (1.0 - t) * arr[tuple(slc_lo)] +  t * arr[tuple(slc_hi)]


def _slice_array(data: QuantityLike,
                 values: Sequence[Optional[QuantityLike]],
                 scales: Sequence[Optional[QuantityLike]],
                 fill_value: Optional[QuantityLike],
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
    for i, (v, s) in enumerate(zip(values, scales, strict=True)):
        if v is not None:
            if s is None:
                raise ValueError("Cannot interpolate to a value when the corresponding scale is not provided.")
            data = _interpolate_dim(data, i, v, s, fill_value)
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
    return args


def _parse_islice_args(*args, shape: tuple[int, ...],):
    """Normalize index-space slice arguments to a tuple of :class:`slice` objects.

    Accepts a mix of ``None``, ``int``, ``slice``, ``(start, stop[, step])`` tuples,
    and ``Ellipsis``, and yields one slice per spatial axis.

    Parameters
    ----------
    *args : None, int, slice, tuple, or Ellipsis
        Index arguments in physical ``(r, θ, φ)`` user order.  Fewer arguments than
        dimensions are padded with ``None`` (full-axis slices).
    shape : tuple[int, ...]
        Physical ``(r, θ, φ)`` shape (i.e. ``reversed(self.shape)`` from storage).

    Yields
    ------
    s : slice
        Normalized slice for each axis.

    Raises
    ------
    ValueError
        If a slice argument produces an empty dimension (``stop <= start``).
    TypeError
        If an argument cannot be converted to a slice (via :func:`_cast_to_slice`).
    """
    sargs = _expand_args(*args, ndim=len(shape))

    for arg, dim_size in zip(sargs, shape, strict=True):
        slice_ = _cast_to_slice(arg)
        start, stop, step = slice_.indices(dim_size)
        if stop <= start:
            raise ValueError(f"Slice argument {arg!r} yields an empty dimension.")
        yield slice_


def _parse_vslice_args(*args,
                       scales: Scales,
                       bounds_error: bool):
    """Convert value-space dimension arguments to ``(value, index_slice)`` pairs.

    For each axis, a bare scalar or :class:`~u.Quantity` triggers a
    value-space lookup: the coordinate scale is searched to find the two bracketing
    indices, and a 2-element slice is returned for later linear interpolation via
    :func:`_interpolate_dim`.  All other argument types (``None``, ``int``,
    ``slice``, tuple) are treated as index-space and passed through to
    :func:`_cast_to_slice` unchanged, with ``None`` returned as the value.

    Parameters
    ----------
    *args : scalar, u.Quantity, int, slice, tuple, None, or Ellipsis
        One argument per spatial axis in physical ``(r, θ, φ)`` order.  Fewer
        arguments than dimensions are padded with ``None`` via :func:`_expand_args`.
    scales : Scales
        Coordinate scale readers in physical ``(r, θ, φ)`` order.  Used to look
        up the native unit and perform bounds checking.
    bounds_error : bool
        If ``True``, raise :class:`ValueError` when a value lies outside the
        range of its coordinate scale.

    Yields
    ------
    value : u.Quantity or None
        The physical target value (in the scale's native unit) for value-space
        axes; ``None`` for index-space axes.
    s : slice
        Corresponding index-space slice.  For value-space axes this is the
        2-element bracketing window ``slice(i-1, i+1)``; for index-space axes
        it is the result of :func:`_cast_to_slice`.

    Raises
    ------
    ValueError
        If *bounds_error* is ``True`` and a value lies outside its scale range.
    """
    sargs = _expand_args(*args, ndim=len(scales))
    for arg, scale in zip(sargs, scales, strict=True):
        value = None
        if np.isscalar(arg):
            arg = arg * scale.unit
        if isinstance(arg, u.Quantity):
            value = arg.to(scale.unit)
            raw_value = value.value
            if bounds_error and (raw_value < scale[0] or raw_value > scale[-1]):
                raise ValueError(f"Value {arg} is out of bounds for scale '{scale.quantity}' "
                                 f"with range {scale.data[0]:.3f} to {scale.data[-1]:.3f} {scale.unit}.")
            index = int(np.clip(np.searchsorted(scale[...], raw_value), 1, scale.size - 1))
            arg = (index - 1, index + 1)
        slice_ = _cast_to_slice(arg)
        start, stop, step = slice_.indices(scale.size)
        if stop <= start:
            raise ValueError(f"Slice argument {arg!r} yields an empty dimension.")
        yield value, slice_


def _apply_units(data: u.Quantity, unit: Optional[str | UnitLike]) -> u.Quantity:
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
        return slice(input, input + 1)
    elif isinstance(input, slice):
        return input
    elif isinstance(input, (list, tuple)):
        return slice(*input)
    else:
        raise TypeError(f"Invalid slice argument: {input!r}. Expected int, slice, or sequence.")


# =============================================================================
# Abstract interface
# =============================================================================

class _HdfInterface(ABC):
    """Abstract base class defining the full public interface for PSI HDF data objects.

    All concrete readers (MAS, POT3D, coordinate scales) and their format mixins
    satisfy this interface.  Consumers should program against this class rather than
    against concrete subclasses.

    Subclasses must implement all abstract properties and the :meth:`read` and
    :meth:`_read` methods.  The ``read`` method in this base class provides shared
    unit-conversion and remeshing logic that concrete implementations invoke via
    ``super().read(*args, **kwargs)``.

    Notes
    -----
    The ``__slots__ = ()`` declaration in this class is intentional: it ensures that
    concrete subclasses using ``__slots__`` do not incur a ``__dict__`` overhead
    from the abstract base.
    """

    __slots__ = ()

    _HDFN: ClassVar[HdfVersionType]                        # provided by format mixin

    def __getitem__(self, args):
        """Index the underlying HDF dataset with physical ``(r, θ, φ)`` axis ordering.

        PSI HDF files are written in Fortran column-major order so that the radial
        index ``r`` varies fastest.  When read into numpy (row-major), this results
        in storage shape ``(Nφ, Nθ, Nr)``.  This method re-reverses the user-supplied
        index tuple so that callers can use natural physical ordering ``(r, θ, φ)``:

        .. code-block:: python

            obj[r_slice, t_slice, p_slice]   # user order
            # internally becomes:
            obj.data[p_slice, t_slice, r_slice]  # numpy storage order

        Parameters
        ----------
        args : int, slice, or tuple
            Index argument(s) in physical ``(r, θ, φ)`` order.  A single non-tuple
            argument is treated as indexing along the first physical axis (``r``).

        Returns
        -------
        out : numpy.ndarray
            Slice of the dataset in numpy storage order ``(Nφ, Nθ, Nr)``.
        """
        if not isinstance(args, tuple):
            args = (args,)

        if self._values is not None:
            return self._values[args[::-1]]
        else:
            odata = self.data[args[::-1]]
            if odata.shape == self.shape:
                self._values = odata
            return odata

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying numpy array in storage order ``(Nφ, Nθ, Nr)``."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """NumPy dtype of the stored array."""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Total number of elements in the array."""
        ...

    @property
    @abstractmethod
    def nbytes(self) -> int:
        """Total memory occupied by the array in bytes."""
        ...

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of array dimensions (always 3 for data objects, 1 for scales)."""
        ...

    @property
    @abstractmethod
    def attrs(self) -> dict:
        """HDF attributes attached to this dataset as a plain Python dict."""
        ...

    @property
    @abstractmethod
    def unit(self) -> u.Unit:
        """Astropy unit that converts one code-unit value to physical unit."""
        ...

    @property
    @abstractmethod
    def mesh(self) -> tuple[Mesh, ...]:
        """Normalized mesh-stagger tuple, one :class:`~psi_io._mesh.Mesh` per axis."""
        ...

    @property
    @abstractmethod
    def quantity(self) -> str:
        """Canonical lower-case quantity identifier (e.g. ``'br'``)."""
        ...

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """Raw array view into the HDF dataset (not yet loaded into RAM).

        For HDF5 objects this is an h5py :class:`~h5py.Dataset`; for HDF4 objects
        it is a pyhdf ``SDS`` object.  Actual data transfer occurs only when the
        returned object is indexed or converted to a numpy array.
        """
        ...

    @property
    def props(self) -> Props:
        """The :class:`~psi_io._models.Props` descriptor for this quantity.

        Returns the full property bundle (name, description, unit, mesh code) from
        the appropriate mapping for this reader's model type and quantity.

        Returns
        -------
        out : Props
            Immutable property descriptor for :attr:`quantity`.
        """
        prop_getter = get_model_prop_caller(self._model)
        return prop_getter(self._quantity)

    @property
    def description(self) -> str:
        """Human-readable description of the physical quantity.

        Looked up from the appropriate property mapping via :data:`_PROP_GETTER_MAPPING`
        using :attr:`_MODEL` and the stored :attr:`quantity` name.

        Returns
        -------
        out : str
            Description string (e.g. ``'Magnetic Field (Radial Component)'``).
        """
        return self.props.desc

    @property
    def is_scalar(self) -> bool:
        """``True`` if the quantity is a scalar field; ``False`` for vector components."""
        return self.props.scalar

    @property
    def is_cached(self) -> bool:
        """Flag indicating whether the data have been loaded into memory."""
        return self._values is not None

    def load(self):
        """Read the full dataset into memory and cache it in ``_values``."""
        self._values = self.data[...]

    @abstractmethod
    def read(self,
             *args,
             unit: Optional[str | UnitLike] = None,
             mesh: Optional[MeshCodeType] = None,
             ) -> tuple[u.Quantity, tuple[slice, ...], tuple[bool, ...]]:
        """Read a slice of data, applying optional unit conversion and remeshing.

        This base implementation handles unit conversion and remesh flag computation.
        Concrete subclasses call ``super().read(...)`` and then attach coordinate
        scales or perform additional post-processing.

        Parameters
        ----------
        *args : int, slice, tuple, or None
            Index arguments in physical ``(r, θ, φ)`` axis order.  Each positional
            argument selects elements along one spatial axis in the order
            ``(r, θ, φ)``.  Accepted forms per axis:

            - ``None`` or omitted — full axis (equivalent to ``slice(None)``).
            - ``int`` — single index (output retains that axis as a length-1
              dimension).
            - ``slice`` — standard Python slice.
            - ``(start, stop)`` or ``(start, stop, step)`` — converted to a slice.
            - ``Ellipsis`` — expands to ``None`` for all remaining axes.

        unit : str or u.Unit, optional
            Requested output unit.  Special string aliases are accepted:

            - ``'native'`` / ``'code'`` / ``'model'`` — return raw code-unit
              values (an :class:`~u.Quantity` whose unit is the custom
              MAS unit, e.g. ``MAS_b``).
            - ``'real'`` / ``'phys'`` / ``'physical'`` — decompose to CGS base
              unit via :func:`~psi_io._units.decompose_mas_units`.
            - Any other string or :class:`~astropy.units.Unit` — passed directly
              to :meth:`~u.Quantity.to`.

            If ``None``, the data are returned in the native code units without
            conversion.

        mesh : MeshCodeType, optional
            Target mesh stagger.  If provided, half-mesh axes in the stored data
            that are on the main mesh in *mesh* are averaged to the main mesh before
            return (via :func:`~psi_io._mesh.remesh_array`).  Attempting to up-sample
            from main to half mesh raises :class:`ValueError`.  If ``None``, no
            remeshing is performed.

        Returns
        -------
        odata : u.Quantity
            Data array with requested unit and remeshing applied.
        sargs : tuple[slice, ...]
            Slice tuple in physical ``(r, θ, φ)`` order, suitable for applying to
            coordinate scale arrays.
        remesh : tuple[bool, ...]
            Boolean flags in physical ``(r, θ, φ)`` order indicating which axes were
            remeshed from half to main mesh.
        """

        sargs = _parse_islice_args(*args, shape=self.shape[::-1])
        if mesh is None:
            remesh = repeat(False, self.ndim)
        else:
            omesh_norm = _normalize_mesh_code(mesh, self.ndim)
            remesh = _parse_remesh(self.mesh, omesh_norm, 'C')

        sargs = tuple(sargs)
        remesh = tuple(remesh)

        odata = _apply_units(self._read(*sargs, remesh=remesh), unit)
        return odata, sargs, remesh

    @abstractmethod
    def _read(self,
              *sargs,
              remesh) -> u.Quantity:
        """Internal read using pre-validated slice args and remesh flags.

        Applies *sargs* directly to the dataset via :meth:`__getitem__` (which
        performs the axis reversal) and then remeshes the result.  Called by
        :meth:`_HdfData.read` when reading coordinate scales to avoid re-parsing
        slice arguments.

        Parameters
        ----------
        *sargs : slice
            Pre-validated slices in physical ``(r, θ, φ)`` order.
        remesh : tuple[bool, ...]
            Remesh flags in physical ``(r, θ, φ)`` order.

        Returns
        -------
        out : u.Quantity
            Sliced and optionally remeshed data in code units.
        """

        return _remesh_array(self[sargs], remesh=remesh, order='F') * self.unit


# =============================================================================
# Scale classes
# =============================================================================

class _HdfScale(_HdfInterface, ABC):
    """Abstract base for HDF coordinate scale variables (r, t, p).

    A scale object wraps a one-dimensional coordinate array stored alongside the
    main data in an HDF file.  It exposes the same :class:`_HdfInterface` API as
    data objects so that coordinate arrays can be sliced and unit-converted with the
    same :meth:`read` call.

    Subclasses supply the format-specific :attr:`data` property and the array
    introspection properties (``shape``, ``dtype``, etc.).
    """

    __slots__ = ()

    def __init__(self,
                 parent,
                 dim_label: str,
                 data_label: str,
                 scale_model: str = 'scale'):
        """Initialize a scale from a parent data reader and dimension metadata.

        Parameters
        ----------
        parent : _HdfData
            The data reader to which this scale belongs.  The scale holds a reference
            to *parent* so it can share the open file handle.
        dim_label : str
            Coordinate axis label — one of ``'r'``, ``'t'``, ``'p'``.  Used to look
            up the physical unit via :func:`get_psi_scale_properties`.
        data_label : str
            Dataset or SDS name within the HDF file where the coordinate values are
            stored.

        Raises
        ------
        ValueError
            If the underlying coordinate dataset is not one-dimensional.
        """
        self._dataref = parent
        self._datalabel: str = data_label
        self._model: str = scale_model
        self._values = None

        self._set_properties(dim_label)

    def _set_properties(self, scale: str):
        """Look up and cache the unit for this coordinate axis."""
        prop_getter = get_model_prop_caller(self._model)
        qprops = prop_getter(scale)
        if self.ndim != qprops.ndim:
            raise ValueError(f"Expected {qprops.ndim}D coordinate variable for scale '{scale}', "
                             f"found {self.ndim}D dataset with shape {self.shape}.")
        self._quantity: PsiScales = qprops.name
        self._unit: u.Unit = qprops.unit
        self._mesh: Mesh = self._dataref.mesh['rtp'.index(self._quantity)],

    @property
    def unit(self) -> u.Unit:
        """Astropy unit for this coordinate axis (``PSI_rsun`` or ``PSI_angle``)."""
        return self._unit

    @property
    def quantity(self) -> PsiScales:
        """Coordinate axis identifier: ``'r'``, ``'t'``, or ``'p'``."""
        return self._quantity

    @property
    def mesh(self) -> tuple[Mesh, ...]:
        """Single-element mesh-stagger tuple for this coordinate axis."""
        return self._mesh

    def read(self,
             *args,
             **kwargs,
             ) -> u.Quantity:
        """Read the coordinate array, returning an astropy Quantity.

        Delegates to :meth:`_HdfInterface.read` and returns only the data
        (discarding the ``sargs`` and ``remesh`` bookkeeping values).

        Parameters
        ----------
        *args
            Slice arguments; see :meth:`_HdfInterface.read`.
        **kwargs
            Keyword arguments forwarded to :meth:`_HdfInterface.read`.

        Returns
        -------
        out : u.Quantity
            Coordinate values with the appropriate PSI unit attached.
        """
        odata, *_ = super().read(*args, **kwargs)
        return odata

    def _read(self,
              *sargs,
              remesh) -> u.Quantity:
        """Internal read using pre-validated slice args."""
        return super()._read(*sargs, remesh=remesh)


class H5Scale(_HdfScale):
    """HDF5 coordinate scale variable backed by an h5py dimension scale dataset."""

    __slots__ = _SCALE_SLOTS

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the coordinate array (always a length-1 tuple for 1-D scales)."""
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of the coordinate array."""
        return self.data.dtype

    @property
    def size(self) -> int:
        """Total number of coordinate points."""
        return self.data.size

    @property
    def nbytes(self) -> int:
        """Total memory occupied by the coordinate array in bytes."""
        return self.data.nbytes

    @property
    def ndim(self) -> int:
        """Number of dimensions (always 1 for coordinate scale arrays)."""
        return self.data.ndim

    @property
    def attrs(self) -> dict:
        """HDF5 attributes attached to this dimension scale dataset."""
        return dict(self.data.attrs)

    @property
    def data(self) -> np.ndarray:
        """h5py Dataset object for this coordinate dimension."""
        return self._dataref._fileref[self._datalabel]


class H4Scale(_HdfScale):
    """HDF4 coordinate scale variable backed by a pyhdf SDS dimension."""

    __slots__ = _SCALE_SLOTS

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the coordinate array (always a length-1 tuple for 1-D scales)."""
        return self.data.info()[2],

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of the coordinate array."""
        return SDC_TYPE_CONVERSIONS[self.data.info()[3]]

    @property
    def size(self) -> int:
        """Total number of coordinate points."""
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        """Total memory occupied by the coordinate array in bytes."""
        return self.size * self.dtype.itemsize

    @property
    def ndim(self) -> int:
        """Number of dimensions (always 1 for coordinate scale arrays)."""
        return self.data.info()[1]

    @property
    def attrs(self) -> dict:
        """HDF4 attributes attached to this SDS dimension."""
        return self.data.attributes()

    @property
    def data(self) -> np.ndarray:
        """pyhdf SDS object for this coordinate dimension."""
        return self._dataref._fileref.select(self._datalabel)


# =============================================================================
# Format mixins (HDF5 and HDF4 file I/O + raw array access)
# =============================================================================

class _H5DataMixin:
    """Mixin providing HDF5 file I/O and raw array access via h5py.

    Concrete data classes inherit from both this mixin and :class:`_HdfData`.
    The mixin supplies the ``_HDFN = 'h5'`` class variable, the
    :meth:`read_file` class method, and properties that delegate to the open
    :class:`h5py.File` handle.
    """

    __slots__ = ()
    _HDFN = 5

    @classmethod
    def read_file(cls, ifile: PathLike):
        """Open an HDF5 file for reading and return the :class:`h5py.File` handle."""
        return h5.File(ifile, 'r')

    def open(self):
        """Re-open the HDF5 file if it was previously closed.  Returns ``self``."""
        if not self._fileref:
            self._fileref = self.read_file(self._filepath)
        return self

    def close(self):
        """Close the HDF5 file handle.  Returns ``self``."""
        if self._fileref is not None:
            self._fileref.close()
            self._fileref = None
        return self

    def delete(self):
        """Close the file handle during garbage collection (called by ``__del__``)."""
        fileref = getattr(self, '_fileref', None)
        if fileref is not None:
            fileref.close()
            self._fileref = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape in storage order ``(Nφ, Nθ, Nr)``."""
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of the stored array."""
        return self.data.dtype

    @property
    def size(self) -> int:
        """Total number of elements in the array."""
        return self.data.size

    @property
    def nbytes(self) -> int:
        """Total memory occupied by the array in bytes."""
        return self.data.nbytes

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return self.data.ndim

    @property
    def attrs(self) -> dict:
        """HDF5 attributes attached to this dataset as a plain Python dict."""
        return dict(self.data.attrs)

    @property
    def data(self) -> np.ndarray:
        """h5py Dataset object providing lazy access to the array."""
        return self._fileref[self._datalabel]

    def _set_scales(self):
        """Construct :class:`H5Scale` objects from h5py dimension scales."""
        self._scales = Scales(*(H5Scale(self, scale, label.label)
                                     for scale, label in zip('rtp', self.data.dims, strict=True)))


class _H4DataMixin:
    """Mixin providing HDF4 file I/O and raw array access via pyhdf.

    Analogous to :class:`_H5DataMixin` but for HDF4 files.  Raises an informative
    error at import time if pyhdf is not installed, via :func:`_except_no_pyhdf`.
    """

    __slots__ = ()
    _HDFN = 4

    @classmethod
    def read_file(cls, ifile: PathLike):
        """Open an HDF4 file for reading and return the pyhdf ``SD`` object."""
        _except_no_pyhdf()
        return h4.SD(str(ifile), h4.SDC.READ)

    def open(self):
        """Re-open the HDF4 file if it was previously closed.  Returns ``self``."""
        if not self._fileref:
            self._fileref = self.read_file(self._filepath)
        return self

    def close(self):
        """Close the HDF4 file handle via ``end()``."""
        if self._fileref is not None:
            self._fileref.end()
            self._fileref = None

    def delete(self):
        """Close the HDF4 file handle during garbage collection."""
        fileref = getattr(self, '_fileref', None)
        if fileref is not None:
            fileref.end()
            self._fileref = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape in storage order ``(Nφ, Nθ, Nr)``."""
        return tuple(self.data.info()[2])

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of the stored array."""
        return SDC_TYPE_CONVERSIONS[self.data.info()[3]]

    @property
    def size(self) -> int:
        """Total number of elements in the array."""
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        """Total memory occupied by the array in bytes."""
        return self.size * self.dtype.itemsize

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return self.data.info()[1]

    @property
    def attrs(self) -> dict:
        """HDF4 attributes attached to this SDS dataset as a plain Python dict."""
        return self.data.attributes()

    @property
    def data(self) -> np.ndarray:
        """pyhdf SDS object providing lazy access to the array."""
        return self._fileref.select(self._datalabel)

    def _set_scales(self):
        """Construct :class:`H4Scale` objects from pyhdf SDS dimensions.

        HDF4 dimension order is reversed relative to HDF5 (Fortran vs. C order),
        so the dimension list is reversed before zipping with ``'rtp'``.
        """
        sds = self.data
        dims = list(reversed(list(sds.dimensions(full=1).items())))
        self._scales = Scales(*tuple(H4Scale(self, scale, k_)
                                     for scale, (k_, v_) in zip('rtp', dims, strict=True)))


# =============================================================================
# Abstract data base
# =============================================================================

class _HdfData(_HdfInterface, ABC):
    """Abstract base for a single PSI HDF dataset (data fields, not coordinate scales).

    Handles file opening, metadata resolution, and scale construction.  Concrete
    subclasses are produced by combining this class with a format mixin
    (:class:`_H5DataMixin` or :class:`_H4DataMixin`) to form :class:`H5Data` and
    :class:`H4Data`.  Use :func:`PsiData` rather than instantiating these directly.
    """

    __slots__ = ()

    def __init__(self,
                 ifile: PathLike,
                 model: ModelType, /,
                 dataset_id: Optional[str] = None,
                 **kwargs):
        """Open an HDF file and parse metadata for one PSI output quantity.

        Parameters
        ----------
        ifile : PathLike
            Path to the HDF4 or HDF5 file.  Must exist and have the extension
            expected by the format mixin (``'.h5'`` or ``'.hdf'``).
        dataset_id : str, optional
            Name of the dataset (SDS in HDF4, group key in HDF5) to open.  Defaults
            to the PSI standard dataset identifier for this format
            (:data:`~psi_io.psi_io.PSI_DATA_ID`).
        **kwargs
            Optional metadata overrides.  Accepted keys (from
            :data:`METADATA_SCHEMA`): ``'quantity'``, ``'sequence'``, ``'unit'``,
            ``'scalar'``, ``'mesh'``.  Caller-supplied values take precedence over
            both file attributes and filename inference.

        Raises
        ------
        FileNotFoundError
            If *ifile* does not exist on disk.
        ValueError
            If *ifile* has the wrong extension for this format mixin, if the
            dataset is not three-dimensional, or if any required metadata field
            cannot be resolved.
        """
        ifile = Path(ifile)
        hdfv = f'h{self._HDFN}'
        if not ifile.is_file():
            raise FileNotFoundError(f"File '{ifile}' does not exist.")
        if ifile.suffix != _HDF_EXT_MAPPING[hdfv]:
            raise ValueError(f"File '{ifile}' does not have the correct extension for "
                             f"{self._HDFN} files (expected '{_HDF_EXT_MAPPING[hdfv]}' extension).")
        if model not in MODEL_TYPE:
            raise ValueError(f"Invalid model type {model!r}. Expected one of: {', '.join(MODEL_TYPE)}.")

        self._filepath: Path = ifile
        self._datalabel: str = dataset_id or PSI_DATA_ID[hdfv]
        self._fileref = self.read_file(ifile)
        self._model: str = model
        self._values = None

        self._set_properties(**self._parse_properties(**kwargs))
        self._set_scales()

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
    def _set_scales(self):
        """Construct and cache the :class:`Scales` named tuple of coordinate readers."""
        ...

    def _parse_properties(self, **kwargs):
        """Resolve metadata fields by merging caller kwargs, file attrs, and filename.

        Merges three sources (highest to lowest priority):

        1. Keyword arguments in *kwargs* that match :data:`METADATA_SCHEMA` keys.
        2. HDF file-level attributes that match :data:`METADATA_SCHEMA` keys.
        3. Quantity name and sequence number inferred from the filename stem.
        4. The canonical :class:`~psi_io._models.Props` defaults for the resolved
           quantity (unit and mesh).

        Parameters
        ----------
        **kwargs
            Caller-supplied metadata overrides.

        Returns
        -------
        out : dict
            Fully populated metadata dict with keys
            ``{'quantity', 'sequence', 'unit', 'mesh'}``; the ``'scalar'`` key from
            :data:`METADATA_SCHEMA` is resolved internally but not forwarded to
            :meth:`_set_properties`.

        Raises
        ------
        ValueError
            If any metadata field is still ``None`` after merging all sources.
        """
        prop_getter = get_model_prop_caller(self._model)

        input_attrs = {k: v for k, v in kwargs.items() if k in METADATA_SCHEMA}
        file_attrs = {k: v for k, v in self.attrs.items() if k in METADATA_SCHEMA}

        quantity = input_attrs.get('quantity',
                                   file_attrs.get('quantity',
                                                  extract_quantity_from_filepath(self._filepath, '')))
        sequence = input_attrs.get('sequence',
                                   file_attrs.get('sequence',
                                                  extract_sequence_from_filepath(self._filepath, 0)))

        native_props = prop_getter(quantity)
        if self.ndim != native_props.ndim:
            raise ValueError(f"Expected {native_props.ndim}D dataset for quantity '{quantity}', "
                             f"found {self.ndim}D dataset with shape {self.shape}.")

        native_attrs = dict(quantity=native_props.name,
                            sequence=sequence,
                            unit=native_props.unit,
                            mesh=native_props.mesh)

        attributes = native_attrs | file_attrs | input_attrs
        if any(v is None for v in attributes.values()):
            missing_meta = ', '.join(k for k, v in attributes.items() if v is None)
            raise ValueError(f"Malformed metadata: {missing_meta} is missing. "
                             f"Provide these as keyword arguments or ensure they "
                             f"are present in the file attributes.")

        return attributes

    def _set_properties(self,
                        quantity: str,
                        sequence: int,
                        unit: str,
                        mesh: MeshCodeType):
        """Store validated and type-coerced metadata on the instance.

        Parameters
        ----------
        quantity : str
            Quantity name; stored as a string.
        sequence : int
            Sequence number; coerced to int.
        unit : str or u.Unit
            Unit; converted to :class:`~astropy.units.Unit` via ``u.Unit(str(unit))``.
        mesh : MeshCodeType
            Mesh stagger; normalized to ``tuple[Mesh, ...]`` via
            :func:`~psi_io._mesh._normalize_mesh_code`.

        Raises
        ------
        ValueError
            If type coercion of any field fails.
        """
        try:
            self._quantity: str = str(quantity)
            self._sequence: int = int(sequence)
            self._unit: u.Unit = u.Unit(str(unit))
            self._mesh: tuple[Mesh, ...] = _normalize_mesh_code(mesh, self.ndim)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Metadata type coercion failed: {e}") from e

    @property
    def unit(self) -> u.Unit:
        """Astropy unit for converting code-unit values to physical unit."""
        return self._unit

    @property
    def mesh(self) -> tuple[Mesh, ...]:
        """Normalized mesh-stagger tuple, one :class:`~psi_io._mesh.Mesh` per axis."""
        return self._mesh

    @property
    def quantity(self) -> str:
        """Canonical lower-case quantity identifier (e.g. ``'br'``)."""
        return self._quantity

    @property
    def sequence(self) -> int:
        """Integer time-step sequence number from the filename or file attributes."""
        return self._sequence

    @property
    def scales(self) -> Scales:
        """Named tuple of coordinate scale readers ``(r, t, p)``.

        Each element is an :class:`H5Scale` or :class:`H4Scale` that exposes the
        same :meth:`read` interface as the main data object, so coordinate arrays
        can be sliced in sync with the data.
        """
        return self._scales

    def read(self,
             *args,
             scales: bool = True,
             **kwargs
             ) -> u.Quantity | tuple[u.Quantity, ...]:
        """Read a slice of data and optionally the corresponding coordinate arrays.

        Delegates slice parsing, remeshing, and unit conversion to
        :meth:`_HdfInterface.read`, then optionally reads the matching slice of
        each coordinate scale.

        Parameters
        ----------
        *args : int, slice, tuple, None, or Ellipsis
            Index arguments in physical ``(r, θ, φ)`` order.  See
            :meth:`_HdfInterface.read` for the full description.
        scales : bool, optional
            If ``True`` (default), also return the corresponding coordinate slice
            for each spatial axis.  If ``False``, return only the data array.
        **kwargs
            Forwarded to :meth:`_HdfInterface.read`; see that method for
            ``units`` and ``mesh`` keyword arguments.

        Returns
        -------
        odata : u.Quantity
            Data array in the requested unit.
        r_scale : u.Quantity
            Radial coordinate values in solar radii (only if ``scales=True``).
        t_scale : u.Quantity
            Co-latitude values in radians (only if ``scales=True``).
        p_scale : u.Quantity
            Longitude values in radians (only if ``scales=True``).

        Examples
        --------
        Read the full array and coordinate grids:

        >>> data, r, t, p = reader.read()                   # doctest: +SKIP

        Read a radial sub-range and convert to physical CGS units:

        >>> data, r, t, p = reader.read(slice(10, 50), unit='physical')  # doctest: +SKIP

        Read data only (no coordinate arrays):

        >>> data = reader.read(scales=False)                # doctest: +SKIP
        """
        odata, sargs, remesh = super().read(*args, **kwargs)
        if not scales:
            return odata
        oscales = (scale._read(sarg, remesh=rmesh) for scale, sarg, rmesh in zip(self.scales, sargs, remesh))
        return odata, *oscales

    def _read(self,
              *sargs,
              remesh) -> u.Quantity:
        """Internal read using pre-validated slice args."""
        return super()._read(*sargs, remesh=remesh)


    def vslice(self,
               *args,
               scales: bool = True,
               unit: Optional[str | UnitLike] = None,
               mesh: Optional[MeshCodeType] = None,
               bounds_error: bool = True,
               fill_value: Optional[QuantityLike] = None,
               ) -> u.Quantity | tuple[u.Quantity, ...]:
        """Read a slice of data by physical coordinate value with optional interpolation.

        Each positional argument may be a physical coordinate value
        (:class:`~u.Quantity` or bare scalar), in which case the
        dataset is reduced to a 2-element window and linearly interpolated to that
        value.  Index-space arguments (``slice``, ``int``, ``None``, ``Ellipsis``)
        are also accepted and passed through without interpolation, making this
        method a superset of :meth:`read`.

        Parameters
        ----------
        *args : u.Quantity, scalar, int, slice, tuple, None, or Ellipsis
            One argument per spatial axis in physical ``(r, θ, φ)`` order.
            u.Quantity or bare scalar → value-space interpolation.
            All other types → index-space slice (see :meth:`read`).
        scales : bool, optional
            If ``True`` (default), also return the corresponding coordinate value
            for each axis.  Value-interpolated axes return the interpolation target
            as a length-1 :class:`~astropy.units.Quantity`; index-space axes return the full slice.
        unit : str or u.Unit, optional
            Output unit; see :meth:`read` for accepted aliases and formats.
        mesh : MeshCodeType, optional
            Target mesh stagger.  Remeshing is skipped for axes that are being
            value-interpolated (interpolation already collapses the half-mesh
            window to a single value).
        bounds_error : bool, optional
            If ``True`` (default), raise :class:`ValueError` when a value argument
            lies outside its coordinate scale range.
        fill_value : QuantityLike or None, optional
            Value substituted when a coordinate value falls outside the 2-element
            interpolation window and *bounds_error* is ``False``.  ``None``
            silently extrapolates.

        Returns
        -------
        odata : u.Quantity
            Sliced and interpolated data in the requested unit.
        r_scale, t_scale, p_scale : u.Quantity
            Coordinate values for each axis (only if ``scales=True``).
            Value-interpolated axes return the interpolation target as a
            length-1 array; index-space axes return the full coordinate slice.

        Raises
        ------
        ValueError
            If *bounds_error* is ``True`` and any value argument is out of range,
            or if a value axis has no corresponding coordinate scale.
        """
        vslice_args = tuple(_parse_vslice_args(*args, scales=self.scales, bounds_error=bounds_error))
        slice_values, sargs = zip(*vslice_args)

        if mesh is None:
            remesh = repeat(False, self.ndim)
        else:
            omesh_norm = _normalize_mesh_code(mesh, self.ndim)
            remesh = _parse_remesh(self.mesh, omesh_norm, 'C')
        remesh = tuple(remesh)
        remesh_xand_svalue = tuple(rm and sv is None for rm, sv in zip(remesh, slice_values))

        pre_slice_data = _apply_units(self._read(*sargs, remesh=remesh_xand_svalue), unit)

        pre_slice_scales = tuple(scale[sarg] * scale.unit if sv is not None else None
                             for scale, sarg, sv in zip(self.scales, sargs, slice_values, strict=True))

        sliced_data = _slice_array(pre_slice_data, slice_values, pre_slice_scales, fill_value)
        if not scales:
            return sliced_data
        sliced_scales = tuple(scale._read(sarg, remesh=rmesh) if sv is None else np.atleast_1d(sv)
                                 for scale, sarg, sv, rmesh in zip(self.scales, sargs, slice_values, remesh, strict=True))
        return sliced_data, *sliced_scales


# =============================================================================
# Concrete data classes
# =============================================================================

class H5Data(_H5DataMixin, _HdfData):
    """HDF5-backed MAS model data reader.

    Combines :class:`_H5DataMixin` (h5py file I/O) with :class:`_HdfData`
    (MAS metadata and :meth:`read` logic).  Use :func:`PsiData` to instantiate
    rather than calling this class directly.
    """

    __slots__ = _DATA_SLOTS


class H4Data(_H4DataMixin, _HdfData):
    """HDF4-backed MAS model data reader.

    Combines :class:`_H4DataMixin` (pyhdf file I/O) with :class:`_HdfData`
    (MAS metadata and :meth:`read` logic).  Requires the optional ``pyhdf``
    dependency.  Use :func:`PsiData` to instantiate rather than calling this class
    directly.
    """

    __slots__ = _DATA_SLOTS

# =============================================================================
# Private helpers
# =============================================================================

_DATA_CLASS_MAP = MappingProxyType({
    '.h5':  H5Data,
    '.hdf': H4Data,
})
"""Read-only mapping from HDF file extension to the concrete data reader class.

Used by :func:`PsiData` to select between :class:`H5Data` (h5py) and
:class:`H4Data` (pyhdf) based on the file's suffix.
"""


def PsiData(ifile: PathLike, /,
            model: ModelType = 'mas',
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
    ifile = Path(ifile)
    cls = _DATA_CLASS_MAP.get(ifile.suffix)
    if cls is None:
        raise ValueError(
            f"Unsupported HDF extension '{ifile.suffix}'"
        ) from None
    return cls(ifile, model.lower(), **kwargs)