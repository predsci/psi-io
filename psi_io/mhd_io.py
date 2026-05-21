r"""Lazy HDF readers for PSI MAS and POT3D magnetohydrodynamic model output.

This module provides a unified, unit-aware interface for reading three-dimensional
MHD model output from Predictive Science Inc.'s MAS and POT3D solvers.  Both HDF4
(``.hdf``) and HDF5 (``.h5``) files are supported through a common API.
"""

from __future__ import annotations

__all__ = ['PsiData',]

import re
from abc import abstractmethod, ABC
from collections import namedtuple
from itertools import repeat
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Literal, ClassVar
import numpy as np
import h5py as h5
import astropy.units as u
from astropy.units.typing import UnitLike

try:
    import pyhdf.SD as h4
except ImportError:
    h4 = None

from psi_io._mesh import (Mesh,
                          MeshCodeType,
                          _normalize_mesh_code,
                          _remesh_array,
                          _parse_remesh, remesh_array,
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
    if Ellipsis in args:
        n_missing = len(shape) - (len(args) - 1)
        idx = args.index(Ellipsis)
        args = args[:idx] + (None,) * n_missing + args[idx + 1:]
    if len(args) < len(shape):
        args += (None,) * (len(shape) - len(args))

    for arg, dim_size in zip(args, shape, strict=True):
        slice_ = _cast_to_slice(arg)
        start, stop, step = slice_.indices(dim_size)
        if stop <= start:
            raise ValueError(f"Slice argument {arg!r} yields an empty dimension.")
        # if do_remesh and (stop - start) // step < 2:
        #     raise ValueError(f"Cannot remesh dimension with slice {slice_} "
        #                      f"because it does not include at least two indices.")
        yield slice_


def _parse_vslice_args(dim, scale):
    """Convert a value-space dimension specifier to an index-space slice.

    If *dim* is a bare float, it is treated as a value in the coordinate's native
    unit and first converted to an :class:`~astropy.units.Quantity`.  If *dim* is
    an :class:`~astropy.units.Quantity`, its value is located in *scale* via binary
    search and a two-element neighborhood slice is returned for interpolation.
    Non-quantity inputs are passed through to :func:`_cast_to_slice` unchanged.

    Parameters
    ----------
    dim : float, astropy.units.Quantity, int, slice, or tuple
        Dimension specifier.  Floats and Quantities trigger value-based lookup;
        all other types are forwarded to :func:`_cast_to_slice`.
    scale : H5Scale or H4Scale
        Coordinate scale providing the values to search and the native unit for
        unit conversion.

    Returns
    -------
    s : slice
        Index-space slice (a two-index neighborhood for value lookups; the original
        slice for index-space inputs).
    val : astropy.units.Quantity or None
        The matched physical value for value-based lookups, or ``None`` for
        index-based inputs.
    """
    val = None
    if isinstance(dim, float):
        dim = dim * scale.unit
    if isinstance(dim, u.Quantity):
        val = dim.to(scale.unit)
        i1 = int(np.clip(np.searchsorted(scale.data, val.value), 1, scale.size - 2))
        dim = (i1-1, i1+1)
    return _cast_to_slice(dim), val


def _cast_to_slice(input):
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
        """Astropy unit that converts one code-unit value to physical units."""
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
             units: Optional[str | UnitLike] = None,
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

        units : str or astropy.units.Unit, optional
            Requested output unit.  Special string aliases are accepted:

            - ``'native'`` / ``'code'`` / ``'model'`` — return raw code-unit
              values (an :class:`~astropy.units.Quantity` whose unit is the custom
              MAS unit, e.g. ``MAS_b``).
            - ``'real'`` / ``'phys'`` / ``'physical'`` — decompose to CGS base
              units via :func:`~psi_io._units.decompose_mas_units`.
            - Any other string or :class:`~astropy.units.Unit` — passed directly
              to :meth:`~astropy.units.Quantity.to`.

            If ``None``, the data are returned in the native code unit without
            conversion.

        mesh : MeshCodeType, optional
            Target mesh stagger.  If provided, half-mesh axes in the stored data
            that are on the main mesh in *mesh* are averaged to the main mesh before
            return (via :func:`~psi_io._mesh.remesh_array`).  Attempting to up-sample
            from main to half mesh raises :class:`ValueError`.  If ``None``, no
            remeshing is performed.

        Returns
        -------
        odata : astropy.units.Quantity
            Data array with requested units and remeshing applied.
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

        odata = self._read(*sargs, remesh=remesh)
        if units is not None:
            ounit = str(units).lower()
            if ounit in _CODE_UNIT_ALIASES:
                pass
            elif ounit in _REAL_UNIT_ALIASES:
                odata = decompose_mas_units(odata)
            else:
                odata = odata.to(units)

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
        out : astropy.units.Quantity
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
        out : astropy.units.Quantity
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
        unit : str or astropy.units.Unit
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
        """Astropy unit for converting code-unit values to physical units."""
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
        odata : astropy.units.Quantity
            Data array in the requested unit.
        r_scale : astropy.units.Quantity
            Radial coordinate values in solar radii (only if ``scales=True``).
        t_scale : astropy.units.Quantity
            Co-latitude values in radians (only if ``scales=True``).
        p_scale : astropy.units.Quantity
            Longitude values in radians (only if ``scales=True``).

        Examples
        --------
        Read the full array and coordinate grids:

        >>> data, r, t, p = reader.read()                   # doctest: +SKIP

        Read a radial sub-range and convert to physical CGS units:

        >>> data, r, t, p = reader.read(slice(10, 50), units='physical')  # doctest: +SKIP

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

    This is the recommended entry point for reading PSI model output.  The
    function inspects the file extension and the *model* argument to select the
    correct concrete reader class, then instantiates and returns it.

    .. rubric:: Returned object

    The returned object is a lazy reader that holds an open file handle.  The
    HDF dataset is *not* copied into memory at construction time — data transfer
    happens only when :meth:`read` is called (or when the underlying
    :attr:`data` handle is explicitly indexed).  The object exposes:

    - :meth:`read` — primary method for loading a slice of the dataset.
    - :attr:`scales` — ``Scales(r, t, p)`` named tuple of coordinate scale
      readers; each element supports the same :meth:`read` interface.
    - :attr:`quantity` — canonical lower-case quantity string (e.g. ``'br'``).
    - :attr:`sequence` — integer time-step sequence number.
    - :attr:`unit` — :class:`~astropy.units.Unit` for converting from code units
      to physical units.
    - :attr:`mesh` — tuple of :class:`~psi_io._mesh.Mesh` flags describing the
      Yee-grid stagger position of each axis.
    - :attr:`description` — human-readable description of the physical quantity.
    - :attr:`props` — full :class:`~psi_io._models.Props` descriptor.

    The object also supports the context-manager protocol; the file handle is
    closed on exit from the ``with`` block.

    .. rubric:: The read method

    .. code-block:: python

        odata[, r, t, p] = reader.read(*args, units=None, mesh=None, scales=True)

    **Positional arguments** — each positional argument selects elements along
    one spatial axis in physical ``(r, θ, φ)`` order.  Supported forms:

    - Omitted / ``None`` — full axis (``slice(None)``).
    - ``int`` — single index; the axis is retained as a length-1 dimension.
    - ``slice`` — standard Python slice.
    - ``(start, stop)`` or ``(start, stop, step)`` — converted to a slice.
    - ``Ellipsis`` — expands to ``None`` for all remaining axes.

    **Keyword arguments**:

    - ``units`` — output unit.  String aliases: ``'native'`` / ``'code'`` /
      ``'model'`` return values in the custom MAS code unit; ``'real'`` /
      ``'phys'`` / ``'physical'`` decompose to CGS base units.  Any other
      string is forwarded to :class:`~astropy.units.Unit` and must be
      parseable by it — this includes SI and CGS unit names (``'Gauss'``,
      ``'nT'``, ``'T'``), compound expressions (``'km/s'``,
      ``'erg/cm**2/s'``), and scale prefixes (``'mG'``, ``'μT'``); see
      :ref:`astropy:unit-format` for the full grammar.  An
      :class:`~astropy.units.Unit` instance may also be passed directly.
    - ``mesh`` — target mesh stagger (:data:`~psi_io._mesh.MeshCodeType`).
      Half-mesh axes that are on the main mesh in *mesh* are averaged to the
      main mesh via :func:`~psi_io._mesh.remesh_array` before return.
    - ``scales`` — if ``True`` (default), return the coordinate slice for each
      axis as additional :class:`~astropy.units.Quantity` values after the data.

    .. note::
        PSI HDF files are written in Fortran column-major order so that numpy
        reads them with shape ``(Nφ, Nθ, Nr)`` — the *last* axis is radial.
        The ``read`` positional arguments and coordinate scales are always
        returned in physical ``(r, θ, φ)`` order regardless of storage order.

    .. warning:: **POT3D unit convention**

        POT3D applies **no normalization** to its output.  The stored values are in
        whatever physical units the input photospheric boundary magnetogram was
        provided in — most commonly Gauss, but this is not encoded in the file.
        The default ``unit`` for POT3D quantities is therefore
        ``dimensionless_unscaled`` (scale factor 1), meaning that calling
        ``read(units='physical')`` will **not** perform a meaningful unit
        conversion unless the correct unit is supplied explicitly via the *unit*
        keyword argument at construction time:

        .. code-block:: python

            reader = PsiData('br001.h5', model='pot3d', unit='Gauss')
            data, r, t, p = reader.read()
            # data.unit == u.Gauss

    Parameters
    ----------
    ifile : PathLike
        Path to the HDF4 (``.hdf``) or HDF5 (``.h5``) file.
    model : {'mas', 'pot3d'}, optional
        PSI model type.  Defaults to ``'mas'``.
    dataset_id : str, optional
        Name of the dataset within the HDF file.  Defaults to the PSI standard
        identifier for the given format.  Only needed when the file uses a
        non-standard dataset name.
    quantity : str, optional
        Override the quantity name inferred from the filename or file attributes
        (e.g. ``'br'``).  Must be a key in the appropriate properties mapping.
    sequence : int, optional
        Override the time-step sequence number inferred from the filename or
        file attributes.
    unit : str or astropy.units.Unit, optional
        Override the code-to-physical unit derived from the quantity's
        :class:`~psi_io._models.Props` entry.  Accepts any string parseable
        by :class:`~astropy.units.Unit`, including named units (``'Gauss'``,
        ``'nT'``, ``'km/s'``), compound expressions (``'erg/cm**2/s'``),
        and scale-prefixed forms (``'mG'``, ``'μT'``); see
        :ref:`astropy:unit-format` for the complete unit grammar.  An
        :class:`~astropy.units.Unit` instance may also be passed directly.
    mesh : MeshCodeType, optional
        Override the mesh stagger inferred from the quantity's
        :class:`~psi_io._models.Props` entry.  Useful for files whose stagger
        differs from the PSI convention.

    Returns
    -------
    out : H5MasData or H4MasData or H5Pot3dData or H4Pot3dData
        An open reader object implementing the full :class:`_HdfInterface` API.
        The concrete type depends on *model* and the file extension.

    Raises
    ------
    ValueError
        If the combination of file extension and *model* is not supported, or if
        required metadata cannot be resolved from the file, its attributes, and
        the supplied keyword arguments.
    FileNotFoundError
        If *ifile* does not exist on disk.

    See Also
    --------
    astropy.units.Unit : Unit constructor — accepts strings, compound
        expressions, and :class:`~astropy.units.Unit` instances.
    astropy.units.Quantity.to : Unit conversion used internally when
        a ``units`` string is supplied to :meth:`read`.

    Examples
    --------
    Read a MAS radial magnetic field file — full array with coordinate scales:

    >>> from psi_io.mhd_io import PsiData                  # doctest: +SKIP
    >>> reader = PsiData('br001001.h5')
    >>> data, r, t, p = reader.read()
    >>> data.unit                                           # code unit (MAS_b)

    Convert to Gauss on the fly:

    >>> data, r, t, p = reader.read(units='Gauss')         # doctest: +SKIP

    Read only the inner radial shell (indices 0–9 in r) in CGS base units,
    without returning coordinate arrays:

    >>> data = reader.read(slice(0, 10), units='physical', scales=False)  # doctest: +SKIP

    Read a POT3D HDF4 file.  The boundary magnetogram was in Gauss, so the
    unit must be declared explicitly — it cannot be inferred from the file:

    >>> reader = PsiData('br001.hdf', model='pot3d', unit='Gauss')  # doctest: +SKIP
    >>> data, r, t, p = reader.read()

    Use as a context manager to guarantee the file handle is released:

    >>> with PsiData('vr001001.h5') as reader:              # doctest: +SKIP
    ...     data, r, t, p = reader.read(units='km/s')

    Inspect metadata without loading data:

    >>> reader = PsiData('rho001001.h5')                    # doctest: +SKIP
    >>> reader.quantity          # 'rho'
    >>> reader.description       # 'Mass Density'
    >>> reader.unit              # MAS_n (code unit)
    >>> reader.mesh              # (Mesh.HALF, Mesh.HALF, Mesh.HALF)
    >>> reader.shape             # (Nφ, Nθ, Nr) — numpy storage order
    """
    ifile = Path(ifile)
    cls = _DATA_CLASS_MAP.get(ifile.suffix)
    if cls is None:
        raise ValueError(
            f"Unsupported HDF extension '{ifile.suffix}'"
        ) from None
    return cls(ifile, model.lower(), **kwargs)