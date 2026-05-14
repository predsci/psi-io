from __future__ import annotations

import re
from abc import abstractmethod, ABC
from collections import namedtuple
from itertools import repeat
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Sequence, Literal, ClassVar
import numpy as np
import h5py as h5
import astropy.units as u

try:
    import pyhdf.SD as h4
except ImportError:
    h4 = None

from psi_io._mesh import (Mesh, MeshCodeType,
                          _normalize_mesh_code, remesh_arr,
                          )
from psi_io._props import (Props,
                           MasQuantities,
                           Pot3dQuantities,
                           PsiScales,
                           _MAS_QUANTITY_PROPS_MAPPING,
                           _POT3D_QUANTITY_PROPS_MAPPING,
                           _PSI_SCALE_PROPS_MAPPING)
from psi_io._units import decompose_mas_units
from psi_io.psi_io import (PathLike,
                           PSI_DATA_ID,
                           SDC_TYPE_CONVERSIONS,
                           _except_no_pyhdf,
                           )

HDF_EXT_MAPPING = {'h5': '.h5', 'h4': '.hdf',}
_DATA_SLOTS = ('_fileref', '_filepath', '_datalabel', '_quantity', '_sequence', '_unit', '_mesh', '_scales')
ModelType = Literal['mas', 'pot3d', 'scale']
HdfVersionType = Literal['h4', 'h5']
_CODE_UNIT_ALIASES = {'native', 'code', 'model'}
_REAL_UNIT_ALIASES = {'real', 'phys', 'physical'}


METADATA_SCHEMA = dict.fromkeys(['quantity', 'sequence', 'unit', 'mesh'])

MATCH_QUANTITIES = '|'.join(re.escape(q) for q in sorted(_MAS_QUANTITY_PROPS_MAPPING.keys(), key=len, reverse=True))
FILEPATH_SCHEMA = rf'^({MATCH_QUANTITIES})(\d{{3}}(?:\d{{3}})?)$'


def get_mas_quantity_properties(variable: MasQuantities) -> Props:
    try:
        return _MAS_QUANTITY_PROPS_MAPPING[variable.lower()]
    except KeyError:
        raise ValueError(f"Invalid variable '{variable}'. "
                         f"Valid options are: {', '.join(_MAS_QUANTITY_PROPS_MAPPING.keys())}") from None


def get_pot3d_quantity_properties(variable: Pot3dQuantities) -> Props:
    try:
        return _POT3D_QUANTITY_PROPS_MAPPING[variable.lower()]
    except KeyError:
        raise ValueError(f"Invalid variable '{variable}'. "
                         f"Valid options are: {', '.join(_POT3D_QUANTITY_PROPS_MAPPING.keys())}") from None


def get_psi_scale_properties(variable: PsiScales) -> Props:
    try:
        return _PSI_SCALE_PROPS_MAPPING[variable.lower()[0]]
    except KeyError:
        raise ValueError(f"Invalid variable '{variable}'. "
                         f"Valid options are: {', '.join(_PSI_SCALE_PROPS_MAPPING.keys())}") from None


def extract_quantity_from_filepath(ifile: Path, default: Optional[str] = None) -> str | None:
    match = re.match(rf'^({MATCH_QUANTITIES})(?=[^a-zA-Z]|$)', ifile.stem, re.IGNORECASE)
    return match.group(1).lower() if match else default


def extract_sequence_from_filepath(ifile: Path, default: Optional[int] = None) -> int | None:
    match = re.search(r'\d{3}(?:\d{3})?', ifile.stem)
    return int(match.group()) if match else default


def parse_mas_filename_schema(ifile: Path):
    matches = re.match(FILEPATH_SCHEMA, ifile.stem, re.IGNORECASE)
    if not matches:
        raise ValueError(f"Filename '{ifile.name}' does not match expected MAS filename schema: "
                         f"'<quantity><sequence>'. Valid quantities are: {', '.join(_MAS_QUANTITY_PROPS_MAPPING.keys())}. "
                         f"Sequence should be a 3 or 6 digit number.")
    quantity, sequence = matches.groups()
    return quantity, int(sequence)

_PROP_MAPPING = {'mas': get_mas_quantity_properties, 'pot3d': get_pot3d_quantity_properties, 'scale': get_psi_scale_properties,}
Scales = namedtuple("Scales", ['r', 't', 'p'])


# =============================================================================
# Abstract interface
# =============================================================================

class _HdfInterface(ABC):
    __slots__ = ()

    _HDFN: ClassVar[HdfVersionType]                        # provided by format mixin
    _MODEL: ClassVar[ModelType]                           # provided by concrete subclass

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = (args,)
        return self.data[tuple(reversed(args))]

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]: ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype: ...

    @property
    @abstractmethod
    def size(self) -> int: ...

    @property
    @abstractmethod
    def nbytes(self) -> int: ...

    @property
    @abstractmethod
    def ndim(self) -> int: ...

    @property
    @abstractmethod
    def attrs(self) -> dict: ...

    @property
    @abstractmethod
    def unit(self) -> u.Unit: ...

    @property
    @abstractmethod
    def mesh(self) -> tuple[Mesh, ...]: ...

    @property
    @abstractmethod
    def quantity(self) -> str: ...

    @property
    def description(self) -> str:
        return _PROP_MAPPING[self._MODEL](self._quantity).desc

    @property
    @abstractmethod
    def data(self) -> np.ndarray: ...

    @property
    def native_properties(self) -> Props:
        return _PROP_MAPPING[self._MODEL](self._quantity)

    @abstractmethod
    def read(self,
             *args,
             units: Optional[str | u.Unit] = None,
             mesh: Optional[MeshCodeType] = None,
             ) -> tuple[u.Quantity, tuple[slice, ...], tuple[bool, ...]]:
        if mesh is None:
            remesh = repeat(False, self.ndim)
        else:
            remesh = _parse_remesh(self.mesh, _normalize_mesh_code(mesh, self.ndim))
        remesh = tuple(remesh)

        sargs = _parse_islice_args(*args, shape=tuple(reversed(self.shape)), remesh=remesh)
        sargs = tuple(sargs)

        odata = remesh_arr(self[sargs], remesh=tuple(reversed(remesh))) * self.unit
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
        return remesh_arr(self[sargs], remesh=remesh) * self.unit


# =============================================================================
# Scale classes
# =============================================================================

class _HdfScale(_HdfInterface, ABC):
    """Abstract base for HDF coordinate scale variables (r, t, p)."""
    __slots__ = ()
    _MODEL = 'scale'

    def __init__(self,
                 parent,
                 dim_label: str,
                 data_label: str,):
        self._dataref = parent
        self._datalabel: str = data_label

        if self.ndim != 1:
            raise ValueError(f"Expected 1D coordinate variable, "
                             f"found {self.ndim}D dataset with shape {self.shape}.")

        self._set_properties(dim_label)

    def _set_properties(self, scale: str):
        try:
            qprops = _PROP_MAPPING[self._MODEL](scale)
            self._quantity: PsiScales = qprops.name
            self._unit: u.Unit = qprops.unit
        except (ValueError, TypeError) as e:
            raise ValueError(f"Metadata type coercion failed: {e}") from e

    @property
    def unit(self) -> u.Unit:
        return self._unit

    @property
    def quantity(self) -> PsiScales:
        return self._quantity

    @property
    def mesh(self) -> tuple[Mesh, ...]:
        return self._dataref.mesh['rtp'.index(self._quantity)],

    def read(self,
             *args,
             **kwargs,
             ) -> u.Quantity:
        odata, *_ = super().read(*args, **kwargs)
        return odata

    def _read(self,
              *sargs,
              remesh) -> u.Quantity:
        return super()._read(*sargs, remesh=remesh)


class H5Scale(_HdfScale):
    """HDF5 coordinate scale variable backed by an h5py dimension."""
    __slots__ = ('_dataref', '_datalabel', '_quantity', '_unit')

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def nbytes(self) -> int:
        return self.data.nbytes

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def attrs(self) -> dict:
        return dict(self.data.attrs)

    @property
    def data(self) -> np.ndarray:
        return self._dataref._fileref[self._datalabel]


class H4Scale(_HdfScale):
    """HDF4 coordinate scale variable backed by a pyhdf SDS dimension."""
    __slots__ = ('_dataref', '_datalabel', '_quantity', '_unit')

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.info()[2],

    @property
    def dtype(self) -> np.dtype:
        return SDC_TYPE_CONVERSIONS[self.data.info()[3]]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    @property
    def ndim(self) -> int:
        return self.data.info()[1]

    @property
    def attrs(self) -> dict:
        return self.data.attributes()

    @property
    def data(self) -> np.ndarray:
        return self._dataref._fileref.select(self._datalabel)


# =============================================================================
# Format mixins (HDF5 and HDF4 file I/O + raw array access)
# =============================================================================

class _H5DataMixin:
    """Mixin providing HDF5 file I/O and raw array access via h5py."""
    __slots__ = ()
    _HDFN = 'h5'

    @classmethod
    def read_file(cls, ifile: PathLike):
        return h5.File(ifile, 'r')

    def open(self):
        if not self._fileref:
            self._fileref = self.read_file(self._filepath)
        return self

    def close(self):
        if self._fileref is not None:
            self._fileref.close()
            self._fileref = None
        return self

    def delete(self):
        fileref = getattr(self, '_fileref', None)
        if fileref is not None:
            fileref.close()
            self._fileref = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def nbytes(self) -> int:
        return self.data.nbytes

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def attrs(self) -> dict:
        return dict(self.data.attrs)

    @property
    def data(self) -> np.ndarray:
        return self._fileref[self._datalabel]

    def _set_scales(self):
        self._scales = Scales(*tuple(H5Scale(self, scale, label.label)
                                     for scale, label in zip('rtp', self.data.dims, strict=True)))


class _H4DataMixin:
    """Mixin providing HDF4 file I/O and raw array access via pyhdf."""
    __slots__ = ()
    _HDFN = 'h4'

    @classmethod
    def read_file(cls, ifile: PathLike):
        _except_no_pyhdf()
        return h4.SD(str(ifile), h4.SDC.READ)

    def open(self):
        if not self._fileref:
            self._fileref = self.read_file(self._filepath)
        return self

    def close(self):
        if self._fileref is not None:
            self._fileref.end()
            self._fileref = None

    def delete(self):
        fileref = getattr(self, '_fileref', None)
        if fileref is not None:
            fileref.end()
            self._fileref = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.info()[2])

    @property
    def dtype(self) -> np.dtype:
        return SDC_TYPE_CONVERSIONS[self.data.info()[3]]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    @property
    def ndim(self) -> int:
        return self.data.info()[1]

    @property
    def attrs(self) -> dict:
        return self.data.attributes()

    @property
    def data(self) -> np.ndarray:
        return self._fileref.select(self._datalabel)

    def _set_scales(self):
        sds = self.data
        dims = list(reversed(list(sds.dimensions(full=1).items())))
        self._scales = Scales(*tuple(H4Scale(self, scale, k_)
                                     for scale, (k_, v_) in zip('rtp', dims, strict=True)))


# =============================================================================
# Abstract data base
# =============================================================================

class _HdfData(_HdfInterface, ABC):
    """
    Abstract base for PSI MHD data files.

    Subclasses must supply:
      - a format mixin (_H5DataMixin or _H4DataMixin) for file I/O
      - a _QUANTITY_MAPPING class attribute mapping quantity names to Props
    """
    __slots__ = _DATA_SLOTS

    def __init__(self,
                 ifile: PathLike, /,
                 dataset_id: Optional[str] = None,
                 **kwargs):
        ifile = Path(ifile)
        if not ifile.is_file():
            raise FileNotFoundError(f"File '{ifile}' does not exist.")
        if ifile.suffix != HDF_EXT_MAPPING[self._HDFN]:
            raise ValueError(f"File '{ifile}' does not have the correct extension for "
                             f"{self._HDFN} files (expected '{HDF_EXT_MAPPING[self._HDFN]}' extension).")

        self._filepath: Path = ifile
        self._datalabel: str = dataset_id or PSI_DATA_ID[self._HDFN]
        self._fileref = self.read_file(ifile)

        if self.ndim != 3:
            raise ValueError(f"Expected 3D MAS dataset, "
                             f"found {self.ndim}D dataset with shape {self.shape}.")

        self._set_properties(**self._parse_properties(**kwargs))
        self._set_scales()

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        return self.close()

    def __del__(self):
        return self.delete()

    @classmethod
    @abstractmethod
    def read_file(cls, ifile: PathLike): ...

    @abstractmethod
    def open(self): ...

    @abstractmethod
    def close(self): ...

    @abstractmethod
    def delete(self): ...

    @abstractmethod
    def _set_scales(self): ...

    def _parse_properties(self, **kwargs):
        input_attrs = {k: v for k, v in kwargs.items() if k in METADATA_SCHEMA}
        file_attrs = {k: v for k, v in self.attrs.items() if k in METADATA_SCHEMA}

        quantity = input_attrs.pop('quantity',
                                   file_attrs.pop('quantity',
                                                  extract_quantity_from_filepath(self._filepath, '')))
        sequence = input_attrs.pop('sequence',
                                   file_attrs.pop('sequence',
                                                  extract_sequence_from_filepath(self._filepath, 0)))

        native_props = _PROP_MAPPING[self._MODEL](quantity)

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
        try:
            self._quantity: str = str(quantity)
            self._sequence: int = int(sequence)
            self._unit: u.Unit = u.Unit(str(unit))
            self._mesh: tuple[Mesh, ...] = _normalize_mesh_code(mesh, self.ndim)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Metadata type coercion failed: {e}") from e

    @property
    def unit(self) -> u.Unit:
        return self._unit

    @property
    def mesh(self) -> tuple[Mesh, ...]:
        return self._mesh

    @property
    def quantity(self) -> str:
        return self._quantity

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def scales(self) -> Scales:
        return self._scales

    def read(self,
             *args,
             scales: bool = True,
             **kwargs
             ) -> u.Quantity | tuple[u.Quantity, ...]:
        odata, sargs, remesh = super().read(*args, **kwargs)
        if not scales:
            return odata
        oscales = (scale._read(sarg, remesh=rmesh) for scale, sarg, rmesh in zip(self.scales, sargs, remesh))
        return odata, *oscales

    def _read(self,
              *sargs,
              remesh) -> u.Quantity:
        return super()._read(*sargs, remesh=remesh)


# =============================================================================
# Concrete data classes
# =============================================================================

class H5MasData(_H5DataMixin, _HdfData):
    """HDF5 MAS model data."""
    __slots__ = _HdfData.__slots__
    _MODEL = 'mas'


class H5Pot3dData(_H5DataMixin, _HdfData):
    """HDF5 POT3D model data."""
    __slots__ = _HdfData.__slots__
    _MODEL = 'pot3d'


class H4MasData(_H4DataMixin, _HdfData):
    """HDF4 MAS model data."""
    __slots__ = _HdfData.__slots__
    _MODEL = 'mas'


class H4Pot3dData(_H4DataMixin, _HdfData):
    """HDF4 POT3D model data."""
    __slots__ = _HdfData.__slots__
    _MODEL = 'pot3d'


# =============================================================================
# Helpers
# =============================================================================

def _parse_remesh(imesh, omesh):
    for im, om in zip(imesh, omesh, strict=True):
        if im == om:
            yield False
        elif im == Mesh.HALF and om == Mesh.MAIN:
            yield True
        elif im == Mesh.MAIN and om == Mesh.HALF:
            raise ValueError(f"Cannot remesh from MAIN mesh to HALF mesh.")


def _parse_islice_args(*args, shape: tuple[int, ...], remesh: tuple[bool, ...]):
    if Ellipsis in args:
        n_missing = len(shape) - (len(args) - 1)
        idx = args.index(Ellipsis)
        args = args[:idx] + (None,) * n_missing + args[idx + 1:]
    if len(args) < len(shape):
        args += (None,) * (len(shape) - len(args))

    for arg, dim_size, do_remesh in zip(args, shape, remesh, strict=True):
        slice_ = _cast_to_slice(arg)
        start, stop, step = slice_.indices(dim_size)
        if do_remesh and (stop - start) // step < 2:
            raise ValueError(f"Cannot remesh dimension with slice {slice_} "
                             f"because it does not include at least two indices.")
        yield slice_


def _parse_vslice_args(dim, scale):
    val = None
    if isinstance(dim, float):
        dim = dim * scale.unit
    if isinstance(dim, u.Quantity):
        val = dim.to(scale.unit)
        i1 = int(np.clip(np.searchsorted(scale.data, val.value), 1, scale.size - 2))
        dim = (i1-1, i1+1)
    return _cast_to_slice(dim), val


def _cast_to_slice(input):
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


_DATA_CLASS_MAP = MappingProxyType({
    ('.h5',  'mas'):   H5MasData,
    ('.h5',  'pot3d'): H5Pot3dData,
    ('.hdf', 'mas'):   H4MasData,
    ('.hdf', 'pot3d'): H4Pot3dData,
})


def PsiData(ifile: PathLike, /,
            model: ModelType = 'mas',
            **kwargs):
    ifile = Path(ifile)
    key = (ifile.suffix, model.lower())
    cls = _DATA_CLASS_MAP.get(key)
    if cls is None:
        raise ValueError(
            f"Unsupported combination of extension '{ifile.suffix}' and model '{model}'. "
            f"Valid combinations: {[f'{ext}/{m}' for ext, m in _DATA_CLASS_MAP]}"
        )
    return cls(ifile, **kwargs)