"""
Object oriented interface for reading HDF files and interacting
with their datasets on the fly.

This is still being actively developed for use with interactive applications.
For general scientific analysis scripting, we recommend using the routines psi-io.psi_io.

Written by Ryder Davidson.
"""
import os
from math import prod
from warnings import warn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Tuple, ItemsView, ValuesView, KeysView

import numpy as np
import h5py as h5

# Optional import checking
try:
    import pyhdf.SD as h4
    H4_AVAILABLE = True
except ImportError:
    H4_AVAILABLE = False

def except_no_pyhdf():
    if not H4_AVAILABLE:
        raise ImportError('The pyhdf package is required to read HDF4 .hdf files!')
    return

# HDF type constants
SDC_TYPE_CONVERSIONS = {
    3: np.dtype("ubyte"),
    4: np.dtype("byte"),
    5: np.dtype("float32"),
    6: np.dtype("float64"),
    20: np.dtype("int8"),
    21: np.dtype("uint8"),
    22: np.dtype("int16"),
    23: np.dtype("uint16"),
    24: np.dtype("int32"),
    25: np.dtype("uint32")
}

HDF4_EXT = {'.hdf', '.h4', '.he4', '.hdf4'}


class _HdfData(ABC):
    __slots__ = ('_file', '_label', '_values', '_scales')

    @abstractmethod
    def __init__(self, file, label: str):
        self._file = file
        self._label = label
        self._values = None

    def __repr__(self):
        return f'<{self.__class__.__name__} ({"" if self.loaded else "¬"}loaded): File[\'{self._label}\']>'

    @abstractmethod
    def __getitem__(self, ix):
        pass

    @property
    def label(self) -> str:
        return self._label

    @property
    def file(self):
        return self._file

    @property
    @abstractmethod
    def shape(self) -> Tuple:
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        pass

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @property
    @abstractmethod
    def nbytes(self) -> int:
        pass

    @property
    @abstractmethod
    def meta(self) -> Dict:
        pass

    @property
    def loaded(self) -> bool:
        return np.any(self._values)

    @property
    def s_(self) -> List:
        return self._scales

    @property
    def scales(self) -> List:
        return self.s_

    @property
    def v_(self) -> Optional[np.ndarray]:
        return self._values

    @property
    def values(self) -> Optional[np.ndarray]:
        return self.v_

    @abstractmethod
    def load(self, recursive: bool = False) -> None:
        # Read values into memory as np.ndarray
        if recursive:
            for scale in self._scales:
                scale.load(recursive)


class _HdfScale(_HdfData, ABC):
    __slots__ = _HdfData.__slots__ + ('_dlabel', '_ddim')

    def __init__(self, file, label: str, dataset_label: str, dataset_dim: int):
        super().__init__(file, label)
        self._dlabel, self._ddim = dataset_label, dataset_dim

    def __repr__(self):
        return f'<{self.__class__.__name__} ({"" if self.loaded else "¬"}loaded): File[\'{self._dlabel}\'][\'{self._ddim}\']>'

    @property
    def dataset_label(self) -> str:
        return self._dlabel

    @property
    def dataset_dim(self) -> int:
        return self._ddim


class _HdfFile(ABC):
    __slots__ = ('_filepath', '_file', '_data', '_is_open')

    @abstractmethod
    def __init__(self, filepath: os.PathLike):
        pass

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self):
        return f'<{self.__class__.__name__} ({"" if self._is_open else "¬"}opened): "{self._filepath}">'

    @classmethod
    @abstractmethod
    def read_file(cls, filepath: os.PathLike):
        pass

    @classmethod
    @abstractmethod
    def read_data(cls, file) -> Dict:
        pass

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def filepath(self) -> Path:
        return self._filepath

    @property
    @abstractmethod
    def meta(self) -> Dict:
        pass

    def keys(self) -> KeysView:
        return self._data.keys()

    def values(self) -> ValuesView:
        return self._data.values()

    def items(self) -> ItemsView:
        return self._data.items()

    @abstractmethod
    def close(self) -> None:
        # Call file closing routine
        self._is_open = False


class H5Data(_HdfData):
    def __init__(self, file: h5.File, label: str):
        super().__init__(file, label)
        if file[label].is_scale:
            self._scales = []
        else:
            self._scales = [H5Scale(file, dim.label, label, i) for i, dim in enumerate(file[label].dims) if dim]

    def __getitem__(self, ix):
        if isinstance(ix, str):
            return self._scales[int(ix)]
        else:
            if self.loaded:
                return self._values[ix]
            else:
                return self._file[self._label][ix]

    @property
    def shape(self) -> Tuple:
        return self._file[self._label].shape

    @property
    def dtype(self) -> np.dtype:
        return self._file[self._label].dtype

    @property
    def ndim(self) -> int:
        return self._file[self._label].ndim

    @property
    def nbytes(self) -> int:
        return self._file[self._label].nbytes

    @property
    def meta(self) -> Dict:
        return dict(self._file[self._label].attrs)

    def load(self, recursive: bool = False) -> None:
        self._values = self._file[self._label][...]
        super().load(recursive)


class H5Scale(_HdfScale, H5Data):
    pass


class H5File(_HdfFile):

    def __init__(self, filepath: os.PathLike):
        filepath_ = Path(filepath)
        file_ = H5File.read_file(filepath_)
        data_ = H5File.read_data(file_)
        self._filepath, self._file, self._data, self._is_open = filepath_, file_, data_, True

    @classmethod
    def read_file(cls, filepath: os.PathLike) -> h5.File:
        return h5.File(filepath, 'r')

    @classmethod
    def read_data(cls, file: h5.File) -> Dict:
        return {k: H5Data(file, k) for k, v in file.items() if not v.is_scale}

    @property
    def meta(self) -> Dict:
        return dict(self._file.attrs)

    def close(self) -> None:
        self._file.close()
        super().close()


class H4Data(_HdfData):

    def __init__(self, file: h4.SD, label: str):
        super().__init__(file, label)
        if file.select(label).iscoordvar():
            self._scales = []
        else:
            self._scales = [H4Scale(file, dim, label, i) for i, dim in
                            enumerate(reversed(file.select(label).dimensions().keys()))]

    def __getitem__(self, ix):
        if isinstance(ix, str):
            return self._scales[int(ix)]
        else:
            if self.loaded:
                return self._values[ix]
            else:
                return self._file.select(self._label)[ix]

    @property
    def shape(self) -> Tuple:
        _, dimensionality, shape, *_ = self._file.select(self._label).info()
        if dimensionality > 1:
            return tuple(shape)
        else:
            return shape,

    @property
    def dtype(self) -> np.dtype:
        return SDC_TYPE_CONVERSIONS[self._file.select(self._label).info()[3]]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.dtype.itemsize

    @property
    def meta(self) -> Dict:
        return self._file.select(self._label).attributes()

    def load(self, recursive: bool = False):
        self._values = self._file.select(self._label)[:]
        super().load(recursive)


class H4Scale(_HdfScale, H4Data):
    pass


class H4File(_HdfFile):

    def __init__(self, filepath: os.PathLike):
        except_no_pyhdf()
        filepath_ = Path(filepath)
        file_ = H4File.read_file(filepath_)
        data_ = H4File.read_data(file_)
        self._filepath, self._file, self._data, self._is_open = filepath_, file_, data_, True

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self):
        return f'<{self.__class__.__name__} ({"opened" if self._is_open else "¬opened"}): "{self._filepath}">'

    @classmethod
    def read_file(cls, filepath: os.PathLike) -> h4.SD:
        return h4.SD(str(filepath), h4.SDC.READ)

    @classmethod
    def read_data(cls, file: h4.SD) -> Dict:
        return {k: H4Data(file, k) for k in file.datasets().keys() if not file.select(k).iscoordvar()}

    @property
    def meta(self) -> Dict:
        return self._file.attributes(full=True)

    def close(self) -> None:
        try:
            self._file.end()
        except TypeError as e:
            warn('HDF File has already been closed')
        finally:
            super().close()


def read_hdf(filepath: os.PathLike) -> H5File | H4File:
    filepath_ = Path(filepath)
    assert filepath_.exists(), f'\'{filepath}\' cannot be found'
    if h5.is_hdf5(filepath_):
        return H5File(filepath_)
    assert filepath_.suffix in HDF4_EXT, f'\'{filepath_.suffix}\' is not a valid HDF extension'
    return H4File(filepath_)


def slice_hdf_by_index(*ix, data: H4Data | H5Data, return_scales: bool = True) -> np.ndarray:
    assert len(ix) == data.ndim, f'Expected {data.ndim} indices, got {len(ix)}'
    slices = [None] * len(ix)
    for i, s in enumerate(ix):
        match s:
            case slice():
                slices[i] = s
            case int():
                slices[i] = slice(s, s + 1)
            case tuple():
                slices[i] = slice(*s)
            case None:
                slices[i] = slice(None)
            case _:
                raise TypeError(f'Invalid index type: {type(s)}. Expected int, tuple, or None.')
    # slices = tuple(slice(i, i + 1) if isinstance(i, int) else slice(*i) for i in ix)
    values = data[tuple(reversed(slices))]
    if return_scales:
        scales = [data.s_[i][s] for i, s in enumerate(slices)]
        return values, *scales
    else:
        return values


def slice_hdf_by_value(*ix, data: H4Data | H5Data, return_scales: bool = True) -> np.ndarray:
    assert len(ix) == data.ndim, f'Expected {data.ndim} indices, got {len(ix)}'
    slices = [None] * len(ix)
    for scale in data.scales:
        if not scale.loaded:
            scale.load()
    for i, s in enumerate(ix):
        match s:
            case float() | int():
                index = np.searchsorted(data.s_[i][:], s)
                if index == data.s_[i].shape[0]:
                    index -= 1
                slices[i] = (int(index), int(index) + 1)
            case tuple():
                index = np.searchsorted(data.s_[i][:], s)
                if index[0] == data.s_[i].shape[0]:
                    index[0] -= 1
                slices[i] = (int(index[0]), int(index[1]))
            case None:
                slices[i] = slice(None)
            case _:
                raise TypeError(f'Invalid index type: {type(s)}. Expected int, tuple, or None.')
    return slice_hdf_by_index(*slices, data=data, return_scales=return_scales)