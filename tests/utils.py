from types import MappingProxyType

import h5py as h5
import numpy as np

from psi_io import psi_io

try:
    import pyhdf.SD as h4
except ImportError:
    pass

FILEEXT = MappingProxyType({
    "h4": ".hdf",
    "h5": ".h5",
})

PRIMES = (11, 13, 17, 19, 23, 29, 31, 37,)


def generate_data_shape(ndim: int):
    return tuple(reversed(PRIMES[:ndim]))


def generate_mock_data(ndim: int, dtype: str, scales: bool = True):
    shape = generate_data_shape(ndim)
    fdata = np.indices(shape, dtype=dtype).sum(axis=0, dtype=dtype)
    sdata = (np.arange(s, dtype=dtype) for s in reversed(shape)) if scales else tuple()
    return fdata, *sdata


def _generate_mock_h4_data(ifile, ndim, dtype, scales):

    fdata, *sdata = generate_mock_data(ndim, dtype, scales)

    h4file = h4.SD(str(ifile), h4.SDC.WRITE | h4.SDC.CREATE | h4.SDC.TRUNC)
    sds_id = h4file.create("Data-Set-2", psi_io.NPTYPES_TO_SDCTYPES[dtype], fdata.shape)

    if scales:
        for i, scale in enumerate(reversed(sdata)):
            sds_id.dim(i).setscale(psi_io.NPTYPES_TO_SDCTYPES[dtype], scale.tolist())

    sds_id.set(fdata)
    sds_id.endaccess()
    h4file.end()

    return ifile


def _generate_mock_h5_data(ifile, ndim, dtype, scales):
    """Generate mock HDF5 data files for testing."""
    # This is a placeholder function. Implement the logic to generate mock HDF5 data files.
    fdata, *sdata = generate_mock_data(ndim, dtype, scales)
    with h5.File(ifile, "w") as h5file:
        h5file.create_dataset("Data", data=fdata, dtype=dtype, shape=fdata.shape)

        if scales:
            for i, scale in enumerate(sdata):
                h5file.create_dataset(f"dim{i+1}", data=scale, dtype=dtype, shape=scale.shape)
                h5file['Data'].dims[i].attach_scale(h5file[f"dim{i+1}"])
                h5file['Data'].dims[i].label = f"dim{i+1}"

    return ifile


def generate_mock_files(ifile,
                       ndim: int = 1,
                       dtype: str = 'float64',
                       scales: bool = True):
    """Generate mock data files for testing."""
    # This is a placeholder function. Implement the logic to generate mock data files.
    return psi_io._dispatch_by_ext(ifile, _generate_mock_h4_data, _generate_mock_h5_data,
                                   ndim, dtype, scales)
