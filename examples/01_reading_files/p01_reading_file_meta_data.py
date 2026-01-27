"""
Reading an HDF5 File Metadata
=============================

Read 1D, 2D, and 3D HDF5 File Metadata and Data with PSI I/O

This example demonstrates how to use both :func:`~psi_io.read_hdf_data` and
:func:`~psi_io.read_hdf_meta` to read an HDF5 file and plot a 2D slice of the 3D
data using Matplotlib.

"""
# sphinx_gallery_start_ignore
import os
if 'SPHINX_GALLERY_BUILD' not in os.environ:
    import matplotlib
    matplotlib.use('TkAgg')
# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt

from psi_io import read_hdf_meta, read_rtp_meta
from psi_io.data import get_1d_data, get_2d_data, get_3d_data

# %%
# Load 1D, 2D, and 3D example data files
data3d_filepath = get_3d_data()
data2d_filepath = get_2d_data()
data1d_filepath = get_1d_data()

# %%
# Read in the file metadata
#
# .. note::
#    The metadata provides information about each dataset in the HDF5 file,
#    including its name, shape, and data type. It also includes information about
#    any associated scales (axes) for each dataset.
#
# Here, we read the metadata first to understand the structure of the file. For a standard
# PSI 3D data file, we expect to find one primary 3D dataset along with three 1D scale datasets
# corresponding to the :math:`r`, :math:`\theta`, and :math:`\phi` axes.


meta3d = read_hdf_meta(data3d_filepath)
meta2d = read_hdf_meta(data2d_filepath)
meta1d = read_hdf_meta(data1d_filepath)

# sphinx_gallery_start_ignore
for meta in (meta1d, meta2d, meta3d):
    print("Dataset Metadata:")

    print(f"+-------+-----------------+-----------------+-----------------+")
    print(f"| Index | Name            | Shape           | Type            |")
    print(f"+-------+-----------------+-----------------+-----------------+")
    for idx, m in enumerate(meta):
        name = m.name.ljust(15)
        shape = str(m.shape).ljust(15)
        dtype = str(m.type).ljust(15)
        print(f"| {idx:<5} | {name} | {shape} | {dtype} |")
    print(f"+-------+-----------------+-----------------+-----------------+")

    print(f"\nScale Metadata for '{meta[0].name}':")

    print(f"+-------+------------+------------+------------+------------+------------+")
    print(f"| Index | Name       | Shape      | Type       | Min        | Max        |")
    print(f"+-------+------------+------------+------------+------------+------------+")
    for idx, m in enumerate(meta[0].scales):
        name = m.name.ljust(10)
        shape = str(m.shape).ljust(10)
        dtype = str(m.type).ljust(10)
        imin = str(m.imin).ljust(10)
        imax = str(m.imax).ljust(10)
        print(f"| {idx:<5} | {name} | {shape} | {dtype} | {imin} | {imax} |")
    print(f"+-------+------------+------------+------------+------------+------------+")
# sphinx_gallery_end_ignore

# %%
# Similarly, we can use :func:`~psi_io.psi_io.read_rtp_meta` to quickly retrieve the same
# metadata but formatted specifically for radial-theta-phi datasets.

from pprint import pprint
meta_rtp = read_rtp_meta(data3d_filepath)
pprint(meta_rtp, sort_dicts=False)
