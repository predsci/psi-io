"""
Reading HDF5 File Metadata
=============================

Read a selection of standard PSI data products in HDF5 format and examine their metadata.

This example demonstrates how to use :func:`~psi_io.psi_io.read_hdf_meta` to inspect the
metadata of various standard PSI data files; these example files are utilized elsewhere in
the documentation and examples; hopefully this example will help users understand the
structure of these files and how to work with them using PSI I/O.

"""

# sphinx_gallery_start_ignore
def pretty_print_meta(filename, meta):
    print(f"Datasets for file: {filename}")

    print(f"+-----------------+-----------------+-----------------+")
    print(f"| Name            | Shape           | Type            |")
    print(f"+-----------------+-----------------+-----------------+")
    for idx, m in enumerate(meta):
        name = m.name.ljust(15)
        shape = str(m.shape).ljust(15)
        dtype = str(m.type).ljust(15)
        print(f"| {name} | {shape} | {dtype} |")
    print(f"+-----------------+-----------------+-----------------+\n")

    print(f"Scales for file: {filename}")
    if any(len(m.scales) > 0 for m in meta):

        print(f"+------------+------------+------------+------------+------------+------------+------------+")
        print(f"| Dataset    | Dim        | Name       | Shape      | Type       | Min        | Max        |")
        print(f"+------------+------------+------------+------------+------------+------------+------------+")
        for m in meta:
            for idx, d in enumerate(m.scales):
                dset = m.name.ljust(10)
                dimname = str(d.name).ljust(10)
                shape = str(d.shape).ljust(10)
                dtype = str(d.type).ljust(10)
                imin = str(d.imin).ljust(10)
                imax = str(d.imax).ljust(10)
                print(f"| {dset} | {idx:<10} | {dimname} | {shape} | {dtype} | {imin} | {imax} |")
        print(f"+------------+------------+------------+------------+------------+------------+------------+")
    else:
        print("     No scales found in this file.")

# sphinx_gallery_end_ignore

from pathlib import Path
from psi_io import read_hdf_meta, data

# %%
# Read in example 1D data file, *viz.* a vignette function (used for radial scaling).

data1d_filepath = data.get_1d_data()
meta1d = read_hdf_meta(data1d_filepath)

# sphinx_gallery_start_ignore
pretty_print_meta(Path(data1d_filepath).name, meta1d)
# sphinx_gallery_end_ignore

# %%
# Read in example 2D data file, *viz.* a long-lat coronal hole map.

data2d_filepath = data.get_2d_data()
meta2d = read_hdf_meta(data2d_filepath)

# sphinx_gallery_start_ignore
pretty_print_meta(Path(data2d_filepath).name, meta2d)
# sphinx_gallery_end_ignore

# %%
# Read in example 3D data file, *viz.* the radial component of the magnetic field.

data3d_filepath = data.get_3d_data()
meta3d = read_hdf_meta(data3d_filepath)

# sphinx_gallery_start_ignore
pretty_print_meta(Path(data3d_filepath).name, meta3d)
# sphinx_gallery_end_ignore

# %%
# Read in example fieldline trace data file – a 2D dataset without coordinate variables.

datafieldline_filepath = data.get_fieldline_data()
metafieldline = read_hdf_meta(datafieldline_filepath)

# sphinx_gallery_start_ignore
pretty_print_meta(Path(datafieldline_filepath).name, metafieldline)
# sphinx_gallery_end_ignore

# %%
# Read in example file from PSI's `Coronal Hole Map Database <https://q.predsci.com/CHMAP-map-browser/>`_
#
# .. note::
#    This file contains not only a 2D coronal hole map (with its associated coordinate variables)
#    but also a variety of additional metadata datasets describing the observation and processing
#    used to create the map.

datasynchronic_map_filepath = data.get_synchronic_map_data()
metasynchronic_map = read_hdf_meta(datasynchronic_map_filepath)

# sphinx_gallery_start_ignore
pretty_print_meta(Path(datasynchronic_map_filepath).name, metasynchronic_map)
# sphinx_gallery_end_ignore

# %%
# Except for the last example file (derived from the Coronal Hole Map Database), each of these
# files are available in both HDF4 and HDF5 formats. You can read the metadata from either format
# using the same :func:`~psi_io.psi_io.read_hdf_meta` function. To illustrate how these files
# differ (internally) between HDF4 and HDF5, we can read the metadata from the HDF4 version of
# the 3D data file using the same function:

data3d_hdf4_filepath = data.get_3d_data(hdf=".hdf")
meta3d_hdf4 = read_hdf_meta(data3d_hdf4_filepath)

# sphinx_gallery_start_ignore
pretty_print_meta(Path(data3d_hdf4_filepath).name, meta3d_hdf4)
# sphinx_gallery_end_ignore

# %%
# Compared to the HDF5 version of the same file, the HDF4 version has slightly different dataset
# naming conventions (*e.g.* the primary dataset is labeled **"Data-Set-2"** instead of **"Data"**, while
# its scales are named **"fakeDim0"**, **"fakeDim1"**, and **"fakeDim2"** instead of **"dim1"**,
# **"dim2"**, **"dim3"**). Also – and perhaps more importantly – the HDF4 version does not respect
# the inherent Fortran ordering of the underlying data arrays
#
# .. note::
#    For more information regarding the differences between HDF4 and HDF5, see :ref:`Overview <overview>`.

