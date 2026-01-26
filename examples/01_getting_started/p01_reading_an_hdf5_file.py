"""
Reading an HDF5 File
====================

Perform forward tracing of magnetic field lines.

This example demonstrates how to use the :func:`~mapflpy.scripts.run_forward_tracing`
function to trace magnetic field lines forward from a set of default starting points.
It also shows how to load magnetic field data files and visualize the traced field lines in 3D.
"""
# sphinx_gallery_start_ignore
import os
if 'SPHINX_GALLERY_BUILD' not in os.environ:
    import matplotlib
    matplotlib.use('TkAgg')
# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt

from psi_io import read_hdf_meta, read_hdf_meta, read_hdf_data
from psi_io.data import get_3d_data

data = get_3d_data()
meta = read_hdf_meta(data)
dset = read_hdf_data(data)

ax = plt.figure().add_subplot()

ax.pcolormesh( dset[3], dset[2], dset[0][..., 0].T, clim=[-10, 10], cmap='seismic', shading='gouraud')

plt.show()