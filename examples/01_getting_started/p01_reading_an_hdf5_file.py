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

from psi_io import read_hdf_meta
from psi_io.data import FETCHER

data = FETCHER.fetch("2143-mast2-cor/br002.h5")
meta = read_hdf_meta(ifile=data)