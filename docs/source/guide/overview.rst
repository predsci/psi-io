.. _overview:

Overview
========

``psi-io`` is a Python package developed by `Predictive Science Inc. <https://www.predsci.com>`_ for
interacting with the PSI HDF data ecosystem. It provides a unified interface for reading and writing
HDF4 and HDF5 files, as well as tools for reading portions of datasets and interpolating data to
arbitrary positions.


PSI Conventions
===============

PSI follows certain conventions for storing data in HDF files. *While these conventions are not strictly
enforced, adhering to them ensures compatibility with PSI tools and libraries.* Much of the scientific
computing done through PSI relies on tools developed in Fortran. As such, the conventions used within PSI's
python ecosystem mirror those used in Fortran-based PSI tools. However, due to the fact that Python uses
C-style (row-major) array ordering, while Fortran uses Fortran-style (column-major) ordering, there are
some differences in how dimension scales are stored between HDF4 and HDF5 files.

To illustrate, a 3D dataset stored in Fortran order would have its dimensions ordered as (X, Y, Z) in Fortran.
In Python, this same dataset would be represented as (Z, Y, X) due to the difference in array ordering conventions.
For 3D datasets which are based in spherical coordinates – *e.g.* the "3D cubes" generated through
`MAS <https://www.predsci.com/corona/model_desc.html>`_ – datasets are represented in python as arrays with shape
(phi, theta, r) and dimension scales r (radius), theta (colatitude), and phi (longitude).

.. note::
    The difference in array ordering conventions between Fortran and Python can lead to confusion when
    working with dimension scales in HDF files. It is important to be aware of these conventions when
    reading or writing HDF files to ensure that data is interpreted correctly.

To further complicate things, HDF4 and HDF5 handle dimension scales differently. HDF5 supports Fortran ordering of
dimension scales, while HDF4 does not *i.e.* the first dimension of the aforementioned 3D dataset would have its
dimension scale corresponding to r in HDF5, but phi in HDF4. Below are the specific conventions used in HDF4 and HDF5 files
within PSI.

HDF4 Conventions
----------------

    - Datasets are stored in the root group ("/") and named "Data-Set-2".
    - Dimension scales are named "fakeDim0", "fakeDim1", ..., "fakeDimN" for N-dimensional datasets.
    - Datasets are Fortran ordered (column-major). However, due to limitations in HDF4, the dimension scales
      are stored in C-order (row-major).

HDF5 Conventions
----------------

    - Datasets are stored in the root group ("/") and named "Data".
    - Dimension scales are named "dim0", "dim1", ..., "dimN" for N-dimensional datasets.
    - Datasets are Fortran ordered (column-major), and dimension scales are also stored in Fortran order.

psi-io Conventions
------------------

To mitigate these idiosyncrasies, ``psi-io`` provides a unified interface for reading and writing HDF4 and HDF5 files
that abstracts away the differences in conventions. When data is read from an HDF (whether HDF4 or HDF5), ``psi-io``
data is always returned in Fortran order with dimension scales correctly associated with their respective dimensions.
When writing data to an HDF file, ``psi-io`` ensures that the appropriate conventions for HDF4 or HDF5 are followed based
on the file extension.

When reading a 3D dataset (either HDF4 or HDF5) with ``psi-io``, the returned array will have shape :math:`(phi, theta, r)` and
the associated dimension scales will be returned as :math:`(r, theta, phi)`. This ensures that the data is always represented
in a consistent manner, regardless of the underlying HDF format.

Using psi-io
============

**Reading Full Datasets & Scales:**
    - :func:`~psi_io.psi_io.rdhdf_1d`
    - :func:`~psi_io.psi_io.rdhdf_2d`
    - :func:`~psi_io.psi_io.rdhdf_3d`
    - :func:`~psi_io.psi_io.read_hdf_data`

**Writing Full Datasets & Scales:**
    - :func:`~psi_io.psi_io.wrhdf_1d`
    - :func:`~psi_io.psi_io.wrhdf_2d`
    - :func:`~psi_io.psi_io.wrhdf_3d`
    - :func:`~psi_io.psi_io.write_hdf_data`

**Reading Subsets of Datasets:**
    - :func:`~psi_io.psi_io.get_scales_1d`
    - :func:`~psi_io.psi_io.get_scales_2d`
    - :func:`~psi_io.psi_io.get_scales_3d`
    - :func:`~psi_io.psi_io.read_hdf_by_index`
    - :func:`~psi_io.psi_io.read_hdf_by_value`
    - :func:`~psi_io.psi_io.read_hdf_by_ivalue`

**Reading File Metadata:**
    - :func:`~psi_io.psi_io.read_hdf_meta`
    - :func:`~psi_io.psi_io.read_rtp_meta`

**Interpolating Data to Arbitrary Positions:**
    - :func:`~psi_io.psi_io.np_interpolate_slice_from_hdf`
    - :func:`~psi_io.psi_io.interpolate_positions_from_hdf`
    - :func:`~psi_io.psi_io.sp_interpolate_slice_from_hdf`
    - :func:`~psi_io.psi_io.interpolate_point_from_1d_slice`
    - :func:`~psi_io.psi_io.interpolate_point_from_2d_slice`

.. note::
   The HDF type (HDF4 or HDF5) is automatically determined by the file extension
   (".hdf" for HDF4 and ".h5" for HDF5) when using ``psi-io`` functions.

.. note::
   Not all PSI FORTRAN tools can read HDF4 files written by the :class:`pyhdf.SD` interface.
   If you have a problem, use the PSI tool ``hdfsd2hdf`` to convert.
