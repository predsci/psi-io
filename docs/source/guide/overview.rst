.. _overview:

Overview
========

``psi-io`` is a Python package developed by `Predictive Science Inc. <https://www.predsci.com>`_ for
interacting with the PSI HDF data ecosystem. It provides a unified interface for reading and writing
HDF4 and HDF5 files, as well as tools for reading portions of datasets and interpolating data to
arbitrary positions.

Quick Start
===========

The following example opens a MAS HDF5 file containing a radial magnetic field, reads the full
3-D dataset remeshed to the main (cell-center) grid in CGS units (Gauss), and returns the data
array alongside its three spherical coordinate scales (:math:`r`, :math:`\theta`, :math:`\varphi`).

.. code-block:: python

   import psi_io

   with psi_io.PsiData('br002.h5') as mas_reader:
         br, r_scale, t_scale, p_scale = mas_reader.read(mesh='main', unit='cgs')


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

When reading a 3D dataset (either HDF4 or HDF5) with ``psi-io``, the returned array will have shape :math:`(\phi, \theta, r)` and
the associated dimension scales will be returned as :math:`(r, \theta, \phi)`. This ensures that the data is always represented
in a consistent manner, regardless of the underlying HDF format.

PSI Model Quantities & Coordinate Scales
========================================

PSI's modeling codes — MAS and POT3D — write output as HDF files containing 3-D
physical fields defined on a structured spherical grid
:math:`(r, \theta, \varphi)`.  Each file also stores three 1-D coordinate arrays
that describe the grid.  The tables below summarize every quantity that
``psi-io`` recognises, together with its physical meaning and units.

Coordinate scales
-----------------

Every MAS and POT3D HDF file includes three 1-D coordinate arrays that define the
spherical grid.  These scale identifiers are shared across PSI codes:

.. list-table::
   :header-rows: 1
   :widths: 10 12 28 22 16 12

   * - Key
     - Symbol
     - Physical quantity
     - Units
     - Range
     - NumPy axis
   * - ``'r'``
     - :math:`r`
     - Radial distance
     - solar radii
     - :math:`r \geq 1\,R_\odot`
     - last (``-1``)
   * - ``'t'``
     - :math:`\theta`
     - Co-latitude (pole = 0)
     - radians
     - :math:`[0,\,\pi]`
     - middle (``-2``)
   * - ``'p'``
     - :math:`\varphi`
     - Longitude
     - radians
     - :math:`[0,\,2\pi]`
     - first (``0``)

Because PSI HDF files are Fortran-ordered (column-major), the ``r`` dimension
varies fastest in memory.  When ``psi-io`` loads a 3-D dataset into NumPy
(row-major), the resulting array shape is :math:`(N_\varphi,\,N_\theta,\,N_r)`:
``r`` is the *last* axis, ``t`` is the middle axis, and ``p`` is the *first* axis.
The three returned scale arrays always correspond to those axes in that order.

MAS model quantities
--------------------

MAS outputs 19 3-D physical fields.  Code-unit values are converted to physical
(CGS) units by multiplying by MAS normalization constants; approximate physical
scales are noted in the unit column.

.. list-table::
   :header-rows: 1
   :widths: 8 12 30 30 10 10

   * - Key
     - Symbol
     - Physical quantity
     - Physical unit (CGS)
     - Type
     - Mesh code
   * - ``vr``
     - :math:`v_r`
     - Radial velocity
     - km s\ :sup:`−1` (≈ 481 km s\ :sup:`−1` per code unit)
     - vector
     - ``0b011``
   * - ``vt``
     - :math:`v_\theta`
     - Co-latitude velocity
     - km s\ :sup:`−1`
     - vector
     - ``0b101``
   * - ``vp``
     - :math:`v_\varphi`
     - Longitude velocity
     - km s\ :sup:`−1`
     - vector
     - ``0b110``
   * - ``br``
     - :math:`B_r`
     - Radial magnetic field
     - Gauss (≈ 2.2 G per code unit)
     - vector
     - ``0b100``
   * - ``bt``
     - :math:`B_\theta`
     - Co-latitude magnetic field
     - Gauss
     - vector
     - ``0b010``
   * - ``bp``
     - :math:`B_\varphi`
     - Longitude magnetic field
     - Gauss
     - vector
     - ``0b001``
   * - ``jr``
     - :math:`J_r`
     - Radial current density
     - A m\ :sup:`−2`
     - vector
     - ``0b011``
   * - ``jt``
     - :math:`J_\theta`
     - Co-latitude current density
     - A m\ :sup:`−2`
     - vector
     - ``0b101``
   * - ``jp``
     - :math:`J_\varphi`
     - Longitude current density
     - A m\ :sup:`−2`
     - vector
     - ``0b110``
   * - ``t``
     - :math:`T`
     - Single-fluid temperature
     - MK (≈ 28 MK per code unit)
     - scalar
     - ``0b111``
   * - ``te``
     - :math:`T_e`
     - Electron temperature
     - MK
     - scalar
     - ``0b111``
   * - ``tp``
     - :math:`T_p`
     - Proton temperature
     - MK
     - scalar
     - ``0b111``
   * - ``rho``
     - :math:`\rho`
     - Plasma density
     - cm\ :sup:`−3` (10\ :sup:`8` cm\ :sup:`−3` per code unit)
     - scalar
     - ``0b111``
   * - ``p``
     - :math:`p`
     - Plasma pressure
     - erg cm\ :sup:`−3`
     - scalar
     - ``0b111``
   * - ``ep``
     - :math:`e^+`
     - Forward Alfvén wave energy density
     - erg cm\ :sup:`−3`
     - scalar
     - ``0b111``
   * - ``em``
     - :math:`e^-`
     - Backward Alfvén wave energy density
     - erg cm\ :sup:`−3`
     - scalar
     - ``0b111``
   * - ``zp``
     - :math:`z^+`
     - Outward Elsässer amplitude
     - km s\ :sup:`−1`
     - scalar
     - ``0b111``
   * - ``zm``
     - :math:`z^-`
     - Inward Elsässer amplitude
     - km s\ :sup:`−1`
     - scalar
     - ``0b111``
   * - ``heat``
     - :math:`Q`
     - Volumetric heating rate
     - erg cm\ :sup:`−3` s\ :sup:`−1`
     - scalar
     - ``0b111``

POT3D model quantities
----------------------

POT3D solves for the potential magnetic field :math:`\mathbf{B} = -\nabla\Psi`
driven by a photospheric boundary magnetogram.  It outputs three spherical field
components:

.. list-table::
   :header-rows: 1
   :widths: 10 12 35 28 15

   * - Key
     - Symbol
     - Physical quantity
     - Physical unit
     - Mesh code
   * - ``br``
     - :math:`B_r`
     - Radial magnetic field
     - input magnetogram units (typically G)
     - ``0b011``
   * - ``bt``
     - :math:`B_\theta`
     - Co-latitude magnetic field
     - input magnetogram units
     - ``0b101``
   * - ``bp``
     - :math:`B_\varphi`
     - Longitude magnetic field
     - input magnetogram units
     - ``0b110``

.. note::

   POT3D mesh codes are the bitwise complement of the corresponding MAS magnetic field
   codes (``POT3D_mesh = 0b111 ^ MAS_mesh``).  Where MAS places each component on the
   face through which it is the outward normal (face-centred), POT3D places the same
   component on the opposite pair of edges (edge-centred):

   - ``br``: MAS ``0b100`` (r-face) → POT3D ``0b011`` (r-edge)
   - ``bt``: MAS ``0b010`` (θ-face) → POT3D ``0b101`` (θ-edge)
   - ``bp``: MAS ``0b001`` (φ-face) → POT3D ``0b110`` (φ-edge)

.. note::

   POT3D does not apply a normalization: values are stored in the same physical
   units as the input boundary magnetogram (most commonly Gauss, but this is
   run-dependent).  When reading POT3D output, always specify the correct unit
   explicitly; ``read(unit='physical')`` without a ``unit`` override will return
   dimensionless values.

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

**Reading & Writing File Metadata:**
    - :func:`~psi_io.psi_io.read_hdf_meta`
    - :func:`~psi_io.psi_io.read_rtp_meta`
    - :func:`~psi_io.psi_io.write_hdf_meta`

**Interpolating Data to Arbitrary Positions:**
    - :func:`~psi_io.psi_io.np_interpolate_slice_from_hdf`
    - :func:`~psi_io.psi_io.interpolate_positions_from_hdf`
    - :func:`~psi_io.psi_io.sp_interpolate_slice_from_hdf`
    - :func:`~psi_io.psi_io.interpolate_point_from_1d_slice`
    - :func:`~psi_io.psi_io.interpolate_point_from_2d_slice`

**Reading Coordinate/Mesh-Aware MHD Model Output:**
    - :func:`~psi_io.mhd_io.PsiData`

.. note::
   The HDF type (HDF4 or HDF5) is automatically determined by the file extension
   (".hdf" for HDF4 and ".h5" for HDF5) when using ``psi-io`` functions.

.. note::
   Not all PSI FORTRAN tools can read HDF4 files written by the :class:`pyhdf.SD` interface.
   If you have a problem, use the PSI tool ``hdfsd2hdf`` to convert.
