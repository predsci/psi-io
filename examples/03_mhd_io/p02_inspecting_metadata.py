"""
Opening MHD Model Files with PsiData
=====================================

Explore a MAS radial-magnetic-field file through the :func:`~psi_io.mhd_io.PsiData`
interface: inspect metadata attributes, trace the connections to :mod:`psi_io.models`,
:mod:`psi_io.mesh`, and :mod:`psi_io.units`, and observe the lazy-loading and
caching behavior.

This example demonstrates:

1. Opening a MAS HDF5 file and exploring the reader's metadata attributes.
2. The role of :mod:`psi_io.models` in defining physical quantity properties.
3. How :mod:`psi_io.mesh` encodes Yee-grid stagger positions for each quantity.
4. How :mod:`psi_io.units` supplies the MAS code-unit normalization factors.
5. Lazy loading — no data leaves the disk until explicitly requested.
6. Automatic caching of full-array reads for quick re-access.

.. note::
   :func:`~psi_io.mhd_io.PsiData` is the *only* public symbol exported by
   :mod:`psi_io.mhd_io`.  HDF4 (``.hdf``) and HDF5 (``.h5``) files are supported
   transparently; the file extension selects the I/O backend.
"""
from pathlib import Path
from psi_data import fetch_mas_data
from psi_io.mhd_io import PsiData

# %%
# **Opening a file**
#
# :func:`~psi_io.mhd_io.PsiData` takes a path to any PSI MAS or POT3D HDF file.
# No data is read at this point — only the filename is parsed and minimal HDF
# metadata is inspected to identify the quantity, units, and mesh code.
#
# To define the metadata data of the given input file, the reader follows a hierarchy of
# inference steps to determine the values of the following core attributes:
#
# ``'name'``
#     Canonical lower-case quantity identifier.
# ``'sequence'``
#     Integer time-step sequence number.
# ``'unit'``
#     Code-to-physical unit for this quantity, as an :class:`~astropy.units.Unit`
#     or a string parseable by it.
# ``'scalar'``
#     ``True`` if the quantity is a scalar field; ``False`` for vector components.
# ``'mesh'``
#     Mesh code (:data:`~psi_io.mesh.MeshCodeType`) describing data staggering.
#
# If these values are not explicitly included in the :func:`~psi_io.mhd_io.PsiData`
# constructor the reader falls back to reading the HDF metadata attributes (if present) and then
# parsing the filename according to the PSI filename schema. The
# reader then cross-references the quantity against the canonical properties defined
# in :mod:`psi_io.models` to infer the remaining metadata attributes.
#
# The ``model`` argument selects which property table to consult.  Passing
# ``model='mas'`` tells the reader to resolve metadata from the MAS quantity
# mapping; without it, the default ``model='custom'`` requires every metadata
# field to be supplied explicitly.

br_filepath = fetch_mas_data(domains="cor", variables="br").cor_br
print(f"Filename : {Path(br_filepath).name}")
reader = PsiData(br_filepath, model='mas')

# %%
# **Core metadata attributes**
#
# The :attr:`name` and :attr:`sequence` attributes are extracted from the
# filename stem using the PSI filename schema (*e.g.* ``br001001.h5`` gives
# ``name='br'``, ``sequence=1001``). Since the provided filename does not
# contain an explicit sequence number, the reader defaults to ``sequence=0``.

print(f"name      : {reader.name!r}")
print(f"sequence  : {reader.sequence}")
print(f"ndim      : {reader.ndim}")
print(f"shape     : {reader.shape}  (Nr × Nθ × Nφ in physical order)")

# %%
# **Array ordering**
#
# The :attr:`order` attribute records the in-memory layout of the dataset, mirroring
# the :attr:`~psi_io.models.ModelProps.order` field: ``'F'`` for Fortran (column-major,
# the PSI default) or ``'C'`` for C (row-major).  PSI HDF files are written Fortran-ordered,
# so the on-disk storage order is ``(Nφ, Nθ, Nr)`` — the *reverse* of the physical
# ``(r, θ, φ)`` order.  This is precisely why :attr:`shape` (reported above in physical
# ``(r, θ, φ)`` order) is the reverse of the raw HDF storage shape, and why every positional
# argument to :meth:`~psi_io.mhd_io.PsiData.read` is supplied in physical order regardless of
# how the bytes are laid out on disk.

print(f"order        : {reader.order!r}  ('F' = Fortran/column-major, PSI default)")
print(f"shape (phys) : {reader.shape}        (r, θ, φ)")
print(f"shape (HDF)  : {reader.shape[::-1]}        (Nφ, Nθ, Nr storage order)")

# %%
# **Connection to** :mod:`psi_io.models`
#
# The reader's metadata is resolved against the
# :class:`~psi_io.models.ModelProps` dataclass stored in :mod:`psi_io.models`,
# which bundles the canonical name, description, native unit, and mesh code for
# every recognised PSI quantity.  The :attr:`name` and :attr:`desc` attributes
# expose this resolved metadata directly.

print(f"name : {reader.name}")
print(f"desc : {reader.desc}")

# %%
# **Connection to** :mod:`psi_io.mesh`
#
# The :attr:`mesh` attribute is a :class:`~psi_io.mesh.Mesh` instance
# (one stagger flag per spatial axis in physical ``(r, θ, φ)`` order) that encodes
# the Yee-grid stagger position of the field quantity.
#
# For the radial magnetic field ``br``, the field is face-centred in the radial
# direction (*half*-mesh) and cell-centred in both angular directions (*main*-mesh):

from psi_io.mesh import Mesh
print(f"mesh : {reader.mesh}")
print(f"  r  → {reader.mesh[0]}")
print(f"  θ  → {reader.mesh[1]}")
print(f"  φ  → {reader.mesh[2]}")

# %%
# **Connection to** :mod:`psi_io.units`
#
# The :attr:`unit` attribute is one of the custom MAS normalization units defined
# in :mod:`psi_io.units`.  Multiplying a code-unit value by this factor converts
# it to physical CGS units.  Here, ``MAS_b`` represents approximately 2.2 Gauss
# per code unit.

from psi_io.units import MAS_b
print(f"unit     : {reader.unit}")
print(f"MAS_b    : {MAS_b}")
print(f"in Gauss : {reader.unit.to('G'):.4f}")

# %%
# **Coordinate scale readers**
#
# The :attr:`scales` attribute is a named tuple of coordinate readers whose field
# names — and their order — come from the :attr:`~psi_io.models.ModelProps.scales`
# field (default ``('r', 't', 'p')``).  This tuple *defines* the physical
# ``(r, θ, φ)`` axis ordering used everywhere else in the API: it is the order of
# the :attr:`shape`, the per-axis flags of :attr:`mesh`, and every positional
# argument accepted by :meth:`~psi_io.mhd_io.PsiData.read` and
# :meth:`~psi_io.mhd_io.PsiData.vslice`.  Each element is itself a lightweight
# reader; calling :meth:`read` on a scale returns the 1-D coordinate array as a
# :class:`~astropy.units.Quantity`.

print(f"scale order : {reader.scales._fields}  (from ModelProps.scales)")

r_scale = reader.scales.r.read()
t_scale = reader.scales.t.read()
p_scale = reader.scales.p.read()
print(f"r scale  : shape={r_scale.shape}  range=[{r_scale[0]:.5f}, {r_scale[-1]:.5f}]")
print(f"θ scale  : shape={t_scale.shape}  range=[{t_scale[0]:.5f}, {t_scale[-1]:.5f}]")
print(f"φ scale  : shape={p_scale.shape}  range=[{p_scale[0]:.5f}, {p_scale[-1]:.5f}]")

# %%
# **Lazy loading**
#
# Reading the coordinate scales above did *not* load the main data array.  The
# :attr:`data_cached` property confirms the primary dataset has not yet been
# transferred from disk:

print(f"data_cached before read : {reader.data_cached}")

# %%
# Calling :meth:`~psi_io.mhd_io.PsiData.read` with no arguments loads the full
# dataset.  Because no spatial restrictions are applied, the result is stored in
# the reader's internal cache:

data_arr, r, t, p = reader.read()
print(f"data shape  : {data_arr.shape}  (Nφ × Nθ × Nr in HDF storage order)")
print(f"data unit   : {data_arr.unit}")
print(f"data_cached : {reader.data_cached}")

# %%
# Subsequent full-array calls return the cached copy without a second disk read.
# The cache is populated only for unrestricted reads; any partial read (*i.e.* any
# call that restricts at least one axis) bypasses and never updates the cache.
