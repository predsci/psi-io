"""
Reading MHD Data: Units, Meshes, and Value-Space Slicing
==========================================================

Demonstrate the full read and vslice API of :func:`~psi_io.mhd_io.PsiData`:
loading data in different unit systems, remeshing from the half-mesh to the
main mesh, and slicing by physical coordinate value using
:class:`~astropy.units.Quantity`.

This example demonstrates:

1. Reading the full dataset in code units, physical CGS units, and a specific unit.
2. Remeshing a half-mesh field to the main mesh via the *mesh* keyword.
3. Reading a radial subset by index.
4. Slicing by physical coordinate value (``vslice``) with a bare scalar and with
   an :class:`~astropy.units.Quantity` object.
5. Combining index-space and value-space arguments in a single ``vslice`` call.

.. note::
   All positional arguments to :meth:`~psi_io.mhd_io.PsiData.read` and
   :meth:`~psi_io.mhd_io.PsiData.vslice` are interpreted in physical
   ``(r, θ, φ)`` order, even though the HDF storage order is ``(Nφ, Nθ, Nr)``.
   The returned data array shape always reflects the HDF storage order.
"""

import astropy.units as u
import numpy as np
from psi_io import data
from psi_io.units import PSI_rsun
from psi_io.mhd_io import PsiData

# %%
# **Create a reader**
#
# Open the example radial magnetic field file.  Metadata — quantity name, physical
# unit, and mesh stagger — is resolved from the filename and HDF attributes.
# No array data is transferred from disk at this step.

br_filepath = data.get_3d_data()
reader = PsiData(br_filepath, model='mas')
print(f"name  : {reader.name!r}")
print(f"shape : {reader.shape}  (Nr × Nθ × Nφ in physical order)")

# %%
# **Reading in code (native) units**
#
# By default, :meth:`~psi_io.mhd_io.PsiData.read` returns the dataset in MAS
# code units.  The values stored on disk are wrapped in the normalization unit
# object so the result is a :class:`~astropy.units.Quantity`.  Passing
# ``unit='native'`` (aliases: ``'code'``, ``'model'``, ``'psi'``) is equivalent.

data_code, r, t, p = reader.read()
print(f"unit (code) : {data_code.unit}")
print(f"shape       : {data_code.shape}")

# %%
# **Reading in physical (CGS) units**
#
# Passing ``unit='physical'`` (aliases: ``'real'``, ``'phys'``, ``'cgs'``)
# decomposes the data into CGS base units.  Any :class:`~astropy.units.Unit`-compatible
# string is also accepted — the example below converts directly to Gauss:

data_phys, r, t, p = reader.read(unit='physical')
print(f"unit (phys) : {data_phys.unit}")

data_gauss, r, t, p = reader.read(unit='Gauss')
print(f"unit (Gauss): {data_gauss.unit}")

# %%
# **Remeshing to the main mesh**
#
# ``br`` is stored on the *half*-mesh in the radial direction (Yee face-centred).
# Passing ``mesh='main'`` shifts every half-mesh axis to the main mesh by
# averaging adjacent array elements; the radial axis shrinks by one element.
# The mesh stagger for each quantity is encoded in its
# :class:`~psi_io.models.ModelProps` descriptor (see :mod:`psi_io.models`), and the
# averaging itself is performed by :func:`~psi_io.mesh.remesh_array` from
# :mod:`psi_io.mesh`.

data_main, r_main, t_main, p_main = reader.read(mesh='main', unit='Gauss')
print(f"original r points  : {r.shape[0]}")
print(f"main-mesh r points : {r_main.shape[0]}")
print(f"data shape (main)  : {data_main.shape}")

# %%
# **Reading a subset by index**
#
# Positional arguments to :meth:`~psi_io.mhd_io.PsiData.read` are index-space
# slice specifiers in physical ``(r, θ, φ)`` order.  Here we read only the
# first ten radial grid points (all θ and φ):

data_rslice, r_rslice, t_rslice, p_rslice = reader.read(slice(0, 10), unit='Gauss')
print(f"r-slice shape : {data_rslice.shape}")
print(f"r scale range : [{r_rslice[0]:.3f}, {r_rslice[-1]:.3f}]")

# %%
# **Value-space slicing with** ``vslice``
#
# :meth:`~psi_io.mhd_io.PsiData.vslice` accepts physical coordinate values as
# positional arguments alongside the usual index-space arguments.  A bare
# scalar is interpreted in the native coordinate unit (solar radii for r,
# radians for θ and φ).
#
# We first inspect the radial scale to pick a coordinate safely within the domain:

r_scale = reader.scales.r.read()
print(f"r domain : [{r_scale[0]:.5f}, {r_scale[-1]:.5f}]")

r_target = float(r_scale.value[len(r_scale) // 2])
print(f"r target : {r_target:.5f} {r_scale.unit}")

data_vs, r_vs, t_vs, p_vs = reader.vslice(r_target, unit='Gauss')
print(f"vslice shape : {data_vs.shape}  (r axis collapsed to 1)")
print(f"r value      : {r_vs[0]:.5f}")

# %%
# Passing an :class:`~astropy.units.Quantity` allows specifying the coordinate
# in any compatible unit.  Here we use :data:`~psi_io.units.PSI_rsun`, the
# solar radius constant used internally by MAS coordinate scales:

r_qty = r_scale[len(r_scale) // 2].to(u.cm)
data_vsq, r_vsq, t_vsq, p_vsq = reader.vslice(r_qty, unit='Gauss')
print(f"input (Qty) r  : {r_qty:.4e}")
print(f"vslice (Qty) r : {r_vsq[0]:.4f}")
np.testing.assert_allclose(data_vs.value, data_vsq.value, rtol=1e-5)
print("Scalar and Quantity results match.")

# %%
# **Mixing value-space and index-space arguments**
#
# ``vslice`` accepts any combination of value-space (Quantity / scalar) and
# index-space (``slice``, ``int``, ``None``) arguments, one per spatial axis
# in ``(r, θ, φ)`` order.  The example below fixes the radial coordinate at
# the chosen target value and reads only the first five co-latitude grid points:

data_mixed, r_mixed, t_mixed, p_mixed = reader.vslice(
    r_qty,           # r: value-space (PSI_rsun) → collapses to 1
    slice(0, 5),     # θ: index-space → first 5 points
    None,            # φ: all points
    unit='Gauss',
)
print(f"mixed shape : {data_mixed.shape}  (r=1; θ=5; φ=full)")
print(f"r           : {r_mixed[0]:.4f}")
print(f"θ range     : [{t_mixed[0]:.4f}, {t_mixed[-1]:.4f}]")
