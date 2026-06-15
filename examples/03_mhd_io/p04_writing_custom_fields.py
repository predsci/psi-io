"""
Computing a Derived Field and Round-Tripping it as a Custom Model
=================================================================

Build a derived quantity from several MAS fields, write it to a new HDF file with
the metadata attributes that :func:`~psi_io.mhd_io.PsiData` needs, and read it back
as a ``model='custom'`` reader — relying entirely on the attributes stored in the
file to reconstruct the metadata.

This example demonstrates:

1. Reading the three magnetic-field components ``br``, ``bt``, ``bp`` onto a common
   mesh so they can be combined.
2. Computing the field magnitude :math:`|B| = \\sqrt{B_r^2 + B_\\theta^2 + B_\\phi^2}`.
3. Writing the result with :func:`~psi_io.psi_io.write_hdf_data`, attaching the
   metadata attributes (``name``, ``desc``, ``unit``, ``scalar``, ``mesh``,
   ``order``, ``scales``) that describe a PSI-style dataset.
4. Reading it back with the default ``model='custom'`` and confirming the file's
   own attributes supply every piece of metadata — no model table required.

.. note::
   The three components live on *different* half-meshes (``br`` is half-mesh in
   :math:`r`, ``bt`` in :math:`\\theta`, ``bp`` in :math:`\\phi`).  They cannot be
   combined element-wise until they share a grid, so every component is first
   remeshed to the common **main** mesh.
"""

import tempfile
from pathlib import Path

import numpy as np
import warnings

from psi_data import fetch_mas_data
from psi_io import PsiData, write_hdf_data
from psi_io.mhd_io import MetaDataWarning

# %%
# Reading the components onto a common mesh
# -----------------------------------------
#
# Each magnetic-field component is stored on its own staggered (Yee) grid: ``br`` is
# face-centred in :math:`r`, ``bt`` in :math:`\theta`, and ``bp`` in :math:`\phi`.
# Their native shapes therefore differ by one point along the staggered axis. Passing
# ``mesh='main'`` to :meth:`~psi_io.mhd_io.PsiData.read` averages each half-mesh axis
# onto the main mesh, so all three end up on the *same* grid and can be combined.
# We read every component in Gauss.

mas_files = fetch_mas_data(domains='cor', variables='br,bt,bp')

components = {}
scales = None
for comp in ('br', 'bt', 'bp'):
    reader = PsiData(getattr(mas_files, f'cor_{comp}'), model='mas')
    native_mesh = reader.mesh
    data, r, t, p = reader.read(mesh='main', unit='Gauss')
    components[comp] = data
    scales = (r, t, p)  # main-mesh scales are shared across components
    print(f"{comp}: native mesh={native_mesh}  →  main-mesh shape={data.shape}")

print(f"shapes aligned: {components['br'].shape == components['bt'].shape == components['bp'].shape}")

# %%
# Computing the field magnitude
# -----------------------------
#
# With every component on the common main mesh, the magnitude is a plain
# element-wise reduction.  Because the inputs are :class:`~astropy.units.Quantity`
# objects in Gauss, the result carries Gauss units automatically.

bmag = np.sqrt(components['br'] ** 2 + components['bt'] ** 2 + components['bp'] ** 2)
print(f"|B| : shape={bmag.shape}, unit={bmag.unit}, max={bmag.max():.3f}")

# %%
# Writing the derived field with metadata
# ---------------------------------------
#
# :func:`~psi_io.psi_io.write_hdf_data` writes the data array and its coordinate
# scales; any extra keyword arguments are attached to the dataset as HDF attributes.
# To make the file self-describing for a later ``model='custom'`` read, we attach
# the full metadata schema:
#
# ``name`` / ``desc``
#     Quantity identifier and human-readable description.
# ``unit``
#     The physical unit of the stored values (``'G'``).  A custom reader has no
#     normalization table to fall back on, so the unit *must* be stored here.
# ``scalar``
#     ``True`` — :math:`|B|` is a scalar field.
# ``mesh``
#     Stagger code. We write the ``'main'`` shorthand string rather than an integer
#     code: :meth:`~psi_io.mesh.Mesh.parse` accepts the ``'main'``/``'half'``
#     shorthands and per-axis token sequences, but an integer attribute read back
#     from HDF returns as a NumPy integer, which the parser does not accept.
# ``order``
#     ``'F'`` — the data array follows PSI's Fortran (column-major) convention.
# ``scales``
#     The ordered axis-name tuple ``('r', 't', 'p')``.  The scale readers use these
#     names to resolve their own units from :mod:`psi_io.units`.
#
# Note the data array is written in HDF storage order ``(Nφ, Nθ, Nr)`` with the
# scales supplied in physical ``(r, t, p)`` order, exactly as the reader returned them.

r, t, p = scales
outfile = Path(tempfile.gettempdir()) / "bmag_custom.h5"
write_hdf_data(
    outfile,
    bmag.value,
    r.value, t.value, p.value,
    name='bmag',
    desc='Magnetic field magnitude',
    unit='G',
    scalar=True,
    mesh='main',
    order='F',
    scales=('r', 't', 'p'),
)
print(f"wrote {outfile.name}")

# %%
# Reading it back as a custom model
# ---------------------------------
#
# :func:`~psi_io.mhd_io.PsiData` defaults to ``model='custom'``, which infers *no*
# metadata from any model table — it relies solely on the dataset attributes (and,
# for anything still missing, explicit keyword arguments).  Because we stored the
# complete schema above, the custom reader reconstructs the quantity name,
# description, unit, mesh, ordering, and coordinate scales entirely from the file.
#
# A single :class:`~psi_io.mhd_io.MetaDataWarning` is expected: ``'custom'`` is
# deliberately not a recognized model, so the reader notes that it is trusting the
# file-supplied metadata rather than a model table. This is informational, not an error.

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    custom = PsiData(outfile)  # model='custom' by default

print(repr(custom))
print(f"model  : {custom.model}")
print(f"unit   : {custom.unit}   (read from the file's 'unit' attribute)")
print(f"mesh   : {custom.mesh}")
print(f"order  : {custom.order}")
print(f"scales : {[s.name for s in custom.scales]}  units={[str(s.unit) for s in custom.scales]}")
for w in caught:
    if issubclass(w.category, MetaDataWarning):
        print(f"METADATA WARNING: {w.message}")

# %%
# Confirming the round trip
# -------------------------
#
# Reading the data back reproduces both the array and its coordinate scales exactly,
# confirming that the attributes written to the file carried all the information the
# custom reader needed.

data_rt, r_rt, t_rt, p_rt = custom.read()
print(f"data round-trip max |Δ| : {np.abs(data_rt.value - bmag.value).max():.3e}")
print(f"r-scale round-trip max |Δ|: {np.abs(r_rt.value - r.value).max():.3e}")
print(f"recovered unit / shape    : {data_rt.unit}, {data_rt.shape}")
