"""
Getting Started with PsiData
============================

This example is intended as a quick reference for interacting with the :func:`~psi_io.mhd_io.PsiData`
lazy reader. The nuances of this API's design and full capabilities is explored in the subsequent
examples.

.. seealso::
   :ref:`sphx_glr_gallery_03_mhd_io_p02_psidata_reading.py` for the full
   read/vslice API, and
   :ref:`sphx_glr_gallery_01_reading_files_p01_reading_file_meta_data.py`
   for metadata inspection.
"""
import warnings

from psi_io import PsiData
from psi_data import fetch_mas_data, fetch_pot3d_data

from psi_io.mhd_io import MetaDataWarning

# %%
# Fetching Data
# -------------
#
# For the purposes of this example, the :func:`~psi_data.fetch_mas_data` and
# :func:`~psi_data.fetch_pot3d_data` functions, found in the :mod:`psi_data` package, are
# used to retrieve example MAS and POT3D HDF files. These functions return a simple namespace of
# filepaths for each variable.

mas_files= fetch_mas_data(domains='cor', variables='br,vr')
pot3d_file = fetch_pot3d_data(variables='br')

print(f"Example MAS Br file        →    .../{mas_files.cor_br.name}")
print(f"Example MAS Vr file        →    .../{mas_files.cor_vr.name}")
print(f"Example POT3D Br file      →    .../{pot3d_file.br.name}")

# %%
# Creating a Reader
# -----------------
#
# The :func:`~psi_io.mhd_io.PsiData` reader API requires a filepath and ``model`` name; if the
# ``model`` name is not recognized, *i.e.* not ``mas`` or ``pot3d``, metadata cannot be inferred
# and must be supplied explicitly (either as keyword arguments to :func:`~psi_io.mhd_io.PsiData`
# or as attributes written to the HDF dataset itself - see :func:`~psi_io.psi_io.write_hdf_meta`
# for more information on how to write metadata to HDF datasets.

mas_br = PsiData(mas_files.cor_br, model='mas')
print(f"{mas_br!r}")

with warnings.catch_warnings(record=True) as w:
    pot3d_br = PsiData(pot3d_file.br, model='pot3d')
    print(f"METADATA WARNING: {w[-1].message}")

print(f"{pot3d_br!r}")

# %%
# .. warning::
#
#    POT3D applies no normalization to its output. The stored values are in whatever physical
#    unit the input photospheric magnetogram used — most commonly Gauss, but this is not encoded
#    in the file. The fallback unit for POT3D is dimensionless_unscaled (scale factor 1).
#    As a result, it is incumbent upon the user to pass in the correct unit at instantiation
#    time, or to set it manually after the fact, to ensure accurate physical interpretation and
#    unit-aware computations.

pot3d_br.unit = 'Gauss'  # manually set the unit to Gauss for accurate physical interpretation
print(f"Updated POT3D Br unit: {pot3d_br.unit}")

# %%
# Passing the Wrong Model or Mesh
# -------------------------------
#
# :func:`~psi_io.mhd_io.PsiData` does **not** verify that the declared ``model`` actually
# matches the file — it trusts ``model`` and infers every piece of metadata (quantity name and
# description, unit normalization, and mesh staggering) from that model's mapping together with
# the filename. Declaring the wrong model therefore *silently* attaches the wrong metadata.
#
# Below we deliberately open the POT3D ``br`` file as a MAS quantity. The reader happily
# constructs, but the inferred unit is now ``MAS_b`` instead of ``POT3D_b`` — a different
# normalization scale factor that would corrupt any physical-unit computation — and the mesh
# stagger is taken from MAS' ``br`` descriptor rather than POT3D's:

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    err_reader = PsiData(pot3d_file.br, model='mas')

print(f"Inferred unit (wrong) : {err_reader.unit}               (should be POT3D_b)")
print(f"Inferred mesh (wrong) : {err_reader.mesh}    (should be MAIN, HALF, HALF)")
for warning in w:
    if issubclass(warning.category, MetaDataWarning):
        print(f"METADATA WARNING: {warning.message}")

# %%
# .. warning::
#
#    **A model/mesh mismatch is mostly silent.** The wrong *unit* is applied with no warning at
#    all, so values look numerically fine but are scaled incorrectly. The only automatic signal
#    is a best-effort :class:`~psi_io.mhd_io.MetaDataWarning` raised during *scale* validation: a
#    ``t`` or ``p`` coordinate declared on the *main* mesh but holding non-zero values at the
#    inner boundary carries the tell-tale offset of a *half*-mesh axis, which is what trips the
#    warnings above. This heuristic catches some — but not all — staggering mistakes, and it
#    never inspects the data unit.
#
#    The mesh code is not cosmetic: it drives the cell-averaging performed by
#    ``read(mesh=...)`` and the half-/main-mesh offsets used to bracket coordinates in
#    :meth:`~psi_io.mhd_io.PsiData.vslice`. A wrong code averages along the wrong axis or shifts
#    the coordinate–data alignment by up to half a grid cell, again without raising an error.
#
#    When the model cannot be inferred — or is wrong — supply the correct metadata explicitly via
#    the ``unit`` and ``mesh`` keywords at instantiation (or set them afterward, as with the unit
#    above), or write the correct attributes into the HDF dataset with
#    :func:`~psi_io.psi_io.write_hdf_meta`.

# %%
# Putting It Together: Reading on the Main Mesh in Physical Units
# --------------------------------------------------------------
#
# The most common end-to-end task is simply *read the whole field, on the main mesh, in physical
# units*. Passing ``mesh='main'`` averages any half-mesh axis onto the main mesh, and
# ``unit='physical'`` decomposes the data to CGS base units. :meth:`~psi_io.mhd_io.PsiData.read`
# returns the data array together with the (remeshed) ``r``, ``t``, ``p`` coordinate scales.
#
# We also use the reader as a **context manager**. ``PsiData`` holds an open OS-level handle to
# the underlying HDF file (opened when the reader is constructed). Using it in a ``with`` block
# guarantees that handle is closed *deterministically* the moment the block exits — including if
# an exception is raised mid-read — rather than lingering until the object is garbage-collected.
# This matters when looping over many files, where leaked handles can otherwise exhaust the
# process's file-descriptor limit or keep files locked. ``__enter__`` (re)opens the handle and
# returns the reader; ``__exit__`` closes it.

with PsiData(mas_files.cor_vr, model='mas') as reader:
    data, r, t, p = reader.read(mesh='main', unit='physical')
    print(f"quantity   : {reader.name} [{reader.desc}]")
    print(f"data       : shape={data.shape}, unit={data.unit}")
    print(f"r / t / p  : {r.shape[0]} × {t.shape[0]} × {p.shape[0]} points")

# Outside the ``with`` block the underlying HDF handle has been closed, but the data we read is
# an ordinary in-memory array and remains fully usable:
print(f"max |Vr|   : {abs(data).max():.3e}")

