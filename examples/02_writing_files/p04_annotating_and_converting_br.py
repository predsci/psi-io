"""
Annotating and Converting the Radial Magnetic Field
=====================================================

Read the radial magnetic field dataset from HDF4, attach physically meaningful
metadata, and produce a PSI-convention HDF5 file.

This example walks through a realistic data-preparation workflow:

1. Read the radial magnetic field (Br) data and its coordinate scales from the
   example HDF4 file.
2. Attach attributes that describe the physical quantity, coordinate system, and
   units using types compatible with both HDF4 and HDF5.
3. Write an annotated HDF4 file.
4. Convert that file to HDF5 using :func:`~psi_io.psi_io.convert_psih4_to_psih5`,
   which preserves the attached attributes in the output.

.. note::
   This example can be amended to include additional or domain-specific metadata
   once the basic pipeline is confirmed to work end-to-end.
"""

import tempfile
from pathlib import Path
import numpy as np
from psi_io import read_hdf_data, write_hdf_data, read_hdf_meta, convert_psih4_to_psih5, data

# %%
# **Step 1 – Read the source HDF4 file**
#
# Fetch the example radial magnetic field file and load the primary dataset
# together with its (r, θ, φ) coordinate scales:

br_filepath = data.get_3d_data(hdf=".hdf")
print(f"Source file : {Path(br_filepath).name}")

br_data, r, t, p = read_hdf_data(br_filepath)
print(f"\nData shape : {br_data.shape}  (nφ × nθ × nr, Fortran-ordered)")
print(f"r scale    : {r.shape},  range = [{r[0]:.4f},  {r[-1]:.4f}]  R☉")
print(f"θ scale    : {t.shape},  range = [{t[0]:.4f},  {t[-1]:.4f}]  rad")
print(f"φ scale    : {p.shape},  range = [{p[0]:.4f}, {p[-1]:.4f}]  rad")

# %%
# **Step 2 – Define the attributes**
#
# All attribute values are chosen to be compatible with HDF4's SDC type system:
# ``float32`` and ``int32`` scalars for numeric quantities, and plain Python
# strings for descriptive fields. ``float16``, ``int64``, and ``uint64`` are
# **not** supported by HDF4 and must be avoided (see the
# :ref:`sphx_glr_gallery_02_writing_files_p02_writing_datasets_with_attributes.py`
# example for a full discussion of type restrictions):

br_attrs = dict(
    variable="br",
    long_name="Radial Magnetic Field",
    coord_system="Carrington",
    r_units="R_sun",
    angle_units="radians",
    b_scale=np.float32(2.2047),     # reference field scale [Gauss]
    cr_number=np.int32(2190),       # Carrington rotation number
)

# %%
# **Step 3 – Write an annotated HDF4 file**
#
# Pass the attributes as keyword arguments to :func:`~psi_io.psi_io.write_hdf_data`.
# The dataset identifier is omitted so the PSI-standard name ``'Data-Set-2'`` is
# used, which is required by :func:`~psi_io.psi_io.convert_psih4_to_psih5` in
# the next step:

_tmp = tempfile.TemporaryDirectory()
tmpdir = Path(_tmp.name)

annotated_hdf = tmpdir / "br_annotated.hdf"
write_hdf_data(annotated_hdf, br_data, r, t, p, **br_attrs)
print(f"Annotated HDF4 written : {annotated_hdf.name}")

# %%
# **Step 4 – Convert to PSI-convention HDF5**
#
# :func:`~psi_io.psi_io.convert_psih4_to_psih5` reads ``'Data-Set-2'`` from the
# HDF4 file, remaps it to ``'Data'``, and carries over the attached attributes:

annotated_h5 = tmpdir / "br_annotated.h5"
convert_psih4_to_psih5(annotated_hdf, annotated_h5)
print(f"Converted to HDF5      : {annotated_h5.name}")

# %%
# **Step 5 – Verify the round-trip**
#
# Confirm that the dataset, scales, and all attributes survived the write-and-convert
# pipeline intact:

meta = read_hdf_meta(annotated_h5)
print(f"\nDataset : {meta[0].name!r}  shape={meta[0].shape}  dtype={meta[0].type}")

print("\nAttributes:")
for key, val in meta[0].attr.items():
    print(f"  {key:<16}: {val!r}")

print("\nScales:")
for s in meta[0].scales:
    print(f"  {s.name!r:<8} shape={s.shape}  "
          f"range=[{s.imin:.4f}, {s.imax:.4f}]")

_tmp.cleanup()
