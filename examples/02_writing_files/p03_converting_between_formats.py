"""
Converting Between HDF4 and HDF5
==================================

Convert PSI-style HDF files between HDF4 (.hdf) and HDF5 (.h5) formats.

This example demonstrates two conversion routines:

- :func:`~psi_io.psi_io.convert` – a general-purpose bidirectional converter that
  preserves all datasets and their attributes while keeping the original dataset
  names intact.
- :func:`~psi_io.psi_io.convert_psih4_to_psih5` – a PSI-convention-aware converter
  that additionally remaps the standard HDF4 primary dataset name (``'Data-Set-2'``)
  to its HDF5 equivalent (``'Data'``).
"""

# sphinx_gallery_start_ignore
def print_meta_summary(label, meta):
    for m in meta:
        print(f"[{label}]  dataset={m.name!r}  shape={m.shape}  dtype={m.type}")
        for s in m.scales:
            print(f"    scale={s.name!r}  shape={s.shape}  "
                  f"range=[{s.imin:.4f}, {s.imax:.4f}]")
# sphinx_gallery_end_ignore

import tempfile
from pathlib import Path
from psi_io import convert, convert_psih4_to_psih5, read_hdf_meta, data

# %%
# Fetch a real PSI HDF4 data file (the radial magnetic field cube) to use as
# the conversion source:

br_hdf4_filepath = data.get_3d_data(hdf=".hdf")
print(f"Source file : {Path(br_hdf4_filepath).name}")

# %%
# Inspect the HDF4 metadata. Note the PSI-standard dataset name ``'Data-Set-2'``
# and scale names ``'fakeDim0'``, ``'fakeDim1'``, ``'fakeDim2'``:

source_meta = read_hdf_meta(br_hdf4_filepath)
# sphinx_gallery_start_ignore
print_meta_summary("HDF4 source", source_meta)
# sphinx_gallery_end_ignore

# %%
# **Generic conversion** with :func:`~psi_io.psi_io.convert`
#
# The generic converter reads every non-scale dataset in the source file and writes
# it to the output file under the **same name**. For a PSI HDF4 file, this means
# the primary dataset is preserved as ``'Data-Set-2'`` in the resulting HDF5 file.
# All associated scale datasets and attributes are also carried over:

with tempfile.TemporaryDirectory() as tmpdir:
    out_generic = Path(tmpdir) / "br_generic.h5"
    convert(br_hdf4_filepath, out_generic)

    generic_meta = read_hdf_meta(out_generic)
    # sphinx_gallery_start_ignore
    print_meta_summary("convert() → HDF5", generic_meta)
    # sphinx_gallery_end_ignore

# %%
# .. note::
#    Because :func:`~psi_io.psi_io.convert` preserves dataset names verbatim, the
#    resulting HDF5 file has a ``'Data-Set-2'`` dataset rather than the ``'Data'``
#    dataset expected by :func:`~psi_io.psi_io.read_hdf_data` and other ``psi-io``
#    reading routines by default. Use :func:`~psi_io.psi_io.convert_psih4_to_psih5`
#    when PSI-convention HDF5 naming is required.

# %%
# **PSI-convention conversion** with :func:`~psi_io.psi_io.convert_psih4_to_psih5`
#
# This converter is designed specifically for PSI-style HDF4 files. It reads the
# ``'Data-Set-2'`` dataset and writes it as ``'Data'`` in the output HDF5 file,
# matching the naming convention expected by all ``psi-io`` reading routines. Scale
# names are also updated from ``'fakeDimN'`` to ``'dimN'``:

with tempfile.TemporaryDirectory() as tmpdir:
    out_psi = Path(tmpdir) / "br_psi.h5"
    convert_psih4_to_psih5(br_hdf4_filepath, out_psi)

    psi_meta = read_hdf_meta(out_psi)
    # sphinx_gallery_start_ignore
    print_meta_summary("convert_psih4_to_psih5() → HDF5", psi_meta)
    # sphinx_gallery_end_ignore

# %%
# **HDF5 → HDF4 conversion**
#
# :func:`~psi_io.psi_io.convert` is bidirectional. Passing an HDF5 file as input
# and an HDF4 path as output performs the reverse conversion. When ``ofile`` is
# omitted, the output file is placed alongside the input file with its extension
# swapped:

with tempfile.TemporaryDirectory() as tmpdir:
    br_h5_filepath = data.get_3d_data(hdf=".h5")
    out_hdf = Path(tmpdir) / "br_back.hdf"
    convert(br_h5_filepath, out_hdf, strict=False)

    back_meta = read_hdf_meta(out_hdf)
    # sphinx_gallery_start_ignore
    print_meta_summary("convert() → HDF4", back_meta)
    # sphinx_gallery_end_ignore
