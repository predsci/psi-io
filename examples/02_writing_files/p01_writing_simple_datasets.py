"""
Writing Simple Datasets
=======================

Write 1D, 2D, and 3D arrays to PSI-style HDF5 or HDF4 files.

This example demonstrates the basics of writing PSI-style HDF files using
:func:`~psi_io.psi_io.write_hdf_data` for generic n-dimensional datasets, and
:func:`~psi_io.psi_io.wrhdf_1d`, :func:`~psi_io.psi_io.wrhdf_2d`,
:func:`~psi_io.psi_io.wrhdf_3d` for dimensionality-specific writing.
"""

import tempfile
from pathlib import Path
import numpy as np
from psi_io import write_hdf_data, wrhdf_1d, wrhdf_2d, wrhdf_3d, read_hdf_data

# %%
# Construct a simple 3D dataset representing a hypothetical scalar field on a
# rectilinear (r, θ, φ) grid. Following PSI's Fortran ordering convention, the
# data array is C-ordered as (nφ, nθ, nr) while the scales are supplied in
# physical dimension order (r, θ, φ):

nr, nt, np_ = 10, 20, 30
r = np.linspace(1.0, 5.0, nr, dtype=np.float32)
t = np.linspace(0.0, np.pi, nt, dtype=np.float32)
p = np.linspace(0.0, 2*np.pi, np_, dtype=np.float32)
f3d = np.random.default_rng(0).random((np_, nt, nr)).astype(np.float32)

# %%
# Write the dataset to an HDF5 file with :func:`~psi_io.psi_io.write_hdf_data`.
# Scales are passed as positional arguments in physical dimension order (r, θ, φ):

with tempfile.TemporaryDirectory() as tmpdir:
    out_h5 = Path(tmpdir) / "output.h5"
    write_hdf_data(out_h5, f3d, r, t, p)

    f_back, r_back, t_back, p_back = read_hdf_data(out_h5)
    print(f"Data shape (read back) : {f_back.shape}")
    print(f"r scale : shape={r_back.shape}, range=[{r_back[0]:.2f}, {r_back[-1]:.2f}]")
    print(f"t scale : shape={t_back.shape}, range=[{t_back[0]:.4f}, {t_back[-1]:.4f}]")
    print(f"p scale : shape={p_back.shape}, range=[{p_back[0]:.4f}, {p_back[-1]:.4f}]")
    print(f"Round-trip exact match : {np.array_equal(f3d, f_back)}")

# %%
# Writing to an HDF4 file uses the identical call – only the file extension changes.
# Dispatch to the correct backend is handled automatically based on the extension:

with tempfile.TemporaryDirectory() as tmpdir:
    out_hdf = Path(tmpdir) / "output.hdf"
    write_hdf_data(out_hdf, f3d, r, t, p)

    f_back_h4 = read_hdf_data(out_hdf, return_scales=False)
    print(f"HDF4 data shape (read back) : {f_back_h4.shape}")
    print(f"Round-trip exact match      : {np.array_equal(f3d, f_back_h4)}")

# %%
# For 1D and 2D datasets the same conventions apply – the scale(s) precede the
# data in the positional argument list:

x = np.linspace(0.0, 2*np.pi, 64, dtype=np.float32)
f1d = np.sin(x)

y = np.linspace(0.0, np.pi, 32, dtype=np.float32)
f2d = np.outer(np.sin(x), np.cos(y)).astype(np.float32)

with tempfile.TemporaryDirectory() as tmpdir:
    write_hdf_data(Path(tmpdir) / "out1d.h5", f1d, x)
    write_hdf_data(Path(tmpdir) / "out2d.h5", f2d, x, y)

    r1d = read_hdf_data(Path(tmpdir) / "out1d.h5")
    r2d = read_hdf_data(Path(tmpdir) / "out2d.h5")
    print(f"1D : data={r1d[0].shape}, scale={r1d[1].shape}")
    print(f"2D : data={r2d[0].shape}, x={r2d[1].shape}, y={r2d[2].shape}")

# %%
# The legacy dimension-specific writers (:func:`~psi_io.psi_io.wrhdf_1d`,
# :func:`~psi_io.psi_io.wrhdf_2d`, :func:`~psi_io.psi_io.wrhdf_3d`) provide a
# backward-compatible interface for existing PSI codes.
#
# .. attention::
#    The argument order of these writers differs from :func:`~psi_io.psi_io.write_hdf_data`:
#    scales are interleaved between the filename and the data array
#    (*filename*, *x*, [*y*, [*z*,]] *f*). They also default to ``sync_dtype=True``,
#    which silently casts scale arrays to match the data dtype, and always write to
#    PSI's standard dataset identifiers (``'Data-Set-2'`` for HDF4, ``'Data'`` for HDF5).

with tempfile.TemporaryDirectory() as tmpdir:
    wrhdf_1d(Path(tmpdir) / "legacy1d.h5", x, f1d)
    wrhdf_2d(Path(tmpdir) / "legacy2d.h5", x, y, f2d)
    wrhdf_3d(Path(tmpdir) / "legacy3d.h5", r, t, p, f3d)

    rl3d = read_hdf_data(Path(tmpdir) / "legacy3d.h5")
    print(f"Legacy 3D : data={rl3d[0].shape}, r={rl3d[1].shape}, t={rl3d[2].shape}, p={rl3d[3].shape}")
    print(f"Round-trip exact match : {np.array_equal(f3d, rl3d[0])}")
