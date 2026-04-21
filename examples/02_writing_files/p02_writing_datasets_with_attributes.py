"""
Writing Datasets with Attributes
=================================

Attach metadata attributes to HDF5 and HDF4 datasets, and understand datatype
restrictions specific to each format.

This example demonstrates how to attach key-value metadata attributes to PSI-style
HDF datasets using the ``**kwargs`` interface of :func:`~psi_io.psi_io.write_hdf_data`.
It also illustrates the datatype restrictions imposed by the HDF4 format and how to
handle attribute write failures gracefully using the ``strict`` parameter.
"""

import tempfile
from pathlib import Path
import numpy as np
from psi_io import write_hdf_data, read_hdf_meta

# %%
# Construct a simple 2D dataset with coordinate scales, representing a binary
# coronal hole map in (θ, φ):

nt, np_ = 180, 360
t = np.linspace(0.0, np.pi, nt, dtype=np.float32)
p = np.linspace(0.0, 2*np.pi, np_, dtype=np.float32)
chmap = np.zeros((np_, nt), dtype=np.float32)

# %%
# **Writing attributes to an HDF5 file**
#
# Attributes are passed as keyword arguments to :func:`~psi_io.psi_io.write_hdf_data`.
# For HDF5 files, attribute values are stored via :mod:`h5py`, which accepts most
# Python and NumPy types without restriction:

with tempfile.TemporaryDirectory() as tmpdir:
    out_h5 = Path(tmpdir) / "chmap.h5"
    write_hdf_data(out_h5, chmap, t, p,
                   description="Binary coronal hole map",
                   source="synthetic",
                   resolution=np.float32(1.0),
                   cr_number=np.int32(2190))

    meta = read_hdf_meta(out_h5)
    print(f"Dataset : {meta[0].name!r},  shape={meta[0].shape}")
    print("Attributes:")
    for key, val in meta[0].attr.items():
        print(f"  {key!r:<16}: {val!r}")

# %%
# .. note::
#    Prefer explicit NumPy scalar types (*e.g.* ``np.float32``, ``np.int32``) over
#    bare Python ``float`` or ``int`` when precision on disk matters. Python ``float``
#    is stored as ``float64``; Python ``int`` is stored as ``int64``.

# %%
# **HDF4 datatype restrictions – primary data and scales**
#
# HDF4 supports only a restricted set of numeric types, mapped through the
# ``SDC`` type system. The supported types are:
#
# +----------+------------------------------------------+
# | Kind     | Supported itemsizes                      |
# +==========+==========================================+
# | integer  | ``int8``, ``int16``, ``int32``           |
# +----------+------------------------------------------+
# | unsigned | ``uint8``, ``uint16``, ``uint32``        |
# +----------+------------------------------------------+
# | float    | ``float32``, ``float64``                 |
# +----------+------------------------------------------+
# | string   | Unicode and byte strings                 |
# +----------+------------------------------------------+
#
# The types ``float16``, ``int64``, and ``uint64`` have no SDC equivalent.
# Attempting to write a ``float16`` primary dataset to an HDF4 file raises a
# :exc:`KeyError` immediately, before any scales or attributes are written:

with tempfile.TemporaryDirectory() as tmpdir:
    out_hdf = Path(tmpdir) / "bad_data_dtype.hdf"
    try:
        write_hdf_data(out_hdf, chmap.astype(np.float16), t, p)
    except KeyError as e:
        print(f"KeyError raised for float16 data: {e}")

# %%
# The same restriction applies to scale arrays. Attempting to pass an ``int64``
# scale to an HDF4 file also raises a :exc:`KeyError`:

with tempfile.TemporaryDirectory() as tmpdir:
    out_hdf = Path(tmpdir) / "bad_scale_dtype.hdf"
    try:
        write_hdf_data(out_hdf, chmap, t.astype(np.float16), p)
    except KeyError as e:
        print(f"KeyError raised for float16 scale: {e}")

# %%
# **HDF4 datatype restrictions – attributes**
#
# The same SDC type constraints apply to attribute values. Passing an ``int64``
# or ``float16`` attribute value to an HDF4 file raises a :exc:`KeyError`.
# Unlike data and scale failures – which always propagate immediately – attribute
# failures are gated by the ``strict`` parameter:

with tempfile.TemporaryDirectory() as tmpdir:
    out_hdf = Path(tmpdir) / "bad_attr_strict.hdf"
    try:
        write_hdf_data(out_hdf, chmap, t, p,
                       cr_number=np.int64(2190))    # int64: no SDC equivalent
    except KeyError as e:
        print(f"KeyError raised for int64 attribute (strict=True): {e}")

# %%
# When ``strict=False`` is set, attribute write failures are downgraded to printed
# warnings. Compatible attributes are still written; only the offending attribute
# is skipped. This is useful when converting files from formats that use wider
# integer or float types than HDF4 supports:

with tempfile.TemporaryDirectory() as tmpdir:
    out_hdf = Path(tmpdir) / "partial_attrs.hdf"
    write_hdf_data(out_hdf, chmap, t, p,
                   description="Binary coronal hole map",  # str    : valid
                   cr_number=np.int64(2190),               # int64  : skipped with warning
                   resolution=np.float32(1.0),             # float32: valid
                   strict=False)

    meta = read_hdf_meta(out_hdf)
    print("Attributes written (incompatible attributes were skipped):")
    for key, val in meta[0].attr.items():
        print(f"  {key!r:<16}: {val!r}")

# %%
# .. note::
#    ``strict`` also controls behavior for HDF5 attribute writes; a :exc:`TypeError`
#    is raised (or warned) when a value cannot be stored as an HDF5 attribute – for
#    example, if the value is an arbitrary Python object that :mod:`h5py` does not
#    know how to serialize.
