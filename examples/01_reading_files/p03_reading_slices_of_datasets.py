"""
Reading Subsets of Datasets
===========================

Read in subsets of datasets and scales from standard PSI data files.

This example demonstrates how to use
    - :func:`~psi_io.psi_io.read_hdf_by_index`,
    - :func:`~psi_io.psi_io.read_hdf_by_value`,
    - :func:`~psi_io.psi_io.read_hdf_by_ivalue`

These functions allow users to read in subsets of datasets based on either index positions
or scale values. This is particularly useful for large datasets where only a portion
of the data is needed for analysis.

.. note::
   To reiterate, these functions **do not** read in the entire dataset; instead, they
   extract only the specified slices based on the provided indices or values. The only
   caveat to this is that when using :func:`~psi_io.psi_io.read_hdf_by_value`, the
   function's scales are read in to determine the appropriate indices corresponding
   to the requested values.
"""

# sphinx_gallery_start_ignore
def pretty_print_meta(filename, meta):
    print(f"Datasets for file: {filename}")

    print(f"+-----------------+-----------------+-----------------+")
    print(f"| Name            | Shape           | Type            |")
    print(f"+-----------------+-----------------+-----------------+")
    for idx, m in enumerate(meta):
        name = m.name.ljust(15)
        shape = str(m.shape).ljust(15)
        dtype = str(m.type).ljust(15)
        print(f"| {name} | {shape} | {dtype} |")
    print(f"+-----------------+-----------------+-----------------+\n")

    print(f"Scales for file: {filename}")
    if any(len(m.scales) > 0 for m in meta):

        print(f"+------------+------------+------------+------------+------------+------------+------------+")
        print(f"| Dataset    | Dim        | Name       | Shape      | Type       | Min        | Max        |")
        print(f"+------------+------------+------------+------------+------------+------------+------------+")
        for m in meta:
            for idx, d in enumerate(m.scales):
                dset = m.name.ljust(10)
                dimname = str(d.name).ljust(10)
                shape = str(d.shape).ljust(10)
                dtype = str(d.type).ljust(10)
                imin = str(d.imin).ljust(10)
                imax = str(d.imax).ljust(10)
                print(f"| {dset} | {idx:<10} | {dimname} | {shape} | {dtype} | {imin} | {imax} |")
        print(f"+------------+------------+------------+------------+------------+------------+------------+")
    else:
        print("     No scales found in this file.")

# sphinx_gallery_end_ignore

from pathlib import Path
from psi_io import data, read_hdf_meta, read_hdf_by_index, read_hdf_by_value, read_hdf_by_ivalue

# %%
# Read in the metadata for a 3D data file (the radial magnetic field data).

br_data_filepath = data.get_3d_data()
metadata = read_hdf_meta(br_data_filepath)
# sphinx_gallery_start_ignore
pretty_print_meta(Path(br_data_filepath).name, metadata)
# sphinx_gallery_end_ignore

# %%
# Using this metadata, we can see that the dataset is a standard PSI 3D data file
# with scales:
#   - **radius** (1st dimension), in units of solar radii (R☉),
#   - **co-latitude** (2nd dimension), in radians,
#   - **longitude** (3rd dimension), in radians.

# %%
# To read in a subset of this data based on index positions, we can use
# :func:`~psi_io.psi_io.read_hdf_by_index`. For example, to read in only
# the first position along the radius dimension (index 0) and all
# positions along the co-latitude and longitude dimensions, we can do:

subset_at_r0 = read_hdf_by_index(br_data_filepath, 0, None, None)
for r in subset_at_r0:
    print(r.shape)

# %%
# Or to read in the first five positions along the phi dimension (indices 0 to 4):

subset_phi0_4 = read_hdf_by_index(br_data_filepath, None, None, (0, 5))
for r in subset_phi0_4:
    print(r.shape)

# %%
# Or to read a single position at the midpoint of each dimension:

mid_indices = (dim.shape[0]//2 for dim in metadata[0].scales)
subset_midpoint = read_hdf_by_index(br_data_filepath, *mid_indices)
for r in subset_midpoint:
    print(r.shape)

# %%
# Alternatively, to read in subsets based on scale values, we can use
# :func:`~psi_io.psi_io.read_hdf_by_value`. For example, to read in data
# at a radius of 2 R☉ and all co-latitude and longitude values, we can do:

subset_at_r2 = read_hdf_by_value(br_data_filepath, 2.0, None, None)
for r in subset_at_r2:
    print(r.shape)

# %%
# .. note::
#    When using :func:`~psi_io.psi_io.read_hdf_by_value`, the minimum number
#    of elements returned along each dimension is two, *viz.* the values that
#    bracket the requested value. Therefore, in the above example, the radius
#    scale returned will contain two values: one just below 2.0 R☉ and one just above.
#    This allows for interpolation between scale values if desired (using the
#    minimum amount of data necessary).

_, rscale, *_ = subset_at_r2
print(rscale)

# %%
# Suppose we haven't read in the metadata and don't know the exact scale values
# (or, alternatively, the desired scale values don't exist in the dataset). In that case,
# we can use :func:`~psi_io.psi_io.read_hdf_by_ivalue`, which reads in data based on
# the nearest integer value of the requested scale values.
#
# .. note::
#    While this function performs a seemingly trivial task (one that is replicable through
#    :func:`~psi_io.psi_io.read_hdf_by_index`), it is provided for user convenience
#    and for interpolation routines.

subset_ivalue = read_hdf_by_ivalue(br_data_filepath, 0.5, None, None)
for r in subset_ivalue:
    print(r.shape)
_, rscale_ivalue, *_ = subset_ivalue
