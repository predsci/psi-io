"""
Interpolating Datasets
======================

Interpolate slices from standard PSI data files.

This example demonstrates how to use
    - :func:`~psi_io.psi_io.np_interpolate_slice_from_hdf`,
    - :func:`~psi_io.psi_io.interpolate_positions_from_hdf`,
"""

# sphinx_gallery_start_ignore
import os
if 'SPHINX_GALLERY_BUILD' not in os.environ:
    import matplotlib
    matplotlib.use('TkAgg')

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
from psi_io import data, read_hdf_meta, np_interpolate_slice_from_hdf, interpolate_positions_from_hdf
import matplotlib.pyplot as plt
import numpy as np

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
#
# Suppose we wish to interpolate the given dataset at a specific radial value,
# say at 30 R☉, across all latitudes and longitudes. The most efficient approach is
# through the function :func:`~psi_io.psi_io.np_interpolate_slice_from_hdf`, which uses
# vectorized NumPy computation for interpolation, while also leveraging
# :func:`~psi_io.psi_io.read_hdf_by_value` to only read in the minimum amount of data
# to perform these computations.

data_at_30 = np_interpolate_slice_from_hdf(br_data_filepath, 30, None, None)
for r in data_at_30:
    print(r.shape)

# %%
# .. note::
#    :func:`~psi_io.psi_io.np_interpolate_slice_from_hdf` results in a dimensional
#    reduction of the dataset. In this case, the original 3D dataset is reduced to a 2D
#    slice at the specified radial value.
#
# Given the metadata fetched above, we can visualize this slice as a long-lat map:

data, scale_t, scale_p = data_at_30

ax = plt.figure().add_subplot()
ax.pcolormesh(np.rad2deg(scale_p),
              90-np.rad2deg(scale_t),
              data.T,
              cmap='bwr',
              shading='gouraud',
              clim=(-5e-4,5e-4))
ax.set_aspect("equal", adjustable="box")
plt.show()

# %%
#
# Alternatively, if we wish to interpolate the dataset at arbitrary
# (radius, co-latitude, longitude) positions, we can use
# :func:`~psi_io.psi_io.interpolate_positions_from_hdf`. For example, to interpolate
# at some (contrived) trajectory:

r_positions = np.linspace(1.0, 30.0, 10)  # from 1 R☉ to 30 R☉
theta_positions = np.linspace(0.0, np.pi, 10)  # from
phi_positions = np.linspace(0.0, 2*np.pi, 10)  # from 0 to 2π
interpolated_values = interpolate_positions_from_hdf(
    br_data_filepath,
    r_positions,
    theta_positions,
    phi_positions
)

for value, position in zip(interpolated_values, zip(r_positions, theta_positions, phi_positions)):
    print(f"(radius={position[0]:.2f} R☉, "
          f"latitude={90-np.rad2deg(position[1]):.2f}°, "
          f"longitude={np.rad2deg(position[2]):.2f}°) ->\n"
          f"    B_r = {value:.4e} MAS UNITS")
