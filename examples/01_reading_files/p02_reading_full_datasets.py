"""
Reading Full Datasets
=====================

Read in complete datasets and scales from standard PSI data files.

This example demonstrates how to use :func:`~psi_io.psi_io.read_hdf_data` to read
complete datasets and their associated scales from various standard PSI data files.
Additionally, it shows how to utilize :func:`~psi_io.psi_io.rdhdf_1d`,
:func:`~psi_io.psi_io.rdhdf_2d`, and :func:`~psi_io.psi_io.rdhdf_3d` for explicitly
reading in datasets of a specific dimensionality.
"""

import numpy as np
from psi_io import read_hdf_data, rdhdf_2d, rdhdf_3d, data

# %%
# Read in example 3D data file (the radial magnetic field data).

br_data_filepath = data.get_3d_data()
result_with_scales = read_hdf_data(br_data_filepath)
for d in result_with_scales:
    print(d.shape)

# %%
# When reading in an HDF file with :func:`~psi_io.psi_io.read_hdf_data`,
# the returned result is a tuple where the first element is the main dataset –
# which for PSI data products is the **Fortran-ordered** array – and (if present)
# the subsequent elements are the scales associated with each dimension of
# the dataset.
#
# If one wishes to only read in the main dataset without the scales – *e.g.* iterating
# over a time-dependent simulation where one knows, *a priori*, the scale values – one can
# use the ``return_scales=False`` keyword argument:

result_no_scales = read_hdf_data(br_data_filepath, return_scales=False)

# %%
# .. attention::
#    When setting ``return_scales=False``, only the main dataset is returned. Therefore,
#    the result is no longer a tuple, but rather the dataset array itself.
#
# Given these conditions, we can check that the above arrays – ``result_no_scales`` and
# ``result_with_scales[0]`` – are identical:

print(f"Dataset: {result_no_scales.shape}")
print(f"result_with_scales[0] == result_no_scales: {np.allclose(result_with_scales[0], result_no_scales)}")


# %%
# Alternatively, we can use the dimensionality-specific readers to read in
# the same data. Here, we use :func:`~psi_io.psi_io.rdhdf_3d` to read in
# the radial magnetic field data file.
#
# .. attention::
#    These readers – *viz.* :func:`~psi_io.psi_io.rdhdf_1d`, :func:`~psi_io.psi_io.rdhdf_2d`,
#    and :func:`~psi_io.psi_io.rdhdf_3d` – are primarily provided for
#    backward compatibility with older PSI data reading code. Their output format
#    differs in that the scales are returned first, followed by the main dataset.

result_legacy = rdhdf_3d(br_data_filepath)
for d in result_legacy:
    print(d.shape)

# %%
# These readers (unlike :func:`~psi_io.psi_io.read_hdf_data`) also enforce the dimensionality
# of the dataset being read. For example, attempting to read this 3D data file with
# :func:`~psi_io.psi_io.rdhdf_2d` will raise an error:

try:
    rdhdf_2d(br_data_filepath)
except ValueError as e:
    print(f"Error: {e}")
