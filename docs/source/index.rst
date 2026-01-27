psi-io Documentation
=====================


``psi-io`` is a Python package developed by Predictive Science Inc. `(PSI) <https://www.predsci.com>`_ for
interacting with the PSI HDF data ecosystem. It provides a unified interface for reading and writing
HDF4 and HDF5 files, as well as tools for reading portions of datasets and interpolating data to
arbitrary positions.

The primary goal of ``psi-io`` is to abstract away the idiosyncrasies involved with reading and writing
HDF4 and HDF5 files, *viz.* those that adhere to PSI conventions. By providing a consistent interface
for working with both HDF4 and HDF5 files, ``psi-io`` enables users to seamlessly interact with PSI's
data products without needing to worry about the underlying file format or conventions used.

To get started, visit the :ref:`installation` guide. For a more in-depth overview of the conventions used
throughout PSI's data products – as well as an inventory of the functions provided through ``psi-io`` –
consult the :ref:`overview` page.

.. toctree::
    :hidden:

    API <api/index>
    Guide <guide/index>
    Examples <gallery/index>
