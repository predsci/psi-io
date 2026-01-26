.. _installation:

Installation
============

.. attention::

    We highly recommend using a virtual environment to manage your Python packages and avoid conflicts with other
    projects. For the best results, we recommend using ``conda`` – *via* Miniforge (preferred), Miniconda, or Anaconda
    – to create and manage your virtual environments.

To get started with **psi-io**, you can install it directly from PyPI:

.. code-block:: bash

    pip install psi-io


Required Dependencies
----------------------
- `Python >= 3.8 <https://www.python.org/>`_
- `numpy <https://numpy.org/>`_
- `h5py <https://docs.h5py.org/en/stable/>`_

Optional Dependencies
----------------------
- `matplotlib <https://matplotlib.org/>`_ (*viz.* for running examples)
- `pooch <https://www.fatiando.org/pooch/>`_ (*viz.* for running examples)
- `pyhdf <https://pypi.org/project/pyhdf/>`_ (for HDF4 support)
- `scipy <https://scipy.org/>`_ (for advanced interpolation methods)
