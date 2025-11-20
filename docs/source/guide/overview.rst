.. _overview:

Overview
========

``mapflpy`` centers around the :class:`~mapflpy.tracer.Tracer` and :class:`~mapflpy.tracer.TracerMP` classes
(both of which inherit from the base :class:`~mapflpy.tracer._Tracer` class interface). The
:class:`~mapflpy.tracer.Tracer` class is for single-threaded tracing, while the :class:`~mapflpy.tracer.TracerMP`
class is for multi-processed tracing using python's :py:mod:`multiprocessing` module.

The ``mapflpy_fortran`` shared-object is built from ``mapfl`` Fortran source code using
:mod:`numpy.f2py` and `Meson <https://mesonbuild.com/>`_. The Fortran routines are not
intended to be called directly by end-users, but rather through the :class:`~mapflpy.tracer.Tracer` and
:class:`~mapflpy.tracer.TracerMP` classes.

``mapfl`` itself is not designed for object-oriented programming; rather, it is generally
called from the command line using a ``.in`` file to specify input arguments, and relies
on global variables to hold state information.

This package wraps the Fortran routines in a "pythonic" interface, allowing for a
more flexible approach to tracing fieldlines. Until there is time (...) to refactor the
Fortran code into a more modular design, these wrapper classes are the most feasible way to
allow a broader audience to use the Fortran tracer.

Tracer vs. TracerMP
-------------------

The :class:`~mapflpy.tracer.Tracer` class directly imports the ``mapflpy_fortran`` cross-compiled Fortran
module. Since imports in Python are singletons, only one instance of the Fortran module can
exist within a process at a time.

As such, the :class:`~mapflpy.tracer.Tracer` class also enforces a singleton pattern (due to the fact that
the Fortran routines rely on global state variables, and multiple instances of the
:class:`~mapflpy.tracer.Tracer` class would lead to conflicts and race-conditions).

The :class:`~mapflpy.tracer.TracerMP` class, on the other hand, spawns multiple processes using the
:py:mod:`multiprocessing` module. Each instance of the :class:`~mapflpy.tracer.TracerMP` class creates
and links to a distinct subprocess, which imports its own instance of the Fortran module. A two-way pipe
is used to communicate between the main process and the subprocess. *This allows multiple instances of the
Fortran routines to be used simultaneously, without conflicts (at the cost of inter-process communication
overhead).* In an attempt to limit this overhead, magnetic field data is passed as filepaths.
These filepaths are passed to the respective subprocess over the pipe, then loaded into memory by the
subprocess itself.

Tracer as Dictionary
--------------------

The base :class:`~mapflpy.tracer._Tracer` class implements the :class:`~collections.abc.MutableMapping` interface,
allowing for **Tracer** and **TracerMP** instances to behave like dictionaries. Under the hood,
the "mapfl.in" parameters are stored in a :class:`~collections.ChainMap` which contains a set of default
parameters, as well as any user-specified changes (or additions) to the parameters.

.. code-block:: python
   :linenos:

    from mapflpy.tracer import Tracer

    tracer = Tracer()
    current_params = dict(tracer)                 # get current parameters as a dictionary

    tracer['verbose_'] = True                     # set verbose_ parameter to True
    tracer.update(ds_min_=0.00001, ds_max_=10.1)  # update multiple parameters at once
    updated_params = dict(tracer)                 # get updated parameters as a dictionary

    tracer.clear()                                # reset parameters to defaults

The one exception to this is the magnetic field data itself. Because this data is
handled differently in the :class:`~mapflpy.tracer.Tracer` and :class:`~mapflpy.tracer.TracerMP`
classes, the magnetic field data should be set using the :attr:`~mapflpy.tracer._Tracer.br`,
:attr:`~mapflpy.tracer._Tracer.bt`, and :attr:`~mapflpy.tracer._Tracer.bp` properties (or,
alternatively, the :meth:`~mapflpy.tracer._Tracer.load_fields` method).

.. warning::
    When using the :class:`~mapflpy.tracer.TracerMP` class, the magnetic field data
    **MUST** be set using filepaths. Numpy arrays are **NOT** supported, *e.g.*

.. code-block:: python
   :linenos:

   from mapflpy.tracer import Tracer
   from psi_io import read_hdf_by_value

   # load magnetic field data from HDF files
   # read_hdf_by_value returns the data array followed by any scale arrays
   # e.g. values, r_scale, t_scale, p_scale
   br, *br_scales = read_hdf_by_value(ifile="br_file.h5")
   bt, *bt_scales = read_hdf_by_value(ifile="bt_file.h5")
   bp = "bp_file.h5"

   tracer = Tracer()
   tracer.br = br, *br_scales
   tracer.bt = bt, *bt_scales
   tracer.bp = bp  # can be a filepath

or

.. code-block:: python
   :linenos:

   from mapflpy.tracer import TracerMP

   tracer_mp = TracerMP()
   tracer_mp.br = "br_file.h5"
   tracer_mp.bt = "bt_file.h5"
   tracer_mp.bp = "bp_file.h5"

Tracing Fieldines
-----------------

Once a :class:`~mapflpy.tracer.Tracer` or :class:`~mapflpy.tracer.TracerMP`
instance has been created, and the magnetic field data has been set, field lines
can be traced using the :meth:`~mapflpy.tracer._Tracer.trace` method.

.. note::
   *Prior to tracing fieldlines the* :meth:`~mapflpy.tracer._Tracer.run` *method
   must be called viz. to populate any changes made to the input params or magnetic
   field data.*

   With that said, the current "staleness" of the tracer instance can be checked using
   the :attr:`~mapflpy.tracer._Tracer.stale` property *i.e.* whether there have been
   any changes to the input parameters or magnetic field data since the last time the
   last :meth:`~mapflpy.tracer._Tracer.run` method was called.

.. code-block:: python
   :linenos:

   from mapflpy.tracer import Tracer

   tracer = Tracer()
   print(tracer.stale)         # True, since nothing has been run yet
   tracer.run()                # run the tracer to initialize
   print(tracer.stale)         # False, since tracer is up-to-date
   tracer['ds_min_'] = 0.0001  # change a parameter
   print(tracer.stale)         # True, since a parameter has changed

.. note::
   If :meth:`~mapflpy.tracer._Tracer.trace` is called while the tracer is stale,
   :meth:`~mapflpy.tracer._Tracer.run` will be called automatically.

.. code-block:: python
   :linenos:

   from mapflpy.tracer import Tracer
   import numpy as np

   lps = [
       np.full(10, 1.01),              # r launch points [R_sun]
       np.linspace(0, np.pi, 10),      # theta launch points [rad]
       np.zeros(10)                    # phi launch points [rad]
   ]

   tracer = Tracer()
   tracer.load_fields( ... )            # load magnetic field data
   tracer.trace(lps, buffer_size=1000)  # will call run() if stale

Traces can be performed "forward" or "backward" by calling the
:meth:`~mapflpy.tracer._Tracer.set_tracing_direction` method with either
``'f'`` (forward) or ``'b'`` (backward).

*Two separate calls must be made to trace in both directions. The resulting
trace geometry must then be combined manually.* With that said, a utility function
:func:`~mapflpy.utils.combine_fwd_bwd_traces` is provided to help with this
common use case.

.. code-block:: python
   :linenos:

   from mapflpy.tracer import Tracer
   from mapflpy.utils import combine_fwd_bwd_traces

   tracer = Tracer()
   lps = [ ... ]                           # launch points
   tracer.load_fields( ... )               # load magnetic field data
   tracer.set_tracing_direction('f')       # set to forward tracing
   fwd_traces = tracer.trace(lps)
   tracer.set_tracing_direction('b')       # set to backward tracing
   bwd_traces = tracer.trace(lps)

   # combine the two traces into one, correctly ordered
   combined = combine_fwd_bwd_traces(fwd_traces, bwd_traces)

Example of iterating through a series of states within a time-dependent run:

.. code-block:: python
   :linenos:

   from mapflpy.tracer import Tracer
   import numpy as np

   traces =[]
   states = range(10, 20)

   lps = [
       np.full(10, 1.01),              # r launch points [R_sun]
       np.linspace(0, np.pi, 10),      # theta launch points [rad]
       np.zeros(10)                    # phi launch points [rad]
   ]

   tracer = Tracer()
   for state in states:
       tracer.load_fields(
           br=f"br_0000{state}.h5",
           bt=f"bt_0000{state}.h5",
           bp=f"bp_0000{state}.h5"
       )
       traces.append(tracer.trace(lps, buffer_size=1000))  # will call run() if stale

The result of :meth:`~mapflpy.tracer._Tracer.trace` is a :class:`~mapflpy.globals.Traces` object –
a named-tuple-like container for the traced fieldlines. This structure contains the following attributes:

- ``geometry`` : an **N x 3 x M** array of the traced fieldline coordinates, *i.e.* the
  radial-theta-phi coordinates of the traces where **N** is the buffer size, and **M** is
  the number of fieldlines. *(Note, to preserve a homogeneous array, field lines shorter
  than **N** are NaN padded).*
- ``start_pos`` : an **M x 3** array of the starting positions of each fieldline.
- ``end_pos`` : an **M x 3** array of the ending positions of each fieldline.
- ``traced_to_boundary`` : a boolean array of length **M** indicating whether each fieldline
  traced to a boundary (True) or was terminated early due to step-size constraints (False).

Using Scripts
-------------

Several scripts are provided in the :mod:`~mapflpy.scripts` module. These standalone functions
are used to perform common "one-off" tracing tasks (similar, in many respects, to how ``mapfl``
itself is used from the command line).

Any additional keyword arguments (see function signature) provided within the calls
to these scripts are passed to the instantiation of the :class:`~mapflpy.tracer.TracerMP` class,
*i.e.* arguments used to set the mapfl parameters.

.. code-block:: python
   :linenos:

   from mapflpy.scripts import run_forward_tracing

   lps = [...]
   bfiles = {
       'br': 'br_file.h5',
       'bt': 'bt_file.h5',
       'bp': 'bp_file.h5'
   }
   traces = run_forward_tracing(
       **bfiles,
       launch_points=lps,  # <-- passed to trace() method
       buffer_size=1000,   # <-- passed to trace() method
       domain_r_max_=100   # <-- example of passing mapfl parameter
   )

or

.. code-block:: python
   :linenos:

   from mapflpy.scripts import run_fwdbwd_tracing

   lps = [ ... ]
   bfiles = {
       'br': 'br_file.h5',
       'bt': 'bt_file.h5',
       'bp': 'bp_file.h5'
   }
   traces = run_fwdbwd_tracing(
       **bfiles,
       launch_points=lps,  # <-- passed to trace() method
       buffer_size=1000,   # <-- passed to trace() method
   )

This module also contains a script for performing interdomain tracing between
two different magnetic field domains (*e.g.* coronal to heliospheric). A more
comprehensive explanation of the function signature can be found in the documentation
for :func:`~mapflpy.scripts.inter_domain_tracing`.

A few general notes about this function:

- requires 6 magnetic field files: 3 for the inner domain, and 3 for the outer domain.
- the ``r_interface`` parameter must be specified to indicate the radial location of the
  recross boundary between the two domains.
- the ``helio_shift`` parameter can be used to account for the longitudinal shift angle
  between the heliospheric domain and the coronal domain.

.. code-block:: python
   :linenos:

   from mapflpy.scripts import inter_domain_tracing
   from math import pi
   lps = [ ... ]
   coronal_bfiles = {
       'br_cor': 'br_coronal.h5',
       'bt_cor': 'bt_coronal.h5',
       'bp_cor': 'bp_coronal.h5'
   }
   heliospheric_bfiles = {
       'br_hel': 'br_helio.h5',
       'bt_hel': 'bt_helio.h5',
       'bp_hel': 'bp_helio.h5'
   }
   traces = inter_domain_tracing(
       **coronal_bfiles,
       **heliospheric_bfiles,
       launch_points=lps,
       r_interface=30.0,         # <-- radial location of domain interface [R_sun]
       helio_shift=pi/6,         # <-- longitudinal shift between domains [rad]
       rtol_=1e-6,               # <-- relative tolerance used to determine when
                                 #     a fieldline has crossed the interface boundary
   )

Utilities
---------

A few utility functions are provided in the :mod:`~mapflpy.utils` module.
:func:`~mapflpy.utils.get_fieldline_endpoints`, :func:`~mapflpy.utils.get_fieldline_npoints`,
and :func:`~mapflpy.utils.trim_fieldline_nan_buffer` can be used on raw fieldline geometry
(numpy arrays) or on :class:`~mapflpy.globals.Traces` objects to extract information about the
traced field lines, or (in the case of :func:`~mapflpy.utils.trim_fieldline_nan_buffer`)
to remove the NaN padding from field lines – returning a list of heterogeneously sized
field lines.

Lastly, :func:`~mapflpy.utils.get_fieldline_polarity` can be used to determine and classify
field lines as:

+-------------------+---------------+------------------------------------------------------------+
| Enum Value        | Integer Value | Description                                                |
+===================+===============+============================================================+
| :data:`R0_R1_NEG` | -2            | Open field line with negative polarity at inner boundary   |
+-------------------+---------------+------------------------------------------------------------+
| :data:`R0_R0`     | -1            | Closed field line                                          |
+-------------------+---------------+------------------------------------------------------------+
| :data:`ERROR`     | 0             | Field line that failed to resolve within the buffer size   |
+-------------------+---------------+------------------------------------------------------------+
| :data:`R1_R1`     | 1             | Disconnected field line                                    |
+-------------------+---------------+------------------------------------------------------------+
| :data:`R0_R1_POS` | 2             | Open field line with positive polarity at inner boundary   |
+-------------------+---------------+------------------------------------------------------------+

The result of this function is an array of type :class:`~mapflpy.globals.Polarity` (an :py:class:`~enum.IntEnum` class)
which provides a mapping between integer codes and string labels for each fieldline.