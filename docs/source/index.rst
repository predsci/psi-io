mapflpy Documentation
=====================

A Python extension for tracing field lines using the Fortran tracer from
`MapFL <https://github.com/predsci/MapFL>`_.

The goal of mapflpy is to provide fast and accurate tracing capabilities for
spherical vector fields inside a convenient Python interface.

``mapflpy`` is designed to work natively with the staggered meshes produced by
Predictive Science Inc.'s codes for simulating the solar corona, and
inner heliosphere (`MAS <https://www.predsci.com/mas>`_ or
`POT3D <https://github.com/predsci/POT3D>`_) , but it should be generally compatible
with any global vector field that can be described on a rectilinear grid in
spherical coordinates.

To get started, visit the :ref:`installation` guide. For a more in-depth analysis of
``mapflpy``'s architecture – the motivations underlying it's design – consult the
:ref:`overview` page.

.. toctree::
    :hidden:

    API <api/index>
    Guide <guide/index>
    Examples <gallery/index>
