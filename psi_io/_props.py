r"""Physical property metadata for PSI MAS and POT3D model quantities.

This module provides the :class:`Props` dataclass and three read-only mappings that
associate each named MAS or POT3D output quantity with:

- a human-readable description,
- the :mod:`astropy.units` unit needed to convert from code units to physical units,
- the mesh stagger code that identifies where on the staggered grid the quantity lives.

Staggered grid overview
-----------------------
MAS and POT3D solve their equations on a three-dimensional **staggered** (Yee-type)
spherical grid :math:`(r, \theta, \varphi)`.  Different physical quantities are
located at different positions within each grid cell so that discrete differential
operators (curl, divergence) are exactly satisfied.

With the convention used in this module, numpy arrays loaded from PSI HDF files have
shape ``(N_\varphi, N_\theta, N_r)`` — longitude index varies slowest, radial index
fastest.  Mesh codes therefore assign bits as follows:

.. list-table::
   :header-rows: 1

   * - Bit position
     - Numpy axis
     - Coordinate
   * - Most-significant bit (MSB)
     - ``axis = -1`` (last)
     - :math:`r` (radial)
   * - Middle bit
     - ``axis = -2``
     - :math:`\theta` (co-latitude)
   * - Least-significant bit (LSB)
     - ``axis = -3`` (first)
     - :math:`\varphi` (longitude)

A bit value of ``1`` means the quantity is on the **half mesh** along that axis
(displaced half a grid spacing), while ``0`` means it is on the **main mesh**.

The resulting grid positions for MAS vector components are:

.. list-table::
   :header-rows: 1

   * - Quantity type
     - Mesh code
     - :math:`r`
     - :math:`\theta`
     - :math:`\varphi`
     - Grid position
   * - ``br``
     - ``0b100``
     - half
     - main
     - main
     - :math:`r`-face center
   * - ``bt``
     - ``0b010``
     - main
     - half
     - main
     - :math:`\theta`-face center
   * - ``bp``
     - ``0b001``
     - main
     - main
     - half
     - :math:`\varphi`-face center
   * - ``vr``, ``jr``
     - ``0b011``
     - main
     - half
     - half
     - :math:`r`-edge center
   * - ``vt``, ``jt``
     - ``0b101``
     - half
     - main
     - half
     - :math:`\theta`-edge center
   * - ``vp``, ``jp``
     - ``0b110``
     - half
     - half
     - main
     - :math:`\varphi`-edge center
   * - Scalars (``t``, ``rho``, ``p``, …)
     - ``0b111``
     - half
     - half
     - half
     - Cell corner

This Yee-type arrangement ensures that :math:`\nabla \cdot \mathbf{B} = 0` is
satisfied exactly at the discrete level: each :math:`B` component lives on the face
through which it is the outward normal.  The current density components
:math:`\mathbf{J} = \nabla \times \mathbf{B}` then live on the corresponding cell
edges.  Scalar thermodynamic quantities (density, temperature, pressure) occupy the
cell corners, which are equivalent to cell centers on the dual grid.

Usage example
-------------
>>> from psi_io._props import get_mas_quantity_properties
>>> props = get_mas_quantity_properties('br')
>>> props.name
'br'
>>> props.desc
'Magnetic Field (Radial Component)'
>>> props.mesh          # doctest: +NORMALIZE_WHITESPACE
(Mesh.HALF, Mesh.MAIN, Mesh.MAIN)

See Also
--------
psi_io._mesh : Defines :class:`~psi_io._mesh.Mesh` and the mesh normalization helpers.
psi_io._units : Provides the custom astropy units referenced by each :class:`Props`.
psi_io.mhd_io : Uses these mappings to auto-configure lazy HDF readers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import astropy.units as u

from psi_io._mesh import _normalize_mesh_code
from psi_io._units import MAS_v, MAS_b, MAS_j, MAS_t, MAS_n, MAS_p, MAS_heat, POT3D_b, PSI_rsun, PSI_angle

MasQuantities = Literal[
    'vr', 'vt', 'vp', 'br', 'bt', 'bp', 'jr', 'jt', 'jp',
    't', 'te', 'tp', 'rho', 'p', 'ep', 'em', 'zp', 'zm', 'heat'
]
"""Literal type alias for the 19 named MAS output quantities.

Each string is the canonical lower-case identifier used as a key in
:data:`_MAS_QUANTITY_PROPS_MAPPING` and as the filename prefix (e.g. ``br001001.h5``).

Velocity components
    ``'vr'``, ``'vt'``, ``'vp'`` — radial, co-latitude, and longitude velocity.

Magnetic field components
    ``'br'``, ``'bt'``, ``'bp'`` — radial, co-latitude, and longitude magnetic field.

Current density components
    ``'jr'``, ``'jt'``, ``'jp'`` — radial, co-latitude, and longitude current density.

Temperature
    ``'t'`` — single-fluid temperature; ``'te'`` — electron temperature;
    ``'tp'`` — proton (ion) temperature (two-temperature model).

Density and pressure
    ``'rho'`` — mass density; ``'p'`` — plasma pressure.

Alfvén wave quantities
    ``'ep'``, ``'em'`` — wave energy density for forward- and backward-propagating
    Alfvén waves (proportional to :math:`|z^+|^2` and :math:`|z^-|^2`);
    ``'zp'``, ``'zm'`` — Elsässer variable amplitudes
    :math:`z^\\pm = v \\pm v_A` for forward/backward propagating waves.

Heating
    ``'heat'`` — local coronal volumetric heating rate.
"""

Pot3dQuantities = Literal['br', 'bt', 'bp',]
"""Literal type alias for the 3 POT3D magnetic field output quantities.

POT3D is a potential-field solver that outputs only the three spherical components of
the magnetic field: ``'br'`` (radial), ``'bt'`` (co-latitude), and ``'bp'``
(longitude).  The staggering convention is the same as for the corresponding MAS
quantities.

.. warning::

    POT3D output files carry **no intrinsic physical unit**.  The values are stored as
    dimensionless quantities (unit :data:`~psi_io._units.POT3D_b` = 1) whose physical
    interpretation depends entirely on the units of the photospheric boundary
    magnetogram used to drive the simulation — typically Gauss, but this is not
    guaranteed.  Always supply the correct unit explicitly via the ``unit`` keyword
    argument of :func:`~psi_io.mhd_io.PsiData`; calling ``read(units='physical')``
    on a reader opened without a ``unit`` override will **not** perform a meaningful
    conversion.
"""

PsiScales = Literal['r', 't', 'p',]
"""Literal type alias for the three PSI coordinate scale identifiers.

``'r'``
    Radial coordinate in units of solar radii (:data:`~psi_io._units.PSI_rsun`).
``'t'``
    Co-latitude :math:`\\theta` in radians (:data:`~psi_io._units.PSI_angle`).
``'p'``
    Longitude :math:`\\varphi` in radians (:data:`~psi_io._units.PSI_angle`).
"""


@dataclass(frozen=True, repr=True)
class Props:
    """Immutable property bundle for a single PSI model quantity.

    Associates a quantity name with its human-readable description, physical unit
    conversion factor, and staggered-grid mesh code.  Instances are frozen (immutable)
    and slot-based for memory efficiency.

    Parameters
    ----------
    name : str
        Canonical lower-case quantity identifier (e.g. ``'br'``, ``'vr'``).  Matches
        the filename prefix used in MAS and POT3D HDF output.
    desc : str
        Human-readable description of the physical quantity.
    unit : astropy.units.Unit
        Astropy unit whose scale factor converts one code unit of this quantity to
        physical units.  For example, :data:`~psi_io._units.MAS_b` ≈ 2.2 Gauss.
    _mesh : int, optional
        Integer mesh code encoding the stagger position on the three-dimensional grid.
        Each binary bit indicates whether the quantity is on the half mesh (``1``) or
        main mesh (``0``) along one coordinate axis.  ``None`` for coordinate scale
        arrays that carry no stagger information (e.g. the radial scale ``'r'``).

    Attributes
    ----------
    mesh : tuple[Mesh, ...] or None
        Normalized form of :attr:`_mesh`, expanded to a length-3 tuple of
        :class:`~psi_io._mesh.Mesh` members.  Returns ``None`` when ``_mesh`` is
        ``None``.

    Notes
    -----
    The arithmetic dunder methods (``__mul__``, ``__rmul__``, ``__rtruediv__``)
    delegate to :attr:`unit`, so ``some_value * props`` is equivalent to
    ``some_value * props.unit`` and returns an :class:`~astropy.units.Quantity`.

    Examples
    --------
    >>> from psi_io._props import Props
    >>> import astropy.units as u
    >>> p = Props('br', 'Radial B field', u.Gauss, 0b100)
    >>> str(p)
    'br'
    >>> p.mesh          # doctest: +NORMALIZE_WHITESPACE
    (Mesh.HALF, Mesh.MAIN, Mesh.MAIN)
    >>> (2.5 * p).unit
    Unit("G")
    """

    name: str
    desc: str
    unit: u.Unit
    _mesh: int = None

    @property
    def mesh(self):
        """Normalized mesh-stagger tuple for this quantity.

        Converts the integer :attr:`_mesh` code to a length-3 tuple of
        :class:`~psi_io._mesh.Mesh` members via
        :func:`~psi_io._mesh._normalize_mesh_code`.

        Returns
        -------
        out : tuple[Mesh, Mesh, Mesh] or None
            One :class:`~psi_io._mesh.Mesh` value per spatial dimension
            ``(r, theta, phi)`` — in the order they appear in
            :func:`~psi_io._mesh.main_mesh` processing (most-significant bit first,
            mapping to the last numpy axis).  Returns ``None`` when :attr:`_mesh` is
            ``None`` (coordinate scale arrays have no stagger).

        Examples
        --------
        >>> from psi_io._props import _MAS_QUANTITY_PROPS_MAPPING
        >>> from psi_io._mesh import Mesh
        >>> _MAS_QUANTITY_PROPS_MAPPING['br'].mesh
        (Mesh.HALF, Mesh.MAIN, Mesh.MAIN)
        >>> _MAS_QUANTITY_PROPS_MAPPING['t'].mesh
        (Mesh.HALF, Mesh.HALF, Mesh.HALF)
        >>> _MAS_QUANTITY_PROPS_MAPPING['vr'].mesh
        (Mesh.MAIN, Mesh.HALF, Mesh.HALF)
        """
        return _normalize_mesh_code(self._mesh, 3) if self._mesh is not None else None

    def __str__(self) -> str:
        """Return the quantity name.

        Returns
        -------
        out : str
            The :attr:`name` field (e.g. ``'br'``, ``'vr'``).

        Examples
        --------
        >>> from psi_io._props import _MAS_QUANTITY_PROPS_MAPPING
        >>> str(_MAS_QUANTITY_PROPS_MAPPING['br'])
        'br'
        """
        return self.name

    def __mul__(self, other):
        """Multiply *other* by this quantity's unit.

        Allows ``props * value`` expressions that return an astropy
        :class:`~astropy.units.Quantity`.

        Parameters
        ----------
        other : numeric or astropy.units.Quantity
            The value to multiply by :attr:`unit`.

        Returns
        -------
        out : astropy.units.Quantity
            ``other * self.unit``.

        Examples
        --------
        >>> from psi_io._props import _MAS_QUANTITY_PROPS_MAPPING
        >>> props = _MAS_QUANTITY_PROPS_MAPPING['vr']
        >>> (props * 1.0).unit   # doctest: +ELLIPSIS
        Unit(...)
        """
        return other * self.unit

    def __rmul__(self, other):
        """Right-multiply *other* by this quantity's unit.

        Allows ``value * props`` expressions that return an astropy
        :class:`~astropy.units.Quantity`.

        Parameters
        ----------
        other : numeric or astropy.units.Quantity
            The value to multiply by :attr:`unit`.

        Returns
        -------
        out : astropy.units.Quantity
            ``other * self.unit``.

        Examples
        --------
        >>> from psi_io._props import _MAS_QUANTITY_PROPS_MAPPING
        >>> props = _MAS_QUANTITY_PROPS_MAPPING['vr']
        >>> (1.0 * props).unit   # doctest: +ELLIPSIS
        Unit(...)
        """
        return other * self.unit

    def __rtruediv__(self, other):
        """Divide *other* by this quantity's unit.

        Allows ``value / props`` expressions that return an astropy
        :class:`~astropy.units.Quantity`.

        Parameters
        ----------
        other : numeric or astropy.units.Quantity
            The numerator; divided by :attr:`unit`.

        Returns
        -------
        out : astropy.units.Quantity
            ``other / self.unit``.

        Examples
        --------
        >>> from psi_io._props import _MAS_QUANTITY_PROPS_MAPPING
        >>> props = _MAS_QUANTITY_PROPS_MAPPING['br']
        >>> (1.0 / props).unit   # doctest: +ELLIPSIS
        Unit(...)
        """
        return other / self.unit


_MAS_QUANTITY_PROPS_MAPPING = MappingProxyType({
    'vr': Props('vr', 'Velocity (Radial Component)', MAS_v, 0b011),
    'vt': Props('vt', 'Velocity (Theta Component)', MAS_v, 0b101),
    'vp': Props('vp', 'Velocity (Phi Component)', MAS_v, 0b110),
    'br': Props('br', 'Magnetic Field (Radial Component)', MAS_b, 0b100),
    'bt': Props('bt', 'Magnetic Field (Theta Component)', MAS_b, 0b010),
    'bp': Props('bp', 'Magnetic Field (Phi Component)', MAS_b, 0b001),
    'jr': Props('jr', 'Current Density (Radial Component)', MAS_j, 0b011),
    'jt': Props('jt', 'Current Density (Theta Component)', MAS_j, 0b101),
    'jp': Props('jp', 'Current Density (Phi Component)', MAS_j, 0b110),
    't': Props('t', 'Temperature', MAS_t, 0b111),
    'te': Props('te', 'Electron Temperature', MAS_t, 0b111),
    'tp': Props('tp', 'Proton Temperature', MAS_t, 0b111),
    'rho': Props('rho', 'Density', MAS_n, 0b111),
    'p': Props('p', 'Pressure', MAS_p, 0b111),
    'ep': Props('ep', 'Wave Energy Density (Parallel to the Field)', MAS_p, 0b111),
    'em': Props('em', 'Wave Energy Density (Anti-Parallel to the Field)', MAS_p, 0b111),
    'zp': Props('zp', 'Positive Charge Density', MAS_v, 0b111),
    'zm': Props('zm', 'Negative Charge Density', MAS_v, 0b111),
    'heat': Props('heat', 'Local Coronal Heating Rate', MAS_heat, 0b111),
})
"""Read-only mapping from MAS quantity name to its :class:`Props` descriptor.

Contains the 19 MAS output quantities grouped by physical type:

**Velocity** — edge-centred in the two transverse directions (see staggered grid
table in the module docstring):

- ``'vr'`` (``0b011``): radial component, half-mesh in :math:`\\theta` and
  :math:`\\varphi`, main in :math:`r`.
- ``'vt'`` (``0b101``): co-latitude component, half in :math:`r` and :math:`\\varphi`,
  main in :math:`\\theta`.
- ``'vp'`` (``0b110``): longitude component, half in :math:`r` and :math:`\\theta`,
  main in :math:`\\varphi`.

**Magnetic field** — face-centred in the normal direction:

- ``'br'`` (``0b100``): radial component, half-mesh in :math:`r`, main elsewhere.
- ``'bt'`` (``0b010``): co-latitude component, half in :math:`\\theta`, main elsewhere.
- ``'bp'`` (``0b001``): longitude component, half in :math:`\\varphi`, main elsewhere.

**Current density** — edge-centred in the same transverse pair as the corresponding
velocity component (follows from :math:`\\mathbf{J} = \\nabla \\times \\mathbf{B}`):

- ``'jr'`` (``0b011``), ``'jt'`` (``0b101``), ``'jp'`` (``0b110``).

**Temperature** — scalar, all-half-mesh (cell corners of the primary grid):

- ``'t'`` — single-fluid (MHD) temperature.
- ``'te'`` — electron temperature (two-temperature model).
- ``'tp'`` — proton/ion temperature (two-temperature model).

**Density and pressure** — scalar, all-half-mesh:

- ``'rho'`` — plasma mass density (code units; divide by :data:`~psi_io._units.FMP`
  to obtain number density).
- ``'p'`` — total plasma pressure.

**Alfvén wave quantities** — produced by MAS's wave-turbulence-driven (WTD) heating
model; all scalar, all-half-mesh:

- ``'ep'``, ``'em'`` — wave energy densities for forward- and backward-propagating
  Alfvén waves (:math:`e^\\pm \\propto |z^\\pm|^2`), in pressure units.
- ``'zp'``, ``'zm'`` — Elsässer variable amplitudes
  :math:`z^\\pm = v \\pm v_A`, in velocity units.

**Heating** — scalar, all-half-mesh:

- ``'heat'`` — local coronal volumetric heating rate in erg cm⁻³ s⁻¹.
"""

_POT3D_QUANTITY_PROPS_MAPPING = MappingProxyType({
    'br': Props('br', 'Magnetic Field (Radial Component)', POT3D_b, 0b011),
    'bt': Props('bt', 'Magnetic Field (Theta Component)', POT3D_b, 0b101),
    'bp': Props('bp', 'Magnetic Field (Phi Component)', POT3D_b, 0b110),
})
"""Read-only mapping from POT3D quantity name to its :class:`Props` descriptor.

POT3D solves for the potential magnetic field
:math:`\\mathbf{B} = -\\nabla \\Psi` driven by a photospheric radial-field boundary
condition.  It outputs three spherical magnetic field components that use the same
stagger codes as the corresponding MAS velocity/current components — i.e. each
component is edge-centred (half-mesh in the two transverse directions):

- ``'br'`` (``0b011``): radial component, half in :math:`\\theta` and :math:`\\varphi`.
- ``'bt'`` (``0b101``): co-latitude component, half in :math:`r` and :math:`\\varphi`.
- ``'bp'`` (``0b110``): longitude component, half in :math:`r` and :math:`\\theta`.

.. warning::

    The unit for all three quantities is :data:`~psi_io._units.POT3D_b`, which has a
    scale factor of 1 (dimensionless-unscaled).  POT3D does not apply a normalization:
    the values stored in the HDF file are whatever physical units the input boundary
    magnetogram was provided in — most commonly Gauss, but this depends entirely on the
    run configuration.  There is **no way** to infer the correct physical unit from the
    file alone.

    As a consequence, ``read(units='physical')`` on a reader opened without a ``unit``
    override will return a :class:`~astropy.units.Quantity` with unit
    ``dimensionless_unscaled`` rather than performing any meaningful conversion.
    Always pass the correct unit explicitly:

    .. code-block:: python

        reader = PsiData('br001.h5', model='pot3d', unit='Gauss')
        data, r, t, p = reader.read()
"""

_PSI_SCALE_PROPS_MAPPING = MappingProxyType({
    'r': Props('r', 'Radial Scale (Solar Radii)', PSI_rsun),
    't': Props('t', 'Theta Scale (Co-Latitude)', PSI_angle),
    'p': Props('p', 'Phi Scale (Longitude)', PSI_angle),
})
"""Read-only mapping from coordinate scale label to its :class:`Props` descriptor.

HDF files from MAS and POT3D store the coordinate grid arrays alongside the data.
These scale arrays have no mesh stagger (``_mesh=None``) since their staggering is
contingent on the dataset that they are associated with.

- ``'r'``: radial coordinate in solar radii
  (:data:`~psi_io._units.PSI_rsun` = :data:`~psi_io._units.RSUN` = 6.96 × 10¹⁰ cm).
- ``'t'``: co-latitude :math:`\\theta \\in [0, \\pi]` in radians
  (:data:`~psi_io._units.PSI_angle`).
- ``'p'``: longitude :math:`\\varphi \\in [0, 2\\pi]` in radians
  (:data:`~psi_io._units.PSI_angle`).
"""