r"""Physical property metadata for PSI's model quantities and scales.

This module provides the :class:`Props` dataclass and three read-only mapping objects
that fully describe every quantity PSI's modeling codes write to HDF files:

- The filename/lookup identifier (e.g., ``'br'``)
- Human-readable description (e.g., ``'MAS Magnetic Field (Radial Component)'``)
- Native code :class:`~astropy.units.Unit` (e.g., ``Unit("2.20689 G")`` or ``Unit("MAS_b")``)
- Dimensionality (e.g., 3 for 3-D output fields)
- Scalar or vector classification
- Mesh staggering code (e.g., ``0b011``)

For metadata access, one should use the provided getter functions rather than the mapping
objects directly.


.. list-table::
   :header-rows: 1

   * - Mapping
     - Model
     - Contents
   * - :func:`get_mas_quantity_properties`
     - `MAS <https://github.com/predsci/MAS>`_ (Magnetohydrodynamic Algorithm outside a Sphere)
     - 19 3-D output fields: velocity, magnetic field, current density, temperature,
       density, pressure, wave energy, Elsässer amplitudes, heating
   * - :func:`get_pot3d_quantity_properties`
     - `POT3D <https://github.com/predsci/POT3D>`_ (High Performance Potential Field Solver)
     - 3 magnetic field components (br, bt, bp)
   * - :func:`get_psi_scale_properties`
     - PSI coordinate grids (shared by MAS, POT3D, and related codes)
     - 3 1-D coordinate scale arrays (r, t, p)

Coordinate scales
-----------------
Every PSI HDF file stores three 1-D coordinate arrays alongside its data.  These
arrays define the spherical grid on which model output is sampled and are standard
across MAS, POT3D, and most other PSI codes:

.. list-table::
   :header-rows: 1
   :widths: 10 12 28 24 16

   * - Key
     - Symbol
     - Physical quantity
     - Units
     - Range
   * - ``'r'``
     - :math:`r`
     - Radial distance
     - solar radii (:data:`~psi_io._units.PSI_rsun`)
     - :math:`r \geq 1\,R_\odot`
   * - ``'t'``
     - :math:`\theta`
     - Co-latitude (pole = 0)
     - radians (:data:`~psi_io._units.PSI_angle`)
     - :math:`[0,\,\pi]`
   * - ``'p'``
     - :math:`\varphi`
     - Longitude
     - radians (:data:`~psi_io._units.PSI_angle`)
     - :math:`[0,\,2\pi]`

PSI HDF files are written in **Fortran order** (column-major), so the *first* Fortran
dimension varies fastest in memory.  When read into NumPy (row-major), the axis order
is reversed: a 3-D array has shape :math:`(N_\varphi, N_\theta, N_r)` and the three
scale arrays map to axes as follows:

.. list-table::
   :header-rows: 1

   * - Key
     - Fortran dimension
     - NumPy axis
     - ``arr.shape`` index
   * - ``'r'``
     - 1st (fastest-varying)
     - last
     - ``-1``
   * - ``'t'``
     - 2nd
     - middle
     - ``-2``
   * - ``'p'``
     - 3rd (slowest-varying)
     - first
     - ``0``


MAS Model Quantities
--------------------
MAS outputs 19 3-D fields in spherical coordinates.  Code-unit values are converted
to physical (CGS/SI) unit by multiplying by the normalization constants defined in
:mod:`psi_io._units`.  Approximate physical scales are given in parentheses.

.. list-table::
   :header-rows: 1
   :widths: 8 12 30 28 10 10

   * - Key
     - Symbol
     - Physical quantity
     - Physical unit (CGS)
     - Type
     - Mesh code
   * - ``vr``
     - :math:`v_r`
     - Radial velocity
     - km s⁻¹ (:data:`~psi_io._units.FN_V` ≈ 481 km s⁻¹)
     - vector
     - ``0b011``
   * - ``vt``
     - :math:`v_\theta`
     - Co-latitude velocity
     - km s⁻¹
     - vector
     - ``0b101``
   * - ``vp``
     - :math:`v_\varphi`
     - Longitude velocity
     - km s⁻¹
     - vector
     - ``0b110``
   * - ``br``
     - :math:`B_r`
     - Radial magnetic field
     - G (:data:`~psi_io._units.FN_B` ≈ 2.2 G)
     - vector
     - ``0b100``
   * - ``bt``
     - :math:`B_\theta`
     - Co-latitude magnetic field
     - G
     - vector
     - ``0b010``
   * - ``bp``
     - :math:`B_\varphi`
     - Longitude magnetic field
     - G
     - vector
     - ``0b001``
   * - ``jr``
     - :math:`J_r`
     - Radial current density
     - A m⁻² (:data:`~psi_io._units.FN_J`)
     - vector
     - ``0b011``
   * - ``jt``
     - :math:`J_\theta`
     - Co-latitude current density
     - A m⁻²
     - vector
     - ``0b101``
   * - ``jp``
     - :math:`J_\varphi`
     - Longitude current density
     - A m⁻²
     - vector
     - ``0b110``
   * - ``t``
     - :math:`T`
     - Single-fluid temperature
     - MK (:data:`~psi_io._units.FN_T` ≈ 28 MK)
     - scalar
     - ``0b111``
   * - ``te``
     - :math:`T_e`
     - Electron temperature
     - MK
     - scalar
     - ``0b111``
   * - ``tp``
     - :math:`T_p`
     - Proton temperature
     - MK
     - scalar
     - ``0b111``
   * - ``rho``
     - :math:`\rho`
     - Plasma density
     - cm⁻³ (:data:`~psi_io._units.FN_N` = 10⁸ cm⁻³)
     - scalar
     - ``0b111``
   * - ``p``
     - :math:`p`
     - Plasma pressure
     - erg cm⁻³ (:data:`~psi_io._units.FN_P`)
     - scalar
     - ``0b111``
   * - ``ep``
     - :math:`e^+`
     - Forward Alfvén wave energy density
     - erg cm⁻³
     - scalar
     - ``0b111``
   * - ``em``
     - :math:`e^-`
     - Backward Alfvén wave energy density
     - erg cm⁻³
     - scalar
     - ``0b111``
   * - ``zp``
     - :math:`z^+`
     - Outward Elsässer amplitude
     - km s⁻¹
     - scalar
     - ``0b111``
   * - ``zm``
     - :math:`z^-`
     - Inward Elsässer amplitude
     - km s⁻¹
     - scalar
     - ``0b111``
   * - ``heat``
     - :math:`Q`
     - Volumetric heating rate
     - erg cm⁻³ s⁻¹ (:data:`~psi_io._units.FN_HEAT`)
     - scalar
     - ``0b111``

POT3D Model Quantities
----------------------
POT3D solves for the potential magnetic field :math:`\mathbf{B} = -\nabla\Psi` driven
by a photospheric radial-field boundary condition.  It outputs three spherical
magnetic field components:

.. list-table::
   :header-rows: 1
   :widths: 10 12 35 25 18

   * - Key
     - Symbol
     - Physical quantity
     - Physical unit
     - Mesh code
   * - ``br``
     - :math:`B_r`
     - Radial magnetic field
     - input magnetogram unit (typically G)
     - ``0b011``
   * - ``bt``
     - :math:`B_\theta`
     - Co-latitude magnetic field
     - input magnetogram unit
     - ``0b101``
   * - ``bp``
     - :math:`B_\varphi`
     - Longitude magnetic field
     - input magnetogram unit
     - ``0b110``

.. note::

   POT3D mesh codes are the bitwise complement of the corresponding MAS magnetic field
   codes: ``POT3D_mesh = 0b111 ^ MAS_mesh``.  Where MAS places each component on the
   face through which it is the outward normal (face-centred), POT3D places the same
   component on the opposite pair of edges (edge-centred).

.. warning::

   POT3D does not apply a normalization: values are in whatever physical unit the
   input boundary magnetogram was provided in (typically Gauss, but run-dependent).
   The code unit :data:`~psi_io._units.POT3D_b` has a scale factor of 1
   (dimensionless-unscaled).  Always supply the correct unit explicitly when
   converting; ``read(unit='physical')`` without a ``unit`` override will return
   dimensionless values.

Staggered Grid Overview
-----------------------
MAS and POT3D solve their equations on a three-dimensional **staggered** (Yee-type)
spherical grid :math:`(r, \theta, \varphi)`.  Different physical quantities are
located at different positions within each grid cell so that discrete differential
operators (curl, divergence) are exactly satisfied.

With the convention used in this module, numpy arrays loaded from PSI HDF files have
shape :math:`(N_{\phi}, N_{\theta}, N_r)` — longitude index varies slowest, radial index
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

See Also
--------

:mod:`psi_io._mesh` :
    Defines :class:`~psi_io._mesh.Mesh` and the mesh normalization helpers.
:mod:`psi_io._units` :
    Provides the custom astropy unit referenced by each :class:`Props`.
:mod:`psi_io.mhd_io` :
    Uses these mappings to auto-configure lazy HDF readers.
"""

from __future__ import annotations

__all__ = [
    "get_mas_quantity_properties",
    "get_pot3d_quantity_properties",
    "get_psi_scale_properties",
    "parse_psi_filename_schema",
    "extract_quantity_from_filepath",
    "extract_sequence_from_filepath"
]

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Optional, Callable

import astropy.units as u

from psi_io._mesh import ArrayOrdering, Mesh
from psi_io._units import MAS_v, MAS_b, MAS_j, MAS_t, MAS_n, MAS_p, MAS_heat, POT3D_b, PSI_rsun, PSI_angle


@dataclass(frozen=True, repr=True)
class ScaleProps:
    """Immutable property bundle for a single PSI model quantity.

    Associates a quantity name with its human-readable description, physical unit,
    dimensionality, scalar/vector classification, and staggered-grid mesh code.
    Instances are frozen (immutable) dataclass instances.

    Parameters
    ----------
    name : str
        Canonical lower-case quantity identifier (e.g. ``'br'``, ``'vr'``).  Matches
        the filename prefix used in MAS and POT3D HDF output.
    desc : str
        Human-readable description of the physical quantity.
    unit : u.Unit
        Astropy unit whose scale factor converts one code unit of this quantity to
        physical unit.  For example, :data:`~psi_io._units.MAS_b` ≈ 2.2 Gauss.
    ndim : int
        Number of spatial dimensions of the output array (``3`` for MAS/POT3D fields,
        ``1`` for coordinate scale arrays).
    scalar : bool
        ``True`` if the quantity is a scalar field (temperature, density, …);
        ``False`` if it is a component of a vector field (velocity, magnetic field, …).
    _mesh : int, optional
        Integer mesh code encoding the stagger position on the three-dimensional grid.
        Each binary bit indicates whether the quantity is on the half mesh (``1``) or
        main mesh (``0``) along one coordinate axis.  ``None`` for coordinate scale
        arrays that carry no stagger information (e.g. the radial scale ``'r'``).

    Attributes
    ----------
    mesh : tuple[Mesh, ...] or None
        Normalized form of :attr:`_mesh`, expanded to a length-:attr:`ndim` tuple of
        :class:`~psi_io._mesh.Mesh` members.  Returns ``None`` when :attr:`_mesh` is
        ``None`` (coordinate scale arrays have no stagger).

    Notes
    -----
    The arithmetic dunder methods (``__mul__``, ``__rmul__``, ``__rtruediv__``)
    delegate to :attr:`unit`, so ``some_value * props`` is equivalent to
    ``some_value * props.unit`` and returns an :class:`~astropy.units.Quantity`.

    Examples
    --------
    >>> from psi_io._models import ModelProps
    >>> import astropy.units as u
    >>> p = ModelProps('br', 'Radial B field', u.Gauss, 3, False, 0b100)
    >>> str(p)
    'br'
    >>> p.ndim, p.scalar
    (3, False)
    >>> p.mesh          # doctest: +NORMALIZE_WHITESPACE
    (Mesh.HALF, Mesh.MAIN, Mesh.MAIN)
    >>> (2.5 * p).unit
    Unit("G")
    """

    name: str
    desc: str
    unit: u.Unit

    def __str__(self) -> str:
        """Return the quantity name.

        Returns
        -------
        out : str
            The :attr:`name` field (e.g. ``'br'``, ``'vr'``).

        Examples
        --------
        >>> from psi_io._models import _MAS_QUANTITY_PROPS_MAPPING
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
        other : numeric or u.Quantity
            The value to multiply by :attr:`unit`.

        Returns
        -------
        out : u.Quantity
            ``other * self.unit``.

        Examples
        --------
        >>> from psi_io._models import _MAS_QUANTITY_PROPS_MAPPING
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
        other : numeric or u.Quantity
            The value to multiply by :attr:`unit`.

        Returns
        -------
        out : u.Quantity
            ``other * self.unit``.

        Examples
        --------
        >>> from psi_io._models import _MAS_QUANTITY_PROPS_MAPPING
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
        other : numeric or u.Quantity
            The numerator; divided by :attr:`unit`.

        Returns
        -------
        out : u.Quantity
            ``other / self.unit``.

        Examples
        --------
        >>> from psi_io._models import _MAS_QUANTITY_PROPS_MAPPING
        >>> props = _MAS_QUANTITY_PROPS_MAPPING['br']
        >>> (1.0 / props).unit   # doctest: +ELLIPSIS
        Unit(...)
        """
        return other / self.unit

    def _asdict(self):
        return asdict(self)


@dataclass(frozen=True, repr=True)
class ModelProps(ScaleProps):
    """Immutable property bundle for a single PSI model quantity.

    Associates a quantity name with its human-readable description, physical unit,
    dimensionality, scalar/vector classification, and staggered-grid mesh code.
    Instances are frozen (immutable) dataclass instances.

    Parameters
    ----------
    name : str
        Canonical lower-case quantity identifier (e.g. ``'br'``, ``'vr'``).  Matches
        the filename prefix used in MAS and POT3D HDF output.
    desc : str
        Human-readable description of the physical quantity.
    unit : u.Unit
        Astropy unit whose scale factor converts one code unit of this quantity to
        physical unit.  For example, :data:`~psi_io._units.MAS_b` ≈ 2.2 Gauss.
    ndim : int
        Number of spatial dimensions of the output array (``3`` for MAS/POT3D fields,
        ``1`` for coordinate scale arrays).
    scalar : bool
        ``True`` if the quantity is a scalar field (temperature, density, …);
        ``False`` if it is a component of a vector field (velocity, magnetic field, …).
    _mesh : int, optional
        Integer mesh code encoding the stagger position on the three-dimensional grid.
        Each binary bit indicates whether the quantity is on the half mesh (``1``) or
        main mesh (``0``) along one coordinate axis.  ``None`` for coordinate scale
        arrays that carry no stagger information (e.g. the radial scale ``'r'``).

    Attributes
    ----------
    mesh : tuple[Mesh, ...] or None
        Normalized form of :attr:`_mesh`, expanded to a length-:attr:`ndim` tuple of
        :class:`~psi_io._mesh.Mesh` members.  Returns ``None`` when :attr:`_mesh` is
        ``None`` (coordinate scale arrays have no stagger).

    Notes
    -----
    The arithmetic dunder methods (``__mul__``, ``__rmul__``, ``__rtruediv__``)
    delegate to :attr:`unit`, so ``some_value * props`` is equivalent to
    ``some_value * props.unit`` and returns an :class:`~astropy.units.Quantity`.

    Examples
    --------
    >>> from psi_io._models import ModelProps
    >>> import astropy.units as u
    >>> p = ModelProps('br', 'Radial B field', u.Gauss, 3, False, 0b100)
    >>> str(p)
    'br'
    >>> p.ndim, p.scalar
    (3, False)
    >>> p.mesh          # doctest: +NORMALIZE_WHITESPACE
    (Mesh.HALF, Mesh.MAIN, Mesh.MAIN)
    >>> (2.5 * p).unit
    Unit("G")
    """

    name: str
    desc: str
    unit: u.Unit
    scalar: bool
    _mesh: int
    order: ArrayOrdering = 'F'
    scales: tuple = tuple('rtp')

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
        >>> from psi_io._models import _MAS_QUANTITY_PROPS_MAPPING
        >>> from psi_io._mesh import Mesh
        >>> _MAS_QUANTITY_PROPS_MAPPING['br'].mesh
        (Mesh.HALF, Mesh.MAIN, Mesh.MAIN)
        >>> _MAS_QUANTITY_PROPS_MAPPING['t'].mesh
        (Mesh.HALF, Mesh.HALF, Mesh.HALF)
        >>> _MAS_QUANTITY_PROPS_MAPPING['vr'].mesh
        (Mesh.MAIN, Mesh.HALF, Mesh.HALF)
        """
        return Mesh(self._mesh, len(self.scales))

    def _asdict(self):
        dout = asdict(self)
        dout.update(mesh=self.mesh)
        del dout['_mesh']
        return dout


# ----------------------------------------------------------------
# MAS Model
# ----------------------------------------------------------------


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

_MAS_QUANTITY_PROPS_MAPPING = MappingProxyType({
    'vr': ModelProps('vr', 'MAS Velocity (Radial Component)', MAS_v, False, 0b011),
    'vt': ModelProps('vt', 'MAS Velocity (Theta Component)', MAS_v, False, 0b101),
    'vp': ModelProps('vp', 'MAS Velocity (Phi Component)', MAS_v, False, 0b110),
    'br': ModelProps('br', 'MAS Magnetic Field (Radial Component)', MAS_b, False, 0b100),
    'bt': ModelProps('bt', 'MAS Magnetic Field (Theta Component)', MAS_b, False, 0b010),
    'bp': ModelProps('bp', 'MAS Magnetic Field (Phi Component)', MAS_b, False, 0b001),
    'jr': ModelProps('jr', 'MAS Current Density (Radial Component)', MAS_j, False, 0b011),
    'jt': ModelProps('jt', 'MAS Current Density (Theta Component)', MAS_j, False, 0b101),
    'jp': ModelProps('jp', 'MAS Current Density (Phi Component)', MAS_j, False, 0b110),
    't': ModelProps('t', 'MAS Temperature', MAS_t, True, 0b111),
    'te': ModelProps('te', 'MAS Electron Temperature', MAS_t, True, 0b111),
    'tp': ModelProps('tp', 'MAS Proton Temperature', MAS_t, True, 0b111),
    'rho': ModelProps('rho', 'MAS Density', MAS_n, True, 0b111),
    'p': ModelProps('p', 'MAS Pressure', MAS_p, True, 0b111),
    'ep': ModelProps('ep', 'MAS Wave Energy Density (Parallel to the Field)', MAS_p, True, 0b111),
    'em': ModelProps('em', 'MAS Wave Energy Density (Anti-Parallel to the Field)', MAS_p, True, 0b111),
    'zp': ModelProps('zp', 'MAS Outward Propagating Wave Amplitude', MAS_v, True, 0b111),
    'zm': ModelProps('zm', 'MAS Inward Propagating Wave Amplitude', MAS_v, True, 0b111),
    'heat': ModelProps('heat', 'MAS Local Coronal Heating Rate', MAS_heat, True, 0b111),
})
"""Read-only mapping from MAS quantity name to its :class:`ModelProps` descriptor."""


def get_mas_quantity_properties(variable: MasQuantities) -> ModelProps:
    """Return the :class:`~psi_io._models.ModelProps` descriptor for a MAS quantity.

    Parameters
    ----------
    variable : MasQuantities
        Case-insensitive MAS quantity name.  Valid values: ``'br'``, ``'bt'``, ``'bp'``,
        ``'vr'``, ``'vt'``, ``'vp'``, ``'jr'``, ``'jt'``, ``'jp'``, ``'t'``, ``'te'``,
        ``'tp'``, ``'rho'``, ``'p'``, ``'ep'``, ``'em'``, ``'zp'``, ``'zm'``, ``'heat'``.

    Returns
    -------
    out : ModelProps
        Immutable metadata descriptor for the requested MAS quantity.

    Raises
    ------
    ValueError
        If *variable* is not a recognized MAS quantity.

    Examples
    --------
    >>> from psi_io.mhd_io import get_mas_quantity_properties
    >>> props = get_mas_quantity_properties('br')
    >>> props.desc
    'Magnetic Field (Radial Component)'
    >>> get_mas_quantity_properties('BR').name   # case-insensitive
    'br'
    """
    try:
        return _MAS_QUANTITY_PROPS_MAPPING[variable.lower()]
    except KeyError:
        raise ValueError(f"Invalid variable {variable!r} for MAS model. "
                         f"Valid options are: {', '.join(_MAS_QUANTITY_PROPS_MAPPING.keys())}") from None


# ----------------------------------------------------------------
# POT3D Model
# ----------------------------------------------------------------


Pot3dQuantities = Literal['br', 'bt', 'bp',]
"""Literal type alias for the 3 POT3D magnetic field output quantities.

POT3D is a potential-field solver that outputs only the three spherical components of
the magnetic field: ``'br'`` (radial), ``'bt'`` (co-latitude), and ``'bp'``
(longitude).  

.. note::

   POT3D mesh codes are the bitwise complement of the corresponding MAS magnetic field
   codes: ``POT3D_mesh = 0b111 ^ MAS_mesh``.  Where MAS places each component on the
   face through which it is the outward normal (face-centred), POT3D places the same
   component on the opposite pair of edges (edge-centred).

.. warning::

    POT3D output files carry **no intrinsic physical unit**.  The values are stored as
    dimensionless quantities (unit :data:`~psi_io._units.POT3D_b` = 1) whose physical
    interpretation depends entirely on the unit of the photospheric boundary
    magnetogram used to drive the simulation — typically Gauss, but this is not
    guaranteed.  Always supply the correct unit explicitly via the ``unit`` keyword
    argument of :func:`~psi_io.mhd_io.PsiData`; calling ``read(unit='physical')``
    on a reader opened without a ``unit`` override will **not** perform a meaningful
    conversion.
"""

_POT3D_QUANTITY_PROPS_MAPPING = MappingProxyType({
    'br': ModelProps('br', 'POT3D Magnetic Field (Radial Component)', POT3D_b, False, 0b011),
    'bt': ModelProps('bt', 'POT3D Magnetic Field (Theta Component)', POT3D_b, False, 0b101),
    'bp': ModelProps('bp', 'POT3D Magnetic Field (Phi Component)', POT3D_b, False, 0b110),
})
"""Read-only mapping from POT3D quantity name to its :class:`ModelProps` descriptor."""


def get_pot3d_quantity_properties(variable: Pot3dQuantities) -> ModelProps:
    """Return the :class:`~psi_io._models.ModelProps` descriptor for a POT3D quantity.

    Parameters
    ----------
    variable : Pot3dQuantities
        Case-insensitive POT3D quantity name.  Valid values: ``'br'``, ``'bt'``,
        ``'bp'``.

    Returns
    -------
    out : ModelProps
        Immutable descriptor for the requested POT3D magnetic field component.

    Raises
    ------
    ValueError
        If *variable* is not a recognized POT3D quantity.

    Examples
    --------
    >>> from psi_io.mhd_io import get_pot3d_quantity_properties
    >>> get_pot3d_quantity_properties('bp').name
    'bp'
    """
    try:
        return _POT3D_QUANTITY_PROPS_MAPPING[variable.lower()]
    except KeyError:
        raise ValueError(f"Invalid variable {variable!r} for POT3D model. "
                         f"Valid options are: {', '.join(_POT3D_QUANTITY_PROPS_MAPPING.keys())}") from None


# ----------------------------------------------------------------
# PSI Coordinate Scales
# ----------------------------------------------------------------


PsiScales = Literal['r', 'radius', 't', 'theta', 'p', 'phi']
"""Literal type alias for the three PSI coordinate scale identifiers.

``'r'`` or ``'radius'``
    Radial coordinate in unit of solar radii (:data:`~psi_io._units.PSI_rsun`).
``'t'`` or ``'theta'``
    Co-latitude :math:`\\theta` in radians (:data:`~psi_io._units.PSI_angle`).
``'p'`` or ``'phi'``
    Longitude :math:`\\varphi` in radians (:data:`~psi_io._units.PSI_angle`).
"""


_BASE_SCALE_PROPS_MAPPING = MappingProxyType({
    'r': ScaleProps('r', 'PSI Radial Scale (Solar Radii)', PSI_rsun,),
    't': ScaleProps('t', 'PSI Theta Scale (Co-Latitude)', PSI_angle,),
    'p': ScaleProps('p', 'PSI Phi Scale (Longitude)', PSI_angle,),
})
"""Read-only mapping from coordinate scale label to its :class:`Props` descriptor."""

_PSI_SCALE_PROPS_MAPPING = MappingProxyType({
    **_BASE_SCALE_PROPS_MAPPING,
    'radius': _BASE_SCALE_PROPS_MAPPING['r'],
    'theta': _BASE_SCALE_PROPS_MAPPING['t'],
    'phi': _BASE_SCALE_PROPS_MAPPING['p'],
})


def get_psi_scale_properties(variable: PsiScales) -> ScaleProps:
    """Return the :class:`~psi_io._models.Props` descriptor for a PSI coordinate scale.

    Parameters
    ----------
    variable : PsiScales
        Coordinate label.  The first character is used for lookup, so ``'r'``,
        ``'radius'``, ``'t'``, ``'theta'``, ``'p'``, and ``'phi'`` are all accepted.

    Returns
    -------
    out : Props
        Descriptor for the requested coordinate axis.

    Raises
    ------
    ValueError
        If the first character of *variable* is not ``'r'``, ``'t'``, or ``'p'``.

    Examples
    --------
    >>> from psi_io import get_psi_scale_properties
    >>> get_psi_scale_properties('r').desc
    'Radial Scale (Solar Radii)'
    >>> get_psi_scale_properties('theta').name   # uses first character only
    't'
    """
    try:
        return _PSI_SCALE_PROPS_MAPPING[variable.lower()]
    except KeyError:
        raise ValueError(f"Invalid variable {variable!r} for PSI coordinate scale. "
                         f"Valid options are: {', '.join(_PSI_SCALE_PROPS_MAPPING.keys())}") from None


# ----------------------------------------------------------------
# Model Collection Factory
# ----------------------------------------------------------------


ModelType = Literal['mas', 'pot3d']
"""Literal type alias for the two recognized PSI model types.

``'mas'``
    MAS (Magnetohydrodynamic Algorithm outside a Sphere) plasma model output.
``'pot3d'``
    POT3D potential-field source-surface (PFSS) magnetic field output.
"""


_PROP_GETTER_MAPPING = MappingProxyType({
    'mas': get_mas_quantity_properties,
    'pot3d': get_pot3d_quantity_properties,})
"""Read-only mapping from model/scale label to its :class:`Props` getter function."""


def get_model_prop_caller(model: ModelType) -> Callable:
    """Return the :class:`Props` getter function for the given model type.

    Parameters
    ----------
    model : ModelType
        Case-insensitive model label.  Valid values: ``'mas'``, ``'pot3d'``,
        ``'scale'``.

    Returns
    -------
    out : Callable
        The getter function associated with *model* — one of
        :func:`get_mas_quantity_properties`, :func:`get_pot3d_quantity_properties`,
        or :func:`get_psi_scale_properties`.

    Raises
    ------
    ValueError
        If *model* is not a recognized model label.

    Examples
    --------
    >>> from psi_io._models import get_model_prop_caller
    >>> caller = get_model_prop_caller('mas')
    >>> caller('br').name
    'br'
    """
    try:
        return _PROP_GETTER_MAPPING[model.lower()]
    except KeyError:
        raise ValueError(f"Invalid model '{model}'. "
                         f"Valid options are: {', '.join(_PROP_GETTER_MAPPING.keys())}") from None


MATCH_QUANTITIES = '|'.join(re.escape(q) for q in sorted(_MAS_QUANTITY_PROPS_MAPPING.keys(), key=len, reverse=True))
"""Regex alternation string matching any valid MAS quantity name (case-insensitive).

Quantities are sorted longest-first to avoid partial matches (e.g. ``'heat'`` must be
tried before ``'h'``).  Used in :data:`FILEPATH_SCHEMA` and
:func:`extract_quantity_from_filepath`.
"""

FILEPATH_SCHEMA = rf'^({MATCH_QUANTITIES})(\d{{3}}(?:\d{{3}})?)$'
"""Regex pattern for the strict MAS filename schema ``<quantity><sequence>``.

The stem (filename without extension) must consist of exactly one recognized MAS
quantity name followed by a 3- or 6-digit decimal sequence number.

Groups:

1. Quantity name (e.g. ``'br'``, ``'heat'``).
2. Sequence number (e.g. ``'001'``, ``'001001'``).

Used by :func:`parse_psi_filename_schema`

See Also
--------
extract_quantity_from_filepath :
    A lenient variant that does not require the sequence suffix.
"""


def extract_quantity_from_filepath(ifile: Path, default: Optional[str] = None) -> str | None:
    """Extract the MAS/POT3D quantity name from a filename stem.

    Searches for the first recognized quantity token anywhere in the stem.  A
    match is accepted only when the token is not immediately preceded or followed
    by another ASCII letter, so partial matches inside longer words are rejected.
    The match is case-insensitive.

    Parameters
    ----------
    ifile : Path
        File path whose stem is examined.  Only the stem (filename without
        extension) is inspected.
    default : str or None, optional
        Value to return when no quantity token is found.  Defaults to ``None``.

    Returns
    -------
    out : str or None
        Lower-case quantity name (e.g. ``'br'``), or *default* if no recognized
        quantity token is found in the stem.

    Examples
    --------
    >>> from psi_io._models import extract_quantity_from_filepath
    >>> from pathlib import Path
    >>> extract_quantity_from_filepath(Path('br001001.h5'))
    'br'
    >>> extract_quantity_from_filepath(Path('heat001.h5'))
    'heat'
    >>> extract_quantity_from_filepath(Path('run_br_001.h5'))
    'br'
    >>> extract_quantity_from_filepath(Path('unknown.h5')) is None
    True
    >>> extract_quantity_from_filepath(Path('unknown.h5'), default='br')
    'br'
    """
    match = re.search(rf'(?<![a-zA-Z])({MATCH_QUANTITIES})(?![a-zA-Z])', ifile.stem, re.IGNORECASE)
    return match.group(1).lower() if match else default


def extract_sequence_from_filepath(ifile: Path, default: Optional[int] = None) -> int | None:
    """Extract the sequence number from a filename stem.

    Searches for the first 3- or 6-digit decimal token in the stem that is not
    immediately preceded or followed by another digit.  A 6-digit run is always
    preferred over a 3-digit run at the same position.

    Parameters
    ----------
    ifile : Path
        File path whose stem is examined.
    default : int or None, optional
        Value to return when no 3- or 6-digit token is found.  Defaults to ``None``.

    Returns
    -------
    out : int or None
        Integer sequence number, or *default* if no match is found.

    Examples
    --------
    >>> from pathlib import Path
    >>> from psi_io.mhd_io import extract_sequence_from_filepath
    >>> extract_sequence_from_filepath(Path('br001001.h5'))
    1001
    >>> extract_sequence_from_filepath(Path('vr001.h5'))
    1
    >>> extract_sequence_from_filepath(Path('nosequence.h5')) is None
    True
    """
    match = re.search(r'(?<!\d)\d{3}(?:\d{3})?(?!\d)', ifile.stem)
    return int(match.group()) if match else default


def parse_psi_filename_schema(ifile: Path):
    """Parse a PSI HDF filename that follows the strict ``<quantity><sequence>`` schema.

    The filename stem must consist of exactly one recognized MAS/POT3D quantity name
    followed immediately by a 3- or 6-digit sequence number, with no other characters.
    The match is case-insensitive.

    Parameters
    ----------
    ifile : Path
        File path to parse.  The stem is matched against :data:`FILEPATH_SCHEMA`.

    Returns
    -------
    quantity : str
        Lower-case quantity name (e.g. ``'br'``).
    sequence : int
        Integer sequence number (e.g. ``1001``).

    Raises
    ------
    ValueError
        If the filename stem does not match the expected schema.

    Examples
    --------
    >>> from pathlib import Path
    >>> from psi_io._models import parse_psi_filename_schema
    >>> parse_psi_filename_schema(Path('br001001.h5'))
    ('br', 1001)
    >>> parse_psi_filename_schema(Path('heat001.hdf'))
    ('heat', 1)
    >>> parse_psi_filename_schema(Path('notvalid.h5'))
    Traceback (most recent call last):
        ...
    ValueError: Filename 'notvalid.h5' does not match expected MAS filename schema: ...
    """
    matches = re.match(FILEPATH_SCHEMA, ifile.stem, re.IGNORECASE)
    if not matches:
        raise ValueError(f"Filename '{ifile.name}' does not match expected MAS filename schema: "
                         f"'<quantity><sequence>'. Valid quantities are: {', '.join(_MAS_QUANTITY_PROPS_MAPPING.keys())}. "
                         f"Sequence should be a 3 or 6 digit number.")
    quantity, sequence = matches.groups()
    return quantity, int(sequence)