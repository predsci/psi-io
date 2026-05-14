r"""Physical constants and normalization factors for PSI MAS model output.

The MAS (Magnetohydrodynamic Algorithm outside a Sphere) code solves the resistive
MHD equations in *dimensionless* form.  Every quantity stored in an HDF output file is
a dimensionless ratio

.. math::

   q_\text{code} = \frac{q_\text{phys}}{q_0}

where :math:`q_0` is the characteristic scale for that quantity.  This module
collects all reference scales so that code-unit values can be converted to physical
(CGS or SI) units via :mod:`astropy.units`.

Normalization scheme
--------------------
MAS adopts a normalization anchored to two observable solar parameters and one
reference number density:

.. list-table::
   :header-rows: 1

   * - Symbol
     - Quantity
     - Value
   * - :data:`FNORML` = :data:`RSUN`
     - Characteristic length
     - :math:`6.96 \times 10^{10}` cm
   * - :data:`G0PHYS`
     - Solar surface gravity
     - :math:`2.74 \times 10^4` cm s⁻²
   * - :data:`FN0PHYS`
     - Reference number density
     - :math:`10^8` cm⁻³

From these three anchors, the characteristic **time** is fixed by requiring the
dimensionless surface gravity :math:`G_0^\text{norm} = 0.823`:

.. math::

   T_0 = \sqrt{\frac{G_0^\text{norm}\, R_\odot}{g_0}} \approx 1446 \;\text{s}

All other normalization factors follow from dimensional analysis:

.. list-table::
   :header-rows: 1

   * - Symbol
     - Quantity
     - Formula
   * - :data:`FN_V`
     - Velocity
     - :math:`V_0 = L_0 / T_0 \approx 481` km s⁻¹
   * - :data:`FN_RHO`
     - Mass density
     - :math:`\rho_0 = m_p n_0`
   * - :data:`FN_T`
     - Temperature
     - :math:`T_0^\text{temp} = m_p V_0^2 / k_B \approx 28` MK
   * - :data:`FN_P`
     - Pressure
     - :math:`P_0 = \rho_0 V_0^2`
   * - :data:`FN_B`
     - Magnetic field
     - :math:`B_0 = V_0 \sqrt{\mu_0 \rho_0} \approx 2.2` G
   * - :data:`FN_J`
     - Current density (SI)
     - :math:`J_0 = B_0 / (\mu_0 L_0)` A m⁻²
   * - :data:`FN_HEAT`
     - Volumetric heating rate
     - :math:`Q_0 = P_0 / T_0` erg cm⁻³ s⁻¹

Electromagnetic units
---------------------
MAS is formulated in Gaussian CGS, where Ampère's law reads
:math:`\mathbf{J} = (c/4\pi)\,\nabla \times \mathbf{B}`.  Because current density
in Gaussian CGS (statampere cm⁻²) is unfamiliar to most users, this module provides
both SI (:data:`FN_J`, :data:`FN_E`) and Gaussian CGS (:data:`FN_J_CGS`,
:data:`FN_E_CGS`) normalizations for these quantities.

Magnetic field is expressed in Gauss throughout, since that system is unambiguous and
Gauss is the conventional unit for coronal magnetic field strengths.

Custom astropy units
--------------------
Each MAS quantity has a dedicated :mod:`astropy.units` unit registered at module
import time (e.g. ``MAS_b``, ``MAS_v``).  These units carry the conversion factor
from code units to physical units so that astropy can chain unit conversions
automatically.

See Also
--------
psi_io._props : Maps each MAS quantity name to its :data:`FN_*` normalization.
psi_io.mhd_io : Lazy readers that apply these normalizations on property access.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np


FAVORED_UNITS = {u.erg, u.cm, u.s, u.K, u.Gauss, u.g, u.rad}
"""Set of CGS units preferred when composing or decomposing MAS quantities.

Used by :func:`compose_mas_units` and :func:`decompose_mas_units` to express
results in the most physically intuitive CGS basis: erg, cm, s, K, Gauss, g, rad.
Gauss is included explicitly because :func:`astropy.units.compose` would otherwise
resolve magnetic field to mixed electromagnetic CGS units.
"""


def compose_mas_units(quantity: u.Quantity) -> u.Quantity:
    """Express a quantity in the preferred MAS CGS unit basis.

    Calls :meth:`astropy.units.UnitBase.compose` with :data:`FAVORED_UNITS` as the
    allowed unit set.  The first (simplest) composed unit that astropy returns is used.

    Parameters
    ----------
    quantity : astropy.units.Quantity
        Input quantity in any unit system.

    Returns
    -------
    out : astropy.units.Quantity
        Equivalent quantity expressed in the MAS-preferred CGS unit basis
        (erg, cm, s, K, Gauss, g, rad).

    See Also
    --------
    decompose_mas_units : Decomposes to CGS *base* units rather than composing.

    Examples
    --------
    >>> import astropy.units as u
    >>> from psi_io._units import compose_mas_units, FN_P
    >>> compose_mas_units(1.0 * u.Pa)        # doctest: +ELLIPSIS
    <Quantity ... erg / cm3>
    """
    return quantity.to(quantity.unit.compose(units=FAVORED_UNITS)[0])


def decompose_mas_units(quantity: u.Quantity) -> u.Quantity:
    """Decompose a quantity into the preferred MAS CGS base units.

    Calls :meth:`astropy.units.Quantity.decompose` with :data:`FAVORED_UNITS` as the
    allowed bases.  Unlike :func:`compose_mas_units`, this always expands compound
    units (e.g. Joule → erg) without attempting to find a compact composed form.

    Parameters
    ----------
    quantity : astropy.units.Quantity
        Input quantity in any unit system.

    Returns
    -------
    out : astropy.units.Quantity
        Equivalent quantity with all units decomposed into the MAS-preferred CGS
        bases.

    See Also
    --------
    compose_mas_units : Composes to a simpler combined CGS unit where possible.

    Examples
    --------
    >>> import astropy.units as u
    >>> from psi_io._units import decompose_mas_units
    >>> decompose_mas_units(1.0 * u.J)       # doctest: +ELLIPSIS
    <Quantity 1.e+07 erg>
    """
    return quantity.decompose(bases=list(FAVORED_UNITS))


# =============================================================================
# Physical constants
# =============================================================================

PI = 3.1415926535897932
"""High-precision value of π.

Stored as a module-level float to avoid repeated calls to :func:`math.pi` or
:data:`numpy.pi` in constant expressions evaluated at import time.
"""

RSUN = 6.96e10 * u.cm
"""Solar radius in centimetres.

The canonical value :math:`R_\\odot = 6.96 \\times 10^{10}` cm used throughout PSI
models as the characteristic length scale.  Also aliased as :data:`FN_LENGTH` and
:data:`FNORML`.
"""

G0PHYS = 0.274e5 * u.cm / u.s**2
"""Solar surface gravitational acceleration in CGS.

:math:`g_0 = 2.74 \\times 10^4` cm s⁻², the standard solar surface gravity.
Together with :data:`RSUN` and :data:`G0NORM`, this fixes the MAS time normalization
:data:`FNORMT`.
"""

FN0PHYS = 1e8 / u.cm**3
"""Reference number density of the coronal base in CGS.

:math:`n_0 = 10^8` cm⁻³, a representative electron (or proton) number density at
:math:`r = 1\\,R_\\odot` for a quiescent solar corona.  Used together with
:data:`FMP` to set the mass-density normalization :data:`FN_RHO`.
"""

FMP = 1.6726e-24 * u.g
"""Proton mass in CGS.

:math:`m_p = 1.6726 \\times 10^{-24}` g.  The MAS density normalization assumes a
fully ionized hydrogen plasma, so mass density is proportional to number density
through :math:`\\rho_0 = m_p n_0`.
"""

BOLTZ = 1.3807e-16 * u.erg / u.K
"""Boltzmann constant in CGS.

:math:`k_B = 1.3807 \\times 10^{-16}` erg K⁻¹.  Used to derive the temperature
normalization :data:`FN_T` from the kinetic pressure relation.
"""

FKSPITZ = 9e-7  # erg cm^-1 s^-1 K^(-7/2)
"""Spitzer parallel thermal conductivity prefactor in CGS.

The Spitzer heat flux along the magnetic field is

.. math::

   \\mathbf{q} = -\\kappa_0 T^{5/2}\\, \\hat{\\mathbf{b}}\\hat{\\mathbf{b}} \\cdot \\nabla T

where :math:`\\kappa_0 = 9 \\times 10^{-7}` erg cm⁻¹ s⁻¹ K⁻⁷/² is the Spitzer
conductivity coefficient for a fully ionized hydrogen plasma.  This constant is stored
as a plain :class:`float` (not an astropy :class:`~astropy.units.Quantity`) because it
is used directly in normalized code expressions; the associated physical units are
erg cm⁻¹ s⁻¹ K⁻⁷/².

See :data:`FN_KAPPA` for the conductivity normalization that converts the dimensionless
code conductivity back to physical units.
"""


# =============================================================================
# Convenience unit containers
# =============================================================================

FLUX_UNIT = u.erg / (u.cm**2 * u.s)
"""Astropy unit for surface energy flux: erg cm⁻² s⁻¹."""

VOLUMETRIC_RATE_UNIT = u.erg / (u.cm**3 * u.s)
"""Astropy unit for volumetric energy deposition rate: erg cm⁻³ s⁻¹."""


# =============================================================================
# Normalization anchors
# =============================================================================

G0NORM = 0.823
"""Dimensionless surface gravity in MAS code units.

MAS defines a dimensionless gravitational acceleration profile

.. math::

   g_\\text{code}(r) = \\frac{G_0^\\text{norm}}{r^2}

so that the surface value :math:`g_\\text{code}(1) = G_0^\\text{norm} = 0.823`.
The physical surface gravity is then :math:`g_0 = G_0^\\text{norm}\\,L_0/T_0^2`,
which gives

.. math::

   T_0 = \\sqrt{\\frac{G_0^\\text{norm}\\,L_0}{g_0}} \\approx 1446\\;\\text{s}.

The value 0.823 is a calibration choice made when deriving :data:`FNORMT`; it is not
a universal solar constant.
"""

FNORML = RSUN
"""Characteristic length scale for MAS: the solar radius.

:math:`L_0 = R_\\odot = 6.96 \\times 10^{10}` cm.  All spatial coordinates in MAS
output are in units of :math:`L_0`.  Aliased as :data:`FN_LENGTH`.
"""


# =============================================================================
# Derived normalization factors
# =============================================================================

FNORMT = np.sqrt(G0NORM * FNORML / G0PHYS)
"""Characteristic time scale for MAS.

Derived from the dimensionless surface gravity :data:`G0NORM`, the characteristic
length :data:`FNORML`, and the physical surface gravity :data:`G0PHYS`:

.. math::

   T_0 = \\sqrt{\\frac{G_0^\\text{norm}\\,R_\\odot}{g_0}}
       \\approx 1446\\;\\text{s} \\approx 24\\;\\text{min}.

Aliased as :data:`FN_TIME`.
"""

FNORMM = FN0PHYS * FMP * np.power(FNORML, 3)
"""Characteristic mass scale for MAS.

:math:`M_0 = n_0\\,m_p\\,L_0^3 = \\rho_0\\,L_0^3`.  Represents the total mass
contained in a cube of side :math:`L_0 = R_\\odot` at the reference density.
"""

FN_N = FN0PHYS
"""Number density normalization: :math:`n_0 = 10^8` cm⁻³.

Code-unit number density 1 corresponds to :math:`10^8` particles cm⁻³.
Note that MAS stores *mass* density (dimensionless :math:`\\rho`), related to the
number density by :math:`\\rho_\\text{code} = (m_p / n_0) \\cdot n`.  See
:data:`FN_RHO` for the mass-density normalization.
"""

FN_RHO = FMP * FN0PHYS
"""Mass density normalization for MAS.

:math:`\\rho_0 = m_p\\,n_0 \\approx 1.67 \\times 10^{-16}` g cm⁻³, assuming a
fully ionized hydrogen plasma where the mass per particle is the proton mass.
"""

FN_T = ((FMP * FNORML**2 / FNORMT**2).to(u.erg) / BOLTZ).to(u.MK)
"""Temperature normalization for MAS.

Derived from the condition that the ideal-gas pressure :math:`P = n\\,k_B\\,T` equals
the dynamic pressure :math:`\\rho_0\\,V_0^2` at code-unit temperature 1:

.. math::

   T_0 = \\frac{m_p\\,V_0^2}{k_B} = \\frac{m_p\\,L_0^2}{k_B\\,T_0^2}
       \\approx 28\\;\\text{MK}.

Code-unit temperatures of order unity therefore correspond to tens of megakelvin,
consistent with hot coronal plasma.
"""

FN_V = (FNORML / FNORMT).to(u.km / u.s)
"""Velocity normalization for MAS.

:math:`V_0 = L_0 / T_0 \\approx 481` km s⁻¹.  This is comparable to the fast solar
wind speed, making code-unit velocities of order unity physically natural for
heliospheric modelling.
"""

# Magnetic field normalization requires a CGS–SI bridge.
# MAS is formulated in Gaussian CGS but astropy's CGS electromagnetic units are
# incomplete.  The SI μ₀ is used here as a conversion bridge: in Gaussian CGS the
# Alfvénic normalization B₀ = V₀ √(4πρ₀) is algebraically equivalent to the SI
# expression B₀ = V₀ √(μ₀ ρ₀) once the result is converted to Gauss.
MU0 = 4 * PI * 1e-7 * u.N / u.A**2
"""Permeability of free space (SI).

:math:`\\mu_0 = 4\\pi \\times 10^{-7}` N A⁻², used here as an algebraic bridge to
compute the magnetic field normalization :data:`FN_B` and current density
normalization :data:`FN_J` in SI, which are then converted to the required CGS or
SI output units.

In Gaussian CGS, the Alfvénic magnetic normalization is
:math:`B_0 = V_0\\sqrt{4\\pi\\rho_0}`, which is numerically equivalent to the SI
expression :math:`B_0 = V_0\\sqrt{\\mu_0\\rho_0}` when both results are expressed in
Gauss.
"""

FN_B = (np.sqrt(MU0 * FN_RHO) * FNORML / FNORMT).to(u.Gauss)
"""Magnetic field normalization for MAS, in Gauss.

The code B field is normalized so that the Alfvén speed equals the reference velocity
:data:`FN_V` when :math:`B_\\text{code} = 1`:

.. math::

   B_0 = V_0\\sqrt{\\mu_0\\,\\rho_0} \\approx 2.2\\;\\text{G}.

This is on the order of typical large-scale coronal magnetic field strengths.
"""

FN_J = (FN_B.to(u.Tesla) / FNORML.to(u.m) / MU0).to(u.ampere / u.m**2)
"""Current density normalization for MAS, in SI (A m⁻²).

From Ampère's law :math:`\\mathbf{J} = \\mu_0^{-1}\\nabla\\times\\mathbf{B}` (SI):

.. math::

   J_0 = \\frac{B_0}{\\mu_0\\,L_0}.

SI units are used here because current density in Gaussian CGS (statampere cm⁻²) is
rarely used in practice.  See :data:`FN_J_CGS` for the Gaussian equivalent.
"""

FN_E = (FN_B.to(u.Tesla) * FN_V.to(u.m / u.s)).to(u.Volt / u.m)
"""Electric field normalization for MAS, in SI (V m⁻¹).

In ideal MHD the electric field is :math:`\\mathbf{E} = -\\mathbf{v}\\times\\mathbf{B}`,
so the natural normalization is

.. math::

   E_0 = V_0\\,B_0.

SI units are used for the same reasons as :data:`FN_J`.  See :data:`FN_E_CGS` for the
Gaussian equivalent.
"""

# ---------------------------------------------------------------------------
# Gaussian CGS conversions for J and E
# ---------------------------------------------------------------------------
C_CGS = 2.99792458e10
"""Speed of light in CGS: :math:`c = 2.998 \\times 10^{10}` cm s⁻¹.

Used to convert between SI and Gaussian electromagnetic units.
"""

STATAMP_TO_AMPERE = 10 / C_CGS
"""Conversion factor from statampere to ampere.

In Gaussian CGS, charge is measured in statcoulombs (esu).  One statampere
(statcoulomb per second) equals :math:`10/c` amperes, where :math:`c` is the speed
of light in cm s⁻¹:

.. math::

   1\\;\\text{statA} = \\frac{10}{c}\\;\\text{A} \\approx 3.336 \\times 10^{-10}\\;\\text{A}.
"""

FN_J_CGS = FN_J.value / (1e4 * STATAMP_TO_AMPERE) * u.statA / u.cm**2
"""Current density normalization for MAS, in Gaussian CGS (statA cm⁻²).

Equivalent to :data:`FN_J` expressed in Gaussian electromagnetic units.  The
conversion from SI to Gaussian CGS is:

.. math::

   1\\;\\text{A m}^{-2} = \\frac{1}{10^4 \\times (10/c)}\\;\\text{statA cm}^{-2}

where :math:`c` is :data:`C_CGS`.
"""

STATVOLT_TO_VOLT = C_CGS * 1e-8
"""Conversion factor from statvolt to volt.

One statvolt (erg per statcoulomb) equals :math:`c \\times 10^{-8}` volts, where
:math:`c` is the speed of light in cm s⁻¹:

.. math::

   1\\;\\text{statV} = c \\times 10^{-8}\\;\\text{V} \\approx 299.8\\;\\text{V}.
"""

FN_E_CGS = FN_E.value / (1e2 * STATVOLT_TO_VOLT) * u.erg / u.statC / u.cm
"""Electric field normalization for MAS, in Gaussian CGS (statV cm⁻¹).

Equivalent to :data:`FN_E` expressed in Gaussian electromagnetic units (statvolt per
centimetre, where 1 statvolt = 1 erg / statcoulomb).  The conversion from SI to
Gaussian CGS is:

.. math::

   1\\;\\text{V m}^{-1} = \\frac{1}{10^2 \\times c \\times 10^{-8}}\\;\\text{statV cm}^{-1}

where :math:`c` is :data:`C_CGS`.
"""

# ---------------------------------------------------------------------------
# Thermodynamic and wave-energy normalizations
# ---------------------------------------------------------------------------

FN_P = compose_mas_units(FMP * FN0PHYS * FNORML**2 / FNORMT**2)
"""Pressure normalization for MAS.

The dynamic pressure :math:`P_0 = \\rho_0\\,V_0^2 = m_p\\,n_0\\,(L_0/T_0)^2`,
expressed in the preferred MAS CGS basis (erg cm⁻³).  Code-unit pressure 1 maps to
this value.
"""

FN_QRAD = compose_mas_units(FMP * np.power(FNORML, 2) / (FN0PHYS * np.power(FNORMT, 3)))
"""Radiative loss function normalization for MAS.

The radiative loss per unit volume in the energy equation takes the form
:math:`Q_\\text{rad} = n_e\\,n_p\\,\\Lambda(T)`, where :math:`\\Lambda(T)` is the
optically thin loss function (erg cm³ s⁻¹).  In MAS code units, the dimensionless
loss function :math:`\\Lambda_\\text{code}` satisfies

.. math::

   \\Lambda_\\text{phys} = \\Lambda_\\text{code} \\times
   \\frac{m_p\\,L_0^2}{n_0\\,T_0^3}.

This normalization ensures the radiative source term enters the dimensionless energy
equation with a coefficient of order unity for typical coronal conditions.
"""

FN_KAPPA = compose_mas_units(BOLTZ * FN0PHYS * np.power(FNORML, 2) / FNORMT)
"""Thermal conductivity normalization for MAS.

The Spitzer parallel conductivity :math:`\\kappa = \\kappa_0 T^{5/2}` (with
:data:`FKSPITZ` = :math:`\\kappa_0`) has units of erg cm⁻¹ s⁻¹ K⁻¹.  In MAS code
units the dimensionless conductivity :math:`\\kappa_\\text{code}` satisfies

.. math::

   \\kappa_\\text{phys} = \\kappa_\\text{code} \\times
   \\frac{k_B\\,n_0\\,L_0^2}{T_0}.

The dimensionless Spitzer coefficient stored in the code is therefore

.. math::

   F_\\kappa^\\text{code} = \\kappa_0
   \\frac{T_0^{5/2}}{k_B\\,n_0\\,L_0^2/T_0\\cdot T_0^{5/2}}
   = \\frac{\\kappa_0\\,T_0^{5/2}}{F_\\kappa}.
"""

FN_FLUX = compose_mas_units(FN_P * FN_V).to(FLUX_UNIT)
"""Surface energy flux normalization for MAS.

:math:`F_0 = P_0\\,V_0` in erg cm⁻² s⁻¹.  This scale governs the Poynting flux,
enthalpy flux, and wave-energy flux that appear in the energy equation boundary
conditions.
"""

FN_HEAT = compose_mas_units(FN_P / FNORMT).to(VOLUMETRIC_RATE_UNIT)
"""Volumetric heating rate normalization for MAS.

:math:`Q_0 = P_0 / T_0` in erg cm⁻³ s⁻¹.  Code-unit heating rate 1 corresponds to
the reference dynamic pressure dissipated over one characteristic time.  This sets
the scale for empirical coronal heating prescriptions and wave-turbulence heating in
MAS.
"""

FN_JB = compose_mas_units(FN_P / FNORML)
"""Lorentz force density (and pressure gradient) normalization for MAS.

:math:`(\\mathbf{J}\\times\\mathbf{B})_0 = P_0 / L_0` in erg cm⁻⁴ (equivalent to
dyne cm⁻³).  Both the magnetic Lorentz force and the plasma pressure gradient appear
in the MAS momentum equation with this normalization, ensuring they enter on equal
footing.
"""

# ---------------------------------------------------------------------------
# Total energy scales
# ---------------------------------------------------------------------------

W = FN_P * FNORML**3
"""Total (integrated) energy normalization for MAS.

:math:`W_0 = P_0\\,L_0^3` in erg.  The total magnetic, kinetic, or thermal energy
integrated over a domain of volume :math:`L_0^3` at the reference pressure.  Used
when computing globally integrated energy budgets.
"""

K = W
"""Kinetic energy normalization for MAS.

Alias for :data:`W`.  The kinetic and pressure-volume energy scales are identical
since :math:`K_0 = \\tfrac{1}{2}\\rho_0 V_0^2 L_0^3 = \\tfrac{1}{2} P_0 L_0^3`;
the factor of one-half is absorbed into the code-unit kinetic energy.
"""

# ---------------------------------------------------------------------------
# Aliases and corotating frame
# ---------------------------------------------------------------------------

FN_LENGTH = FNORML
"""Alias for :data:`FNORML`: the characteristic length scale :math:`L_0 = R_\\odot`."""

FN_TIME = FNORMT
"""Alias for :data:`FNORMT`: the characteristic time scale :math:`T_0 \\approx 1446` s."""

OMEGA_COROTATE = 0.004144 * u.rad / FN_TIME
"""Default angular velocity of the MAS corotating frame, in physical units.

MAS can be run in a frame rotating with the Sun.  The dimensionless code rotation
rate 0.004144 rad per time unit converts to

.. math::

   \\Omega_\\text{corotate} = \\frac{0.004144}{T_0} \\approx 2.87 \\times 10^{-6}\\;\\text{rad s}^{-1},

which matches the solar sidereal rotation rate (period ≈ 25.4 days).
"""


# =============================================================================
# Helium abundance corrections
# =============================================================================

def get_helium_fractions(he_frac: float) -> dict[str, float]:
    """Compute fractional abundance multipliers for a hydrogen–helium plasma.

    MAS models the solar corona as a gas of electrons (e), protons (p), and alpha
    particles (a).  Given the fractional helium abundance :math:`f = n_a / n_e`, this
    function returns dimensionless multipliers that relate the MAS code-unit density
    and pressure to the individual species number densities.

    The multipliers are derived from two constraints:

    1. **Charge neutrality**: :math:`n_e = n_p + 2 n_a`, giving
       :math:`n_p/n_e = 1/(1+2f)` and :math:`n_a/n_e = f/(1+2f)`.
    2. **Alpha particle properties**: mass :math:`4 m_p`, charge :math:`2e`.

    Parameters
    ----------
    he_frac : float
        Helium fraction :math:`f = n_a / n_e`, the ratio of alpha-particle number
        density to electron number density.  Typical coronal value: ``0.05``.

    Returns
    -------
    fractions : dict[str, float]
        Dictionary with the following keys:

        ``'he_rho'``
            Mass-density multiplier.  The code mass density :math:`\\rho_\\text{code}`
            is related to the electron number density by
            :math:`\\rho_\\text{code} = n_e \\cdot` ``he_rho``, where

            .. math::

               \\texttt{he\\_rho} = \\frac{n_p + 4 n_a}{n_e} = \\frac{1 + 4f}{1 + 2f}.

        ``'he_p'``
            Total pressure multiplier for a single-temperature plasma.  For an ideal
            gas the total particle pressure is :math:`p = (n_e + n_p + n_a)\\,k_B T`,
            so

            .. math::

               \\texttt{he\\_p} = \\frac{n_e + n_p + n_a}{n_e} = \\frac{2 + 3f}{1 + 2f}.

        ``'he_np'``
            Proton fraction :math:`n_p / n_e = 1/(1 + 2f)`.  Needed for radiative
            loss rates that scale as :math:`n_e n_p`.

        ``'he_p_e'``
            Electron pressure multiplier (equal to 1.0).  In the two-temperature
            model the electron partial pressure is :math:`p_e = n_e k_B T_e`, so
            the multiplier relative to the electron density is always unity.

        ``'he_p_p'``
            Ion pressure multiplier.  The combined proton + alpha pressure is
            :math:`p_i = (n_p + n_a)\\,k_B T_i`, giving

            .. math::

               \\texttt{he\\_p\\_p} = \\frac{n_p + n_a}{n_e} = \\frac{1 + f}{1 + 2f}.

    Notes
    -----
    These multipliers are *not* stored as module-level constants because they depend on
    ``he_frac``, which varies between simulations.  The caller is responsible for
    passing the correct helium fraction from the model metadata.

    Examples
    --------
    >>> from psi_io._units import get_helium_fractions
    >>> fracs = get_helium_fractions(0.0)   # pure hydrogen plasma
    >>> fracs['he_rho']
    1.0
    >>> fracs['he_p']
    2.0
    >>> fracs = get_helium_fractions(0.05)  # 5 % helium by electron fraction
    >>> round(fracs['he_rho'], 4)
    1.1818
    """
    # Mass Density Multiplier: A = (np + 4*na)/ne
    # --> rho_mas = ne_mas*he_rho
    he_rho = (1 + 4 * he_frac) / (1 + 2 * he_frac)

    # Total number of particles: n = ne + np + na
    # --> for 1T: p_mas = ne_mas*T_mas*he_p = rho_mas*T_mas*he_p/he_rho
    he_p = (2 + 3 * he_frac) / (1 + 2 * he_frac)

    # Number of protons (used by radloss, which needs ne*np)
    he_np = 1 / (1 + 2 * he_frac)

    # Electron pressure multiplier (used in 2T: Pe = ne*kB*Te → multiplier = 1)
    he_p_e = 1.0

    # Ion pressure multiplier (protons + alphas, used in 2T: Pi = (np+na)*kB*Ti)
    he_p_p = (1 + he_frac) / (1 + 2 * he_frac)

    return {
        'he_rho': he_rho,
        'he_p': he_p,
        'he_np': he_np,
        'he_p_e': he_p_e,
        'he_p_p': he_p_p,
    }


# =============================================================================
# Custom astropy unit definitions for MAS and POT3D code quantities
# =============================================================================

MAS_b = u.def_unit(
    [f"MAS_{q}" for q in ("b", "br", "bt", "bp")],
    FN_B,
    doc="PSI's MAS magnetic field normalization unit.",
    format={"latex": r"B_\mathrm{MAS}"},
)
"""Custom astropy unit for MAS magnetic field quantities.

One ``MAS_b`` equals :data:`FN_B` ≈ 2.2 Gauss.  Registered for the quantity names
``MAS_b``, ``MAS_br``, ``MAS_bt``, and ``MAS_bp`` so that astropy can automatically
convert code-unit field values to Gauss or Tesla.
"""

MAS_v = u.def_unit(
    [f"MAS_{q}" for q in ("v", "vr", "vt", "vp", "zp", "zm")],
    FN_V,
    doc="PSI's MAS velocity normalization unit.",
    format={"latex": r"v_\mathrm{MAS}"},
)
"""Custom astropy unit for MAS velocity quantities.

One ``MAS_v`` equals :data:`FN_V` ≈ 481 km s⁻¹.  Registered for velocity components
(``vr``, ``vt``, ``vp``) and Elsässer wave variables (``zp``, ``zm``).
"""

MAS_j = u.def_unit(
    [f"MAS_{q}" for q in ("j", "jr", "jt", "jp")],
    FN_J,
    doc="PSI's MAS current density normalization unit.",
    format={"latex": r"J_\mathrm{MAS}"},
)
"""Custom astropy unit for MAS current density quantities.

One ``MAS_j`` equals :data:`FN_J` in A m⁻² (SI).  Registered for the vector
components ``jr``, ``jt``, ``jp``.
"""

MAS_t = u.def_unit(
    [f"MAS_{q}" for q in ("t", "te", "tp")],
    FN_T,
    doc="PSI's MAS temperature normalization unit.",
    format={"latex": r"T_\mathrm{MAS}"},
)
"""Custom astropy unit for MAS temperature quantities.

One ``MAS_t`` equals :data:`FN_T` ≈ 28 MK.  Registered for the single-temperature
(``t``), electron-temperature (``te``), and proton-temperature (``tp``) variables.
"""

MAS_n = u.def_unit(
    [f"MAS_{q}" for q in ("n", "rho")],
    FN_N,
    doc="PSI's MAS (number) density normalization unit.",
    format={"latex": r"n_\mathrm{MAS}"},
)
"""Custom astropy unit for MAS number density and mass density.

One ``MAS_n`` equals :data:`FN_N` = 10⁸ cm⁻³.  Registered for both the number
density (``n``) and the code mass-density (``rho``) variables; callers should divide
``rho`` by :data:`FMP` to obtain a physical number density in cm⁻³.
"""

MAS_p = u.def_unit(
    [f"MAS_{q}" for q in ("p", "ep", "em")],
    FN_P,
    doc="PSI's MAS pressure normalization unit.",
    format={"latex": r"p_\mathrm{MAS}"},
)
"""Custom astropy unit for MAS pressure and wave-energy quantities.

One ``MAS_p`` equals :data:`FN_P` in erg cm⁻³.  Registered for the thermal pressure
(``p``), Elsässer energy densities (``ep``, ``em``), and Alfvén wave pressure
variables.
"""

MAS_heat = u.def_unit(
    [f"MAS_{q}" for q in ("heat",)],
    FN_HEAT,
    doc="PSI's MAS volumetric heating rate normalization unit.",
    format={"latex": r"heat_\mathrm{MAS}"},
)
"""Custom astropy unit for MAS volumetric heating rate.

One ``MAS_heat`` equals :data:`FN_HEAT` in erg cm⁻³ s⁻¹.
"""

POT3D_b = u.def_unit(
    [f"POT3D_{q}" for q in ("b", "br", "bt", "bp")],
    1 * u.dimensionless_unscaled,
    doc="PSI's POT3D magnetic field normalization unit.",
    format={"latex": r"B_\mathrm{POT3D}"},
)
"""Custom astropy unit for POT3D magnetic field quantities.

One ``POT3D_b`` equals 1 (dimensionless).  POT3D is a potential-field solver driven
by photospheric magnetogram boundary conditions; its output magnetic field is already
expressed in the same physical units as the input map (typically Gauss), so no
additional conversion factor is applied.
"""

PSI_rsun = u.def_unit(
    [f"PSI_{q}" for q in ("rsun", "radius", "r")],
    RSUN,
    doc="PSI's solar radius normalization unit.",
    format={"latex": r"R_\odot"},
)
"""Custom astropy unit for PSI radial coordinate grids.

One ``PSI_rsun`` equals :data:`RSUN` = 6.96 × 10¹⁰ cm.  Radial coordinate arrays
in MAS and POT3D HDF files are stored in units of solar radii; this unit allows
astropy to convert them to cm, km, or AU automatically.
"""

PSI_angle = u.def_unit(
    [f"PSI_{q}" for q in ("angle", "t", "p", "theta", "phi")],
    1 * u.rad,
    doc="PSI's long-lat angle unit.",
    format={"latex": r"R_\odot"},
)
"""Custom astropy unit for PSI angular coordinate grids.

One ``PSI_angle`` equals 1 radian.  Colatitude (θ) and longitude (φ) coordinate
arrays in MAS and POT3D HDF files are stored in radians; this unit allows astropy to
convert them to degrees where needed.
"""


u.add_enabled_units([MAS_b, MAS_v, MAS_j, MAS_t, MAS_n, MAS_p, MAS_heat, POT3D_b, PSI_rsun, PSI_angle])