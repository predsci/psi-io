from __future__ import annotations

from types import MappingProxyType

import astropy.units as u
import numpy as np


FAVORED_UNITS = {u.erg, u.cm, u.s, u.K, u.Gauss, u.g, u.rad}


def compose_mas_units(quantity: u.Quantity) -> u.Quantity:
    """
    Compose the input units with the favored MAS style CGS units
    """
    return quantity.to(quantity.unit.compose(units=FAVORED_UNITS)[0])


def decompose_mas_units(quantity: u.Quantity) -> u.Quantity:
    """
    Decompose the input units into the favored MAS style CGS units
    """
    return quantity.decompose(bases=list(FAVORED_UNITS))


# constants with physical units
PI = 3.1415926535897932
RSUN = 6.96e10*u.cm
G0PHYS = 0.274e5*u.cm/u.s**2
FN0PHYS = 1e8/u.cm**3
FMP = 1.6726e-24*u.g
BOLTZ = 1.3807e-16*u.erg/u.K
FKSPITZ = 9e-7  # TODO: LOOK UP THESE UNITS!

# custom units
FLUX_UNIT = u.erg/(u.cm**2*u.s)
VOLUMETRIC_RATE_UNIT = u.erg/(u.cm**3*u.s)

# normalization choices
G0NORM = 0.823
FNORML = RSUN

# -------------------------------------------------
# derived MAS normalizations
# -------------------------------------------------
FNORMT = np.sqrt(G0NORM*FNORML/G0PHYS)
FNORMM = FN0PHYS*FMP*np.power(FNORML, 3)
FN_N = FN0PHYS
FN_RHO = FMP*FN0PHYS
FN_T = ((FMP*FNORML**2/FNORMT**2).to(u.erg)/BOLTZ).to(u.MK)
FN_V = (FNORML/FNORMT).to(u.km/u.s)

# B conversion is tricky since CGS has inconsistent electromagnetic units...
MU0 = 4*PI*1e-7*u.N/u.A**2  # SI definition of mu0
FN_B = (np.sqrt(MU0*FN_RHO)*FNORML/FNORMT).to(u.Gauss)

# Convert J to SI units since that is what most people use for current density
FN_J = (FN_B.to(u.Tesla)/FNORML.to(u.m)/MU0).to(u.ampere/u.m**2)

# Convert E to SI units since that is what most people use for current density
FN_E = (FN_B.to(u.Tesla)*FN_V.to(u.m/u.s)).to(u.Volt/u.m)

# J and E in CGS units (convert statamp/statvolts and the length, this buries the c factor)
C_CGS = 2.99792458e10
STATAMP_TO_AMPERE = 10/C_CGS
FN_J_CGS = FN_J.value/(1e4*STATAMP_TO_AMPERE)*u.statA/u.cm**2 #statamp/cm**2
STATVOLT_TO_VOLT = C_CGS*1e-8
FN_E_CGS = FN_E.value/(1e2*STATVOLT_TO_VOLT)*u.erg/u.statC/u.cm  # statvolt/cm

# use compose to set these into typical solar units automatically
FN_P = compose_mas_units(FMP*FN0PHYS*FNORML**2/FNORMT**2)
FN_QRAD = compose_mas_units(FMP*np.power(FNORML, 2)/(FN0PHYS*np.power(FNORMT, 3)))
FN_KAPPA = compose_mas_units(BOLTZ*FN0PHYS*np.power(FNORML, 2)/FNORMT)
FN_FLUX = compose_mas_units(FN_P*FN_V).to(FLUX_UNIT)
FN_HEAT = compose_mas_units(FN_P/FNORMT).to(VOLUMETRIC_RATE_UNIT)

# misc
FN_JB = compose_mas_units(FN_P/FNORML)

# total energy conversions
W = FN_P*FNORML**3
K = W

# aliases
FN_LENGTH = FNORML
FN_TIME = FNORMT

# default rotation rate we use for the corotating frame (convert the MAS value to physical)
OMEGA_COROTATE=0.004144*u.rad/FN_TIME


def get_helium_fractions(he_frac):
    """
    Function to compute fractional abundances based on he_frac
    - these values ARE NOT stored in the module

    Assume gas of electrons (e), protons (p), and alphas (a).
    alpha particle mass = 4*mp, alpha particle charge = 2

    From charge neutrality: ne = np + 2na
    --> np/ne = 1/(1+2f)
    --> na/ne = f/(1+2f)
    """
    # Mass Density Multiplier: A = (np + 4*na)/ne
    # --> rho_mas = ne_mas*he_rho
    he_rho = (1 + 4*he_frac)/(1 + 2*he_frac)

    # Total number of particles: n = ne + np + na
    # --> for 1T: p_mas = ne_mas*T_mas*he_p = rho_mas*T_mas*he_p/he_rho
    he_p = (2 + 3*he_frac)/(1 + 2*he_frac)

    # Number of protons (used by radloss, which needs ne*np): np = ne*1/(1+2f) = rho_mas/he_rho/(1+2f) = rho_mas/(1+4f)
    he_np = 1/(1 + 2*he_frac)

    # Number of electrons, ne: (used in 2T advance, since Te = Telectrons)
    # --> Pe = rho_mas/he_rho*he_p_e
    he_p_e = 1.0

    # Number of ions (protons and alphas), np+na: (used in 2T advance, since Tp == Tprotons + Talphas)
    # --> Pi = rho_mas/he_rho*he_p_p
    he_p_p = (1 + he_frac)/(1 + 2*he_frac)

    he_dict = {
        'he_rho': he_rho,
        'he_p': he_p,
        'he_np': he_np,
        'he_p_e': he_p_e,
        'he_p_p': he_p_p
    }

    return he_dict


MAS_b = u.def_unit(
    [f"MAS_{q}" for q in ("b", "br", "bt", "bp")],
    FN_B,
    doc="PSI's MAS magnetic field normalization unit.",
    format={"latex": r"B_\mathrm{MAS}"},
)
MAS_v = u.def_unit(
    [f"MAS_{q}" for q in ("v", "vr", "vt", "vp", "zp", "zm")],
    FN_V,
    doc="PSI's MAS velocity normalization unit.",
    format={"latex": r"v_\mathrm{MAS}"},
)
MAS_j = u.def_unit(
    [f"MAS_{q}" for q in ("j", "jr", "jt", "jp")],
    FN_J,
    doc="PSI's MAS current density normalization unit.",
    format={"latex": r"J_\mathrm{MAS}"},
)
MAS_t = u.def_unit(
    [f"MAS_{q}" for q in ("t", "te", "tp")],
    FN_T,
    doc="PSI's MAS temperature normalization unit.",
    format={"latex": r"T_\mathrm{MAS}"},
)
MAS_n = u.def_unit(
    [f"MAS_{q}" for q in ("n", "rho")],
    FN_N,
    doc="PSI's MAS (number) density normalization unit.",
    format={"latex": r"n_\mathrm{MAS}"},
)
MAS_p = u.def_unit(
    [f"MAS_{q}" for q in ("p", "ep", "em")],
    FN_P,
    doc="PSI's MAS pressure normalization unit.",
    format={"latex": r"p_\mathrm{MAS}"},
)
MAS_heat = u.def_unit(
    [f"MAS_{q}" for q in ("heat",)],
    FN_HEAT,
    doc="PSI's MAS volumetric heating rate normalization unit.",
    format={"latex": r"heat_\mathrm{MAS}"},
)
POT3D_b = u.def_unit(
    [f"POT3D_{q}" for q in ("b", "br", "bt", "bp")],
    1 * u.dimensionless_unscaled,
    doc="PSI's POT3D magnetic field normalization unit.",
    format={"latex": r"B_\mathrm{POT3D}"},
)
PSI_rsun = u.def_unit(
    [f"PSI_{q}" for q in ("rsun", "radius", "r")],
    RSUN,
    doc="PSI's solar radius normalization unit.",
    format={"latex": r"R_\odot"},
)
PSI_angle = u.def_unit(
    [f"PSI_{q}" for q in ("angle", "t", "p", "theta", "phi")],
    1 * u.rad,
    doc="PSI's long-lat angle unit.",
    format={"latex": r"R_\odot"},
)


u.add_enabled_units([MAS_b, MAS_v, MAS_j, MAS_t, MAS_n, MAS_p, MAS_heat, POT3D_b, PSI_rsun, PSI_angle])

