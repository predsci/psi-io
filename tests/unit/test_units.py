"""Unit tests for psi_io._units."""

from __future__ import annotations

import math

import astropy.units as u
import pytest
from numpy.testing import assert_allclose

from psi_io._units import (
    BOLTZ,
    C_CGS,
    FAVORED_UNITS,
    FMP,
    FN0PHYS,
    FN_B,
    FN_HEAT,
    FN_J,
    FN_J_CGS,
    FN_N,
    FN_P,
    FN_RHO,
    FN_T,
    FN_V,
    FNORML,
    FNORMT,
    G0NORM,
    G0PHYS,
    MAS_b,
    MAS_heat,
    MAS_j,
    MAS_n,
    MAS_p,
    MAS_t,
    MAS_v,
    OMEGA_COROTATE,
    PI,
    POT3D_b,
    PSI_angle,
    PSI_rsun,
    RSUN,
    STATAMP_TO_AMPERE,
    compose_mas_units,
    decompose_mas_units,
    get_helium_fractions,
)


class TestPhysicalConstants:
    def test_pi_value(self):
        assert_allclose(PI, math.pi, rtol=1e-10)

    def test_rsun_value_cm(self):
        assert_allclose(RSUN.to(u.cm).value, 6.96e10, rtol=1e-10)

    def test_rsun_unit(self):
        assert RSUN.unit == u.cm

    def test_g0phys_value(self):
        assert_allclose(G0PHYS.to(u.cm / u.s**2).value, 2.74e4, rtol=1e-10)

    def test_fn0phys_value(self):
        assert_allclose(FN0PHYS.to(u.cm**-3).value, 1e8, rtol=1e-10)

    def test_fmp_value(self):
        assert_allclose(FMP.to(u.g).value, 1.6726e-24, rtol=1e-8)

    def test_boltz_converts_to_si(self):
        assert_allclose(BOLTZ.to(u.J / u.K).value, 1.380649e-23, rtol=1e-3)

    def test_c_cgs_value(self):
        assert_allclose(C_CGS, 2.99792458e10, rtol=1e-10)

    def test_statamp_to_ampere_formula(self):
        assert_allclose(STATAMP_TO_AMPERE, 10.0 / C_CGS, rtol=1e-10)


class TestDerivedNormalizations:
    def test_fnorml_is_rsun(self):
        assert FNORML is RSUN

    def test_fnormt_is_quantity(self):
        assert isinstance(FNORMT, u.Quantity)

    def test_fnormt_approx_1446s(self):
        assert_allclose(FNORMT.to(u.s).value, 1446, rtol=0.01)

    def test_fn_v_approx_481kms(self):
        assert_allclose(FN_V.to(u.km / u.s).value, 481, rtol=0.01)

    def test_fn_rho_formula(self):
        expected = (FMP * FN0PHYS).to(u.g / u.cm**3).value
        assert_allclose(FN_RHO.to(u.g / u.cm**3).value, expected, rtol=1e-10)

    def test_fn_n_equals_fn0phys(self):
        assert_allclose(FN_N.to(u.cm**-3).value, FN0PHYS.to(u.cm**-3).value, rtol=1e-10)

    def test_fn_t_approx_28mk(self):
        assert_allclose(FN_T.to(u.MK).value, 28, rtol=0.10)

    def test_fn_b_unit_is_gauss(self):
        assert FN_B.unit == u.Gauss

    def test_fn_b_approx_2gauss(self):
        assert_allclose(FN_B.to(u.Gauss).value, 2.2, rtol=0.10)

    def test_fn_j_positive(self):
        assert FN_J.to(u.A / u.m**2).value > 0

    def test_fn_j_cgs_is_quantity(self):
        assert isinstance(FN_J_CGS, u.Quantity)

    def test_fn_p_positive(self):
        assert FN_P.to(u.erg / u.cm**3).value > 0

    def test_fn_heat_positive(self):
        assert FN_HEAT.to(u.erg / (u.cm**3 * u.s)).value > 0

    def test_g0norm_value(self):
        assert_allclose(G0NORM, 0.823, rtol=1e-10)

    def test_omega_corotate_solar_period(self):
        period = (2 * math.pi * u.rad / OMEGA_COROTATE).to(u.day)
        assert_allclose(period.value, 25.4, rtol=0.05)


class TestFavoredUnits:
    def test_is_tuple(self):
        assert isinstance(FAVORED_UNITS, tuple)

    def test_contains_expected_units(self):
        for unit in (u.erg, u.cm, u.s, u.K, u.Gauss, u.g, u.rad):
            assert unit in FAVORED_UNITS

    def test_length(self):
        assert len(FAVORED_UNITS) == 7


class TestComposeMasUnits:
    def test_returns_quantity(self):
        assert isinstance(compose_mas_units(1.0 * u.J), u.Quantity)

    def test_value_preserved(self):
        assert_allclose(compose_mas_units(3.5 * u.Pa).to(u.Pa).value, 3.5, rtol=1e-10)

    def test_joule_converts_to_erg_basis(self):
        compose_mas_units(1.0 * u.J).to(u.erg)  # should not raise

    def test_already_favored_unit_unchanged(self):
        assert_allclose(compose_mas_units(5.0 * u.erg).to(u.erg).value, 5.0, rtol=1e-10)


class TestDecomposeMasUnits:
    def test_returns_quantity(self):
        assert isinstance(decompose_mas_units(1.0 * u.J), u.Quantity)

    def test_joule_to_erg(self):
        assert_allclose(decompose_mas_units(1.0 * u.J).to(u.erg).value, 1e7, rtol=1e-10)

    def test_value_preserved(self):
        assert_allclose(decompose_mas_units(2.0 * u.J).to(u.J).value, 2.0, rtol=1e-10)

    def test_km_per_s_to_cm_per_s(self):
        assert_allclose(decompose_mas_units(1.0 * u.km / u.s).to(u.cm / u.s).value, 1e5, rtol=1e-10)


class TestCustomAstropyUnits:
    @pytest.mark.parametrize("unit_obj,unit_str", [
        (MAS_b,    "MAS_b"),
        (MAS_v,    "MAS_v"),
        (MAS_j,    "MAS_j"),
        (MAS_t,    "MAS_t"),
        (MAS_n,    "MAS_n"),
        (MAS_p,    "MAS_p"),
        (MAS_heat, "MAS_heat"),
        (POT3D_b,  "POT3D_b"),
        (PSI_rsun, "PSI_rsun"),
        (PSI_angle,"PSI_angle"),
    ])
    def test_unit_is_registered(self, unit_obj, unit_str):
        assert isinstance(unit_obj, u.UnitBase)
        u.Unit(unit_str)  # should not raise

    @pytest.mark.parametrize("component_str", ["MAS_br", "MAS_bt", "MAS_bp"])
    def test_mas_b_component_names_registered(self, component_str):
        u.Unit(component_str)  # should not raise

    @pytest.mark.parametrize("component_str", ["MAS_vr", "MAS_vt", "MAS_vp", "MAS_zp", "MAS_zm"])
    def test_mas_v_component_names_registered(self, component_str):
        u.Unit(component_str)

    def test_mas_b_converts_to_gauss(self):
        assert_allclose((1.0 * MAS_b).to(u.Gauss).value, FN_B.to(u.Gauss).value, rtol=1e-6)

    def test_mas_v_converts_to_km_per_s(self):
        assert_allclose((1.0 * MAS_v).to(u.km / u.s).value, FN_V.to(u.km / u.s).value, rtol=1e-6)

    def test_mas_t_converts_to_mk(self):
        assert_allclose((1.0 * MAS_t).to(u.MK).value, FN_T.to(u.MK).value, rtol=1e-6)

    def test_mas_n_value_is_1e8_per_cm3(self):
        assert_allclose((1.0 * MAS_n).to(u.cm**-3).value, 1e8, rtol=1e-6)

    def test_pot3d_b_is_dimensionless(self):
        assert_allclose((1.0 * POT3D_b).to(u.dimensionless_unscaled).value, 1.0, rtol=1e-10)

    def test_psi_rsun_value(self):
        assert_allclose((1.0 * PSI_rsun).to(u.cm).value, RSUN.to(u.cm).value, rtol=1e-10)

    def test_psi_angle_one_radian(self):
        assert_allclose((1.0 * PSI_angle).to(u.rad).value, 1.0, rtol=1e-10)


class TestGetHeliumFractions:
    def test_pure_hydrogen_he_rho(self):
        assert_allclose(get_helium_fractions(0.0)["he_rho"], 1.0, rtol=1e-10)

    def test_pure_hydrogen_he_p(self):
        assert_allclose(get_helium_fractions(0.0)["he_p"], 2.0, rtol=1e-10)

    def test_pure_hydrogen_he_np(self):
        assert_allclose(get_helium_fractions(0.0)["he_np"], 1.0, rtol=1e-10)

    def test_pure_hydrogen_he_p_e(self):
        assert_allclose(get_helium_fractions(0.0)["he_p_e"], 1.0, rtol=1e-10)

    def test_pure_hydrogen_he_p_p(self):
        assert_allclose(get_helium_fractions(0.0)["he_p_p"], 1.0, rtol=1e-10)

    def test_5pct_he_rho(self):
        assert_allclose(get_helium_fractions(0.05)["he_rho"], 1.2 / 1.1, rtol=1e-10)

    def test_5pct_he_p(self):
        assert_allclose(get_helium_fractions(0.05)["he_p"], 2.15 / 1.1, rtol=1e-10)

    def test_5pct_he_np(self):
        assert_allclose(get_helium_fractions(0.05)["he_np"], 1.0 / 1.1, rtol=1e-10)

    def test_he_p_e_always_one(self):
        for f in [0.0, 0.01, 0.05, 0.1, 0.5]:
            assert_allclose(get_helium_fractions(f)["he_p_e"], 1.0, rtol=1e-10)

    def test_returns_dict_with_all_keys(self):
        fracs = get_helium_fractions(0.05)
        assert isinstance(fracs, dict)
        assert set(fracs.keys()) == {"he_rho", "he_p", "he_np", "he_p_e", "he_p_p"}

    def test_charge_neutrality(self):
        for f in [0.0, 0.05, 0.1]:
            fracs = get_helium_fractions(f)
            assert_allclose(fracs["he_np"] * (1 + 2 * f), 1.0, rtol=1e-10)

    def test_he_rho_increases_with_helium(self):
        prev = get_helium_fractions(0.0)["he_rho"]
        for f in [0.01, 0.05, 0.1, 0.2]:
            curr = get_helium_fractions(f)["he_rho"]
            assert curr > prev
            prev = curr
