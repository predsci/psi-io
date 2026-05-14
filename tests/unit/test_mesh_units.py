"""Unit tests for psi_io._mesh and psi_io._units modules."""

from __future__ import annotations

import math

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from psi_io._mesh import (
    Mesh,
    _MESH_CODE_REVERSE_MAPPING,
    _average_adjacent,
    _normalize_mesh_code,
    main_mesh,
    remesh_arr,
)
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


# ===========================================================================
# _mesh module tests
# ===========================================================================

class TestMeshEnum:
    def test_main_value(self):
        assert Mesh.MAIN.value == 0

    def test_half_value(self):
        assert Mesh.HALF.value == 1

    def test_str_main(self):
        assert str(Mesh.MAIN) == "MAIN"

    def test_str_half(self):
        assert str(Mesh.HALF) == "HALF"

    def test_members_are_distinct(self):
        assert Mesh.MAIN is not Mesh.HALF

    def test_from_value_zero(self):
        assert Mesh(0) is Mesh.MAIN

    def test_from_value_one(self):
        assert Mesh(1) is Mesh.HALF


class TestMeshCodeReverseMapping:
    def test_is_immutable(self):
        with pytest.raises(TypeError):
            _MESH_CODE_REVERSE_MAPPING["new_key"] = 0  # type: ignore[index]

    def test_half_tokens_map_to_one(self):
        for token in ("1", "h", "half", "true"):
            assert _MESH_CODE_REVERSE_MAPPING[token] == 1, f"token={token!r}"

    def test_main_tokens_map_to_zero(self):
        for token in ("0", "m", "main", "false"):
            assert _MESH_CODE_REVERSE_MAPPING[token] == 0, f"token={token!r}"

    def test_all_expected_keys_present(self):
        expected = {"1", "h", "half", "true", "0", "m", "main", "false"}
        assert set(_MESH_CODE_REVERSE_MAPPING.keys()) == expected


class TestNormalizeMeshCode:
    def test_main_string_1d(self):
        assert _normalize_mesh_code("main", ndim=1) == (Mesh.MAIN,)

    def test_main_string_3d(self):
        assert _normalize_mesh_code("main", ndim=3) == (Mesh.MAIN, Mesh.MAIN, Mesh.MAIN)

    def test_half_string_2d(self):
        assert _normalize_mesh_code("half", ndim=2) == (Mesh.HALF, Mesh.HALF)

    def test_half_string_3d(self):
        assert _normalize_mesh_code("half", ndim=3) == (Mesh.HALF, Mesh.HALF, Mesh.HALF)

    def test_int_zero_3d(self):
        assert _normalize_mesh_code(0, ndim=3) == (Mesh.MAIN, Mesh.MAIN, Mesh.MAIN)

    def test_int_all_ones_3d(self):
        assert _normalize_mesh_code(0b111, ndim=3) == (Mesh.HALF, Mesh.HALF, Mesh.HALF)

    def test_int_100_3d(self):
        assert _normalize_mesh_code(0b100, ndim=3) == (Mesh.HALF, Mesh.MAIN, Mesh.MAIN)

    def test_int_010_3d(self):
        assert _normalize_mesh_code(0b010, ndim=3) == (Mesh.MAIN, Mesh.HALF, Mesh.MAIN)

    def test_int_001_3d(self):
        assert _normalize_mesh_code(0b001, ndim=3) == (Mesh.MAIN, Mesh.MAIN, Mesh.HALF)

    def test_int_011_3d(self):
        assert _normalize_mesh_code(0b011, ndim=3) == (Mesh.MAIN, Mesh.HALF, Mesh.HALF)

    def test_int_101_3d(self):
        assert _normalize_mesh_code(0b101, ndim=3) == (Mesh.HALF, Mesh.MAIN, Mesh.HALF)

    def test_int_110_3d(self):
        assert _normalize_mesh_code(0b110, ndim=3) == (Mesh.HALF, Mesh.HALF, Mesh.MAIN)

    def test_sequence_ints(self):
        assert _normalize_mesh_code([1, 0, 1], ndim=3) == (Mesh.HALF, Mesh.MAIN, Mesh.HALF)

    def test_sequence_string_tokens(self):
        assert _normalize_mesh_code(["h", "m", "h"], ndim=3) == (Mesh.HALF, Mesh.MAIN, Mesh.HALF)

    def test_sequence_all_main(self):
        assert _normalize_mesh_code(["main", "main", "main"], ndim=3) == (Mesh.MAIN, Mesh.MAIN, Mesh.MAIN)

    def test_returns_tuple(self):
        assert isinstance(_normalize_mesh_code(0, ndim=3), tuple)

    def test_tuple_length_matches_ndim(self):
        for ndim in (1, 2, 3, 4):
            assert len(_normalize_mesh_code(0, ndim=ndim)) == ndim

    def test_elements_are_mesh_instances(self):
        for element in _normalize_mesh_code(0b101, ndim=3):
            assert isinstance(element, Mesh)

    def test_sequence_wrong_length_raises(self):
        with pytest.raises(ValueError):
            _normalize_mesh_code([1, 0], ndim=3)

    def test_sequence_too_long_raises(self):
        with pytest.raises(ValueError):
            _normalize_mesh_code([1, 0, 1, 0], ndim=3)

    def test_invalid_token_raises(self):
        with pytest.raises((ValueError, KeyError)):
            _normalize_mesh_code(["2", "0", "1"], ndim=3)


class TestAverageAdjacent:
    def test_1d_shape(self):
        assert _average_adjacent(np.ones(5), axis=0).shape == (4,)

    def test_1d_values(self):
        assert_allclose(_average_adjacent(np.array([1.0, 3.0, 5.0]), axis=0), [2.0, 4.0])

    def test_2d_along_axis0(self):
        assert _average_adjacent(np.arange(12.0).reshape(4, 3), axis=0).shape == (3, 3)

    def test_2d_along_axis1(self):
        assert _average_adjacent(np.arange(12.0).reshape(4, 3), axis=1).shape == (4, 2)

    def test_negative_axis(self):
        assert _average_adjacent(np.ones((4, 3)), axis=-1).shape == (4, 2)

    def test_3d_shape_axis0(self):
        assert _average_adjacent(np.ones((6, 5, 4)), axis=0).shape == (5, 5, 4)

    def test_3d_shape_last_axis(self):
        assert _average_adjacent(np.ones((6, 5, 4)), axis=-1).shape == (6, 5, 3)

    def test_uniform_array_unchanged(self):
        assert_allclose(_average_adjacent(np.full((10,), 7.5), axis=0), 7.5)

    def test_arithmetic_mean_2d(self):
        a = np.array([[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]])
        assert_allclose(_average_adjacent(a, axis=0), [[2.0, 4.0], [6.0, 8.0]])


class TestRemeshArr:
    def test_bool_false_unchanged(self):
        a = np.ones((4, 5))
        assert remesh_arr(a, remesh=False).shape == (4, 5)

    def test_bool_true_all_axes(self):
        assert remesh_arr(np.ones((4, 5)), remesh=True).shape == (3, 4)

    def test_sequence_second_axis(self):
        assert remesh_arr(np.ones((4, 5)), remesh=[False, True]).shape == (4, 4)

    def test_sequence_first_axis(self):
        assert remesh_arr(np.ones((4, 5)), remesh=[True, False]).shape == (3, 5)

    def test_3d_selective(self):
        assert remesh_arr(np.ones((6, 5, 4)), remesh=[True, False, True]).shape == (5, 5, 3)

    def test_values_averaged(self):
        a = np.array([[1.0, 3.0], [5.0, 7.0]])
        assert_allclose(remesh_arr(a, remesh=[False, True]), [[2.0], [6.0]])

    def test_uniform_value_preserved(self):
        assert_allclose(remesh_arr(np.full((8, 8), 3.14), remesh=[True, True]), 3.14)


class TestMainMesh:
    def test_all_main_unchanged(self):
        assert main_mesh(np.ones((4, 5, 6)), mesh_code=0).shape == (4, 5, 6)

    def test_msb_reduces_last_axis(self):
        assert main_mesh(np.ones((4, 5, 6)), mesh_code=0b100).shape == (4, 5, 5)

    def test_middle_bit_reduces_middle_axis(self):
        assert main_mesh(np.ones((4, 5, 6)), mesh_code=0b010).shape == (4, 4, 6)

    def test_lsb_reduces_first_axis(self):
        assert main_mesh(np.ones((4, 5, 6)), mesh_code=0b001).shape == (3, 5, 6)

    def test_all_half_reduces_all(self):
        assert main_mesh(np.ones((4, 5, 6)), mesh_code=0b111).shape == (3, 4, 5)

    def test_011_two_axes(self):
        assert main_mesh(np.ones((4, 5, 6)), mesh_code=0b011).shape == (3, 4, 6)

    def test_110_two_axes(self):
        assert main_mesh(np.ones((4, 5, 6)), mesh_code=0b110).shape == (4, 4, 5)

    def test_101_two_axes(self):
        assert main_mesh(np.ones((4, 5, 6)), mesh_code=0b101).shape == (3, 5, 5)

    def test_module_docstring_example(self):
        # From module docstring: br = ones((128, 64, 57)), mesh_code=0b100
        br = np.ones((128, 64, 57))
        assert main_mesh(br, mesh_code=0b100).shape == (128, 64, 56)

    def test_values_averaged_correctly(self):
        data = np.zeros((1, 1, 4))
        data[0, 0, :] = [0.0, 2.0, 4.0, 6.0]
        out = main_mesh(data, mesh_code=0b100)
        assert out.shape == (1, 1, 3)
        assert_allclose(out[0, 0, :], [1.0, 3.0, 5.0])

    def test_uniform_value_preserved(self):
        assert_allclose(main_mesh(np.full((5, 6, 7), 2.5), mesh_code=0b111), 2.5)

    def test_string_shorthand_main(self):
        assert main_mesh(np.ones((4, 5, 6)), mesh_code="main").shape == (4, 5, 6)

    def test_string_shorthand_half(self):
        assert main_mesh(np.ones((4, 5, 6)), mesh_code="half").shape == (3, 4, 5)

    def test_sequence_code_accepted(self):
        assert main_mesh(np.ones((4, 5, 6)), mesh_code=[1, 0, 0]).shape == (4, 5, 5)

    def test_1d_half_reduces(self):
        out = main_mesh(np.array([1.0, 3.0, 5.0]), mesh_code=1)
        assert out.shape == (2,)
        assert_allclose(out, [2.0, 4.0])


# ===========================================================================
# _units module tests
# ===========================================================================

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
        # Solar sidereal rotation ~25.4 days
        period = (2 * math.pi * u.rad / OMEGA_COROTATE).to(u.day)
        assert_allclose(period.value, 25.4, rtol=0.05)


class TestFavoredUnits:
    def test_is_set(self):
        assert isinstance(FAVORED_UNITS, set)

    def test_contains_expected_units(self):
        for unit in (u.erg, u.cm, u.s, u.K, u.Gauss, u.g, u.rad):
            assert unit in FAVORED_UNITS


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
        (MAS_b, "MAS_b"),
        (MAS_v, "MAS_v"),
        (MAS_j, "MAS_j"),
        (MAS_t, "MAS_t"),
        (MAS_n, "MAS_n"),
        (MAS_p, "MAS_p"),
        (MAS_heat, "MAS_heat"),
        (POT3D_b, "POT3D_b"),
        (PSI_rsun, "PSI_rsun"),
        (PSI_angle, "PSI_angle"),
    ])
    def test_unit_is_registered(self, unit_obj, unit_str):
        assert isinstance(unit_obj, u.UnitBase)
        u.Unit(unit_str)  # should not raise

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