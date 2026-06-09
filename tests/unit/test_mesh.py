"""Unit tests for psi_io._mesh."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from psi_io._mesh import (
    Mesh,
    _MESH_CODE_REVERSE_MAPPING,
    _average_adjacent,
    _remesh_array,
    remesh_array,
)


class TestMeshDataclass:
    """Tests for the Mesh frozen dataclass (replaces the old Enum tests)."""

    def test_parse_main_string_3d(self):
        m = Mesh.parse('main', ndim=3)
        assert m.code == 0
        assert m.ndim == 3

    def test_parse_half_string_3d(self):
        m = Mesh.parse('half', ndim=3)
        assert m.code == 0b111
        assert m.ndim == 3

    def test_parse_int_zero_3d(self):
        m = Mesh.parse(0, ndim=3)
        assert m.code == 0

    def test_parse_int_all_ones_3d(self):
        m = Mesh.parse(0b111, ndim=3)
        assert m.code == 0b111

    def test_parse_returns_mesh_instance(self):
        m = Mesh.parse(0b100, ndim=3)
        assert isinstance(m, Mesh)

    def test_mesh_instances_are_distinct(self):
        m1 = Mesh.parse(0, ndim=3)
        m2 = Mesh.parse(0b111, ndim=3)
        assert m1 != m2

    def test_frozen_immutable(self):
        m = Mesh.parse(0b100, ndim=3)
        with pytest.raises((AttributeError, TypeError)):
            m.code = 0  # type: ignore[misc]

    def test_str_all_main(self):
        m = Mesh.parse(0, ndim=3)
        assert str(m) == 'MAIN, MAIN, MAIN'

    def test_str_all_half(self):
        m = Mesh.parse(0b111, ndim=3)
        assert str(m) == 'HALF, HALF, HALF'

    def test_str_100(self):
        m = Mesh.parse(0b100, ndim=3)
        assert str(m) == 'HALF, MAIN, MAIN'

    def test_len_equals_ndim(self):
        m = Mesh.parse(0b100, ndim=3)
        assert len(m) == 3

    def test_bool_false_for_all_main(self):
        m = Mesh.parse(0, ndim=3)
        assert not m

    def test_bool_true_for_half(self):
        m = Mesh.parse(0b100, ndim=3)
        assert m


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


class TestMeshParse:
    """Tests for Mesh.parse (replaces old _normalize_mesh_code tests)."""

    def test_main_string_1d(self):
        m = Mesh.parse('main', ndim=1)
        assert m == Mesh(0, 1)

    def test_main_string_3d(self):
        m = Mesh.parse('main', ndim=3)
        assert m == Mesh(0, 3)
        assert list(m) == [False, False, False]

    def test_half_string_2d(self):
        m = Mesh.parse('half', ndim=2)
        assert m == Mesh(0b11, 2)
        assert list(m) == [True, True]

    def test_half_string_3d(self):
        m = Mesh.parse('half', ndim=3)
        assert list(m) == [True, True, True]

    def test_int_zero_3d(self):
        m = Mesh.parse(0, ndim=3)
        assert list(m) == [False, False, False]

    def test_int_all_ones_3d(self):
        m = Mesh.parse(0b111, ndim=3)
        assert list(m) == [True, True, True]

    def test_int_100_3d(self):
        m = Mesh.parse(0b100, ndim=3)
        assert list(m) == [True, False, False]

    def test_int_010_3d(self):
        m = Mesh.parse(0b010, ndim=3)
        assert list(m) == [False, True, False]

    def test_int_001_3d(self):
        m = Mesh.parse(0b001, ndim=3)
        assert list(m) == [False, False, True]

    def test_int_011_3d(self):
        m = Mesh.parse(0b011, ndim=3)
        assert list(m) == [False, True, True]

    def test_int_101_3d(self):
        m = Mesh.parse(0b101, ndim=3)
        assert list(m) == [True, False, True]

    def test_int_110_3d(self):
        m = Mesh.parse(0b110, ndim=3)
        assert list(m) == [True, True, False]

    def test_sequence_ints(self):
        m = Mesh.parse([1, 0, 1], ndim=3)
        assert list(m) == [True, False, True]

    def test_sequence_string_tokens(self):
        m = Mesh.parse(['h', 'm', 'h'], ndim=3)
        assert list(m) == [True, False, True]

    def test_sequence_all_main(self):
        m = Mesh.parse(['main', 'main', 'main'], ndim=3)
        assert list(m) == [False, False, False]

    def test_returns_mesh_instance(self):
        m = Mesh.parse(0, ndim=3)
        assert isinstance(m, Mesh)

    def test_ndim_matches(self):
        for ndim in (1, 2, 3, 4):
            m = Mesh.parse(0, ndim=ndim)
            assert m.ndim == ndim

    def test_iteration_yields_bools(self):
        m = Mesh.parse(0b101, ndim=3)
        for element in m:
            assert isinstance(element, bool)

    def test_sequence_wrong_length_raises(self):
        with pytest.raises(ValueError):
            Mesh.parse([1, 0], ndim=3)

    def test_sequence_too_long_raises(self):
        with pytest.raises(ValueError):
            Mesh.parse([1, 0, 1, 0], ndim=3)

    def test_invalid_token_raises(self):
        with pytest.raises((ValueError, KeyError)):
            Mesh.parse(['2', '0', '1'], ndim=3)


class TestMeshInstanceMethods:
    """Tests for Mesh instance behavior: indexing, reversal, and remeshing."""

    def test_getitem_returns_single_axis_mesh(self):
        m = Mesh.parse(0b100, ndim=3)   # half, main, main
        assert bool(m[0]) is True
        assert bool(m[1]) is False
        assert bool(m[2]) is False

    def test_getitem_str(self):
        m = Mesh.parse(0b100, ndim=3)
        assert str(m[0]) == 'HALF'
        assert str(m[1]) == 'MAIN'

    def test_index_returns_int_code(self):
        assert int(Mesh.parse(0b100, ndim=3)) == 4
        assert int(Mesh.parse(0b011, ndim=3)) == 3

    def test_reverse_flips_axis_order(self):
        m = Mesh.parse(0b100, ndim=3)        # half, main, main
        assert list(m.reverse()) == [False, False, True]

    def test_reverse_is_involution(self):
        m = Mesh.parse(0b101, ndim=3)
        assert m.reverse().reverse() == m

    def test_len_matches_ndim(self):
        assert len(Mesh.parse('main', ndim=3)) == 3
        assert len(Mesh.parse('main', ndim=2)) == 2

    def test_rshift_to_main_flags_half_axes(self):
        m = Mesh.parse(0b100, ndim=3)        # half, main, main
        main = Mesh.parse('main', ndim=3)
        assert (m >> main) == (True, False, False)

    def test_rshift_none_is_no_remesh(self):
        m = Mesh.parse(0b111, ndim=3)
        assert (m >> None) == (False, False, False)

    def test_rshift_to_self_is_no_remesh(self):
        m = Mesh.parse(0b101, ndim=3)
        assert (m >> m) == (False, False, False)

    def test_remesh_matches_rshift(self):
        m = Mesh.parse(0b110, ndim=3)
        main = Mesh.parse('main', ndim=3)
        assert m.remesh(main) == (m >> main)


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
    """Tests for the low-level _remesh_array helper."""

    def test_bool_false_unchanged(self):
        a = np.ones((4, 5))
        assert _remesh_array(a, remesh=False).shape == (4, 5)

    def test_bool_true_all_axes(self):
        assert _remesh_array(np.ones((4, 5)), remesh=True).shape == (3, 4)

    def test_sequence_second_axis(self):
        assert _remesh_array(np.ones((4, 5)), remesh=[False, True], order='C').shape == (4, 4)

    def test_sequence_first_axis(self):
        assert _remesh_array(np.ones((4, 5)), remesh=[True, False], order='C').shape == (3, 5)

    def test_3d_selective(self):
        assert _remesh_array(np.ones((6, 5, 4)), remesh=[True, False, True], order='C').shape == (5, 5, 3)

    def test_values_averaged(self):
        a = np.array([[1.0, 3.0], [5.0, 7.0]])
        assert_allclose(_remesh_array(a, remesh=[False, True], order='C'), [[2.0], [6.0]])

    def test_uniform_value_preserved(self):
        assert_allclose(_remesh_array(np.full((8, 8), 3.14), remesh=[True, True]), 3.14)


class TestRemeshArray:
    """Tests for the public remesh_array function (imesh/omesh API)."""

    # ------------------------------------------------------------------
    # No-op cases
    # ------------------------------------------------------------------

    def test_omesh_none_is_noop(self):
        arr = np.ones((4, 5, 6))
        assert remesh_array(arr, imesh=0b100).shape == (4, 5, 6)

    def test_same_mesh_is_noop(self):
        arr = np.ones((4, 5, 6))
        assert remesh_array(arr, imesh=0b100, omesh=0b100).shape == (4, 5, 6)

    def test_all_main_to_all_main_unchanged(self):
        assert remesh_array(np.ones((4, 5, 6)), imesh=0, omesh='main').shape == (4, 5, 6)

    # ------------------------------------------------------------------
    # Fortran-order (default) bit → axis mapping
    # MSB → last numpy axis (r), middle → theta, LSB → first axis (phi)
    # ------------------------------------------------------------------

    def test_msb_reduces_last_axis(self):
        assert remesh_array(np.ones((4, 5, 6)), imesh=0b100, omesh='main').shape == (4, 5, 5)

    def test_middle_bit_reduces_middle_axis(self):
        assert remesh_array(np.ones((4, 5, 6)), imesh=0b010, omesh='main').shape == (4, 4, 6)

    def test_lsb_reduces_first_axis(self):
        assert remesh_array(np.ones((4, 5, 6)), imesh=0b001, omesh='main').shape == (3, 5, 6)

    def test_all_half_reduces_all(self):
        assert remesh_array(np.ones((4, 5, 6)), imesh=0b111, omesh='main').shape == (3, 4, 5)

    def test_011_reduces_first_and_middle(self):
        assert remesh_array(np.ones((4, 5, 6)), imesh=0b011, omesh='main').shape == (3, 4, 6)

    def test_110_reduces_middle_and_last(self):
        assert remesh_array(np.ones((4, 5, 6)), imesh=0b110, omesh='main').shape == (4, 4, 5)

    def test_101_reduces_first_and_last(self):
        assert remesh_array(np.ones((4, 5, 6)), imesh=0b101, omesh='main').shape == (3, 5, 5)

    # ------------------------------------------------------------------
    # Alternative input forms for imesh
    # ------------------------------------------------------------------

    def test_string_shorthand_main_is_noop(self):
        assert remesh_array(np.ones((4, 5, 6)), imesh='main', omesh='main').shape == (4, 5, 6)

    def test_string_shorthand_half_reduces_all(self):
        assert remesh_array(np.ones((4, 5, 6)), imesh='half', omesh='main').shape == (3, 4, 5)

    def test_sequence_imesh_accepted(self):
        # [1, 0, 0] normalises to (HALF, MAIN, MAIN), same as 0b100
        assert remesh_array(np.ones((4, 5, 6)), imesh=[1, 0, 0], omesh='main').shape == (4, 5, 5)

    def test_1d_half_reduces(self):
        out = remesh_array(np.array([1.0, 3.0, 5.0]), imesh=1, omesh='main')
        assert out.shape == (2,)
        assert_allclose(out, [2.0, 4.0])

    # ------------------------------------------------------------------
    # C-order
    # ------------------------------------------------------------------

    def test_c_order_msb_reduces_first_axis(self):
        # With order='C', MSB maps to the first numpy axis
        arr = np.ones((4, 5, 6))
        assert remesh_array(arr, imesh=0b100, omesh='main', order='C').shape == (3, 5, 6)

    # ------------------------------------------------------------------
    # Value correctness
    # ------------------------------------------------------------------

    def test_module_docstring_example(self):
        br = np.ones((128, 64, 57))  # shape (Nφ, Nθ, Nr); Nr is on half-mesh
        assert remesh_array(br, imesh=0b100, omesh='main').shape == (128, 64, 56)

    def test_values_averaged_correctly(self):
        data = np.zeros((1, 1, 4))
        data[0, 0, :] = [0.0, 2.0, 4.0, 6.0]
        out = remesh_array(data, imesh=0b100, omesh='main')
        assert out.shape == (1, 1, 3)
        assert_allclose(out[0, 0, :], [1.0, 3.0, 5.0])

    def test_uniform_value_preserved(self):
        assert_allclose(remesh_array(np.full((5, 6, 7), 2.5), imesh=0b111, omesh='main'), 2.5)

    # ------------------------------------------------------------------
    # Error conditions
    # ------------------------------------------------------------------

    def test_main_to_half_raises(self):
        with pytest.raises(ValueError, match="HALF"):
            remesh_array(np.ones((4, 5, 6)), imesh=0b000, omesh=0b100)
