"""Unit tests for psi_io.mhd_io."""

from __future__ import annotations

import numpy as np
import pytest

import astropy.units as u
import h5py

from psi_io._mesh import Mesh
from psi_io._units import MAS_b, PSI_rsun, PSI_angle
from psi_io.mhd_io import (
    _HDF_EXT_MAPPING,
    METADATA_SCHEMA,
    _CODE_UNIT_ALIASES,
    _REAL_UNIT_ALIASES,
    _apply_units,
    _cast_to_slice,
    _expand_args,
    _interpolate_dim,
    _parse_islice_args,
    _parse_vslice_args,
    _slice_array,
    CacheWarning,
    MetaDataWarning,
    H5Data,
    PsiData,
)

try:
    import scipy  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope='module')
def psi_h5_mas_file(tmp_path_factory):
    """Minimal PSI-compatible HDF5 file named as a MAS br field (br001001.h5).

    Dataset "Data" has numpy shape (Nphi=8, Ntheta=9, Nr=7) — Fortran-ordered
    storage means dims[0] carries the r scale, dims[1] theta, dims[2] phi.
    """
    d = tmp_path_factory.mktemp("mhd_io_mas")
    fpath = d / "br001001.h5"
    nr, nt, np_ = 7, 9, 8
    data = np.ones((np_, nt, nr), dtype=np.float32)
    with h5py.File(fpath, 'w') as f:
        ds = f.create_dataset("Data", data=data)
        for i, (label, size) in enumerate([("dim1", nr), ("dim2", nt), ("dim3", np_)]):
            sc = f.create_dataset(label, data=np.linspace(0.0, 1.0, size, dtype=np.float32))
            ds.dims[i].attach_scale(sc)
            ds.dims[i].label = label
    return fpath


@pytest.fixture(scope='module')
def psi_h5_pot3d_file(tmp_path_factory):
    """Minimal PSI-compatible HDF5 file named as a POT3D br field (br001.h5)."""
    d = tmp_path_factory.mktemp("mhd_io_pot3d")
    fpath = d / "br001.h5"
    nr, nt, np_ = 7, 9, 8
    data = np.ones((np_, nt, nr), dtype=np.float32)
    with h5py.File(fpath, 'w') as f:
        ds = f.create_dataset("Data", data=data)
        for i, (label, size) in enumerate([("dim1", nr), ("dim2", nt), ("dim3", np_)]):
            sc = f.create_dataset(label, data=np.linspace(0.0, 1.0, size, dtype=np.float32))
            ds.dims[i].attach_scale(sc)
            ds.dims[i].label = label
    return fpath


@pytest.fixture(scope='module')
def mas_reader(psi_h5_mas_file):
    """Open H5Data (MAS) reader; closed after the module finishes."""
    reader = PsiData(psi_h5_mas_file, model='mas')
    yield reader
    reader.close()


@pytest.fixture(scope='module')
def pot3d_reader(psi_h5_pot3d_file):
    """Open H5Data (POT3D) reader; closed after the module finishes."""
    reader = PsiData(psi_h5_pot3d_file, model='pot3d')
    yield reader
    reader.close()


# ===========================================================================
# Module-level constants
# ===========================================================================

class TestHdfExtMapping:
    def test_h5_extension(self):
        assert _HDF_EXT_MAPPING['h5'] == '.h5'

    def test_h4_extension(self):
        assert _HDF_EXT_MAPPING['h4'] == '.hdf'

    def test_only_two_keys(self):
        assert set(_HDF_EXT_MAPPING.keys()) == {'h5', 'h4'}


class TestUnitAliases:
    def test_code_unit_aliases_contains_expected(self):
        for alias in ('native', 'code', 'model'):
            assert alias in _CODE_UNIT_ALIASES

    def test_real_unit_aliases_contains_expected(self):
        for alias in ('real', 'phys', 'physical'):
            assert alias in _REAL_UNIT_ALIASES

    def test_alias_sets_are_disjoint(self):
        assert _CODE_UNIT_ALIASES.isdisjoint(_REAL_UNIT_ALIASES)


class TestMetadataSchema:
    def test_has_nine_keys(self):
        assert set(METADATA_SCHEMA.keys()) == {
            'name', 'desc', 'unit', 'scalar', 'mesh',
            'order', 'sequence', 'model', 'scales'
        }

    def test_all_values_none(self):
        assert all(v is None for v in METADATA_SCHEMA.values())


# ===========================================================================
# _cast_to_slice
# ===========================================================================

class TestCastToSlice:
    def test_none_gives_full_slice(self):
        assert _cast_to_slice(None) == slice(None)

    def test_int_gives_single_element_slice(self):
        assert _cast_to_slice(3) == slice(3, 4)

    def test_int_zero(self):
        assert _cast_to_slice(0) == slice(0, 1)

    def test_slice_returned_unchanged(self):
        s = slice(2, 10, 3)
        assert _cast_to_slice(s) is s

    def test_two_tuple_gives_slice(self):
        assert _cast_to_slice((2, 8)) == slice(2, 8)

    def test_three_tuple_gives_slice_with_step(self):
        assert _cast_to_slice((1, 9, 2)) == slice(1, 9, 2)

    def test_list_two_elements(self):
        assert _cast_to_slice([0, 5]) == slice(0, 5)

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError):
            _cast_to_slice(3.5)

    def test_plain_object_raises_type_error(self):
        # plain object() is not a recognized type (not int, None, slice, or Collection)
        with pytest.raises(TypeError):
            _cast_to_slice(object())


# ===========================================================================
# _parse_islice_args
# ===========================================================================
# _parse_islice_args zips over (*args, shape, remesh), so all three must have
# the same length (ndim).  Use _expand_args first to produce ndim positional args.

class TestParseISliceArgs:
    def test_all_none_args_returns_full_slices(self):
        # All three axes unrestricted: pass None for each axis explicitly
        shape = (10, 20, 30)
        remesh = (False, False, False)
        slices = list(_parse_islice_args(None, None, None, shape=shape, remesh=remesh))
        assert slices == [slice(None), slice(None), slice(None)]

    def test_first_axis_restricted(self):
        # Only first axis (r in PSI storage) is sliced
        shape = (10, 20, 30)
        remesh = (False, False, False)
        slices = list(_parse_islice_args(slice(0, 5), None, None, shape=shape, remesh=remesh))
        assert slices[0] == slice(0, 5)
        assert slices[1] == slice(None)
        assert slices[2] == slice(None)

    def test_empty_slice_raises(self):
        with pytest.raises(ValueError):
            list(_parse_islice_args(slice(5, 5), None, None,
                                   shape=(10, 20, 30), remesh=(False, False, False)))

    def test_single_element_slice_ok(self):
        slices = list(_parse_islice_args(slice(0, 1), None, None,
                                        shape=(10, 20, 30), remesh=(False, False, False)))
        assert slices[0] == slice(0, 1)

    def test_int_arg_gives_one_element_slice(self):
        slices = list(_parse_islice_args(0, None, None,
                                        shape=(10, 20, 30), remesh=(False, False, False)))
        assert slices[0] == slice(0, 1)


# ===========================================================================
# PsiData factory
# ===========================================================================

class TestPsiDataFactory:
    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PsiData(tmp_path / "nonexistent.h5", model='mas')

    def test_returns_h5_mas_data(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        assert isinstance(reader, H5Data)
        reader.close()

    def test_returns_h5_pot3d_data(self, psi_h5_pot3d_file):
        reader = PsiData(psi_h5_pot3d_file, model='pot3d')
        assert isinstance(reader, H5Data)
        reader.close()

    def test_model_attribute_set_to_mas(self, psi_h5_mas_file):
        # When model='mas' is specified, the reader's model attribute should be 'mas'
        reader = PsiData(psi_h5_mas_file, model='mas')
        assert reader.model == 'mas'
        reader.close()

    def test_unsupported_model_raises(self, psi_h5_mas_file):
        with pytest.raises((ValueError, KeyError)):
            PsiData(psi_h5_mas_file, model='unknown')

    def test_wrong_extension_raises(self, tmp_path, psi_h5_mas_file):
        # Rename .h5 to .hdf — H4Data will raise on the wrong extension
        wrong = tmp_path / "br001001.hdf"
        wrong.write_bytes(psi_h5_mas_file.read_bytes())
        with pytest.raises(Exception):
            PsiData(wrong, model='mas')


# ===========================================================================
# H5MasData properties (via PsiData)
# ===========================================================================

class TestH5MasDataProperties:
    def test_name(self, mas_reader):
        assert mas_reader.name == 'br'

    def test_sequence(self, mas_reader):
        assert mas_reader.sequence == 1001

    def test_unit_is_mas_b(self, mas_reader):
        assert mas_reader.unit == MAS_b

    def test_mesh_is_mesh_instance(self, mas_reader):
        mesh = mas_reader.mesh
        assert isinstance(mesh, Mesh)

    def test_mesh_br_stagger(self, mas_reader):
        # br → 0b100 → HALF, MAIN, MAIN
        assert list(mas_reader.mesh) == [True, False, False]

    def test_ndim(self, mas_reader):
        assert mas_reader.ndim == 3

    def test_shape_is_3_tuple(self, mas_reader):
        assert len(mas_reader.shape) == 3

    def test_size_matches_shape(self, mas_reader):
        import math
        assert mas_reader.size == math.prod(mas_reader.shape)

    def test_nbytes_positive(self, mas_reader):
        assert mas_reader.nbytes > 0

    def test_dtype_is_numpy_dtype(self, mas_reader):
        assert isinstance(mas_reader.dtype, np.dtype)

    def test_attrs_is_dict(self, mas_reader):
        assert isinstance(mas_reader.attrs, dict)

    def test_desc_is_string(self, mas_reader):
        assert isinstance(mas_reader.desc, str)
        assert len(mas_reader.desc) > 0

    def test_desc_mentions_field(self, mas_reader):
        assert 'Radial' in mas_reader.desc or 'radial' in mas_reader.desc

    def test_scales_has_r_t_p(self, mas_reader):
        scales = mas_reader.scales
        assert hasattr(scales, 'r')
        assert hasattr(scales, 't')
        assert hasattr(scales, 'p')

    def test_scale_names(self, mas_reader):
        scales = mas_reader.scales
        assert scales.r.name == 'r'
        assert scales.t.name == 't'
        assert scales.p.name == 'p'

    def test_scale_units_are_psi_units(self, mas_reader):
        scales = mas_reader.scales
        assert scales.r.unit == PSI_rsun
        assert scales.t.unit == PSI_angle
        assert scales.p.unit == PSI_angle


# ===========================================================================
# H5Pot3dData properties
# ===========================================================================

class TestH5Pot3dDataProperties:
    def test_name(self, pot3d_reader):
        assert pot3d_reader.name == 'br'

    def test_sequence(self, pot3d_reader):
        assert pot3d_reader.sequence == 1

    def test_mesh_br_pot3d_stagger(self, pot3d_reader):
        # POT3D br → 0b011 → MAIN, HALF, HALF
        assert list(pot3d_reader.mesh) == [False, True, True]

    def test_ndim(self, pot3d_reader):
        assert pot3d_reader.ndim == 3

    def test_desc_is_string(self, pot3d_reader):
        desc = pot3d_reader.desc
        assert isinstance(desc, str) and len(desc) > 0


# ===========================================================================
# Context manager
# ===========================================================================

class TestContextManager:
    def test_enter_returns_reader(self, psi_h5_mas_file):
        with PsiData(psi_h5_mas_file, model='mas') as reader:
            assert reader is not None
            assert reader.name == 'br'

    def test_exit_closes_file(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.close()
        assert reader._ref is None


# ===========================================================================
# open / close
# ===========================================================================

class TestOpenClose:
    def test_close_sets_ref_none(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.close()
        assert reader._ref is None

    def test_reopen_after_close(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.close()
        reader.open()
        assert reader._ref is not None
        reader.close()

    def test_open_returns_self(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        result = reader.open()
        assert result is reader
        reader.close()


# ===========================================================================
# __getitem__ axis-reversal
# ===========================================================================

class TestGetitem:
    def test_returns_numpy_array(self, mas_reader):
        result = mas_reader[:]
        assert isinstance(result, np.ndarray)

    def test_single_index_reduces_axis(self, mas_reader):
        # Physical index 0 along r (last numpy axis) maps to storage axis 2
        result = mas_reader[0:1]
        assert isinstance(result, np.ndarray)

    def test_full_slice_element_count_matches(self, mas_reader):
        # __getitem__ returns full data in HDF storage order (reversed from .shape)
        # Verify the total element count matches mas_reader.size
        full = mas_reader[slice(None), slice(None), slice(None)]
        assert full.size == mas_reader.size


# ===========================================================================
# _expand_args
# ===========================================================================

class TestExpandArgs:
    def test_empty_args_pads_with_none(self):
        assert _expand_args(ndim=3) == (None, None, None)

    def test_single_arg_pads_trailing(self):
        assert _expand_args(1, ndim=3) == (1, None, None)

    def test_ellipsis_only_expands_to_all_none(self):
        assert _expand_args(..., ndim=3) == (None, None, None)

    def test_ellipsis_trailing_pads_remainder(self):
        assert _expand_args(1, ..., ndim=3) == (1, None, None)

    def test_ellipsis_leading_pads_before_last(self):
        assert _expand_args(..., 2, ndim=3) == (None, None, 2)

    def test_ellipsis_middle_fills_gaps(self):
        assert _expand_args(1, ..., 2, ndim=4) == (1, None, None, 2)

    def test_exact_fill_no_padding(self):
        assert _expand_args(1, 2, 3, ndim=3) == (1, 2, 3)


# ===========================================================================
# _apply_units
# ===========================================================================

class TestApplyUnits:
    @pytest.fixture
    def qty(self):
        return 2.0 * MAS_b

    def test_none_returns_data_unchanged(self, qty):
        assert _apply_units(qty, None) is qty

    def test_native_alias_preserves_unit(self, qty):
        assert _apply_units(qty, 'native').unit == qty.unit

    def test_code_alias_preserves_unit(self, qty):
        assert _apply_units(qty, 'code').unit == qty.unit

    def test_model_alias_preserves_unit(self, qty):
        assert _apply_units(qty, 'model').unit == qty.unit

    def test_psi_alias_preserves_unit(self, qty):
        assert _apply_units(qty, 'psi').unit == qty.unit

    def test_real_alias_decomposes_to_cgs(self, qty):
        result = _apply_units(qty, 'real')
        assert result.unit != qty.unit
        np.testing.assert_allclose(result.to(u.Gauss).value,
                                   qty.to(u.Gauss).value, rtol=1e-5)

    def test_phys_alias_decomposes(self, qty):
        assert _apply_units(qty, 'phys').unit != qty.unit

    def test_physical_alias_decomposes(self, qty):
        assert _apply_units(qty, 'physical').unit != qty.unit

    def test_cgs_alias_decomposes(self, qty):
        assert _apply_units(qty, 'cgs').unit != qty.unit

    def test_gauss_string_converts(self, qty):
        result = _apply_units(qty, 'Gauss')
        assert result.unit == u.Gauss
        np.testing.assert_allclose(result.value, qty.to(u.Gauss).value, rtol=1e-5)

    def test_unit_instance_converts(self, qty):
        result = _apply_units(qty, u.Gauss)
        assert result.unit == u.Gauss

    def test_aliases_are_case_insensitive(self, qty):
        assert _apply_units(qty, 'NATIVE').unit == qty.unit


# ===========================================================================
# _interpolate_dim
# ===========================================================================

class TestInterpolateDim:
    def test_midpoint_gives_average(self):
        arr = np.array([[0.0, 2.0], [4.0, 6.0]]) * u.m
        scale = np.array([0.0, 1.0]) * u.m
        result = _interpolate_dim(arr, axis=0, value=0.5 * u.m, scale=scale)
        assert result.shape == (1, 2)
        np.testing.assert_allclose(result.value, [[2.0, 4.0]])

    def test_zero_parameter_returns_lo_slice(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]]) * u.m
        scale = np.array([0.0, 1.0]) * u.m
        result = _interpolate_dim(arr, axis=0, value=0.0 * u.m, scale=scale)
        assert result.shape == (1, 2)
        np.testing.assert_allclose(result.value, [[1.0, 2.0]])

    def test_one_parameter_returns_hi_slice(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]]) * u.m
        scale = np.array([0.0, 1.0]) * u.m
        result = _interpolate_dim(arr, axis=0, value=1.0 * u.m, scale=scale)
        assert result.shape == (1, 2)
        np.testing.assert_allclose(result.value, [[3.0, 4.0]])

    def test_axis_1_collapses_correct_dimension(self):
        arr = np.array([[0.0, 1.0], [2.0, 3.0]]) * u.m
        scale = np.array([0.0, 1.0]) * u.m
        result = _interpolate_dim(arr, axis=1, value=0.5 * u.m, scale=scale)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result.value, [[0.5], [2.5]])

    def test_extrapolation_beyond_scale(self):
        arr = np.array([[0.0, 1.0]]) * u.m
        scale = np.array([0.0, 1.0]) * u.m
        result = _interpolate_dim(arr, axis=1, value=2.0 * u.m, scale=scale)
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result.value, [[2.0]])


# ===========================================================================
# _slice_array
# ===========================================================================
# _slice_array signature: (data, scales, values, order)
# where scales = 2-element coordinate windows per axis
# and values = target coordinate scalars per axis (None to skip)

class TestSliceArray:
    def test_all_none_values_returns_array_unchanged(self):
        arr = np.ones((2, 3, 4)) * u.m
        result = _slice_array(arr, [None, None, None], [None, None, None], order='F')
        assert result.shape == arr.shape
        np.testing.assert_allclose(result.value, arr.value)

    def test_r_interpolation_fortran_order_collapses_last_axis(self):
        # Storage order (phi=2, theta=3, r=2); physical order: r first
        # In F order, physical r is at last numpy axis (axis 2)
        arr = np.zeros((2, 3, 2)) * u.m
        arr[..., 1] = 1.0 * u.m   # r=1 is all ones, r=0 is all zeros
        scale_r = np.array([0.0, 1.0]) * u.m
        # scales first (2-element window), values second (target)
        result = _slice_array(arr, [scale_r, None, None], [0.5 * u.m, None, None], order='F')
        assert result.shape == (2, 3, 1)
        np.testing.assert_allclose(result.value, 0.5)

    def test_phi_interpolation_fortran_order_collapses_first_axis(self):
        # Physical phi is the last physical coord; storage phi is first axis (index 0)
        arr = np.zeros((2, 3, 4)) * u.m
        arr[1, ...] = 1.0 * u.m   # phi=1 all ones, phi=0 all zeros
        scale_p = np.array([0.0, 1.0]) * u.m
        result = _slice_array(arr, [None, None, scale_p], [None, None, 0.25 * u.m], order='F')
        assert result.shape == (1, 3, 4)
        np.testing.assert_allclose(result.value, 0.25)

    def test_c_order_maps_first_physical_to_first_storage(self):
        arr = np.zeros((2, 3)) * u.m
        arr[1, :] = 1.0 * u.m
        scale = np.array([0.0, 1.0]) * u.m
        result = _slice_array(arr, [scale, None], [0.5 * u.m, None], order='C')
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result.value, 0.5)


# ===========================================================================
# _parse_vslice_args (uses live scale readers from mas_reader)
# ===========================================================================

class TestParseVsliceArgs:
    def test_none_arg_gives_full_slice_and_no_value(self, mas_reader):
        value, s = next(iter(_parse_vslice_args(None, scales=mas_reader.scales,
                                                remesh=(False, False, False))))
        assert value == (None, None)
        assert s == slice(None)

    def test_bare_scalar_gives_quantity_value_and_two_element_window(self, mas_reader):
        value, s = next(iter(_parse_vslice_args(0.4, scales=mas_reader.scales,
                                                remesh=(False, False, False))))
        assert isinstance(value, u.Quantity)
        assert (s.stop - s.start) == 2

    def test_quantity_arg_gives_two_element_window(self, mas_reader):
        target = 0.4 * PSI_rsun
        value, s = next(iter(_parse_vslice_args(target, scales=mas_reader.scales,
                                                remesh=(False, False, False))))
        assert value is not None
        assert (s.stop - s.start) == 2

    def test_index_space_slice_passes_through_as_none_value(self, mas_reader):
        value, s = next(iter(_parse_vslice_args(slice(0, 3), scales=mas_reader.scales,
                                                remesh=(False, False, False))))
        assert value == (None, None)
        assert s == slice(0, 3)

    def test_yields_one_pair_per_axis(self, mas_reader):
        pairs = list(_parse_vslice_args(None, None, None, scales=mas_reader.scales,
                                        remesh=(False, False, False)))
        assert len(pairs) == mas_reader.ndim


# ===========================================================================
# read() method
# ===========================================================================

class TestRead:
    def test_returns_quantity(self, mas_reader):
        result = mas_reader.read(scales=False)
        assert isinstance(result, u.Quantity)

    def test_shape_matches_storage_order(self, mas_reader):
        # read() returns data in HDF storage (numpy C) order; _shape is the raw HDF shape
        result = mas_reader.read(scales=False)
        assert result.shape == mas_reader._shape

    def test_default_unit_is_mas_b(self, mas_reader):
        result = mas_reader.read(scales=False)
        assert result.unit == MAS_b

    def test_scales_false_returns_quantity_not_tuple(self, mas_reader):
        result = mas_reader.read(scales=False)
        assert not isinstance(result, tuple)

    def test_scales_true_returns_four_tuple(self, mas_reader):
        result = mas_reader.read()
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_r_scale_shape(self, mas_reader):
        # shape is in physical (r, t, p) order: shape[0]=Nr, shape[1]=Nt, shape[2]=Np
        _, r, _, _ = mas_reader.read()
        assert r.shape == (mas_reader.shape[0],)

    def test_t_scale_shape(self, mas_reader):
        _, _, t, _ = mas_reader.read()
        assert t.shape == (mas_reader.shape[1],)

    def test_p_scale_shape(self, mas_reader):
        _, _, _, p = mas_reader.read()
        assert p.shape == (mas_reader.shape[2],)

    def test_r_scale_unit_is_psi_rsun(self, mas_reader):
        _, r, _, _ = mas_reader.read()
        assert r.unit == PSI_rsun

    def test_t_scale_unit_is_psi_angle(self, mas_reader):
        _, _, t, _ = mas_reader.read()
        assert t.unit == PSI_angle

    def test_p_scale_unit_is_psi_angle(self, mas_reader):
        _, _, _, p = mas_reader.read()
        assert p.unit == PSI_angle

    def test_unit_gauss_converts(self, mas_reader):
        result = mas_reader.read(unit='Gauss', scales=False)
        assert result.unit == u.Gauss

    def test_unit_native_returns_code_units(self, mas_reader):
        result = mas_reader.read(unit='native', scales=False)
        assert result.unit == MAS_b

    def test_unit_physical_decomposes_from_mas_b(self, mas_reader):
        result = mas_reader.read(unit='physical', scales=False)
        assert result.unit != MAS_b

    def test_partial_r_slice_reduces_data_and_scale(self, mas_reader):
        data, r, t, p = mas_reader.read(slice(0, 3))
        assert data.shape[-1] == 3
        assert r.shape == (3,)

    def test_partial_r_slice_does_not_affect_t_or_p(self, mas_reader):
        # shape is in physical (r, t, p) order: shape[1]=Nt, shape[2]=Np
        data, r, t, p = mas_reader.read(slice(0, 3))
        assert t.shape == (mas_reader.shape[1],)
        assert p.shape == (mas_reader.shape[2],)

    def test_int_index_retains_axis_as_length_one(self, mas_reader):
        data = mas_reader.read(0, scales=False)
        assert data.shape[-1] == 1

    def test_ellipsis_with_leading_r_slice(self, mas_reader):
        data = mas_reader.read(slice(0, 2), ..., scales=False)
        assert data.shape[-1] == 2

    def test_data_values_all_ones(self, mas_reader):
        result = mas_reader.read(scales=False)
        np.testing.assert_allclose(result.value, 1.0)


# ===========================================================================
# Caching behaviour
# ===========================================================================

class TestCachingBehaviour:
    def test_not_cached_initially(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        assert not reader.data_cached
        reader.close()

    def test_full_read_populates_cache(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.read(scales=False)
        assert reader.data_cached
        reader.close()

    def test_partial_read_does_not_populate_cache(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.read(slice(0, 3), scales=False)
        assert not reader.data_cached
        reader.close()

    def test_load_method_populates_cache(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.load()
        assert reader.data_cached
        reader.close()

    def test_cached_and_uncached_reads_agree(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        first = reader.read(scales=False)
        assert reader.data_cached
        second = reader.read(scales=False)
        np.testing.assert_array_equal(first.value, second.value)
        reader.close()


# ===========================================================================
# Cache modes and lifecycle (load / clear / cache setter)
# ===========================================================================

class TestCacheModes:
    def test_eager_cache_loads_at_construction(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas', cache='eager')
        assert reader.data_cached
        reader.close()

    def test_lazy_cache_not_loaded_at_construction(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas', cache='lazy')
        assert not reader.data_cached
        reader.close()

    def test_none_cache_never_caches(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas', cache=None)
        reader.read(scales=False)
        assert not reader.data_cached
        reader.close()

    def test_invalid_cache_mode_raises(self, psi_h5_mas_file):
        with pytest.raises(ValueError):
            PsiData(psi_h5_mas_file, model='mas', cache='sometimes')

    def test_load_with_cache_none_warns_and_no_effect(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas', cache=None)
        with pytest.warns(CacheWarning):
            reader.load()
        assert not reader.data_cached
        reader.close()

    def test_clear_with_eager_cache_warns(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas', cache='eager')
        with pytest.warns(CacheWarning):
            reader.clear()
        assert not reader.data_cached
        reader.close()

    def test_cache_setter_to_eager_triggers_load(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas', cache='lazy')
        assert not reader.data_cached
        reader.cache = 'eager'
        assert reader.data_cached
        reader.close()

    def test_cache_setter_to_none_triggers_clear(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.read(scales=False)
        assert reader.data_cached
        reader.cache = None
        assert not reader.data_cached
        reader.close()

    def test_cached_property_aliases_data_cached(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        assert reader.cached == reader.data_cached
        reader.read(scales=False)
        assert reader.cached == reader.data_cached is True
        reader.close()


# ===========================================================================
# clear() selective flags and recursion
# ===========================================================================

class TestClear:
    def test_clear_data_only_keeps_interp(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.load(interp=True)
        assert reader.data_cached and reader.interp_cached
        reader.clear(data=True, interp=False)
        assert not reader.data_cached
        assert reader.interp_cached
        reader.close()

    def test_clear_interp_only_keeps_data(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.load(interp=True)
        reader.clear(data=False, interp=True)
        assert reader.data_cached
        assert not reader.interp_cached
        reader.close()

    def test_clear_recursive_clears_scales(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.load()
        assert all(scale.data_cached for scale in reader.scales)
        reader.clear(recursive=True)
        assert not any(scale.data_cached for scale in reader.scales)
        reader.close()


# ===========================================================================
# Interpolator cache (interp_cached / load(interp=True))
# ===========================================================================

@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy is required for interpolation")
class TestInterpCache:
    def test_interp_not_cached_initially(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        assert not reader.interp_cached
        reader.close()

    def test_load_interp_true_builds_interpolator(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.load(interp=True)
        assert reader.interp_cached
        reader.close()

    def test_load_interp_false_does_not_build(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.load(interp=False)
        assert reader.data_cached
        assert not reader.interp_cached
        reader.close()

    def test_data_and_interp_caches_independent(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.load()
        assert reader.data_cached
        assert not reader.interp_cached
        reader.close()


# ===========================================================================
# interp() method
# ===========================================================================

@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy is required for interpolation")
class TestInterp:
    def _midpoints(self, reader):
        r = reader.scales.r.read()
        t = reader.scales.t.read()
        p = reader.scales.p.read()
        return np.column_stack([
            [r.value[len(r) // 2]],
            [t.value[len(t) // 2]],
            [p.value[len(p) // 2]],
        ])

    def test_interp_returns_quantity(self, mas_reader):
        positions = self._midpoints(mas_reader)
        result = mas_reader.interp(positions)
        assert isinstance(result, u.Quantity)

    def test_interp_uniform_field_returns_constant(self, mas_reader):
        # The mock field is all ones in code units
        positions = self._midpoints(mas_reader)
        result = mas_reader.interp(positions)
        np.testing.assert_allclose(result.value, 1.0, rtol=1e-5)

    def test_interp_output_length_matches_positions(self, mas_reader):
        r = mas_reader.scales.r.read()
        t = mas_reader.scales.t.read()
        p = mas_reader.scales.p.read()
        positions = np.column_stack([
            [r.value[1], r.value[2]],
            [t.value[1], t.value[2]],
            [p.value[1], p.value[2]],
        ])
        result = mas_reader.interp(positions)
        assert result.shape == (2,)

    def test_interp_unit_conversion(self, mas_reader):
        positions = self._midpoints(mas_reader)
        result = mas_reader.interp(positions, unit='Gauss')
        assert result.unit == u.Gauss

    def test_interp_with_cache_populates_interp_cache(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas', cache='lazy')
        r = reader.scales.r.read()
        t = reader.scales.t.read()
        p = reader.scales.p.read()
        positions = np.column_stack([
            [r.value[len(r) // 2]],
            [t.value[len(t) // 2]],
            [p.value[len(p) // 2]],
        ])
        reader.load()
        reader.interp(positions)
        assert reader.interp_cached
        reader.close()


# ===========================================================================
# vslice() method
# ===========================================================================

class TestVslice:
    def test_bare_scalar_collapses_r_axis_to_one(self, mas_reader):
        result = mas_reader.vslice(0.4, scales=False)
        assert result.shape[-1] == 1

    def test_bare_scalar_data_value_correct_for_uniform_field(self, mas_reader):
        # All-ones data → interpolated value is 1.0 regardless of position
        result = mas_reader.vslice(0.4, scales=False)
        np.testing.assert_allclose(result.value, 1.0)

    def test_returns_quantity(self, mas_reader):
        result = mas_reader.vslice(0.4, scales=False)
        assert isinstance(result, u.Quantity)

    def test_scales_true_returns_four_tuple(self, mas_reader):
        result = mas_reader.vslice(0.4)
        assert isinstance(result, tuple) and len(result) == 4

    def test_value_interpolated_r_scale_is_length_one(self, mas_reader):
        _, r, _, _ = mas_reader.vslice(0.4)
        assert r.shape == (1,)

    def test_value_interpolated_r_scale_matches_target(self, mas_reader):
        _, r, _, _ = mas_reader.vslice(0.4)
        np.testing.assert_allclose(r.value, [0.4], rtol=1e-5)

    def test_index_space_axes_return_full_coordinate(self, mas_reader):
        # shape is in physical (r, t, p) order: shape[1]=Nt, shape[2]=Np
        _, r, t, p = mas_reader.vslice(0.4)
        assert t.shape == (mas_reader.shape[1],)
        assert p.shape == (mas_reader.shape[2],)

    def test_scales_false_returns_quantity_not_tuple(self, mas_reader):
        result = mas_reader.vslice(0.4, scales=False)
        assert not isinstance(result, tuple)

    def test_quantity_arg_accepted(self, mas_reader):
        result = mas_reader.vslice(0.4 * PSI_rsun, scales=False)
        assert result.shape[-1] == 1

    def test_out_of_bounds_raises_by_default(self, mas_reader):
        with pytest.raises(ValueError):
            mas_reader.vslice(2.0, scales=False)

    def test_out_of_bounds_no_raise_with_bounds_error_false(self, mas_reader):
        result = mas_reader.vslice(2.0, bounds_error=False, scales=False)
        assert isinstance(result, u.Quantity)

    def test_unit_conversion_applied(self, mas_reader):
        result = mas_reader.vslice(0.4, unit='Gauss', scales=False)
        assert result.unit == u.Gauss

    def test_index_space_arg_gives_same_result_as_read(self, mas_reader):
        result_vslice = mas_reader.vslice(slice(0, 3), scales=False)
        result_read = mas_reader.read(slice(0, 3), scales=False)
        np.testing.assert_array_equal(result_vslice.value, result_read.value)

    def test_mixed_value_and_index_space_args(self, mas_reader):
        # shape is in physical (r, t, p) order: shape[1]=Nt
        data, r, t, p = mas_reader.vslice(0.4, None, None)
        assert data.shape[-1] == 1
        assert r.shape == (1,)
        assert t.shape == (mas_reader.shape[1],)


# ===========================================================================
# Coordinate scale readers
# ===========================================================================

class TestCoordinateScaleReaders:
    def test_r_scale_read_returns_quantity(self, mas_reader):
        assert isinstance(mas_reader.scales.r.read(), u.Quantity)

    def test_r_scale_unit_is_psi_rsun(self, mas_reader):
        assert mas_reader.scales.r.read().unit == PSI_rsun

    def test_t_scale_unit_is_psi_angle(self, mas_reader):
        assert mas_reader.scales.t.read().unit == PSI_angle

    def test_p_scale_unit_is_psi_angle(self, mas_reader):
        assert mas_reader.scales.p.read().unit == PSI_angle

    def test_r_scale_shape(self, mas_reader):
        # shape is in physical (r, t, p) order: shape[0]=Nr
        assert mas_reader.scales.r.read().shape == (mas_reader.shape[0],)

    def test_t_scale_shape(self, mas_reader):
        assert mas_reader.scales.t.read().shape == (mas_reader.shape[1],)

    def test_p_scale_shape(self, mas_reader):
        assert mas_reader.scales.p.read().shape == (mas_reader.shape[2],)

    def test_r_scale_read_with_slice(self, mas_reader):
        assert mas_reader.scales.r.read(slice(0, 3)).shape == (3,)

    def test_scale_values_monotonically_increasing(self, mas_reader):
        r = mas_reader.scales.r.read()
        assert np.all(np.diff(r.value) > 0)
