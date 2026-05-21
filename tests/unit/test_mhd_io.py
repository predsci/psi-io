"""Unit tests for psi_io.mhd_io — everything except the read() method."""

from __future__ import annotations

import numpy as np
import pytest

import astropy.units as u
import h5py

from psi_io._mesh import Mesh
from psi_io._units import MAS_b
from psi_io.mhd_io import (
    _HDF_EXT_MAPPING,
    METADATA_SCHEMA,
    Scales,
    _CODE_UNIT_ALIASES,
    _REAL_UNIT_ALIASES,
    _cast_to_slice,
    _parse_islice_args,
    H5Data,
    PsiData,
)


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
    def test_has_five_keys(self):
        assert set(METADATA_SCHEMA.keys()) == {'quantity', 'sequence', 'unit', 'scalar', 'mesh'}

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

    def test_invalid_string_raises_type_error(self):
        with pytest.raises(TypeError):
            _cast_to_slice("all")


# ===========================================================================
# _parse_islice_args
# ===========================================================================

class TestParseISliceArgs:
    def test_no_args_returns_full_slices(self):
        shape = (10, 20, 30)
        slices = list(_parse_islice_args(shape=shape))
        assert slices == [slice(None), slice(None), slice(None)]

    def test_single_arg_first_axis(self):
        shape = (10, 20, 30)
        slices = list(_parse_islice_args(slice(0, 5), shape=shape))
        assert slices[0] == slice(0, 5)
        assert slices[1] == slice(None)
        assert slices[2] == slice(None)

    def test_ellipsis_expands_to_full_slices(self):
        shape = (10, 20, 30)
        slices = list(_parse_islice_args(slice(0, 3), ..., shape=shape))
        assert slices[0] == slice(0, 3)
        assert slices[1] == slice(None)
        assert slices[2] == slice(None)

    def test_empty_slice_raises(self):
        with pytest.raises(ValueError):
            list(_parse_islice_args(slice(5, 5), shape=(10, 20, 30)))

    def test_single_element_slice_ok(self):
        slices = list(_parse_islice_args(slice(0, 1), shape=(10, 20, 30)))
        assert slices[0] == slice(0, 1)


# ===========================================================================
# Scales namedtuple
# ===========================================================================

class TestScalesNamedTuple:
    def test_has_r_t_p_fields(self):
        s = Scales(r=1, t=2, p=3)
        assert s.r == 1
        assert s.t == 2
        assert s.p == 3

    def test_positional_construction(self):
        s = Scales(10, 20, 30)
        assert s[0] == 10
        assert s[1] == 20
        assert s[2] == 30

    def test_field_names(self):
        assert Scales._fields == ('r', 't', 'p')


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

    def test_default_model_is_mas(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file)
        assert isinstance(reader, H5Data)
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
    def test_quantity(self, mas_reader):
        assert mas_reader.quantity == 'br'

    def test_sequence(self, mas_reader):
        assert mas_reader.sequence == 1001

    def test_unit_is_mas_b(self, mas_reader):
        assert mas_reader.unit == MAS_b

    def test_mesh_is_tuple_of_mesh(self, mas_reader):
        mesh = mas_reader.mesh
        assert isinstance(mesh, tuple)
        assert all(isinstance(m, Mesh) for m in mesh)

    def test_mesh_br_stagger(self, mas_reader):
        # br → 0b100 → (HALF, MAIN, MAIN)
        assert mas_reader.mesh == (Mesh.HALF, Mesh.MAIN, Mesh.MAIN)

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

    def test_description_is_string(self, mas_reader):
        assert isinstance(mas_reader.description, str)
        assert len(mas_reader.description) > 0

    def test_description_mentions_field(self, mas_reader):
        assert 'Radial' in mas_reader.description or 'radial' in mas_reader.description

    def test_native_properties_is_props(self, mas_reader):
        from psi_io._models import Props
        assert isinstance(mas_reader.props, Props)

    def test_native_properties_quantity_matches(self, mas_reader):
        assert mas_reader.props.name == mas_reader.quantity

    def test_scales_is_scales_namedtuple(self, mas_reader):
        assert isinstance(mas_reader.scales, Scales)

    def test_scales_has_r_t_p(self, mas_reader):
        scales = mas_reader.scales
        assert hasattr(scales, 'r')
        assert hasattr(scales, 't')
        assert hasattr(scales, 'p')

    def test_scale_quantities(self, mas_reader):
        scales = mas_reader.scales
        assert scales.r.quantity == 'r'
        assert scales.t.quantity == 't'
        assert scales.p.quantity == 'p'

    def test_scale_units_are_psi_units(self, mas_reader):
        from psi_io._units import PSI_rsun, PSI_angle
        scales = mas_reader.scales
        assert scales.r.unit == PSI_rsun
        assert scales.t.unit == PSI_angle
        assert scales.p.unit == PSI_angle


# ===========================================================================
# H5Pot3dData properties
# ===========================================================================

class TestH5Pot3dDataProperties:
    def test_quantity(self, pot3d_reader):
        assert pot3d_reader.quantity == 'br'

    def test_sequence(self, pot3d_reader):
        assert pot3d_reader.sequence == 1

    def test_mesh_br_pot3d_stagger(self, pot3d_reader):
        # POT3D br → 0b011 → (MAIN, HALF, HALF)
        assert pot3d_reader.mesh == (Mesh.MAIN, Mesh.HALF, Mesh.HALF)

    def test_ndim(self, pot3d_reader):
        assert pot3d_reader.ndim == 3

    def test_description_is_string(self, pot3d_reader):
        desc = pot3d_reader.description
        assert isinstance(desc, str) and len(desc) > 0


# ===========================================================================
# Context manager
# ===========================================================================

class TestContextManager:
    def test_enter_returns_reader(self, psi_h5_mas_file):
        with PsiData(psi_h5_mas_file, model='mas') as reader:
            assert reader is not None
            assert reader.quantity == 'br'

    def test_exit_closes_file(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.close()
        assert reader._fileref is None


# ===========================================================================
# open / close
# ===========================================================================

class TestOpenClose:
    def test_close_sets_fileref_none(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.close()
        assert reader._fileref is None

    def test_reopen_after_close(self, psi_h5_mas_file):
        reader = PsiData(psi_h5_mas_file, model='mas')
        reader.close()
        reader.open()
        assert reader._fileref is not None
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

    def test_full_slice_shape_matches_storage(self, mas_reader):
        # __getitem__ with no restriction returns full array in storage order
        full = mas_reader[slice(None), slice(None), slice(None)]
        assert full.shape == mas_reader.shape
