"""Unit tests for psi_io._props and the pure helper functions in psi_io.mhd_io."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import pytest

from psi_io._mesh import Mesh
from psi_io._props import (
    Props,
    MasQuantities,
    _MAS_QUANTITY_PROPS_MAPPING,
    _POT3D_QUANTITY_PROPS_MAPPING,
    _PSI_SCALE_PROPS_MAPPING,
)
from psi_io.mhd_io import (
    HDF_EXT_MAPPING,
    get_mas_quantity_properties,
    get_pot3d_quantity_properties,
    get_psi_scale_properties,
    extract_quantity_from_filepath,
    extract_sequence_from_filepath,
    parse_mas_filename_schema,
)


# ===========================================================================
# _props module tests
# ===========================================================================

class TestProps:
    """Tests for the Props dataclass itself."""

    def test_fields_accessible(self):
        p = Props('br', 'Radial B', u.Gauss, 0b100)
        assert p.name == 'br'
        assert p.desc == 'Radial B'
        assert p.unit == u.Gauss
        assert p._mesh == 0b100

    def test_str_returns_name(self):
        p = Props('vr', 'Radial V', u.km / u.s, 0)
        assert str(p) == 'vr'

    def test_mesh_property_returns_tuple(self):
        p = Props('br', 'desc', u.Gauss, 0b100)
        mesh = p.mesh
        assert isinstance(mesh, tuple)
        assert len(mesh) == 3
        assert all(isinstance(m, Mesh) for m in mesh)

    def test_mesh_property_none_when_no_mesh(self):
        p = Props('r', 'Radial Scale', u.cm)
        assert p.mesh is None

    def test_frozen_immutable(self):
        p = Props('br', 'Radial B', u.Gauss, 0b100)
        with pytest.raises((AttributeError, TypeError)):
            p.name = 'bt'  # type: ignore[misc]

    def test_mul_returns_quantity(self):
        p = Props('br', 'desc', u.Gauss, 0b100)
        result = p * 2.0
        assert isinstance(result, u.Quantity)

    def test_rmul_returns_quantity(self):
        p = Props('br', 'desc', u.Gauss, 0b100)
        result = 2.0 * p
        assert isinstance(result, u.Quantity)

    def test_rtruediv_returns_quantity(self):
        p = Props('br', 'desc', u.Gauss, 0b100)
        result = 1.0 / p
        assert isinstance(result, u.Quantity)

    def test_mul_value_correct(self):
        p = Props('br', 'desc', u.Gauss, 0b100)
        result = p * 3.5
        assert result.value == pytest.approx(3.5)
        assert result.unit == u.Gauss


class TestMasQuantityPropsMapping:
    """Tests for _MAS_QUANTITY_PROPS_MAPPING contents."""

    ALL_MAS_QUANTITIES = [
        'vr', 'vt', 'vp', 'br', 'bt', 'bp', 'jr', 'jt', 'jp',
        't', 'te', 'tp', 'rho', 'p', 'ep', 'em', 'zp', 'zm', 'heat'
    ]

    def test_all_19_quantities_present(self):
        assert len(_MAS_QUANTITY_PROPS_MAPPING) == 19

    @pytest.mark.parametrize("qty", ALL_MAS_QUANTITIES)
    def test_each_quantity_has_name(self, qty):
        assert _MAS_QUANTITY_PROPS_MAPPING[qty].name == qty

    @pytest.mark.parametrize("qty", ALL_MAS_QUANTITIES)
    def test_each_quantity_has_non_empty_desc(self, qty):
        assert len(_MAS_QUANTITY_PROPS_MAPPING[qty].desc) > 0

    @pytest.mark.parametrize("qty", ALL_MAS_QUANTITIES)
    def test_each_quantity_has_unit(self, qty):
        assert _MAS_QUANTITY_PROPS_MAPPING[qty].unit is not None

    @pytest.mark.parametrize("qty", ALL_MAS_QUANTITIES)
    def test_each_quantity_has_mesh_code(self, qty):
        assert _MAS_QUANTITY_PROPS_MAPPING[qty]._mesh is not None

    def test_b_field_components_have_face_stagger(self):
        # br: half in r (MSB=1), bt: half in t (middle=1), bp: half in p (LSB=1)
        assert _MAS_QUANTITY_PROPS_MAPPING['br']._mesh == 0b100
        assert _MAS_QUANTITY_PROPS_MAPPING['bt']._mesh == 0b010
        assert _MAS_QUANTITY_PROPS_MAPPING['bp']._mesh == 0b001

    def test_velocity_components_have_edge_stagger(self):
        assert _MAS_QUANTITY_PROPS_MAPPING['vr']._mesh == 0b011
        assert _MAS_QUANTITY_PROPS_MAPPING['vt']._mesh == 0b101
        assert _MAS_QUANTITY_PROPS_MAPPING['vp']._mesh == 0b110

    def test_scalar_quantities_all_half(self):
        for qty in ('t', 'te', 'tp', 'rho', 'p', 'ep', 'em', 'zp', 'zm', 'heat'):
            assert _MAS_QUANTITY_PROPS_MAPPING[qty]._mesh == 0b111, f"{qty} mesh != 0b111"

    def test_is_immutable(self):
        with pytest.raises(TypeError):
            _MAS_QUANTITY_PROPS_MAPPING['br'] = None  # type: ignore[index]


class TestPot3dQuantityPropsMapping:
    """Tests for _POT3D_QUANTITY_PROPS_MAPPING contents."""

    def test_three_quantities_present(self):
        assert len(_POT3D_QUANTITY_PROPS_MAPPING) == 3

    @pytest.mark.parametrize("qty", ['br', 'bt', 'bp'])
    def test_each_quantity_name(self, qty):
        assert _POT3D_QUANTITY_PROPS_MAPPING[qty].name == qty

    @pytest.mark.parametrize("qty", ['br', 'bt', 'bp'])
    def test_each_quantity_has_mesh(self, qty):
        assert _POT3D_QUANTITY_PROPS_MAPPING[qty]._mesh is not None


class TestPsiScalePropsMapping:
    """Tests for _PSI_SCALE_PROPS_MAPPING contents."""

    def test_three_scales_present(self):
        assert len(_PSI_SCALE_PROPS_MAPPING) == 3

    def test_r_scale_name(self):
        assert _PSI_SCALE_PROPS_MAPPING['r'].name == 'r'

    def test_t_scale_name(self):
        assert _PSI_SCALE_PROPS_MAPPING['t'].name == 't'

    def test_p_scale_name(self):
        assert _PSI_SCALE_PROPS_MAPPING['p'].name == 'p'

    def test_r_scale_has_no_mesh(self):
        assert _PSI_SCALE_PROPS_MAPPING['r'].mesh is None

    def test_scales_have_units(self):
        for key in ('r', 't', 'p'):
            assert _PSI_SCALE_PROPS_MAPPING[key].unit is not None


# ===========================================================================
# mhd_io pure helper function tests
# ===========================================================================

class TestGetMasQuantityProperties:
    def test_returns_props(self):
        assert isinstance(get_mas_quantity_properties('br'), Props)

    def test_correct_quantity_returned(self):
        assert get_mas_quantity_properties('br').name == 'br'

    def test_case_insensitive(self):
        assert get_mas_quantity_properties('BR').name == 'br'
        assert get_mas_quantity_properties('Br').name == 'br'

    def test_all_quantities_accessible(self):
        for qty in ('vr', 'vt', 'vp', 'br', 'bt', 'bp', 'jr', 'jt', 'jp',
                    't', 'te', 'tp', 'rho', 'p', 'ep', 'em', 'zp', 'zm', 'heat'):
            p = get_mas_quantity_properties(qty)
            assert p.name == qty

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid variable"):
            get_mas_quantity_properties('notafield')

    def test_invalid_empty_raises(self):
        with pytest.raises((ValueError, KeyError)):
            get_mas_quantity_properties('')


class TestGetPot3dQuantityProperties:
    def test_returns_props(self):
        assert isinstance(get_pot3d_quantity_properties('br'), Props)

    def test_correct_quantity_returned(self):
        assert get_pot3d_quantity_properties('bp').name == 'bp'

    def test_case_insensitive(self):
        assert get_pot3d_quantity_properties('BR').name == 'br'

    def test_all_three_quantities(self):
        for qty in ('br', 'bt', 'bp'):
            assert get_pot3d_quantity_properties(qty).name == qty

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid variable"):
            get_pot3d_quantity_properties('t')


class TestGetPsiScaleProperties:
    def test_returns_props(self):
        assert isinstance(get_psi_scale_properties('r'), Props)

    def test_radial_scale(self):
        assert get_psi_scale_properties('r').name == 'r'

    def test_theta_scale(self):
        assert get_psi_scale_properties('t').name == 't'

    def test_phi_scale(self):
        assert get_psi_scale_properties('p').name == 'p'

    def test_first_char_only(self):
        # Uses only first character, so 'theta' → 't'
        assert get_psi_scale_properties('theta').name == 't'
        assert get_psi_scale_properties('phi').name == 'p'
        assert get_psi_scale_properties('radius').name == 'r'

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid variable"):
            get_psi_scale_properties('x')


class TestExtractQuantityFromFilepath:
    @pytest.mark.parametrize("stem,expected", [
        ('br001001.h5', 'br'),
        ('vr001.h5', 'vr'),
        ('heat001.h5', 'heat'),
        ('t001001.h5', 't'),
        ('rho001001.hdf', 'rho'),
        ('BR001001.h5', 'br'),   # case-insensitive
    ])
    def test_known_quantities(self, stem, expected):
        assert extract_quantity_from_filepath(Path(stem)) == expected

    def test_unknown_returns_none(self):
        assert extract_quantity_from_filepath(Path('unknown.h5')) is None

    def test_unknown_returns_default(self):
        assert extract_quantity_from_filepath(Path('unknown.h5'), default='br') == 'br'

    def test_empty_stem_returns_default(self):
        assert extract_quantity_from_filepath(Path('.h5'), default='vr') == 'vr'

    def test_longer_quantity_wins_over_prefix(self):
        # 'heat' must match before 'h'
        assert extract_quantity_from_filepath(Path('heat001.h5')) == 'heat'


class TestExtractSequenceFromFilepath:
    @pytest.mark.parametrize("stem,expected", [
        ('br001001.h5', 1001),
        ('vr001.h5', 1),
        ('heat123456.h5', 123456),
        ('t001001.hdf', 1001),
    ])
    def test_known_sequences(self, stem, expected):
        assert extract_sequence_from_filepath(Path(stem)) == expected

    def test_no_digits_returns_none(self):
        assert extract_sequence_from_filepath(Path('nosequence.h5')) is None

    def test_no_digits_returns_default(self):
        assert extract_sequence_from_filepath(Path('nosequence.h5'), default=0) == 0

    def test_returns_int(self):
        result = extract_sequence_from_filepath(Path('br001001.h5'))
        assert isinstance(result, int)


class TestParseMasFilenameSchema:
    @pytest.mark.parametrize("stem,qty,seq", [
        ('br001001.h5', 'br', 1001),
        ('vr001.h5', 'vr', 1),
        ('heat001.hdf', 'heat', 1),
        ('rho001001.h5', 'rho', 1001),
        ('t001.h5', 't', 1),
    ])
    def test_valid_filenames(self, stem, qty, seq):
        quantity, sequence = parse_mas_filename_schema(Path(stem))
        assert quantity == qty
        assert sequence == seq
        assert isinstance(sequence, int)

    def test_case_insensitive(self):
        quantity, sequence = parse_mas_filename_schema(Path('BR001001.h5'))
        assert quantity == 'BR'  # raw match group, not lowercased
        assert sequence == 1001

    def test_invalid_name_raises_value_error(self):
        with pytest.raises(ValueError, match="does not match"):
            parse_mas_filename_schema(Path('notvalid.h5'))

    def test_bare_quantity_no_sequence_raises(self):
        with pytest.raises(ValueError):
            parse_mas_filename_schema(Path('br.h5'))

    def test_short_sequence_too_few_digits_raises(self):
        # Sequence must be 3 or 6 digits; 2 digits should fail
        with pytest.raises(ValueError):
            parse_mas_filename_schema(Path('br01.h5'))

    def test_returns_tuple_of_str_and_int(self):
        qty, seq = parse_mas_filename_schema(Path('br001001.h5'))
        assert isinstance(qty, str)
        assert isinstance(seq, int)


class TestHdfExtMapping:
    def test_h5_extension(self):
        assert HDF_EXT_MAPPING['h5'] == '.h5'

    def test_h4_extension(self):
        assert HDF_EXT_MAPPING['h4'] == '.hdf'

    def test_only_two_keys(self):
        assert set(HDF_EXT_MAPPING.keys()) == {'h5', 'h4'}