"""Unit tests for psi_io.models."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import pytest

from psi_io.mesh import Mesh
from psi_io.models import (
    ModelProps,
    ScaleProps,
    _MAS_QUANTITY_PROPS_MAPPING,
    _POT3D_QUANTITY_PROPS_MAPPING,
    _PSI_SCALE_PROPS_MAPPING,
    extract_quantity_from_filepath,
    extract_sequence_from_filepath,
    get_mas_quantity_properties,
    get_model_prop_caller,
    get_pot3d_quantity_properties,
    get_psi_scale_properties,
    parse_psi_filename_schema,
)


# ===========================================================================
# ModelProps dataclass
# ===========================================================================

class TestModelProps:
    def test_fields_accessible(self):
        p = ModelProps('br', 'Radial B', u.Gauss, False, 0b100)
        assert p.name == 'br'
        assert p.desc == 'Radial B'
        assert p.unit == u.Gauss
        assert p._mesh == 0b100

    def test_str_returns_name(self):
        p = ModelProps('vr', 'Radial V', u.km / u.s, False, 0)
        assert str(p) == 'vr'

    def test_mesh_property_returns_mesh(self):
        p = ModelProps('br', 'desc', u.Gauss, False, 0b100)
        mesh = p.mesh
        assert isinstance(mesh, Mesh)

    def test_mesh_property_br_stagger(self):
        p = ModelProps('br', 'desc', u.Gauss, False, 0b100)
        # mesh is a Mesh object; iterate to get per-axis booleans
        assert list(p.mesh) == [True, False, False]  # HALF, MAIN, MAIN

    def test_frozen_immutable(self):
        p = ModelProps('br', 'Radial B', u.Gauss, False, 0b100)
        with pytest.raises((AttributeError, TypeError)):
            p.name = 'bt'  # type: ignore[misc]

    def test_mul_returns_quantity(self):
        p = ModelProps('br', 'desc', u.Gauss, False, 0b100)
        result = p * 2.0
        assert isinstance(result, u.Quantity)

    def test_rmul_returns_quantity(self):
        p = ModelProps('br', 'desc', u.Gauss, False, 0b100)
        result = 2.0 * p
        assert isinstance(result, u.Quantity)

    def test_rtruediv_returns_quantity(self):
        p = ModelProps('br', 'desc', u.Gauss, False, 0b100)
        result = 1.0 / p
        assert isinstance(result, u.Quantity)

    def test_mul_value_correct(self):
        p = ModelProps('br', 'desc', u.Gauss, False, 0b100)
        result = p * 3.5
        assert result.value == pytest.approx(3.5)
        assert result.unit == u.Gauss

    def test_scalar_stored(self):
        p = ModelProps('t', 'Temperature', u.MK, True, 0b111)
        assert p.scalar is True


# ===========================================================================
# ScaleProps dataclass
# ===========================================================================

class TestScaleProps:
    def test_fields_accessible(self):
        p = ScaleProps('r', 'Radial Scale', u.cm)
        assert p.name == 'r'
        assert p.desc == 'Radial Scale'
        assert p.unit == u.cm

    def test_str_returns_name(self):
        p = ScaleProps('r', 'Radial Scale', u.cm)
        assert str(p) == 'r'

    def test_frozen_immutable(self):
        p = ScaleProps('r', 'Radial Scale', u.cm)
        with pytest.raises((AttributeError, TypeError)):
            p.name = 't'  # type: ignore[misc]

    def test_mul_returns_quantity(self):
        p = ScaleProps('r', 'desc', u.cm)
        result = p * 2.0
        assert isinstance(result, u.Quantity)

    def test_rmul_returns_quantity(self):
        p = ScaleProps('r', 'desc', u.cm)
        result = 2.0 * p
        assert isinstance(result, u.Quantity)


# ===========================================================================
# _MAS_QUANTITY_PROPS_MAPPING
# ===========================================================================

class TestMasQuantityPropsMapping:
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


# ===========================================================================
# _POT3D_QUANTITY_PROPS_MAPPING
# ===========================================================================

class TestPot3dQuantityPropsMapping:
    def test_three_quantities_present(self):
        assert len(_POT3D_QUANTITY_PROPS_MAPPING) == 3

    @pytest.mark.parametrize("qty", ['br', 'bt', 'bp'])
    def test_each_quantity_name(self, qty):
        assert _POT3D_QUANTITY_PROPS_MAPPING[qty].name == qty

    @pytest.mark.parametrize("qty", ['br', 'bt', 'bp'])
    def test_each_quantity_has_mesh(self, qty):
        assert _POT3D_QUANTITY_PROPS_MAPPING[qty]._mesh is not None

    def test_pot3d_mesh_codes_are_complement_of_mas(self):
        # POT3D mesh = 0b111 ^ MAS_mesh
        from psi_io.models import _MAS_QUANTITY_PROPS_MAPPING as mas
        for qty in ('br', 'bt', 'bp'):
            assert _POT3D_QUANTITY_PROPS_MAPPING[qty]._mesh == (0b111 ^ mas[qty]._mesh)


# ===========================================================================
# _PSI_SCALE_PROPS_MAPPING
# ===========================================================================

class TestPsiScalePropsMapping:
    def test_six_entries_present(self):
        # r, t, p + aliases radius, theta, phi
        assert len(_PSI_SCALE_PROPS_MAPPING) == 6

    def test_r_scale_name(self):
        assert _PSI_SCALE_PROPS_MAPPING['r'].name == 'r'

    def test_t_scale_name(self):
        assert _PSI_SCALE_PROPS_MAPPING['t'].name == 't'

    def test_p_scale_name(self):
        assert _PSI_SCALE_PROPS_MAPPING['p'].name == 'p'

    def test_alias_radius_same_as_r(self):
        assert _PSI_SCALE_PROPS_MAPPING['radius'] is _PSI_SCALE_PROPS_MAPPING['r']

    def test_alias_theta_same_as_t(self):
        assert _PSI_SCALE_PROPS_MAPPING['theta'] is _PSI_SCALE_PROPS_MAPPING['t']

    def test_alias_phi_same_as_p(self):
        assert _PSI_SCALE_PROPS_MAPPING['phi'] is _PSI_SCALE_PROPS_MAPPING['p']

    def test_scales_have_units(self):
        for key in ('r', 't', 'p'):
            assert _PSI_SCALE_PROPS_MAPPING[key].unit is not None

    def test_scale_props_are_scale_props_instances(self):
        for key in ('r', 't', 'p'):
            assert isinstance(_PSI_SCALE_PROPS_MAPPING[key], ScaleProps)


# ===========================================================================
# get_mas_quantity_properties
# ===========================================================================

class TestGetMasQuantityProperties:
    def test_returns_model_props(self):
        assert isinstance(get_mas_quantity_properties('br'), ModelProps)

    def test_correct_quantity_returned(self):
        assert get_mas_quantity_properties('br').name == 'br'

    def test_case_insensitive(self):
        assert get_mas_quantity_properties('BR').name == 'br'
        assert get_mas_quantity_properties('Br').name == 'br'

    def test_all_quantities_accessible(self):
        for qty in ('vr', 'vt', 'vp', 'br', 'bt', 'bp', 'jr', 'jt', 'jp',
                    't', 'te', 'tp', 'rho', 'p', 'ep', 'em', 'zp', 'zm', 'heat'):
            assert get_mas_quantity_properties(qty).name == qty

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid variable"):
            get_mas_quantity_properties('notafield')

    def test_invalid_empty_raises(self):
        with pytest.raises((ValueError, KeyError)):
            get_mas_quantity_properties('')


# ===========================================================================
# get_pot3d_quantity_properties
# ===========================================================================

class TestGetPot3dQuantityProperties:
    def test_returns_model_props(self):
        assert isinstance(get_pot3d_quantity_properties('br'), ModelProps)

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


# ===========================================================================
# get_psi_scale_properties
# ===========================================================================

class TestGetPsiScaleProperties:
    def test_returns_scale_props(self):
        assert isinstance(get_psi_scale_properties('r'), ScaleProps)

    def test_radial_scale(self):
        assert get_psi_scale_properties('r').name == 'r'

    def test_theta_scale(self):
        assert get_psi_scale_properties('t').name == 't'

    def test_phi_scale(self):
        assert get_psi_scale_properties('p').name == 'p'

    def test_alias_names(self):
        assert get_psi_scale_properties('theta').name == 't'
        assert get_psi_scale_properties('phi').name == 'p'
        assert get_psi_scale_properties('radius').name == 'r'

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid variable"):
            get_psi_scale_properties('x')


# ===========================================================================
# extract_quantity_from_filepath
# ===========================================================================

class TestExtractQuantityFromFilepath:
    @pytest.mark.parametrize("stem,expected", [
        ('br001001.h5',  'br'),
        ('vr001.h5',     'vr'),
        ('heat001.h5',   'heat'),
        ('t001001.h5',   't'),
        ('rho001001.hdf','rho'),
        ('BR001001.h5',  'br'),   # case-insensitive
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
        assert extract_quantity_from_filepath(Path('heat001.h5')) == 'heat'


# ===========================================================================
# extract_sequence_from_filepath
# ===========================================================================

class TestExtractSequenceFromFilepath:
    @pytest.mark.parametrize("stem,expected", [
        ('br001001.h5',   1001),
        ('vr001.h5',      1),
        ('heat123456.h5', 123456),
        ('t001001.hdf',   1001),
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


# ===========================================================================
# parse_psi_filename_schema
# ===========================================================================

class TestParsePsiFilenameSchema:
    @pytest.mark.parametrize("stem,qty,seq", [
        ('br001001.h5',  'br',   1001),
        ('vr001.h5',     'vr',   1),
        ('heat001.hdf',  'heat', 1),
        ('rho001001.h5', 'rho',  1001),
        ('t001.h5',      't',    1),
    ])
    def test_valid_filenames(self, stem, qty, seq):
        quantity, sequence = parse_psi_filename_schema(Path(stem))
        assert quantity.lower() == qty
        assert sequence == seq
        assert isinstance(sequence, int)

    def test_case_insensitive_preserves_raw_match(self):
        quantity, sequence = parse_psi_filename_schema(Path('BR001001.h5'))
        assert quantity == 'BR'   # raw match group, not lowercased
        assert sequence == 1001

    def test_invalid_name_raises_value_error(self):
        with pytest.raises(ValueError, match="does not match"):
            parse_psi_filename_schema(Path('notvalid.h5'))

    def test_bare_quantity_no_sequence_raises(self):
        with pytest.raises(ValueError):
            parse_psi_filename_schema(Path('br.h5'))

    def test_short_sequence_too_few_digits_raises(self):
        with pytest.raises(ValueError):
            parse_psi_filename_schema(Path('br01.h5'))

    def test_returns_tuple_of_str_and_int(self):
        qty, seq = parse_psi_filename_schema(Path('br001001.h5'))
        assert isinstance(qty, str)
        assert isinstance(seq, int)


# ===========================================================================
# get_model_prop_caller
# ===========================================================================

class TestGetModelPropCaller:
    def test_mas_returns_mas_getter(self):
        caller = get_model_prop_caller('mas')
        assert caller is get_mas_quantity_properties

    def test_pot3d_returns_pot3d_getter(self):
        caller = get_model_prop_caller('pot3d')
        assert caller is get_pot3d_quantity_properties

    def test_case_insensitive(self):
        assert get_model_prop_caller('MAS') is get_mas_quantity_properties

    def test_resolved_getter_works(self):
        caller = get_model_prop_caller('mas')
        assert caller('br').name == 'br'

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            get_model_prop_caller('custom')


# ===========================================================================
# _asdict serialization
# ===========================================================================

class TestAsdict:
    def test_scaleprops_asdict_keys(self):
        d = get_psi_scale_properties('r')._asdict()
        assert set(d.keys()) == {'name', 'desc', 'unit'}

    def test_modelprops_asdict_replaces_mesh_int_with_mesh(self):
        d = get_mas_quantity_properties('br')._asdict()
        assert '_mesh' not in d
        assert 'mesh' in d

    def test_modelprops_asdict_mesh_is_mesh_instance(self):
        d = get_mas_quantity_properties('br')._asdict()
        assert isinstance(d['mesh'], Mesh)

    def test_modelprops_asdict_roundtrip_fields(self):
        d = get_mas_quantity_properties('vr')._asdict()
        assert d['name'] == 'vr'
        assert d['scalar'] is False
