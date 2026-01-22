import pytest
import numpy as np
from numpy.testing import assert_array_equal

from psi_io import (read_hdf_data,
                    read_hdf_meta,
                    read_rtp_meta,
                    read_hdf_by_index,
                    read_hdf_by_value,
                    read_hdf_by_ivalue,
                    rdhdf_1d, rdhdf_2d, rdhdf_3d,
                    get_scales_1d, get_scales_2d, get_scales_3d
                    )
from tests.utils import generate_data_shape, generate_mock_data

def test_read_hdf_meta(hdf_version, datatype, dimensionality, scales_included, generated_files):
    result, *_ = read_hdf_meta(generated_files[datatype][dimensionality][scales_included])
    expected_shape = generate_data_shape(dimensionality)
    assert result.shape == expected_shape
    assert result.type == datatype
    if scales_included:
        if hdf_version == 'h4':
            for i, scale in enumerate(result.scales):
                assert scale.type == datatype
                assert scale.shape[0] == expected_shape[i]
                assert scale.imin == 0
                assert scale.imax == expected_shape[i]-1
        else:
            for i, scale in enumerate(reversed(result.scales)):
                assert scale.type == datatype
                assert scale.shape[0] == expected_shape[i]
                assert scale.imin == 0
                assert scale.imax == expected_shape[i]-1
    else:
        assert len(result.scales) == 0


def test_read_rtp_meta(hdf_version, datatype, generated_files):
    result = read_rtp_meta(generated_files[datatype][3][True])
    expected_shape = generate_data_shape(3)
    for dim, esize in zip('ptr', expected_shape):
        assert result[dim] == (esize, 0, esize-1)


def test_read_hdf_data(hdf_version, datatype, dimensionality, scales_included, generated_files):
    result = read_hdf_data(generated_files[datatype][dimensionality][scales_included])
    expected = generate_mock_data(dimensionality, datatype, scales_included)
    for darray, earray in zip(result, expected):
        assert darray.shape == earray.shape
        assert darray.dtype == earray.dtype
        assert_array_equal(darray, earray)


@pytest.mark.h4
def test_h4_h5_data_equivalence(datatype, dimensionality, scales_included, combined_files):
    h4result = read_hdf_data(combined_files['h4'][datatype][dimensionality][scales_included])
    h5result = read_hdf_data(combined_files['h5'][datatype][dimensionality][scales_included])
    for h4array, h5array in zip(h4result, h5result):
        assert_array_equal(h4array, h5array)


def test_read_hdf_meta_vs_data(hdf_version, datatype, dimensionality, scales_included, generated_files):
    meta_result, *_ = read_hdf_meta(generated_files[datatype][dimensionality][scales_included])
    data_result = read_hdf_data(generated_files[datatype][dimensionality][scales_included])
    expected_shape = generate_data_shape(dimensionality)

    if scales_included:
        assert meta_result.shape == data_result[0].shape == expected_shape
        assert meta_result.type == data_result[0].dtype == datatype
        if hdf_version == 'h4':
            for result_mscale, result_dscale, esize in zip(reversed(meta_result.scales), data_result[1:], reversed(expected_shape)):
                assert result_mscale.shape[0] == result_dscale.shape[0] == esize
                assert result_mscale.type == result_dscale.dtype == datatype
        else:
            for result_mscale, result_dscale, esize in zip(meta_result.scales, data_result[1:], reversed(expected_shape)):
                assert result_mscale.shape[0] == result_dscale.shape[0] == esize
                assert result_mscale.type == result_dscale.dtype == datatype
    else:
        assert len(meta_result.scales) == 0
        assert meta_result.type == data_result[0].dtype == datatype
        assert meta_result.shape == data_result[0].shape == expected_shape


def test_read_rdhdf_1d(hdf_version, datatype, generated_files):
    result_scale, result_data = rdhdf_1d(generated_files[datatype][1][True])
    expected_data, expected_scale = generate_mock_data(1, datatype, True)
    assert_array_equal(result_data, expected_data)


def test_read_rdhdf_2d(hdf_version, datatype, generated_files):
    *result_scale, result_data = rdhdf_2d(generated_files[datatype][2][True])
    expected_data, *expected_scale = generate_mock_data(2, datatype, True)
    assert_array_equal(result_data, expected_data)


def test_read_rdhdf_3d(hdf_version, datatype, generated_files):
    *result_scale, result_data = rdhdf_3d(generated_files[datatype][3][True])
    expected_data, *expected_scale = generate_mock_data(3, datatype, True)
    assert_array_equal(result_data, expected_data)
    for rs, es in zip(result_scale, expected_scale):
        assert_array_equal(rs, es)


def test_get_scales_1d(hdf_version, datatype, generated_files):
    result_scale, *_ = get_scales_1d(generated_files[datatype][1][True])
    expected_shape = generate_data_shape(1)
    assert_array_equal(result_scale, np.arange(expected_shape[0], dtype=datatype))


def test_get_scales_2d(hdf_version, datatype, generated_files):
    result_scale = get_scales_2d(generated_files[datatype][2][True])
    expected_shape = generate_data_shape(2)
    for rscale, escale in zip(result_scale, reversed(expected_shape)):
        assert_array_equal(rscale, np.arange(escale, dtype=datatype))


def test_get_scales_3d(hdf_version, datatype, generated_files):
    result_scale = get_scales_3d(generated_files[datatype][3][True])
    expected_shape = generate_data_shape(3)
    for rscale, escale in zip(result_scale, reversed(expected_shape)):
        assert_array_equal(rscale, np.arange(escale, dtype=datatype))


def test_read_hdf_by_index(hdf_version, datatype, dimensionality, scales_included, generated_files):
    filepath = generated_files[datatype][dimensionality][scales_included]
    expected_shape = generate_data_shape(dimensionality)
    slices = tuple(dim//2 for dim in expected_shape)
    data_result = read_hdf_by_index(filepath, *slices)
    if scales_included:
        assert_array_equal(data_result[0], np.array(np.sum(slices), dtype=datatype))
        for darray, slice_ in zip(data_result[1:], slices):
            assert_array_equal(darray, np.array([slice_], dtype=datatype))
    else:
        assert_array_equal(data_result, np.array(np.sum(slices), dtype=datatype))


def test_read_hdf_by_value(hdf_version, datatype, dimensionality, generated_files):
    filepath = generated_files[datatype][dimensionality][True]
    expected_shape = generate_data_shape(dimensionality)
    slices = tuple(dim/2 for dim in expected_shape)
    data_result, *scales_result = read_hdf_by_value(filepath, *slices)
    for rscale, slice_ in zip(scales_result, slices):
        assert_array_equal(rscale, np.arange(np.floor(slice_), np.ceil(slice_ + 1), dtype=datatype))


def test_read_hdf_by_ivalue(hdf_version, datatype, dimensionality, generated_files):
    filepath = generated_files[datatype][dimensionality][True]
    expected_shape = generate_data_shape(dimensionality)
    slices = tuple(dim/2 for dim in expected_shape)
    data_result, *scales_result = read_hdf_by_ivalue(filepath, *slices)
    for rscale, slice_ in zip(scales_result, slices):
        assert_array_equal(rscale, np.arange(np.floor(slice_), np.ceil(slice_ + 1), dtype=datatype))


def test_read_hdf_by_ivalue_value_equivalence(hdf_version, datatype, generated_files):
    filepath = generated_files[datatype][3][True]
    test_indices = (None, (5, 7), None,)
    test_values = (None, 5.5, None,)
    by_index_result = read_hdf_by_index(filepath, *test_indices)
    by_value_result = read_hdf_by_value(filepath, *test_values)
    by_ivalue_result = read_hdf_by_ivalue(filepath, *test_values)
    for rbyindex, rbyvalue, rbyivalue in zip(by_index_result, by_value_result, by_ivalue_result):
        assert_array_equal(rbyindex, rbyvalue)
        assert_array_equal(rbyvalue, rbyivalue)


def test_rdhdf1d_by_read_hdf_data(hdf_version, datatype, generated_files):
    filepath = generated_files[datatype][1][True]
    rdhdf_result = rdhdf_1d(filepath)
    bydata_result = read_hdf_data(filepath)
    for r1d, rdata in zip((rdhdf_result[-1], *rdhdf_result[:-1]), bydata_result):
        assert_array_equal(r1d, rdata)


def test_rdhdf2d_by_read_hdf_data(hdf_version, datatype, generated_files):
    filepath = generated_files[datatype][2][True]
    rdhdf_result = rdhdf_2d(filepath)
    bydata_result = read_hdf_data(filepath)
    for r1d, rdata in zip((rdhdf_result[-1], *rdhdf_result[:-1]), bydata_result):
        assert_array_equal(r1d, rdata)


def test_rdhdf3d_by_read_hdf_data(hdf_version, datatype, generated_files):
    filepath = generated_files[datatype][3][True]
    rdhdf_result = rdhdf_3d(filepath)
    bydata_result = read_hdf_data(filepath)
    for r1d, rdata in zip((rdhdf_result[-1], *rdhdf_result[:-1]), bydata_result):
        assert_array_equal(r1d, rdata)
