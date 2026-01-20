import pytest

from psi_io import read_hdf_meta


def test_nd_default_dataids(datafile, default_dataid,):
    result, *_ = read_hdf_meta(datafile)
    assert result.name == default_dataid


def test_nd_default_scaleids(datafile, scaleids,):
    result, *_ = read_hdf_meta(datafile)
    for scale, scalecheck in zip(result.scales, scaleids):
        assert scale.name == scalecheck


def test_fl_default_datasetid(filepath_fls_data, default_dataid,):
    result, *_ = read_hdf_meta(filepath_fls_data)
    assert result.name == default_dataid


def test_fl_default_scaleid(filepath_fls_data,):
    result, *_ = read_hdf_meta(filepath_fls_data)
    assert len(result.scales) == 0


def test_nd_invalid_dataids(datafile, error_types,):
    with pytest.raises(error_types):
        read_hdf_meta(datafile, dataset_id="NULL")


def test_fl_invalid_dataids(filepath_fls_data, error_types,):
    with pytest.raises(error_types):
        read_hdf_meta(filepath_fls_data, dataset_id="NULL")


def test_nd_valid_dataids(datafile, scaleids,):
    assert read_hdf_meta(datafile, dataset_id=scaleids[0])


def test_h4_h5_data_equivalence(datakind, datafunc,):
    pytest.importorskip("pyhdf")
    if datakind == "sm":
        pytest.skip("No HDF4 synchronic map data available for comparison.")

    result_h4, *_ = read_hdf_meta(datafunc(hdf=".hdf"))
    result_h5, *_ = read_hdf_meta(datafunc(hdf=".h5"))
    assert result_h4.type == result_h5.type
    assert result_h4.shape == result_h5.shape
    assert len(result_h4.scales) == len(result_h5.scales)
    for scale_h4, scale_h5 in zip(reversed(result_h4.scales), result_h5.scales):
        for k in ("type", "shape", "imin", "imax"):
            assert getattr(scale_h4, k) == getattr(scale_h5, k)