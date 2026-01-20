import pytest

from psi_io import read_rtp_meta


RTP_RETURN_VALUES = (
    (151, 0.9995602369308472, 30.511646270751953),
    (100, 0.0, 3.1415927410125732),
    (181, 0.0, 6.2831854820251465),
)


def test_rtp_meta_with_3d_data(filepath_3d_data,):
    meta = read_rtp_meta(filepath_3d_data)
    for k, v in zip("rtp", RTP_RETURN_VALUES):
        assert k in meta
        for result, comparison in zip(meta[k], v):
            assert result == comparison


def test_rtp_meta_with_non3d_data(filepath_2d_data, error_types,):
    with pytest.raises(error_types):
        read_rtp_meta(filepath_2d_data)