from psi_io import read_hdf_meta

def test_datasetid_none(filepath_2d_data):
    result = read_hdf_meta(ifile=filepath_2d_data)
    debug=1