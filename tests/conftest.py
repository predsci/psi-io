import pytest

@pytest.fixture(scope="session", params=["hdf", "h5"])
def filepath_1d_data(request):
    from psi_io.data import get_1d_data
    return get_1d_data(hdf=request.param)


@pytest.fixture(scope="session", params=["hdf", "h5"])
def filepath_2d_data(request):
    from psi_io.data import get_2d_data
    return get_2d_data(hdf=request.param)


@pytest.fixture(scope="session", params=["hdf", "h5"])
def filepath_3d_data(request):
    from psi_io.data import get_3d_data
    return get_3d_data(hdf=request.param)


@pytest.fixture(scope="session", params=["hdf", "h5"])
def filepath_fls_data(request):
    from psi_io.data import get_fieldline_data
    return get_fieldline_data(hdf=request.param)


@pytest.fixture(scope="session")
def filepath_1d_hdf4():
    from psi_io.data import get_1d_data
    return get_1d_data(hdf="hdf")


@pytest.fixture(scope="session")
def filepath_1d_hdf5():
    from psi_io.data import get_1d_data
    return get_1d_data(hdf="h5")