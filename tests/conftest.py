import importlib
import sys
from pathlib import Path

import pytest

# ADD TYPING CASTING FOR RDHDF_2d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
print(sys.path)

from tests.utils import generate_mock_data, generate_mock_files

HDF_VERSION_MAPPINGS = {
    "h4": {
        "extension": ".hdf",
        "data_id": "Data-Set-2",
        "scale_id": "fakeDim"
    },
    "h5": {
        "extension": ".h5",
        "data_id": "Data",
        "scale_id": "dim"
    },
}

VERSION_PARAMS = ["h4", "h5"]
DATATYPE_PARAMS = ["float32", "float64", "int16", "int32"]
DIMENSIONALITY_PARAMS = [1, 2, 3]
SCALE_PARAMS = [True, False]

@pytest.fixture(scope="session")
def hdf4_available():
    return importlib.util.find_spec("pyhdf") is not None


@pytest.fixture(scope="session")
def dataid_mapping():
    from psi_io.psi_io import PSI_DATA_ID
    return PSI_DATA_ID


@pytest.fixture(scope="session")
def scaleid_mapping():
    from psi_io.psi_io import PSI_SCALE_ID
    return PSI_SCALE_ID


@pytest.fixture(scope="session",
                params=[
                    pytest.param("h4", marks=pytest.mark.h4),
                    pytest.param("h5", marks=pytest.mark.h5),
                ])
def hdf_version(request) -> str:
    return request.param


@pytest.fixture(scope="session",
                params=DATATYPE_PARAMS)
def datatype(request):
    return request.param


@pytest.fixture(scope="session",
                params=DIMENSIONALITY_PARAMS)
def dimensionality(request):
    return request.param


@pytest.fixture(scope="session",
                params=SCALE_PARAMS)
def scales_included(request):
    return request.param


@pytest.fixture(scope="session")
def error_types(hdf_version: str):
    if hdf_version == "h5":
        return KeyError
    else:
        from pyhdf.error import HDF4Error
        return HDF4Error


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # One temp directory for the whole test session
    return tmp_path_factory.mktemp("psi_io_test_data")


@pytest.fixture(scope="session")
def generated_files(hdf_version, data_dir: Path) -> dict[str, Path]:
    """
    Create all test files once per test session and return their paths.
    """

    filepaths = {
        pdatatypes: {
            pdimensions: {
                pscales: None
                for pscales in SCALE_PARAMS
            } for pdimensions in DIMENSIONALITY_PARAMS
        } for pdatatypes in DATATYPE_PARAMS
    }

    for pdatatypes in DATATYPE_PARAMS:
        for pdimensions in DIMENSIONALITY_PARAMS:
            for pscales in SCALE_PARAMS:
                ifile = data_dir / f"mock_{pdatatypes}_{pdimensions}d_{'withscales' if pscales else 'noscales'}{HDF_VERSION_MAPPINGS[hdf_version]['extension']}"
                if not ifile.exists():
                    generate_mock_files(
                        ifile,
                        pdimensions,
                        pdatatypes,
                        pscales,
                    )
                filepaths[pdatatypes][pdimensions][pscales] = ifile

    return filepaths


@pytest.fixture(scope="session")
def combined_files(data_dir: Path) -> dict[str, Path]:
    """
    Create all test files once per test session and return their paths.
    """

    filepaths = {
        phdfversions: {
            pdatatypes: {
                pdimensions: {
                    pscales: None
                    for pscales in SCALE_PARAMS
                } for pdimensions in DIMENSIONALITY_PARAMS
            } for pdatatypes in DATATYPE_PARAMS
        } for phdfversions in VERSION_PARAMS
    }

    for phdfversions in VERSION_PARAMS:
        for pdatatypes in DATATYPE_PARAMS:
            for pdimensions in DIMENSIONALITY_PARAMS:
                for pscales in SCALE_PARAMS:
                    ifile = data_dir / f"mock_{pdatatypes}_{pdimensions}d_{'withscales' if pscales else 'noscales'}{HDF_VERSION_MAPPINGS[phdfversions]['extension']}"
                    if not ifile.exists():
                        generate_mock_files(
                            ifile,
                            pdimensions,
                            pdatatypes,
                            pscales,
                        )
                    filepaths[phdfversions][pdatatypes][pdimensions][pscales] = ifile

    return filepaths


@pytest.fixture(scope="session")
def expensive_file(hdf_version, data_dir: Path) -> dict[str, Path]:
    large_filepath = data_dir / f"expensive_file{HDF_VERSION_MAPPINGS[hdf_version]['extension']}"
    if not large_filepath.exists():
        generate_mock_files(
            large_filepath,
            5,
            "float32",
            True,
        )
    return large_filepath


@pytest.fixture(scope="session")
def cheap_file(hdf_version, data_dir: Path) -> dict[str, Path]:
    small_filepath = data_dir / f"cheap_file{HDF_VERSION_MAPPINGS[hdf_version]['extension']}"
    if not small_filepath.exists():
        generate_mock_files(
            small_filepath,
            3,
            "float32",
            True,
        )
    return small_filepath
