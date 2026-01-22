import pytest

from psi_io.psi_io import read_hdf_meta, read_hdf_by_index, read_hdf_by_value, read_hdf_by_ivalue

@pytest.mark.parametrize(
    "slice_indices",
    [(None, None, None),
    (0, None, None),
    (None, 0, None),
    (None, None, 0),
    (0, 0, None),
    (0, None, 0),
    (None, 0, 0),
    (0, 0, 0),],
    ids=["no_dims", "dim1", "dim2", "dim3", "dim1_dim2", "dim1_dim3", "dim2_dim3", "dim1_dim2_dim3"]
)
@pytest.mark.read_hdf_by_index
def test_read_hdf_by_index(benchmark, hdf_version, generated_files, slice_indices):
    benchmark.pedantic(
        read_hdf_by_index,
        args=(generated_files['float64'][3][True], *slice_indices),
        iterations=50,   # per round
        rounds=10,       # number of rounds
        warmup_rounds=2,
    )
    
    
@pytest.mark.parametrize(
    "slice_values",
    [(None, None, None),
    (5.5, None, None),
    (None, 5.5, None),
    (None, None, 5.5),
    (5.5, 5.5, None),
    (5.5, None, 5.5),
    (None, 5.5, 5.5),
    (5.5, 5.5, 5.5),],
    ids=["no_dims", "dim1", "dim2", "dim3", "dim1_dim2", "dim1_dim3", "dim2_dim3", "dim1_dim2_dim3"]
)
@pytest.mark.read_hdf_by_value
def test_read_hdf_by_value(benchmark, hdf_version, generated_files, slice_values):
    benchmark.pedantic(
        read_hdf_by_value,
        args=(generated_files['float64'][3][True], *slice_values),
        iterations=50,   # per round
        rounds=10,       # number of rounds
        warmup_rounds=2,
    )


@pytest.mark.parametrize(
    "slice_values",
    [(None, None, None),
    (5.5, None, None),
    (None, 5.5, None),
    (None, None, 5.5),
    (5.5, 5.5, None),
    (5.5, None, 5.5),
    (None, 5.5, 5.5),
    (5.5, 5.5, 5.5),],
    ids=["no_dims", "dim1", "dim2", "dim3", "dim1_dim2", "dim1_dim3", "dim2_dim3", "dim1_dim2_dim3"]
)
@pytest.mark.read_hdf_by_ivalue
def test_read_hdf_by_ivalue(benchmark, hdf_version, generated_files, slice_values):
    benchmark.pedantic(
        read_hdf_by_ivalue,
        args=(generated_files['float64'][3][True], *slice_values),
        iterations=50,   # per round
        rounds=10,       # number of rounds
        warmup_rounds=2,
    )
