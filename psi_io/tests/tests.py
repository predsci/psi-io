"""
Ryder's manual tests copied from a previous iteration of psihdf so that we
have a starting point for building a proper testsuite.

NOT YET INTENDED FOR USE! This will need to be refactored for automated testing.
"""
import timeit
from matplotlib import pyplot as plt

from ..psi_io import *

ITERATION_NUMBER_PWR = 3

def log_h4_unit_test(func):
    def wrapper(*args, **kwargs):
        print("# " + "-" * 64)
        print(f"# TEST H4: {func.__name__}")
        print("# " + "-" * 64)
        print()

        try:
            assert args[0].endswith('.hdf'), "Must pass a hdf4 (.hdf) filepath"
            func(*args, **kwargs)
        except Exception as e:
            print(f"# !!!FAILED!!!: {e}")
        finally:
            print("# " + "-" * 64)
            print(f"# TEST COMPLETE")
            print("# " + "-" * 64)
            print()

    return wrapper


def log_h5_unit_test(func):
    def wrapper(*args, **kwargs):
        print("# " + "-" * 64)
        print(f"# TEST H5: {func.__name__}")
        print("# " + "-" * 64)
        print()

        try:
            assert args[0].endswith('.h5'), "Must pass a hdf5 (.h5) filepath"
            func(*args, **kwargs)
        except Exception as e:
            print(f"# !!!FAILED!!!: {e}")
        finally:
            print("# " + "-" * 64)
            print(f"# TEST COMPLETE")
            print("# " + "-" * 64)
            print()

    return wrapper

def validate_outputs_old_vs_new(ifile: Path | str):
    r0, t0, p0, f0 = rdhdf_3d(ifile)
    f1, r1, t1, p1 = read_hdf_by_value(ifile)
    for val in zip((r0, t0, p0, f0), (r1, t1, p1, f1), ('radial scale', 'theta scale', 'phi scale', 'values')):
        print(f'{val[2].upper()} ' + '-'*32)

        print('\n[[dtype]]')
        print(f'OLD: {val[0].dtype}')
        print(f'NEW: {val[1].dtype}')
        if val[0].dtype == val[1].dtype:
            print('PASSED')
        else:
            print('FAILED')

        print('\n[[shape]]')
        print(f'OLD: {val[0].shape}')
        print(f'NEW: {val[1].shape}')
        if val[0].shape == val[1].shape:
            print('PASSED')

            print("\n[[equivalence]]")
            if np.all(val[0] == val[1]):
                print('PASSED')
            else:
                print('FAILED')
        else:
            print('FAILED')
        print()

@log_h4_unit_test
def validate_h4_outputs_old_vs_new(ifile: Path | str):
    validate_outputs_old_vs_new(ifile)

@log_h5_unit_test
def validate_h5_outputs_old_vs_new(ifile: Path | str):
    validate_outputs_old_vs_new(ifile)

def benchmark_old_vs_new_byvalue_xdim_single(ifile: Path | str):
    value = 2.01234567
    rdhdf_3d(ifile)
    read_hdf_by_value(ifile, values=(value, None, None))
    avg_time_old = []
    avg_time_new = []
    total_namespace = globals().copy()
    total_namespace.update(locals())
    for i in range(ITERATION_NUMBER_PWR):
        print(f'\n[[AVERAGE FOR {10**i} ITERATIONS]]')
        avg_time_old_ = timeit.timeit('rdhdf_3d(ifile)', globals=total_namespace, number=10**i) / 10**i
        avg_time_new_ = timeit.timeit('read_hdf_by_value(ifile, values=(value, None, None))', globals=total_namespace, number=10**i) / 10**i
        print(f'OLD: {avg_time_old_} sec')
        print(f'NEW: {avg_time_new_} sec')
        print(f'OLD/NEW speedup: {avg_time_old_/avg_time_new_}')
        avg_time_old.append(avg_time_old_)
        avg_time_new.append(avg_time_new_)
    methods = ['rdhdf_3d', 'read_hdf_by_value']
    times = [avg_time_old, avg_time_new]
    plt.xscale('log')
    plt.plot([10**i for i in range(ITERATION_NUMBER_PWR)], avg_time_old, color='red', label="rdhdf_rd")
    plt.plot([10**i for i in range(ITERATION_NUMBER_PWR)], avg_time_new, color='green', label='read_hdf_by_value')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Performance Comparison (XDIM SINGLE)')
    plt.legend()
    plt.show()

@log_h4_unit_test
def benchmark_h4_old_vs_new_byvalue_xdim_single(ifile: Path | str):
    benchmark_old_vs_new_byvalue_xdim_single(ifile)
    
@log_h5_unit_test
def benchmark_h5_old_vs_new_byvalue_xdim_single(ifile: Path | str):
    benchmark_old_vs_new_byvalue_xdim_single(ifile)

def benchmark_old_vs_new_byvalue_xdim_range(ifile: Path | str):
    value = (2.01234567, 19.76543210)
    rdhdf_3d(ifile)
    read_hdf_by_value(ifile, values=(None, value, None))
    avg_time_old = []
    avg_time_new = []
    total_namespace = globals().copy()
    total_namespace.update(locals())
    for i in range(ITERATION_NUMBER_PWR):
        print(f'\n[[AVERAGE FOR {10**i} ITERATIONS]]')
        avg_time_old_ = timeit.timeit('rdhdf_3d(ifile)', globals=total_namespace, number=10**i) / 10**i
        avg_time_new_ = timeit.timeit('read_hdf_by_value(ifile, values=(value, None, None))', globals=total_namespace, number=10**i) / 10**i
        print(f'OLD: {avg_time_old_} sec')
        print(f'NEW: {avg_time_new_} sec')
        print(f'OLD/NEW speedup: {avg_time_old_/avg_time_new_}')
        avg_time_old.append(avg_time_old_)
        avg_time_new.append(avg_time_new_)
    methods = ['rdhdf_3d', 'read_hdf_by_value']
    times = [avg_time_old, avg_time_new]
    plt.xscale('log')
    plt.plot([10**i for i in range(ITERATION_NUMBER_PWR)], avg_time_old, color='red', label="rdhdf_rd")
    plt.plot([10**i for i in range(ITERATION_NUMBER_PWR)], avg_time_new, color='green', label='read_hdf_by_value')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Performance Comparison (XDIM RANGE)')
    plt.legend()
    plt.show()
    
@log_h4_unit_test
def benchmark_h4_old_vs_new_byvalue_xdim_range(ifile: Path | str):
    benchmark_old_vs_new_byvalue_xdim_range(ifile)
    
@log_h5_unit_test
def benchmark_h5_old_vs_new_byvalue_xdim_range(ifile: Path | str):
    benchmark_old_vs_new_byvalue_xdim_range(ifile)


def benchmark_old_vs_new_byvalue_ydim_single(ifile: Path | str):
    value = 2.01234567
    rdhdf_3d(ifile)
    read_hdf_by_value(ifile, values=(None, value, None))
    avg_time_old = []
    avg_time_new = []
    total_namespace = globals().copy()
    total_namespace.update(locals())
    for i in range(ITERATION_NUMBER_PWR):
        print(f'\n[[AVERAGE FOR {10 ** i} ITERATIONS]]')
        avg_time_old_ = timeit.timeit('rdhdf_3d(ifile)', globals=total_namespace, number=10 ** i) / 10 ** i
        avg_time_new_ = timeit.timeit('read_hdf_by_value(ifile, values=(None, value, None))', globals=total_namespace, number=10 ** i) / 10 ** i
        print(f'OLD: {avg_time_old_} sec')
        print(f'NEW: {avg_time_new_} sec')
        print(f'OLD/NEW speedup: {avg_time_old_ / avg_time_new_}')
        avg_time_old.append(avg_time_old_)
        avg_time_new.append(avg_time_new_)
    methods = ['rdhdf_3d', 'read_hdf_by_value']
    times = [avg_time_old, avg_time_new]
    plt.xscale('log')
    plt.plot([10 ** i for i in range(ITERATION_NUMBER_PWR)], avg_time_old, color='red', label="rdhdf_rd")
    plt.plot([10 ** i for i in range(ITERATION_NUMBER_PWR)], avg_time_new, color='green', label='read_hdf_by_value')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Performance Comparison (YDIM SINGLE)')
    plt.legend()
    plt.show()


@log_h4_unit_test
def benchmark_h4_old_vs_new_byvalue_ydim_single(ifile: Path | str):
    benchmark_old_vs_new_byvalue_ydim_single(ifile)


@log_h5_unit_test
def benchmark_h5_old_vs_new_byvalue_ydim_single(ifile: Path | str):
    benchmark_old_vs_new_byvalue_ydim_single(ifile)
    
def benchmark_old_vs_new_byvalue_ydim_range(ifile: Path | str):
    value = 2.01234567
    rdhdf_3d(ifile)
    read_hdf_by_value(ifile, values=(None, value, None))
    avg_time_old = []
    avg_time_new = []
    total_namespace = globals().copy()
    total_namespace.update(locals())
    for i in range(ITERATION_NUMBER_PWR):
        print(f'\n[[AVERAGE FOR {10 ** i} ITERATIONS]]')
        avg_time_old_ = timeit.timeit('rdhdf_3d(ifile)', globals=total_namespace, number=10 ** i) / 10 ** i
        avg_time_new_ = timeit.timeit('read_hdf_by_value(ifile, values=(None, value, None))', globals=total_namespace, number=10 ** i) / 10 ** i
        print(f'OLD: {avg_time_old_} sec')
        print(f'NEW: {avg_time_new_} sec')
        print(f'OLD/NEW speedup: {avg_time_old_ / avg_time_new_}')
        avg_time_old.append(avg_time_old_)
        avg_time_new.append(avg_time_new_)
    methods = ['rdhdf_3d', 'read_hdf_by_value']
    times = [avg_time_old, avg_time_new]
    plt.xscale('log')
    plt.plot([10 ** i for i in range(ITERATION_NUMBER_PWR)], avg_time_old, color='red', label="rdhdf_rd")
    plt.plot([10 ** i for i in range(ITERATION_NUMBER_PWR)], avg_time_new, color='green', label='read_hdf_by_value')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Performance Comparison (YDIM RANGE)')
    plt.legend()
    plt.show()


@log_h4_unit_test
def benchmark_h4_old_vs_new_byvalue_ydim_range(ifile: Path | str):
    benchmark_old_vs_new_byvalue_ydim_range(ifile)


@log_h5_unit_test
def benchmark_h5_old_vs_new_byvalue_ydim_range(ifile: Path | str):
    benchmark_old_vs_new_byvalue_ydim_range(ifile)


def benchmark_old_vs_new_byvalue_zdim_single(ifile: Path | str):
    value = 2.01234567
    rdhdf_3d(ifile)
    read_hdf_by_value(ifile, values=(None, None, value))
    avg_time_old = []
    avg_time_new = []
    total_namespace = globals().copy()
    total_namespace.update(locals())
    for i in range(ITERATION_NUMBER_PWR):
        print(f'\n[[AVERAGE FOR {10 ** i} ITERATIONS]]')
        avg_time_old_ = timeit.timeit('rdhdf_3d(ifile)', globals=total_namespace, number=10 ** i) / 10 ** i
        avg_time_new_ = timeit.timeit('read_hdf_by_value(ifile, values=(None, None, value))', globals=total_namespace, number=10 ** i) / 10 ** i
        print(f'OLD: {avg_time_old_} sec')
        print(f'NEW: {avg_time_new_} sec')
        print(f'OLD/NEW speedup: {avg_time_old_ / avg_time_new_}')
        avg_time_old.append(avg_time_old_)
        avg_time_new.append(avg_time_new_)
    methods = ['rdhdf_3d', 'read_hdf_by_value']
    times = [avg_time_old, avg_time_new]
    plt.xscale('log')
    plt.plot([10 ** i for i in range(ITERATION_NUMBER_PWR)], avg_time_old, color='red', label="rdhdf_rd")
    plt.plot([10 ** i for i in range(ITERATION_NUMBER_PWR)], avg_time_new, color='green', label='read_hdf_by_value')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Performance Comparison (ZDIM SINGLE)')
    plt.legend()
    plt.show()


@log_h4_unit_test
def benchmark_h4_old_vs_new_byvalue_zdim_single(ifile: Path | str):
    benchmark_old_vs_new_byvalue_zdim_single(ifile)


@log_h5_unit_test
def benchmark_h5_old_vs_new_byvalue_zdim_single(ifile: Path | str):
    benchmark_old_vs_new_byvalue_zdim_single(ifile)


def benchmark_old_vs_new_byvalue_zdim_range(ifile: Path | str):
    value = 2.01234567
    rdhdf_3d(ifile)
    read_hdf_by_value(ifile, values=(None, None, value))
    avg_time_old = []
    avg_time_new = []
    total_namespace = globals().copy()
    total_namespace.update(locals())
    for i in range(ITERATION_NUMBER_PWR):
        print(f'\n[[AVERAGE FOR {10 ** i} ITERATIONS]]')
        avg_time_old_ = timeit.timeit('rdhdf_3d(ifile)', globals=total_namespace, number=10 ** i) / 10 ** i
        avg_time_new_ = timeit.timeit('read_hdf_by_value(ifile, values=(None, None, value))', globals=total_namespace, number=10 ** i) / 10 ** i
        print(f'OLD: {avg_time_old_} sec')
        print(f'NEW: {avg_time_new_} sec')
        print(f'OLD/NEW speedup: {avg_time_old_ / avg_time_new_}')
        avg_time_old.append(avg_time_old_)
        avg_time_new.append(avg_time_new_)
    methods = ['rdhdf_3d', 'read_hdf_by_value']
    times = [avg_time_old, avg_time_new]
    plt.xscale('log')
    plt.plot([10 ** i for i in range(ITERATION_NUMBER_PWR)], avg_time_old, color='red', label="rdhdf_rd")
    plt.plot([10 ** i for i in range(ITERATION_NUMBER_PWR)], avg_time_new, color='green', label='read_hdf_by_value')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Performance Comparison (ZDIM RANGE)')
    plt.legend()
    plt.show()


@log_h4_unit_test
def benchmark_h4_old_vs_new_byvalue_zdim_range(ifile: Path | str):
    benchmark_old_vs_new_byvalue_zdim_range(ifile)


@log_h5_unit_test
def benchmark_h5_old_vs_new_byvalue_zdim_range(ifile: Path | str):
    benchmark_old_vs_new_byvalue_zdim_range(ifile)