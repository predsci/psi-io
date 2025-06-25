# psi-io

---

Python utilities for interacting with scientific data formats used by 
Predictive Science Inc. (PSI).

Primarily this includes the HDF file format used by several of PSI's
scientific codes for simulating the solar surface, solar corona, and inner 
heliosphere, including:

- [MAS](https://www.predsci.com/mas/)
- [POT3D](https://github.com/predsci/POT3D)
- [HipFT](https://github.com/predsci/hipft)


## Installation

---

```bash
pip install psi-io
```


## Usage

---

### Basic Reading
Read the entire dataset from a 3D Br HDF file output by a POT3D solution (spherical r,t,p):
```python
import psi_io
r, t, p, br = psi_io.rdhdf_3d('br.h5')
```
Here `r`, `t`, and `p` are the 1D coordinate arrays. `br` is the 3D data array.

There are equivalent routines `rdhdf_1d`, and `rdhdf_2d` for 1D and 2D HDF files.

### Optimized Reading and Interpolation
One can also read a specific subset of the datasets from disk.

Extract a wedge of data that just spans a range of interest:

```python
r_range = [1.0, 1.2]
t_range = [0.5, 1.0]
p_range = [2.0, 3.0]
brx, rx, tx, px = psi_io.read_hdf_by_value(r_range, t_range, p_range, ifile='br.h5')
```

Get 2D t,p slice of data interpolated to a specific radius (e.g. 2.0 Rs).
```python
br_slice, t, p = psi_io.np_interpolate_slice_from_hdf(2.5, None, None, ifile='br.h5')
```

Interpolate the data to specific r, t, p positions (supplied as 1D, 2D, or 3D arrays).
```python
import numpy as np
r_vals = np.array([1.0, 1.1])
t_vals = np.array([0.7, 0.7])
t_vals = np.array([1.5, 1.5])
br_vals = psi_io.interpolate_positions_from_hdf(r_vals, t_vals, t_vals, ifile='br.h5')
```

### Writing
Write a new 3D file:
```python
psi_io.wrhdf_3d('br_mod.h5', r, t, p, br_mod)
```
As before `r`, `t`, and `p` are the 1D coordinate arrays. `br_mod` is the 3D data array.

There are equivalent routines `wrhdf_1d`, and `wrhdf_2d` for 1D and 2D HDF files.

## Requirements

---

This package requires the python HDF5 interface, `h5py`, to work with `.h5` files. 

If you are working with HDF4 `.hdf` files then you must also have the optional
`pyhdf` HDF4 python interface installed.

Because these packages require the underlying C HDF libraries to be installed, 
we generally recommend using conda to install these dependencies. This often
makes life much easier than installing from source:

for HDF5
```bash
conda install h5py
```
for HDF4
```bash
conda install pyhdf
```

To isolate these to a specific environment, see `environment.yml`

## Disclaimer

---

This package is currently in a pre-release state as we transition from having
copies of these routines everywhere to centralizing them in a pip-installable
Python package. Although the basic readers are unlikely to change for historical
compatibility reasons, some of the newer things like optimized i/o and object 
oriented hdf reader interface may be more likely to evolve quickly. 

Automated tests and basic documentation are coming soon!
