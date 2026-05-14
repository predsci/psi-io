from __future__ import annotations

import enum
from types import MappingProxyType
from typing import Sequence, Any, Union, Literal

import numpy as np

MeshCodeType = Union[int, Literal['main', 'half'], Sequence[Any]]

class Mesh(enum.Enum):
    HALF = 1
    MAIN = 0

    def __str__(self):
        return str(self.name)

_MESH_CODE_REVERSE_MAPPING = MappingProxyType({
    '1': 1, 'h': 1, 'half': 1, 'true': 1,
    '0': 0, 'm': 0, 'main': 0, 'false': 0
})

def _normalize_mesh_code(mesh_code: MeshCodeType, ndim: int) -> tuple[Mesh, ...]:
    if isinstance(mesh_code, int):
        mesh_code = format(mesh_code, f'0{ndim}b')
    elif mesh_code == 'main':
        mesh_code = '0' * ndim
    elif mesh_code == 'half':
        mesh_code = '1' * ndim
    elif len(mesh_code) != ndim:
        raise ValueError(f'Mesh code length {len(mesh_code)} does not match data ndim {ndim}.')
    try:
        return tuple(Mesh(_MESH_CODE_REVERSE_MAPPING[str(c).lower()]) for c in mesh_code)
    except KeyError as e:
        raise ValueError(f"Invalid mesh code character '{e.args[0]}'. "
                         f"Valid characters are: {', '.join(_MESH_CODE_REVERSE_MAPPING.keys())}") from None


def _average_adjacent(arr: np.ndarray, axis: int) -> np.ndarray:
    slc_lo = [slice(None)] * arr.ndim
    slc_hi = [slice(None)] * arr.ndim
    slc_lo[axis] = slice(None, -1)
    slc_hi[axis] = slice(1, None)
    return 0.5 * (arr[tuple(slc_lo)] + arr[tuple(slc_hi)])


def remesh_arr(data: np.ndarray, remesh: Sequence[bool] | bool) -> np.ndarray:
    if isinstance(remesh, bool):
        remesh = [remesh] * data.ndim
    for i, shift in enumerate(remesh):
        if shift:
            data = _average_adjacent(data, i)
    return data


def main_mesh(data: np.ndarray,
              mesh_code: int | Sequence) -> np.ndarray:
    mesh_code = _normalize_mesh_code(mesh_code, data.ndim)
    for i, code in enumerate(mesh_code):
        if code.value:
            data = _average_adjacent(data, -i-1)
    return data


