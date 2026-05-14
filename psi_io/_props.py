from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import astropy.units as u

from psi_io._mesh import _normalize_mesh_code
from psi_io._units import MAS_v, MAS_b, MAS_j, MAS_t, MAS_n, MAS_p, MAS_heat, POT3D_b, PSI_rsun, PSI_angle

MasQuantities = Literal[
    'vr', 'vt', 'vp', 'br', 'bt', 'bp', 'jr', 'jt', 'jp',
    't', 'te', 'tp', 'rho', 'p', 'ep', 'em', 'zp', 'zm', 'heat'
]
Pot3dQuantities = Literal['br', 'bt', 'bp',]
PsiScales = Literal['r', 't', 'p',]


@dataclass(frozen=True, slots=True, repr=True)
class Props:
    name: str
    desc: str
    unit: u.Unit
    _mesh: int = None

    @property
    def mesh(self):
        return _normalize_mesh_code(self._mesh, 3) if self._mesh is not None else None

    def __str__(self):
        return self.name

    def __mul__(self, other):
        return other * self.unit

    def __rmul__(self, other):
        return other * self.unit

    def __rtruediv__(self, other):
        return other / self.unit


_MAS_QUANTITY_PROPS_MAPPING = MappingProxyType({
    'vr': Props('vr', 'Velocity (Radial Component)', MAS_v, 0b011),
    'vt': Props('vt', 'Velocity (Theta Component)', MAS_v, 0b101),
    'vp': Props('vp', 'Velocity (Phi Component)', MAS_v, 0b110),
    'br': Props('br', 'Magnetic Field (Radial Component)', MAS_b, 0b100),
    'bt': Props('bt', 'Magnetic Field (Theta Component)', MAS_b, 0b010),
    'bp': Props('bp', 'Magnetic Field (Phi Component)', MAS_b, 0b001),
    'jr': Props('jr', 'Current Density (Radial Component)', MAS_j, 0b011),
    'jt': Props('jt', 'Current Density (Theta Component)', MAS_j, 0b101),
    'jp': Props('jp', 'Current Density (Phi Component)', MAS_j, 0b110),
    't': Props('t', 'Temperature', MAS_t, 0b111),
    'te': Props('te', 'Electron Temperature', MAS_t, 0b111),
    'tp': Props('tp', 'Proton Temperature', MAS_t, 0b111),
    'rho': Props('rho', 'Density', MAS_n, 0b111),
    'p': Props('p', 'Pressure', MAS_p, 0b111),
    'ep': Props('ep', 'Wave Energy Density (Parallel to the Field)', MAS_p, 0b111),
    'em': Props('em', 'Wave Energy Density (Anti-Parallel to the Field)', MAS_p, 0b111),
    'zp': Props('zp', 'Positive Charge Density', MAS_v, 0b111),
    'zm': Props('zm', 'Negative Charge Density', MAS_v, 0b111),
    'heat': Props('heat', 'Local Coronal Heating Rate', MAS_heat, 0b111),
})

_POT3D_QUANTITY_PROPS_MAPPING = MappingProxyType({
    'br': Props('br', 'Magnetic Field (Radial Component)', POT3D_b, 0b011),
    'bt': Props('bt', 'Magnetic Field (Theta Component)', POT3D_b, 0b101),
    'bp': Props('bp', 'Magnetic Field (Phi Component)', POT3D_b, 0b110),
})

_PSI_SCALE_PROPS_MAPPING = MappingProxyType({
    'r': Props('r', 'Radial Scale (Solar Radii)', PSI_rsun),
    't': Props('t', 'Theta Scale (Co-Latitude)', PSI_angle),
    'p': Props('p', 'Phi Scale (Longitude)', PSI_angle),
})
