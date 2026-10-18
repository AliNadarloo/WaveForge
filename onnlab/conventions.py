# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Explicit right-handed mapping from local (u,v,w) to a CST coordinate frame.

Training always propagates along local +w. Choosing an export axis rotates
positions; it neither mirrors the transverse frame nor conjugates the phase.
"""
import numpy as np

# Columns are local u, local v, propagation w, expressed in global CST x,y,z.
BASES={
    "+z": ((1,0,0),(0,1,0),(0,0,1)),
    "-z": ((1,0,0),(0,-1,0),(0,0,-1)),
    "+x": ((0,1,0),(0,0,1),(1,0,0)),
    "-x": ((0,1,0),(0,0,-1),(-1,0,0)),
    "+y": ((0,0,1),(1,0,0),(0,1,0)),
    "-y": ((0,0,1),(-1,0,0),(0,-1,0)),
}


def cst_coordinates(c,layer):
    u=(np.arange(c.nx)-(c.nx-1)/2)*c.dx_um
    v=(np.arange(c.ny)-(c.ny-1)/2)*c.dy_um
    V,U=np.meshgrid(v,u,indexing="ij")
    distance=sum(c.distances_wl[:layer+1])*c.wavelength*1e6
    basis=np.asarray(BASES[c.cst_axis],dtype=float)
    points=U[...,None]*basis[0]+V[...,None]*basis[1]+distance*basis[2]
    return points


def export_explanation(c):
    u,v,w=BASES[c.cst_axis]
    return (f"CST propagation direction: {c.cst_axis}. Training uses local +w.\n"
            f"Local u → CST {u}; local v → CST {v}; propagation w → CST {w}.\n"
            "For +z: surfaces lie in xy, illumination travels from −z toward +z.\n"
            "Columns advance from negative to positive local u; rows advance from negative to positive local v.\n"
            "CST_phase_layer_N.txt gives explicit x_um y_um z_um phase_rad phase_deg amplitude at each CELL CENTER.\n"
            "The input plane is the origin along propagation; layer positions include the configured spacings.\n"
            "The app uses exp(+iωt), outgoing exp(−ikR), and surface transmission t=exp(+iφ).\n"
            "The phase is exported without conjugation. Match the transmission-phase / reference-plane definition of your CST unit-cell library.\n"
            "CST tables use signed phase [−π,π] / [−180°,180°]; best_meta_layer_N.txt retains the original [0,2π) radians format.\n"
            "Polarization 1 / 2 are generic scalar channels; the unit-cell geometry determines their physical basis.")
