# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Transparent solver advice. Recommendations are heuristics, not convergence proofs."""
import math


def recommendation(c):
    n=c.nx*c.ny
    out=max(round(c.ny*c.output_scale),c.ny)*max(round(c.nx*c.output_scale),c.nx)
    dense_mib=8*(c.layers*n*n+n*out)/1024**2
    paper=(c.nx==c.ny==32 and c.layers==1 and c.aperture=="Logic gates" and
           c.beam=="Plane wave" and c.target=="Logic detectors" and
           c.dx_um==c.dy_um==150 and c.frequency_thz==1 and c.distances_wl==[1.2,10.8])
    fine_feature=(c.target=="OAM mode" or c.beam=="OAM (LG p=0)" or
                  c.target in ("Text intensity","Image intensity") or min(c.distances_wl)<2 or
                  max(c.dx,c.dy)>c.wavelength/2)
    if paper:
        method="Dense RS";reason="Original paper geometry: retain the exact transfer matrix and loss comparison."
    elif fine_feature or c.output_scale>1:
        method="ASM • evanescent decay"
        reason="Subcell structure, a short propagation segment, or a larger output window benefits from padded, oversampled angular-spectrum propagation."
    elif dense_mib>64 or c.layers>2 or n>4096:
        method="FFT RS";reason="Larger grid / cascade: FFT convolution avoids quadratic dense-matrix storage while retaining the sampled RS kernel."
    else:
        method="Dense RS";reason="Small sampled problem: dense RS is straightforward to inspect and inexpensive."
    asm_h=c.samples_per_cell*max(c.ny,round(c.ny*c.output_scale))*c.padding_factor
    asm_w=c.samples_per_cell*max(c.nx,round(c.nx*c.output_scale))*c.padding_factor
    asm_mib=8*(c.layers+1)*asm_h*asm_w/1024**2
    if method.startswith("ASM") and asm_mib>512:
        if c.output_scale==1:
            method="FFT RS";reason="Requested ASM sampling would be expensive; FFT RS fits the equal-grid task with lower memory. Recheck subcell accuracy if needed."
        else:
            reason+=" Reduce ASM sampling / padding or output scale: the current estimate is large."
    advice=f"Recommended: {method}\n{reason}\nTransfer functions only: dense ≈ {dense_mib:.1f} MiB; ASM ≈ {asm_mib:.1f} MiB. Autograd and fields need additional memory."
    if min(c.dx,c.dy)/c.samples_per_cell>c.wavelength/2:
        advice+="\nASM sampling is coarser than λ/2; increase samples per cell for propagating-angle coverage."
    if max(c.distances_wl)*c.wavelength > min(c.nx*c.dx,c.ny*c.dy)*c.padding_factor:
        advice+="\nLong propagation relative to the padded window: compare larger padding and field of view to check wraparound / truncation."
    advice+="\nVerify convergence by increasing sampling and padding. Advice does not switch methods automatically."
    return method,advice
