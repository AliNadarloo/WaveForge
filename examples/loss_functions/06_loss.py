# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
# Two observation planes
import torch
import math

def loss(ctx):
    # Additional propagation starts immediately AFTER the last metasurface.
    near = ctx["at_z"](5.0 * ctx["wavelength"])
    far = ctx["at_z"](12.0 * ctx["wavelength"])
    mask = ctx["roi"](0., 0., 2.0 * ctx["wavelength"])
    def concentration(field):
        intensity = field.abs()**2
        return (intensity*mask).sum((-2,-1)) / (intensity.sum((-2,-1)) + 1e-16)
    return (2 - concentration(near) - concentration(far)).mean()

def plots(ctx):
    return {"near_intensity": ctx["at_z"](5*ctx["wavelength"])[0].abs()**2,
            "far_intensity": ctx["at_z"](12*ctx["wavelength"])[0].abs()**2}
