# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
# Circular phase matching
import torch
import math

def loss(ctx):
    # Avoid discontinuities at ±pi. Weight phase error by target intensity.
    phase_error = 1 - torch.cos(ctx["phase"] - torch.angle(ctx["target_field"]))
    return (phase_error * ctx["target_intensity"]).sum((-2,-1)).mean()

def plots(ctx):
    return {"wrapped_phase_error": torch.angle(ctx["field"][0] * ctx["target_field"].conj())}
