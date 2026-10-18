# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
# Smooth fabricated phase
import torch
import math

def loss(ctx):
    prediction = ctx["intensity"] / ctx["total_power"][:,None,None]
    fit = ((prediction-ctx["target_intensity"])**2).mean()*prediction[0].numel()**2
    # Circular phase smoothness respects wrapping and remains differentiable.
    smooth = sum((1-torch.cos(p[:,1:]-p[:,:-1])).mean() +
                 (1-torch.cos(p[1:,:]-p[:-1,:])).mean() for p in ctx["surface_phases"])
    return fit + 0.01*smooth
