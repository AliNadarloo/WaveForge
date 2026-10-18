# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
# Beam steering / centroid
import torch
import math

def loss(ctx):
    normalized = ctx["intensity"] / ctx["total_power"][:,None,None]
    cx = (normalized*ctx["X"]).sum((-2,-1)) / ctx["wavelength"]
    cy = (normalized*ctx["Y"]).sum((-2,-1)) / ctx["wavelength"]
    return ((cx-3.)**2 + cy**2).mean()

def metrics(ctx):
    return {"centroid_x_wavelengths": (ctx["intensity"]*ctx["X"]).sum() / ctx["total_power"].sum() / ctx["wavelength"]}
