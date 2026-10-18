# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
# Detector efficiency + leakage
import torch
import math

def loss(ctx):
    # Maximize absolute collected energy and penalize power elsewhere.
    mask = ctx["roi"](0.0, 0.0, 2.0 * ctx["wavelength"])
    collected = (ctx["intensity"] * mask).sum((-2,-1))
    outside = ctx["total_power"] - collected
    return (-collected / ctx["input_power"] + 0.2 * outside / ctx["input_power"]).mean()

def plots(ctx):
    return {"central_detector": ctx["roi"](0., 0., 2.*ctx["wavelength"])}
