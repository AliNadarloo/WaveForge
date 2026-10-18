# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
# Paper logic gates
import torch
import math

def loss(ctx):
    # Same balanced per-case loss as the original paper code.
    e = ctx["detector_fractions"]
    value = torch.zeros((), device=e.device)
    for i in range(len(e)):
        hot = ctx["detector_one"] if ctx["label_values"][i] else 1-ctx["detector_one"]
        t0, t1 = (0.0, ctx["desired_fraction"]) if hot == 1 else (ctx["desired_fraction"], 0.0)
        value = value + ctx["sample_weights"][i] * ((e[i,0]-t0)**2 + (e[i,1]-t1)**2 + ctx["outside_weight"]*e[i,2]**2)
    return value / len(e)

def metrics(ctx):
    e = ctx["detector_fractions"]
    expected = torch.where(ctx["labels"].bool(), ctx["detector_one"], 1-ctx["detector_one"])
    return {"accuracy": (e[:,:2].argmax(1) == expected).float().mean(),
            "outside_fraction": e[:,2].mean()}

def plots(ctx):
    return {"detector_fractions": ctx["detector_fractions"]}
