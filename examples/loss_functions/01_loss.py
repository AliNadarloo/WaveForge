# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
# Focus / image intensity
import torch
import math

def loss(ctx):
    # Match the normalized intensity at the output plane.
    predicted = ctx["intensity"] / ctx["total_power"][:, None, None]
    return ((predicted - ctx["target_intensity"]) ** 2).mean() * predicted[0].numel()**2

def metrics(ctx):
    return {"captured_power": ctx["total_power"].mean() / ctx["input_power"].mean()}

def plots(ctx):
    return {"target_intensity": ctx["target_intensity"],
            "intensity_error": ctx["intensity"][0] / ctx["total_power"][0] - ctx["target_intensity"]}
