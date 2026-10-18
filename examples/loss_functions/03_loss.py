# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
# OAM / complex mode overlap
import torch
import math

def loss(ctx):
    # Target: OAM mode. A complex overlap includes phase and radial profile.
    overlap = (ctx["target_field"].conj() * ctx["field"]).sum((-2,-1)).abs()**2
    return (1 - overlap / ctx["total_power"]).mean()

def metrics(ctx):
    ell, powers = ctx["oam_spectrum"](ctx["field"])
    return {"target_angular_purity": powers[:, ell == ctx["target_charge"]].sum(1).mean()}

def plots(ctx):
    ell, powers = ctx["oam_spectrum"](ctx["field"])
    return {"oam_spectrum": powers[0], "target_phase": torch.angle(ctx["target_field"])}
