# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
def loss(ctx):
    # Correct circle should receive power; competing circles should stay dark.
    fractions = ctx["digit_fractions"]
    correct = fractions.gather(1, ctx["labels"][:, None]).squeeze(1)
    return -torch.log(correct.clamp(min=1e-10)).mean()

def metrics(ctx):
    return {"batch_accuracy": (ctx["digit_fractions"].argmax(1) == ctx["labels"]).float().mean()}

def plots(ctx):
    return {"digit_detector_fractions": ctx["digit_fractions"],
            "numbered_detector_map": (ctx["digit_detector_masks"] * torch.arange(1,11,device=ctx["field"].device)[:,None,None]).sum(0)}
