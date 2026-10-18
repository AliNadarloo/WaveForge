# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Bundled handwritten digits and ten circular output detectors. No downloads at runtime."""
from pathlib import Path
import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
from .paths import application_directory


def data_path():
    root=application_directory()
    return root/"examples/digits/handwritten_digits.npz"


def load_digits(c,device):
    with np.load(data_path(),allow_pickle=False) as source:
        images=source["images"].copy();labels=source["labels"].copy()
    # Dataset rows are top-to-bottom. Our fields use Cartesian y increasing upward.
    images=np.flip(images,axis=1).copy().astype(np.float32)/16
    images=torch.as_tensor(images,device=device)[:,None]
    h,w=max(8,round(c.ny*.75)),max(8,round(c.nx*.75))
    images=F.interpolate(images,size=(h,w),mode="bilinear",align_corners=False)[:,0]
    images=F.pad(images,((c.nx-w)//2,c.nx-w-(c.nx-w)//2,(c.ny-h)//2,c.ny-h-(c.ny-h)//2))
    return images,torch.as_tensor(labels,device=device,dtype=torch.long)


def split_indices(labels,seed,per_class=0):
    """Deterministic stratified 80/20 split; held-out samples never enter updates."""
    values=labels.detach().cpu().numpy() if isinstance(labels,torch.Tensor) else np.asarray(labels)
    rng=np.random.default_rng(seed);train=[];test=[]
    for digit in range(10):
        ids=np.flatnonzero(values==digit);rng.shuffle(ids)
        n_test=max(1,round(.2*len(ids)))
        test.extend(ids[:n_test]);available=ids[n_test:]
        train.extend(available[:per_class] if per_class else available)
    return np.asarray(train,dtype=int),np.asarray(test,dtype=int)


def detector_layout(c,X,Y):
    angles=torch.arange(10,device=X.device)*2*math.pi/10+math.pi/2
    centers=torch.stack((torch.cos(angles),torch.sin(angles)),1)*c.digit_ring_radius_wl*c.wavelength
    masks=((X[None]-centers[:,0,None,None])**2+(Y[None]-centers[:,1,None,None])**2
           <=(c.digit_detector_radius_wl*c.wavelength)**2).float()
    if not bool((masks.sum((-2,-1))>0).all()):
        raise ValueError("Each digit detector must cover field pixels. Increase its radius or sampling.")
    if bool((masks.sum(0)>1).any()):raise ValueError("Digit detector circles overlap on the sampled grid.")
    return masks,centers


def detector_fractions(field,masks):
    intensity=field.abs().square()
    energies=torch.einsum("bhw,khw->bk",intensity,masks)
    return energies/(intensity.sum((-2,-1))[:,None]+1e-16)


def detector_overlay(ctx,c):
    if "digit_detector_centers" not in ctx:return []
    h,w=c.output_shape;r=c.digit_detector_radius_wl*c.wavelength
    return [(float(x)/(w*c.field_dx)+.5,float(y)/(h*c.field_dy)+.5,
             r/(w*c.field_dx),r/(h*c.field_dy),i)
            for i,(x,y) in enumerate(ctx["digit_detector_centers"].cpu().tolist())]
