# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Scalar Rayleigh–Sommerfeld propagation, using the paper's exp(+iωt) convention.

Dense RS is the original quadrature. FFT RS is a zero-padded linear convolution
of exactly the same sampled kernel (not an angular-spectrum replacement).
No trainable amplitude or polarization coupling is assumed.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def choose_device(preference="Auto (CUDA / CPU)"):
    if preference == "CPU":
        return torch.device("cpu")
    try:
        if torch.cuda.is_available():
            torch.empty(1, device="cuda")  # catches an unusable CUDA runtime
            return torch.device("cuda")
    except RuntimeError:
        pass
    return torch.device("cpu")


def grid(ny, nx, dx, dy, device):
    # Use linspace exactly as in code_public/train.py, including rounding.
    x = torch.linspace(-nx*dx/2+dx/2, nx*dx/2-dx/2, nx, device=device)
    y = torch.linspace(-ny*dy/2+dy/2, ny*dy/2-dy/2, ny, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    return X, Y


def kernel(dx, dy, z, wavelength, area):
    R = torch.sqrt(dx*dx + dy*dy + float(z)**2).clamp(min=1e-12)
    Rc = R.to(torch.complex64)
    inv_R = 1/Rc
    k = 2*math.pi/wavelength
    h = (1/(2*math.pi)) * (inv_R+1j*k) * (float(z)/R).to(torch.complex64)
    return h * torch.exp(-1j*(k*Rc)) * inv_R * area


class Propagator(nn.Module):
    def __init__(self, config, src_shape, dst_shape, z, device):
        super().__init__()
        self.src_shape, self.dst_shape = src_shape, dst_shape
        self.method = config.propagation
        if self.method.startswith("ASM"):
            # Padded angular spectrum in the shared exp(+iωt) convention:
            # outgoing exp(-ikz*z) for propagating modes and evanescent decay.
            self.canvas_shape=tuple(max(a,b)*config.padding_factor for a,b in zip(src_shape,dst_shape))
            hy,hx=self.canvas_shape
            sy,sx=(hy-src_shape[0])//2,(hx-src_shape[1])//2
            self.source_pad=(sx,hx-src_shape[1]-sx,sy,hy-src_shape[0]-sy)
            self.crop=((hy-dst_shape[0])//2,(hx-dst_shape[1])//2)
            fx=torch.fft.fftfreq(hx,d=config.field_dx,device=device)
            fy=torch.fft.fftfreq(hy,d=config.field_dy,device=device)
            FY,FX=torch.meshgrid(fy,fx,indexing="ij")
            k=2*math.pi/config.wavelength
            transverse=(2*math.pi*FX)**2+(2*math.pi*FY)**2
            kz=torch.sqrt((k*k-transverse).clamp(min=0))
            alpha=torch.sqrt((transverse-k*k).clamp(min=0))
            h=torch.exp(-1j*kz*float(z))
            h=h*(torch.exp(-alpha*abs(float(z))) if "decay" in self.method else (transverse<k*k))
        elif self.method == "Dense RS":
            xs, ys = grid(*src_shape, config.dx, config.dy, device)
            xo, yo = grid(*dst_shape, config.dx, config.dy, device)
            h = kernel(xo.reshape(-1, 1)-xs.reshape(1, -1),
                       yo.reshape(-1, 1)-ys.reshape(1, -1), z,
                       config.wavelength, config.dx*config.dy)
        else:
            ny, nx = src_shape
            # Convolution index ny-1,nx-1 represents zero displacement.
            y = torch.arange(1-ny, ny, device=device)*config.dy
            x = torch.arange(1-nx, nx, device=device)*config.dx
            Y, X = torch.meshgrid(y, x, indexing="ij")
            h = kernel(X, Y, z, config.wavelength, config.dx*config.dy)
            self.fft_shape = tuple(2**math.ceil(math.log2(3*n-2)) for n in src_shape)
            h = torch.fft.fft2(h, s=self.fft_shape)
        self.register_buffer("H", h)

    def forward(self, field):
        if self.method.startswith("ASM"):
            padded=F.pad(field,self.source_pad)
            propagated=torch.fft.ifft2(torch.fft.fft2(padded)*self.H)
            y,x=self.crop;ny,nx=self.dst_shape
            return propagated[...,y:y+ny,x:x+nx]
        if self.method == "Dense RS":
            if field.ndim>2:
                return (field.reshape(field.shape[0],-1) @ self.H.T).reshape(field.shape[0],*self.dst_shape)
            return (self.H @ field.reshape(-1)).reshape(self.dst_shape)
        out = torch.fft.ifft2(torch.fft.fft2(field, s=self.fft_shape)*self.H)
        ny, nx = self.src_shape
        return out[...,ny-1:2*ny-1, nx-1:2*nx-1]


class DiffractiveNetwork(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.paper_order=config.name.startswith("Paper") and config.propagation=="Dense RS"
        shape = (config.ny, config.nx)
        phases=[]
        for _ in range(config.layers):
            phase=torch.zeros(shape,device=device)
            if config.phase_initialization=="Small random":phase=.05*torch.randn(shape,device=device)
            elif config.phase_initialization=="Random [-pi, pi]":phase=(2*torch.rand(shape,device=device)-1)*math.pi
            phases.append(nn.Parameter(phase))
        self.phases=nn.ParameterList(phases)
        sampled_shape=(config.ny*config.sampling,config.nx*config.sampling)
        shapes = [sampled_shape]*(config.layers+1) + [config.output_shape]
        self.propagators = nn.ModuleList([
            Propagator(config, shapes[i], shapes[i+1], z*config.wavelength, device)
            for i, z in enumerate(config.distances_wl)
        ])

    def forward(self, inputs):
        # Batch general datasets into one GPU operation per plane. Keep the exact
        # original per-sample evaluation order for the dense paper presets.
        if len(inputs)>1 and not self.paper_order:
            field=inputs
            if self.config.sampling>1:
                field=field.repeat_interleave(self.config.sampling,-2).repeat_interleave(self.config.sampling,-1)
            incident=[];transmitted=[]
            for propagator,phase in zip(self.propagators,self.phases):
                field=propagator(field);incident.append(field)
                transmission=torch.exp(1j*phase.to(torch.complex64))
                if self.config.sampling>1:
                    transmission=transmission.repeat_interleave(self.config.sampling,0).repeat_interleave(self.config.sampling,1)
                field=field*transmission;transmitted.append(field)
            return self.propagators[-1](field),incident,transmitted
        outputs, incident, transmitted = [], [[] for _ in self.phases], [[] for _ in self.phases]
        for field in inputs:  # original per-sample evaluation order retained
            if self.config.sampling>1:
                field=field.repeat_interleave(self.config.sampling,0).repeat_interleave(self.config.sampling,1)
            for i, phase in enumerate(self.phases):
                field = self.propagators[i](field)
                incident[i].append(field)
                transmission=torch.exp(1j*phase.to(torch.complex64))
                if self.config.sampling>1:
                    transmission=transmission.repeat_interleave(self.config.sampling,0).repeat_interleave(self.config.sampling,1)
                field = field * transmission
                transmitted[i].append(field)
            outputs.append(self.propagators[-1](field))
        return torch.stack(outputs), [torch.stack(v) for v in incident], [torch.stack(v) for v in transmitted]
