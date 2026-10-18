# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Differentiable objective context and examples from simple to advanced.

Custom Python runs locally with normal Python privileges. Use code you trust.
Return a real scalar Tensor from loss(ctx); do not detach it or use NumPy there.
Optional metrics(ctx) update live and at completion; plots(ctx) run at completion.
"""
import math
import torch
import torch.nn.functional as F
from .physics import grid, Propagator
from .inputs import build_target


DIGIT_OBJECTIVE = '''def loss(ctx):
    # Correct circle should receive power; competing circles should stay dark.
    fractions = ctx["digit_fractions"]
    correct = fractions.gather(1, ctx["labels"][:, None]).squeeze(1)
    return -torch.log(correct.clamp(min=1e-10)).mean()

def metrics(ctx):
    return {"batch_accuracy": (ctx["digit_fractions"].argmax(1) == ctx["labels"]).float().mean()}

def plots(ctx):
    return {"digit_detector_fractions": ctx["digit_fractions"],
            "numbered_detector_map": (ctx["digit_detector_masks"] * torch.arange(1,11,device=ctx["field"].device)[:,None,None]).sum(0)}
'''

EXAMPLES = {
"Focus / image intensity": '''def loss(ctx):
    # Match the normalized intensity at the output plane.
    predicted = ctx["intensity"] / ctx["total_power"][:, None, None]
    return ((predicted - ctx["target_intensity"]) ** 2).mean() * predicted[0].numel()**2

def metrics(ctx):
    return {"captured_power": ctx["total_power"].mean() / ctx["input_power"].mean()}

def plots(ctx):
    return {"target_intensity": ctx["target_intensity"],
            "intensity_error": ctx["intensity"][0] / ctx["total_power"][0] - ctx["target_intensity"]}
''',
"Paper logic gates": '''def loss(ctx):
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
''',
"OAM / complex mode overlap": '''def loss(ctx):
    # Target: OAM mode. A complex overlap includes phase and radial profile.
    overlap = (ctx["target_field"].conj() * ctx["field"]).sum((-2,-1)).abs()**2
    return (1 - overlap / ctx["total_power"]).mean()

def metrics(ctx):
    ell, powers = ctx["oam_spectrum"](ctx["field"])
    return {"target_angular_purity": powers[:, ell == ctx["target_charge"]].sum(1).mean()}

def plots(ctx):
    ell, powers = ctx["oam_spectrum"](ctx["field"])
    return {"oam_spectrum": powers[0], "target_phase": torch.angle(ctx["target_field"])}
''',
"Circular phase matching": '''def loss(ctx):
    # Avoid discontinuities at ±pi. Weight phase error by target intensity.
    phase_error = 1 - torch.cos(ctx["phase"] - torch.angle(ctx["target_field"]))
    return (phase_error * ctx["target_intensity"]).sum((-2,-1)).mean()

def plots(ctx):
    return {"wrapped_phase_error": torch.angle(ctx["field"][0] * ctx["target_field"].conj())}
''',
"Detector efficiency + leakage": '''def loss(ctx):
    # Maximize absolute collected energy and penalize power elsewhere.
    mask = ctx["roi"](0.0, 0.0, 2.0 * ctx["wavelength"])
    collected = (ctx["intensity"] * mask).sum((-2,-1))
    outside = ctx["total_power"] - collected
    return (-collected / ctx["input_power"] + 0.2 * outside / ctx["input_power"]).mean()

def plots(ctx):
    return {"central_detector": ctx["roi"](0., 0., 2.*ctx["wavelength"])}
''',
"Two observation planes": '''def loss(ctx):
    # Additional propagation starts immediately AFTER the last metasurface.
    near = ctx["at_z"](5.0 * ctx["wavelength"])
    far = ctx["at_z"](12.0 * ctx["wavelength"])
    mask = ctx["roi"](0., 0., 2.0 * ctx["wavelength"])
    def concentration(field):
        intensity = field.abs()**2
        return (intensity*mask).sum((-2,-1)) / (intensity.sum((-2,-1)) + 1e-16)
    return (2 - concentration(near) - concentration(far)).mean()

def plots(ctx):
    return {"near_intensity": ctx["at_z"](5*ctx["wavelength"])[0].abs()**2,
            "far_intensity": ctx["at_z"](12*ctx["wavelength"])[0].abs()**2}
''',
"Smooth fabricated phase": '''def loss(ctx):
    prediction = ctx["intensity"] / ctx["total_power"][:,None,None]
    fit = ((prediction-ctx["target_intensity"])**2).mean()*prediction[0].numel()**2
    # Circular phase smoothness respects wrapping and remains differentiable.
    smooth = sum((1-torch.cos(p[:,1:]-p[:,:-1])).mean() +
                 (1-torch.cos(p[1:,:]-p[:-1,:])).mean() for p in ctx["surface_phases"])
    return fit + 0.01*smooth
''',
"Beam steering / centroid": '''def loss(ctx):
    normalized = ctx["intensity"] / ctx["total_power"][:,None,None]
    cx = (normalized*ctx["X"]).sum((-2,-1)) / ctx["wavelength"]
    cy = (normalized*ctx["Y"]).sum((-2,-1)) / ctx["wavelength"]
    return ((cx-3.)**2 + cy**2).mean()

def metrics(ctx):
    return {"centroid_x_wavelengths": (ctx["intensity"]*ctx["X"]).sum() / ctx["total_power"].sum() / ctx["wavelength"]}
'''
}
EXAMPLES["Digit recognition / ten detectors"] = DIGIT_OBJECTIVE
SOURCE_CREDIT = "# Made by Ali Nadarloo.\n# SPDX-License-Identifier: MIT\n"
EXAMPLES = {name: SOURCE_CREDIT+code for name,code in EXAMPLES.items()}
DIGIT_OBJECTIVE = EXAMPLES["Digit recognition / ten detectors"]


def recipe_name(source):
    """Recognize recipes saved before the attribution header was introduced."""
    return next((name for name,code in EXAMPLES.items()
                 if source.strip() in (code.strip(),code.removeprefix(SOURCE_CREDIT).strip())),None)

PARAMETERS = '''loss(ctx) must return a finite, differentiable, real scalar torch.Tensor.
metrics(ctx) optionally returns {name: scalar}; plots(ctx) returns {name: 1D or 2D array}.
All field tensors live on the selected CPU/GPU. Shape is [batch, H, W].
Dense / FFT RS: H,W are cell counts. ASM: field grids have samples_per_cell
samples along each cell axis. Trainable surface_phases remain [Ny_cells,Nx_cells].

field                  complex output electric field [B, Hout, Wout]
intensity              |field|² (relative units), not normalized [B, Hout, Wout]
amplitude, phase       |field| and angle(field), phase in radians
input_field            complex incident field at the aperture [B, Ny, Nx]
input_power            sum |input_field|² per sample [B]
total_power            sum output intensity + 1e-16 per sample [B]
pixel_area             dx*dy in m²; multiply summed intensity for integrated units
X, Y                   output pixel-center coordinates in metres [Hout, Wout]
input_X, input_Y       input pixel-center coordinates in metres [Ny, Nx]
target_intensity       normalized target intensity, sum=1 [Hout, Wout]
target_field           unit-norm complex target field [Hout, Wout]
target_charge          desired OAM topological charge (integer)
labels                 gate truth values 0 or 1 [B]
label_values           same constant labels as a Python list; avoids GPU synchronization in Python branches
sample_names           gate/input labels or 'input' (list of strings)
sample_weights         original class-balancing weights [B]
detector_masks         [3, Hout, Wout]: detector 0, detector 1, outside
detector_fractions     [B,3]: energies divided by total sampled output energy
detector_one           1 means right detector represents logic 1
desired_fraction       paper target 0.975
outside_weight         paper leakage penalty 0.20
surface_phases         list of trainable real [Ny,Nx] phases, radians
layer_incident         list of complex [B,Ny,Nx] fields before each surface
layer_transmitted      list of complex [B,Ny,Nx] fields after each surface
wavelength, dx, dy     wavelength and FIELD sample pitches in metres
cell_dx, cell_dy       physical unit-cell pitches in metres (coarser than ASM dx/dy)
epoch                  current training epoch (0 for initial/final evaluation)
roi(x,y,radius)         circular output mask (arguments in metres)
at_z(distance_m)       differentiable field at distance from LAST surface [B,Hout,Wout]
oam_spectrum(field)    (ell [-8..8], powers [B,17]); angular power about target x/y

Digit detectors only:
digit_fractions        [B,10], integrated circle powers / sampled total output power
digit_detector_masks   [10,Hout,Wout], circle 0 starts at the top; IDs increase counterclockwise
digit_detector_centers [10,2], physical x/y in metres; labels contain integers 0..9
Mini-batch training uses training inputs only. Held-out evaluation never updates phases.
Final fields/metrics(ctx)/plots(ctx) contain one held-out representative per digit;
the built-in training_accuracy / heldout_accuracy cover the complete selected splits.

Python globals: torch, math, normal Python builtins. Local code has normal file
and process access: load only code you trust. Never use .item(), .detach(), NumPy,
or an integer argmax to construct a differentiable loss. Those are fine in metrics.

OAM spectrum: polar resampling inside a circle inscribed in the output window,
64 azimuth samples and 24 radial bins; radial-weighted angular Fourier power.
The displayed -8..8 spectrum is truncated and need not sum to 1. This is angular
purity, while the overlap objective also measures the chosen radial mode.
Scalar free-space quadrature is a design model, not a full-wave unit-cell solver.
Subwavelength near-field sampling and high spatial frequencies need convergence
checks. Increasing output scale can reveal energy outside the sampled window.
'''


class Objective:
    def __init__(self, source):
        namespace = {"torch": torch, "math": math}
        exec(compile(source, "<user_loss.py>", "exec"), namespace)
        if not callable(namespace.get("loss")):
            raise ValueError("Define a function named loss(ctx).")
        self.loss = namespace["loss"]
        self.metrics = namespace.get("metrics", lambda ctx: {})
        self.plots = namespace.get("plots", lambda ctx: {})

    def checked_loss(self, ctx):
        loss = self.loss(ctx)
        if not isinstance(loss, torch.Tensor) or loss.numel()!=1 or loss.is_complex():
            raise ValueError("loss(ctx) must return one real scalar torch.Tensor.")
        if not bool(torch.isfinite(loss)):
            raise ValueError("Loss is NaN or infinite. Check your objective and inputs.")
        if not loss.requires_grad:
            raise ValueError("Loss has no gradient. Keep it in PyTorch and do not detach the field.")
        return loss.reshape(())


def oam_spectrum(field, c, X, Y):
    h, w = field.shape[-2:]
    device = field.device
    center_x, center_y = c.target_x_wl*c.wavelength, c.target_y_wl*c.wavelength
    radius = min(w*c.field_dx/2-abs(center_x), h*c.field_dy/2-abs(center_y))*0.9
    if radius <= 0:
        raise ValueError("OAM center must be inside the output window.")
    r = (torch.arange(24,device=device)+0.5)/24*radius
    theta = torch.arange(64,device=device)*2*math.pi/64
    xp = center_x+r[:,None]*torch.cos(theta)
    yp = center_y+r[:,None]*torch.sin(theta)
    coords = torch.stack((2*xp/(c.field_dx*(w-1)), 2*yp/(c.field_dy*(h-1))), -1)
    batch = field.shape[0]
    channels = torch.stack((field.real,field.imag),1)
    samples = F.grid_sample(channels, coords[None].expand(batch,-1,-1,-1), align_corners=True)
    complex_samples = torch.complex(samples[:,0],samples[:,1])
    spectrum = torch.fft.fft(complex_samples,dim=-1)/64
    power = (spectrum.abs().square()*r[None,:,None]).sum(1)
    total = power.sum(-1,keepdim=True).clamp(min=1e-16)
    ell = torch.arange(-8,9,device=device)
    return ell, power[:,ell.remainder(64)]/total


class ContextBuilder:
    def __init__(self, config, model, inputs, names, labels, weights):
        self.c, self.model, self.inputs = config, model, inputs
        self.names, self.labels, self.weights = names, labels, weights
        self.label_values=labels.cpu().tolist()
        self.input_field=inputs
        if config.sampling>1:
            self.input_field=inputs.repeat_interleave(config.sampling,-2).repeat_interleave(config.sampling,-1)
        self.input_power=self.input_field.abs().square().sum((-2,-1))+1e-16
        self.extra = {}
        device = inputs.device
        self.X,self.Y = grid(*config.output_shape,config.field_dx,config.field_dy,device)
        self.iX,self.iY = grid(config.ny*config.sampling,config.nx*config.sampling,config.field_dx,config.field_dy,device)
        self.target_i,self.target_f = build_target(config,self.X,self.Y)
        self.digit_masks=self.digit_centers=None
        if config.target=="Digit detectors":
            from .digits import detector_layout
            self.digit_masks,self.digit_centers=detector_layout(config,self.X,self.Y)
            self.target_i=self.digit_masks.sum(0)/self.digit_masks.sum()
            self.target_f=self.target_i.sqrt().to(torch.complex64)
        r,off = config.detector_radius_wl*config.wavelength,config.detector_offset_wl*config.wavelength
        m0 = (self.X+off)**2+self.Y**2 <= r*r
        m1 = (self.X-off)**2+self.Y**2 <= r*r
        self.masks = torch.stack((m0,m1,~(m0|m1))).float()
        if config.target == "Logic detectors" and (not m0.any() or not m1.any()):
            raise ValueError("Detector radius is too small to cover a pixel.")

    def __call__(self, epoch=0, indices=None):
        c = self.c
        selected=self.inputs if indices is None else self.inputs[indices]
        labels=self.labels if indices is None else self.labels[indices]
        names=self.names if indices is None else [self.names[i] for i in indices]
        label_values=self.label_values if indices is None else [self.label_values[i] for i in indices]
        field, incident, transmitted = self.model(selected)
        intensity = field.abs().square()
        total = intensity.sum((-2,-1))+1e-16
        fractions = (intensity[:,None]*self.masks[None]).sum((-2,-1))/total[:,None]

        def at_z(distance_m):
            z = float(distance_m)
            if not math.isfinite(z) or z <= 0:
                raise ValueError("at_z distance must be positive and finite.")
            if z not in self.extra:
                self.extra[z] = Propagator(c,(c.ny*c.sampling,c.nx*c.sampling),c.output_shape,z,field.device)
            return torch.stack([self.extra[z](f) for f in transmitted[-1]])

        result=dict(field=field,intensity=intensity,amplitude=field.abs(),phase=torch.angle(field),
                    input_field=self.input_field if indices is None else self.input_field[indices],
                    input_power=self.input_power if indices is None else self.input_power[indices],
                    total_power=total,pixel_area=c.field_dx*c.field_dy,X=self.X,Y=self.Y,input_X=self.iX,input_Y=self.iY,
                    target_intensity=self.target_i,target_field=self.target_f,target_charge=c.target_charge,
                    labels=labels,label_values=label_values,sample_names=names,
                    sample_weights=self.weights if indices is None else self.weights[indices],
                    detector_masks=self.masks,detector_fractions=fractions,detector_one=c.detector_one,
                    desired_fraction=c.desired_fraction,outside_weight=c.outside_weight,
                    surface_phases=list(self.model.phases),layer_incident=incident,layer_transmitted=transmitted,
                    wavelength=c.wavelength,dx=c.field_dx,dy=c.field_dy,cell_dx=c.dx,cell_dy=c.dy,epoch=epoch,
                    roi=lambda x,y,radius: (((self.X-x)**2+(self.Y-y)**2)<=radius**2).float(),
                    at_z=at_z,oam_spectrum=lambda f: oam_spectrum(f,c,self.X,self.Y))
        if self.digit_masks is not None:
            # Contract over pixels without allocating [batch,10,height,width].
            energies=torch.einsum("bhw,khw->bk",intensity,self.digit_masks)
            result.update(digit_detector_masks=self.digit_masks,digit_detector_centers=self.digit_centers,
                          digit_fractions=energies/total[:,None])
        return result
