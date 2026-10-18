# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Input apertures, illumination and target fields. Arrays always use [y, x]."""
from pathlib import Path
import math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from .physics import grid

GATES = {"OR": lambda a: bool(a[0] or a[1]), "AND": lambda a: bool(a[0] and a[1]),
         "XOR": lambda a: bool(a[0] ^ a[1]), "NOR": lambda a: not(a[0] or a[1]),
         "NAND": lambda a: not(a[0] and a[1]), "XNOR": lambda a: not(a[0] ^ a[1]),
         "NOT": lambda a: not a[0], "BUFFER": lambda a: bool(a[0])}


def gate_aperture(gate, bits, ny, nx):
    """Paper's four rows and columns, scaled by cell boundaries for arbitrary grids.

    At 32×32 these reproduce util_xypol_4030.generate_aperture exactly.
    Complementary gates use the same aperture in a separate polarization run.
    """
    out = np.zeros((ny, nx), dtype=np.float32)
    row = {"XOR": 0, "XNOR": 0, "AND": 1, "NAND": 1, "OR": 2, "NOR": 2, "NOT": 3, "BUFFER": 3}[gate]
    single = gate in ("NOT", "BUFFER")
    column = bits[0] if single else 2*bits[0]+bits[1]
    columns = 2 if single else 4
    out[round(row*ny/4):round((row+1)*ny/4), round(column*nx/columns):round((column+1)*nx/columns)] = 1
    return out


def text_image(text, shape):
    if not text.strip():
        raise ValueError("Enter nonempty text.")
    ny, nx = shape
    font = None
    for path in ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"]:
        if Path(path).exists():
            font = ImageFont.truetype(path, 120)
            break
    font = font or ImageFont.load_default()
    box = font.getbbox(text)
    canvas = Image.new("L", (max(1, box[2]-box[0])+16, max(1, box[3]-box[1])+16))
    ImageDraw.Draw(canvas).text((8-box[0], 8-box[1]), text, fill=255, font=font)
    # Preserve aspect ratio and leave a border for diffraction.
    canvas.thumbnail((max(nx-4, 1), max(ny-4, 1)), Image.Resampling.LANCZOS)
    output = Image.new("L", (nx, ny))
    output.paste(canvas, ((nx-canvas.width)//2, (ny-canvas.height)//2))
    return np.flipud(np.asarray(output, dtype=np.float32)/255).copy()


def load_array(path, shape, complex_allowed=True):
    """NPY/CSV/TXT numeric arrays, or Pillow raster formats. No pickle loading.

    Raster brightness is amplitude for apertures and intensity for targets.
    Raster images are flipped vertically so their visual orientation is upright
    in the Cartesian origin='lower' plots. Numeric files retain their row order.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError("Choose an existing image or numeric array file.")
    if p.suffix.lower() == ".npy":
        a = np.load(p, allow_pickle=False)
    elif p.suffix.lower() in (".csv", ".txt"):
        a = np.loadtxt(p, delimiter="," if p.suffix.lower()==".csv" else None, dtype=complex)
        if np.all(a.imag == 0):
            a = a.real
    else:
        with Image.open(p) as im:
            rgba = im.convert("RGBA")
            im = rgba.convert("L")
            a = np.asarray(im.resize((shape[1], shape[0]), Image.Resampling.LANCZOS), dtype=np.float32)/255
            alpha = np.asarray(rgba.getchannel("A").resize((shape[1], shape[0]), Image.Resampling.LANCZOS), dtype=np.float32)/255
            a = np.flipud(a*alpha).copy()
    if a.ndim != 2 or a.shape != shape:
        raise ValueError(f"Numeric arrays must have shape {shape} [y, x]; got {a.shape}.")
    if not np.isfinite(a).all() or np.max(np.abs(a)) == 0:
        raise ValueError("Input must be finite and nonzero.")
    if np.iscomplexobj(a):
        if not complex_allowed:
            raise ValueError("Intensity targets must be real and nonnegative.")
    elif np.min(a) < 0:
        raise ValueError("Real amplitudes / intensities must be nonnegative.")
    return a


def beam_field(c, X, Y):
    r = torch.sqrt(X*X+Y*Y)
    w = c.waist_wl*c.wavelength
    g = torch.exp(-(r/w)**2)
    if c.beam == "Gaussian":
        b = g
    elif c.beam == "OAM (LG p=0)":
        b = (math.sqrt(2)*r/w)**abs(c.charge)*g*torch.exp(1j*c.charge*torch.atan2(Y, X))
        b = b / b.abs().max().clamp(min=1e-16)
    elif c.beam == "Bessel–Gauss":
        b = torch.special.bessel_j0(4*r/w)*g
    else:
        b = torch.ones_like(X)
    k = 2*math.pi/c.wavelength
    # Negative transverse phase is consistent with exp(+iωt).
    tilt = -k*(X*math.sin(math.radians(c.tilt_x_deg))+Y*math.sin(math.radians(c.tilt_y_deg)))
    return b.to(torch.complex64)*torch.exp(1j*tilt)


def build_inputs(c, device):
    X, Y = grid(c.ny, c.nx, c.dx, c.dy, device)
    beam = beam_field(c, X, Y)
    if c.aperture=="Handwritten digits":
        from .digits import load_digits
        apertures,labels=load_digits(c,device)
        fields=apertures.to(torch.complex64)*beam
        powers=fields.abs().square().sum((-2,-1))
        if bool((powers<1e-20).any()):raise ValueError("Digit illumination is dark; increase the beam waist.")
        fields=fields/torch.sqrt(powers[:,None,None])
        names=[f"digit_{int(v)}_sample_{i:04d}" for i,v in enumerate(labels.cpu().tolist())]
        return fields,names,labels,torch.ones(len(labels),device=device)
    cases = []
    if c.aperture == "Logic gates":
        for gate in c.gates:
            patterns = [[0], [1]] if gate in ("NOT", "BUFFER") else [[0,0], [0,1], [1,0], [1,1]]
            for bits in patterns:
                cases.append((f"{gate}_{''.join(map(str,bits))}", gate_aperture(gate,bits,c.ny,c.nx), int(GATES[gate](bits))))
    else:
        if c.aperture == "Circle":
            a = ((X*X+Y*Y) <= (c.aperture_radius_wl*c.wavelength)**2).float().cpu().numpy()
        elif c.aperture == "Double slit":
            radius = c.aperture_radius_wl*c.wavelength
            a = (((X.abs()-radius/2).abs()<radius/6) & (Y.abs()<radius)).float().cpu().numpy()
        elif c.aperture == "Text / number":
            a = text_image(c.text, (c.ny,c.nx))
        elif c.aperture == "Image / array":
            a = load_array(c.aperture_file, (c.ny,c.nx))
        else:
            a = np.ones((c.ny,c.nx), dtype=np.float32)
        cases.append(("input", a, 0))
    fields = torch.stack([torch.as_tensor(a, device=device, dtype=torch.complex64)*beam for _,a,_ in cases])
    if (fields.abs().square().sum((-2,-1)) < 1e-20).any():
        raise ValueError("One or more input fields are dark; increase the aperture or beam waist.")
    labels = torch.tensor([v for _,_,v in cases], device=device)
    n_on = int(labels.sum()); n_off = len(cases)-n_on
    weights = torch.tensor([len(cases)/(2*max(1,n_on if v else n_off)) for _,_,v in cases], device=device)
    return fields, [name for name,_,_ in cases], labels, weights


def build_target(c, X, Y):
    x, y = X-c.target_x_wl*c.wavelength, Y-c.target_y_wl*c.wavelength
    r = torch.sqrt(x*x+y*y)
    w = c.target_waist_wl*c.wavelength
    if c.target == "Image intensity":
        intensity = torch.as_tensor(load_array(c.target_file, tuple(X.shape), False), device=X.device, dtype=torch.float32)
        field = torch.sqrt(intensity).to(torch.complex64)
    elif c.target == "Text intensity":
        intensity = torch.as_tensor(text_image(c.target_text, tuple(X.shape)), device=X.device)
        field = torch.sqrt(intensity).to(torch.complex64)
    elif c.target == "OAM mode":
        field = (math.sqrt(2)*r/w)**abs(c.target_charge)*torch.exp(-(r/w)**2)*torch.exp(1j*c.target_charge*torch.atan2(y,x))
        intensity = field.abs().square()
    else:
        intensity = torch.exp(-2*(r/w)**2)
        field = torch.sqrt(intensity).to(torch.complex64)
    intensity = intensity / intensity.sum().clamp(min=1e-16)
    field = field / torch.sqrt(field.abs().square().sum()).clamp(min=1e-16)
    return intensity, field
