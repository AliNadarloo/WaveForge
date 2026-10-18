# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""All user settings and validation. Lengths are entered in micrometres or wavelengths."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import math


@dataclass
class Config:
    name: str = "Gaussian focus"
    polarization_count: int = 1
    channel2: dict = field(default_factory=dict)
    nx: int = 32
    ny: int = 32
    layers: int = 1
    dx_um: float = 150.0
    dy_um: float = 150.0
    frequency_thz: float = 1.0
    distances_wl: list = field(default_factory=lambda: [1.2, 10.8])
    output_scale: float = 1.0
    propagation: str = "Dense RS"
    samples_per_cell: int = 4
    padding_factor: int = 3
    cst_axis: str = "+z"
    device: str = "Auto (CUDA / CPU)"
    epochs: int = 300
    learning_rate: float = 0.008
    optimizer: str = "Adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    seed: int = 7
    phase_initialization: str = "Zeros (paper)"
    aperture: str = "Open"
    aperture_radius_wl: float = 5.0
    aperture_file: str = ""
    text: str = "ONN"
    beam: str = "Gaussian"
    waist_wl: float = 4.0
    charge: int = 1
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0
    gates: list = field(default_factory=lambda: ["OR", "AND", "XOR", "BUFFER"])
    target: str = "Focus"
    target_file: str = ""
    target_text: str = "ONN"
    target_charge: int = 1
    target_waist_wl: float = 2.0
    target_x_wl: float = 0.0
    target_y_wl: float = 0.0
    detector_radius_wl: float = 2.4
    detector_offset_wl: float = 4.68
    detector_one: int = 1
    desired_fraction: float = 0.975
    outside_weight: float = 0.20
    loss_code: str = ""
    initial_phases: str = ""
    digit_batch_size: int = 64
    digit_samples_per_class: int = 0
    digit_ring_radius_wl: float = 5.0
    digit_detector_radius_wl: float = 0.9

    @property
    def wavelength(self):
        return 299_792_458.0 / (self.frequency_thz * 1e12)

    @property
    def dx(self):
        return self.dx_um * 1e-6

    @property
    def dy(self):
        return self.dy_um * 1e-6

    @property
    def output_shape(self):
        s=self.sampling
        return (s*max(round(self.ny * self.output_scale), self.ny),
                s*max(round(self.nx * self.output_scale), self.nx))

    @property
    def sampling(self):
        return self.samples_per_cell if self.propagation.startswith("ASM") else 1

    @property
    def field_dx(self):return self.dx/self.sampling

    @property
    def field_dy(self):return self.dy/self.sampling

    def to_dict(self):
        return asdict(self)

    def resolve_paths(self,base):
        for key in ["aperture_file","target_file","initial_phases"]:
            value=getattr(self,key)
            if value and not Path(value).is_absolute():setattr(self,key,str((Path(base)/value).resolve()))
            value=self.channel2.get(key,"")
            if value and not Path(value).is_absolute():self.channel2[key]=str((Path(base)/value).resolve())
        return self

    def validate(self):
        if self.polarization_count not in (1,2):
            raise ValueError("Choose one or two scalar polarizations.")
        if self.cst_axis not in ("+z","-z","+x","-x","+y","-y"):
            raise ValueError("Choose a supported CST propagation axis.")
        if self.optimizer not in ("Adam","AdamW","SGD + momentum","RMSprop","Adagrad"):
            raise ValueError("Choose Adam, AdamW, SGD + momentum, RMSprop or Adagrad.")
        if self.phase_initialization not in ("Zeros (paper)","Small random","Random [-pi, pi]"):
            raise ValueError("Choose a supported phase initialization.")
        if not 0<=self.momentum<1 or self.weight_decay<0:
            raise ValueError("Momentum must be in [0,1); weight decay must be nonnegative.")
        if self.polarization_count == 2:
            for channel in self.channels():
                channel.validate()
        for name, value in self.to_dict().items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not (8 <= self.nx <= 256 and 8 <= self.ny <= 256 and 1 <= self.layers <= 8):
            raise ValueError("Use 8–256 cells on each axis and 1–8 surfaces.")
        if min(self.dx_um, self.dy_um, self.frequency_thz, self.waist_wl,
               self.target_waist_wl, self.aperture_radius_wl, self.learning_rate) <= 0:
            raise ValueError("Pitches, frequency, waists, aperture radius and learning rate must be positive.")
        if not (1 <= self.epochs <= 100000 and 1 <= self.output_scale <= 3):
            raise ValueError("Epochs must be 1–100000; output scale must be 1–3.")
        if len(self.distances_wl) != self.layers + 1 or any(not math.isfinite(z) or z <= 0 for z in self.distances_wl):
            raise ValueError(f"Enter {self.layers + 1} positive spacings: input → surfaces → output (in λ).")
        if self.propagation not in ("Dense RS", "FFT RS", "ASM • evanescent decay", "ASM • propagating only"):
            raise ValueError("Choose Dense RS, FFT RS, or a padded ASM variant.")
        if not (isinstance(self.samples_per_cell,int) and 1<=self.samples_per_cell<=8 and
                isinstance(self.padding_factor,int) and 1<=self.padding_factor<=8):
            raise ValueError("ASM uses 1–8 samples per cell and 1–8 padding factor.")
        if self.propagation.startswith("ASM"):
            source=(self.ny*self.sampling,self.nx*self.sampling)
            if any((a-b)%2 for a,b in zip(source,self.output_shape)):
                raise ValueError("ASM input/output grids need the same parity. Use an even samples-per-cell value or adjust output scale.")
            memory=8*(self.layers+1)*self.padding_factor**2*math.prod(self.output_shape)
            if memory>1024**3:
                raise ValueError("ASM transfer functions exceed 1 GiB. Reduce sampling, padding, layers or output window.")
        if self.propagation == "FFT RS" and self.output_scale != 1:
            raise ValueError("FFT RS uses equal plane grids; set output scale to 1 or choose Dense RS.")
        n = self.nx * self.ny
        dense_bytes = 8 * ((self.layers) * n*n + n * math.prod(self.output_shape))
        if self.propagation == "Dense RS" and dense_bytes > 512 * 1024**2:
            raise ValueError("Dense matrices exceed 512 MiB. Choose FFT RS with output scale 1, or reduce the grid.")
        if self.detector_one not in (0, 1) or not (0 < self.desired_fraction <= 1) or self.outside_weight < 0:
            raise ValueError("Invalid detector assignment or loss weights.")
        if self.aperture == "Logic gates":
            valid = {"OR", "AND", "XOR", "NOR", "NAND", "XNOR", "NOT", "BUFFER"}
            if not self.gates or any(g not in valid for g in self.gates):
                raise ValueError("Select one or more supported gates.")
            complementary = [{"OR", "NOR"}, {"AND", "NAND"}, {"XOR", "XNOR"}, {"BUFFER", "NOT"}]
            if any(pair <= set(self.gates) for pair in complementary):
                raise ValueError("Complementary gates share the same aperture. Train the x and y channels separately.")
        if self.target == "Logic detectors":
            if self.aperture != "Logic gates":
                raise ValueError("Logic detector targets require Logic gates inputs.")
            r, off = self.detector_radius_wl*self.wavelength, self.detector_offset_wl*self.wavelength
            if r <= 0 or off <= r or off+r > self.output_shape[1]*self.field_dx/2 or r > self.output_shape[0]*self.field_dy/2:
                raise ValueError("Two nonoverlapping detectors must fit within the output plane. Adjust radius / offset / grid.")
        if self.aperture=="Handwritten digits" or self.target=="Digit detectors":
            if self.aperture!="Handwritten digits" or self.target!="Digit detectors":
                raise ValueError("Handwritten digits use the ten-circle Digit detectors target.")
            if not 1<=self.digit_batch_size<=256 or not 0<=self.digit_samples_per_class<=140:
                raise ValueError("Use batch size 1–256 and 0–140 training samples per digit (0 means all).")
            radius,ring=self.digit_detector_radius_wl,self.digit_ring_radius_wl
            half=min(self.output_shape[1]*self.field_dx,self.output_shape[0]*self.field_dy)/(2*self.wavelength)
            if radius<=0 or ring<=0 or radius>=ring*math.sin(math.pi/10) or ring+radius>half:
                raise ValueError("Ten nonoverlapping digit circles must fit within the output window. Reduce detector / ring radius or enlarge the window.")
        if abs(self.tilt_x_deg) >= 80 or abs(self.tilt_y_deg) >= 80:
            raise ValueError("Beam tilts must lie between −80 and +80 degrees.")

    def channels(self):
        """Independent scalar channels share geometry and optimizer settings."""
        data=self.to_dict();data.update(polarization_count=1,channel2={})
        first=Config(**data)
        if self.polarization_count==1:
            return [first]
        second=data.copy()
        second.update({k:v for k,v in self.channel2.items() if k in CHANNEL_FIELDS})
        second["name"]=self.name+" / Polarization 2"
        first.name=self.name+" / Polarization 1"
        return [first,Config(**second)]


SHARED_FIELDS={"name","polarization_count","channel2","nx","ny","layers","dx_um","dy_um",
               "frequency_thz","distances_wl","output_scale","propagation","device","epochs",
               "learning_rate","seed"}
SHARED_FIELDS.update({"optimizer","momentum","weight_decay"})
SHARED_FIELDS.update({"samples_per_cell","padding_factor"})
SHARED_FIELDS.add("cst_axis")
SHARED_FIELDS.add("phase_initialization")
SHARED_FIELDS.add("digit_batch_size")
CHANNEL_FIELDS=set(Config.__dataclass_fields__)-SHARED_FIELDS
