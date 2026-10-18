# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Runnable examples. Paper presets lock in the authoritative public trainer settings."""
from .config import Config
from .objectives import EXAMPLES
from pathlib import Path
import sys
from .paths import application_directory

PRESET_NAMES = ["Gaussian focus", "Paper • Polarization 1", "Paper • Polarization 2", "Paper • two polarizations",
                "OAM conversion", "Text hologram", "Image hologram", "Double-slit steering", "Two-plane focus", "ASM • subcell OAM",
                "Advanced • four-layer digit router"]


def preset(name):
    c = Config(name=name, loss_code=EXAMPLES["Focus / image intensity"])
    if name=="Advanced • four-layer digit router":
        c.layers=4;c.distances_wl=[1.,3.,3.,3.,8.];c.propagation="FFT RS"
        c.aperture="Handwritten digits";c.target="Digit detectors";c.beam="Plane wave"
        c.epochs=100;c.learning_rate=.02;c.phase_initialization="Small random"
        c.loss_code=EXAMPLES["Digit recognition / ten detectors"]
    elif name.startswith("Paper"):
        c.epochs=3000; c.beam="Plane wave"; c.aperture="Logic gates"; c.target="Logic detectors"
        c.gates=["NOR","NAND","XNOR","NOT"] if "Polarization 2" in name or "y polarization" in name else ["OR","AND","XOR","BUFFER"]
        c.loss_code=EXAMPLES["Paper logic gates"]
        if "two polarizations" in name:
            c.polarization_count=2
            second=preset("Paper • Polarization 2")
            from .config import CHANNEL_FIELDS
            c.channel2={k:v for k,v in second.to_dict().items() if k in CHANNEL_FIELDS}
    elif name in ("OAM conversion","ASM • subcell OAM"):
        c.target="OAM mode"; c.layers=2; c.distances_wl=[1.2,4.0,10.8]; c.epochs=500
        c.loss_code=EXAMPLES["OAM / complex mode overlap"]
        c.phase_initialization="Small random"
        if name.startswith("ASM"):
            c.propagation="ASM • evanescent decay";c.samples_per_cell=4;c.padding_factor=3
    elif name in ("Text hologram", "Image hologram"):
        c.target="Text intensity" if name=="Text hologram" else "Image intensity"
        c.layers=2; c.distances_wl=[1.2,4.0,10.8]; c.epochs=500
        if name=="Image hologram":
            root=application_directory()
            c.target_file=str(root/"examples/target.png")
    elif name == "Double-slit steering":
        c.aperture="Double slit"; c.beam="Plane wave"; c.target_x_wl=3
        c.loss_code=EXAMPLES["Beam steering / centroid"]
    elif name == "Two-plane focus":
        c.layers=2; c.distances_wl=[1.2,4.,12.]
        c.loss_code=EXAMPLES["Two observation planes"]
    return c
