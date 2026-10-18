# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Presentation preferences; independent of the optical model and optimizer."""
import json
from pathlib import Path

THEMES = {
    "Deep space": dict(background="#090f1b",panel="#111c2d",plot="#0c1524",rail="#070c15",
                       text="#dce6f3",muted="#8294ae",accent="#6af6e0",border="#293a53"),
    "Nebula violet": dict(background="#120f20",panel="#211a36",plot="#191428",rail="#0d0b17",
                          text="#eee8ff",muted="#a89bbf",accent="#c6a3ff",border="#44365e"),
    "Solar amber": dict(background="#17120d",panel="#292016",plot="#20180f",rail="#100d09",
                        text="#fff1dc",muted="#baa78b",accent="#ffc76a",border="#51412b"),
    "Daylight": dict(background="#f3f6fa",panel="#ffffff",plot="#ffffff",rail="#e7edf4",
                     text="#17273b",muted="#53677e",accent="#007b78",border="#c2cfdd"),
}
COLORMAPS=["inferno","magma","viridis","plasma","cividis","hot","jet","winter",
           "cool","gray","turbo","twilight","hsv","RdBu_r","Spectral","coolwarm"]
DEFAULTS=dict(theme="Deep space",input_cmap="magma",intensity_cmap="inferno",phase_cmap="twilight")


def style_figure(figure,prefs):
    """Use the live workspace palette for saved axes, legends and color bars."""
    colors=THEMES[prefs.get("theme",DEFAULTS["theme"])]
    figure.set_facecolor(colors["plot"])
    for ax in figure.axes:
        ax.set_facecolor(colors["plot"])
        ax.tick_params(colors=colors["muted"])
        ax.title.set_color(colors["muted"])
        ax.xaxis.label.set_color(colors["muted"])
        ax.yaxis.label.set_color(colors["muted"])
        ax.xaxis.get_offset_text().set_color(colors["muted"])
        ax.yaxis.get_offset_text().set_color(colors["muted"])
        for spine in ax.spines.values():spine.set_color(colors["border"])
        for line in ax.get_xgridlines()+ax.get_ygridlines():line.set_color(colors["border"])
        legend=ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor(colors["panel"])
            legend.get_frame().set_edgecolor(colors["border"])
            for text in legend.get_texts():text.set_color(colors["text"])
    return colors


def load_preferences(path):
    prefs=DEFAULTS.copy()
    try:
        saved=json.loads(Path(path).read_text(encoding="utf-8"))
        if saved.get("theme") in THEMES:prefs["theme"]=saved["theme"]
        for key in ("input_cmap","intensity_cmap","phase_cmap"):
            if saved.get(key) in COLORMAPS:prefs[key]=saved[key]
    except (OSError,ValueError,TypeError):pass
    return prefs


def recolor(text,theme,previous="Deep space"):
    """Replace palette tokens simultaneously, avoiding replacement collisions."""
    import re
    old,new=THEMES[previous],THEMES[theme]
    colors={old[k].lower():new[k] for k in old}
    # Related shades in the original stylesheet share the same semantic role.
    if previous=="Deep space":
        for color in ["#142238","#101c2d","#101b2c","#14293a","#101827","#142238"]:colors[color]=new["panel"]
        for color in ["#20314a","#2b415f","#243650","#233650","#27394e","#253950"]:colors[color]=new["border"]
        for color in ["#5ad5c7","#78f6df","#69e8d3","#66f2d5","#67ead5","#69f3d4","#a6e9e0"]:colors[color]=new["accent"]
        for color in ["#8196af","#8199b9","#526780"]:colors[color]=new["muted"]
    return re.sub(r"#[0-9a-fA-F]{6}",lambda m:colors.get(m[0].lower(),m[0]),text)
