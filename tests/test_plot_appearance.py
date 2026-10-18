# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Saved figures retain the live palette, display rules and original samples."""
from pathlib import Path
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from onnlab.appearance import DEFAULTS,THEMES
from onnlab.export import save_plot


@pytest.mark.parametrize("theme",list(THEMES))
def test_saved_fields_and_surface_cells_match_live_rules(monkeypatch,tmp_path,theme):
    snapshots=[]
    def capture(fig,path,**kwargs):
        ax=fig.axes[0];im=ax.images[0]
        snapshots.append(dict(background=fig.get_facecolor(),axes=ax.get_facecolor(),
            title=ax.title.get_color(),cmap=im.get_cmap().name,interpolation=im.get_interpolation(),
            stage=im._interpolation_stage,data=np.asarray(im.get_array()).copy(),limits=im.get_clim()))
    monkeypatch.setattr(Figure,"savefig",capture)
    prefs=dict(DEFAULTS,theme=theme,intensity_cmap="hot",phase_cmap="jet")
    data=np.arange(64,dtype=float).reshape(8,8)/20
    for phase,meta in [(False,False),(True,False),(True,True)]:
        save_plot(tmp_path/"field.png",data,"Field",phase=phase,metasurface=meta,
                  cmap=prefs["phase_cmap" if phase else "intensity_cmap"],plot_settings=prefs)
        im=snapshots[-1]
        assert im["background"]==to_rgba(THEMES[theme]["plot"])
        assert im["axes"]==to_rgba(THEMES[theme]["plot"])
        assert im["title"]==THEMES[theme]["muted"]
        assert im["cmap"]==("jet" if phase else "hot")
        assert im["interpolation"]==("nearest" if meta else "bilinear")
        assert im["stage"]==("rgba" if phase and not meta else "data")
        np.testing.assert_array_equal(im["data"],data)
        if phase:assert im["limits"]==(-np.pi,np.pi)
