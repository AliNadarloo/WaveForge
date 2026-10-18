# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Portable, inspectable results: no opaque pickles or dependency on the parent project."""
from datetime import datetime
from pathlib import Path
import csv
import json
import re
import sys
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .conventions import cst_coordinates,export_explanation
from .appearance import DEFAULTS,THEMES,style_figure
from .paths import application_directory


def array(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(name))[:80] or "plot"


def runtime_directory():
    # A double-clicked EXE writes next to itself; source writes next to main.py.
    return application_directory()


def save_plot(path, data, title, extent=None, phase=False, line_x=None,cmap=None,circles=None,
              plot_settings=None,metasurface=False):
    prefs=DEFAULTS.copy();prefs.update(plot_settings or {})
    colors=THEMES[prefs["theme"]]
    fig=Figure(figsize=(6.4,4.8),layout="constrained")
    FigureCanvasAgg(fig)
    ax=fig.add_subplot()
    if data.ndim==1:
        ax.plot(np.arange(len(data)) if line_x is None else line_x,data,color=colors["accent"],linewidth=1.8)
        ax.set_xlabel("Epoch (pre-update loss)" if "loss" in path.stem else "Index / angular order")
        ax.grid(alpha=.2)
    elif data.ndim==2:
        image=ax.imshow(data,origin="lower",extent=extent,cmap=cmap or ("twilight" if phase else "inferno"),
                        vmin=-np.pi if phase else None,vmax=np.pi if phase else None,aspect="equal",
                        interpolation="nearest" if metasurface else "bilinear",
                        interpolation_stage="rgba" if phase and not metasurface else "data")
        fig.colorbar(image,ax=ax,label="rad" if phase else "Relative units")
        ax.set_xlabel("x (mm)" if extent else "x cell")
        ax.set_ylabel("y (mm)" if extent else "y cell")
        if circles:
            from matplotlib.patches import Circle
            for x,y,r,digit in circles:
                ax.add_patch(Circle((x,y),r,fill=False,edgecolor=colors["accent"],linewidth=.8))
                ax.text(x,y,str(digit),color=colors["accent"],fontsize=8,ha="center",va="center")
    else:
        raise ValueError(f"Plot {title} must be a 1D or 2D array.")
    ax.set_title(title)
    style_figure(fig,prefs)
    fig.savefig(path,dpi=160)
    fig.clear()


def export_result(result, base=None, directory=None,plot_settings=None):
    c,ctx=result.config,result.context
    base=Path(base) if base else runtime_directory()/"results"
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out=Path(directory) if directory else base/f"{stamp}_{safe_name(c.name)}"
    out.mkdir(parents=True,exist_ok=False)
    prefs=DEFAULTS.copy();prefs.update(plot_settings or {})
    (out/"plot_settings.json").write_text(json.dumps(prefs,indent=2),encoding="utf-8")
    def plot(path,data,title,extent=None,phase=False,line_x=None,kind="intensity",circles=None,metasurface=False):
        save_plot(path,data,title,extent,phase,line_x,cmap=prefs["phase_cmap" if phase else kind+"_cmap"],
                  circles=circles,plot_settings=prefs,metasurface=metasurface)
    (out/"config.json").write_text(json.dumps(c.to_dict(),indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"CST_EXPORT_README.txt").write_text(export_explanation(c),encoding="utf-8")
    if hasattr(result,"channels"):
        for i,channel in enumerate(result.channels,1):
            export_result(channel,directory=out/f"Polarization_{i}",plot_settings=prefs)
        summary=dict(application="WaveForge 1.1.2",polarization_count=2,metrics=result.metrics,
                     device=result.device,seconds=result.seconds,cancelled=result.cancelled,
                     channels_completed=len(result.channels),plot_errors=[],
                     note="Independent scalar channels; polarization basis is determined by physical unit-cell design.")
        (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        return out
    source=c.loss_code if "Made by Ali Nadarloo" in c.loss_code else "# Made by Ali Nadarloo.\n"+c.loss_code
    (out/"loss.py").write_text(source,encoding="utf-8")
    phases=np.stack([array(p) for p in result.model.phases])
    wrapped=np.angle(np.exp(1j*phases))
    np.savez_compressed(out/"phases.npz",phases_rad=phases,wrapped_rad=wrapped,
                        dx_m=c.dx,dy_m=c.dy,wavelength_m=c.wavelength)
    for i,p in enumerate(phases,1):
        # Original trainer exports modulo 2π in radians, rows=y and columns=x.
        phase=np.mod(p,2*np.pi)
        iy=np.arange(c.ny)-c.ny//2; ix=np.arange(c.nx)-c.nx//2
        rows=np.array([[iy[y],ix[x],phase[y,x],1.0] for y in range(c.ny) for x in range(c.nx)])
        np.savetxt(out/f"best_meta_layer_{i}.txt",rows,fmt=["%d","%d","%.9f","%.6f"],header="i j phase(rad) amp",comments="")
        np.savetxt(out/f"phase_layer_{i}_radians.csv",phase,delimiter=",")
        np.savetxt(out/f"phase_layer_{i}_degrees.csv",np.mod(np.rad2deg(p),360),delimiter=",")
        xyz=cst_coordinates(c,i-1).reshape(-1,3)
        signed=wrapped[i-1].reshape(-1)
        cst=np.column_stack((xyz,signed,np.rad2deg(signed),np.ones(signed.shape)))
        np.savetxt(out/f"CST_phase_layer_{i}.txt",cst,fmt="%.10g",
                   header="x_um y_um z_um phase_rad phase_deg amplitude",comments="")
        plot(out/f"phase_layer_{i}.png",wrapped[i-1],f"Surface {i} phase",phase=True,metasurface=True)
    np.savetxt(out/"training_loss.csv",np.column_stack((np.arange(1,len(result.history)+1),result.history)),delimiter=",",header="epoch,pre_update_loss",comments="")
    if result.history:
        plot(out/"training_loss.png",np.asarray(result.history),"Optimization loss")
        normalized=np.asarray(result.history)/max(abs(result.history[0]),1e-16)
        plot(out/"normalized_loss.png",normalized,"Loss / absolute initial loss")
    intensity,field,inputs=array(ctx["intensity"]),array(ctx["field"]),array(ctx["input_field"])
    np.savez_compressed(out/"fields.npz",input_field=inputs,output_field=field,
                        intensity=intensity,target_field=array(ctx["target_field"]),
                        target_intensity=array(ctx["target_intensity"]),
                        layer_incident=np.stack([array(f) for f in ctx["layer_incident"]]),
                        layer_transmitted=np.stack([array(f) for f in ctx["layer_transmitted"]]),
                        x_m=array(ctx["X"]),y_m=array(ctx["Y"]))
    h,w=c.output_shape
    extent=[-w*c.field_dx*500,w*c.field_dx*500,-h*c.field_dy*500,h*c.field_dy*500]
    ih,iw=c.ny,c.nx
    iextent=[-iw*c.dx*500,iw*c.dx*500,-ih*c.dy*500,ih*c.dy*500]
    fractions=array(ctx["detector_fractions"])
    report=[]
    circles=[]
    if "digit_detector_centers" in ctx:
        circles=[(x*1000,y*1000,c.digit_detector_radius_wl*c.wavelength*1000,i)
                 for i,(x,y) in enumerate(array(ctx["digit_detector_centers"]))]
    for i,name in enumerate(ctx["sample_names"]):
        slug=safe_name(name)
        plot(out/f"{slug}_input_amplitude.png",np.abs(inputs[i]),f"{name} • input amplitude",iextent,kind="input")
        plot(out/f"{slug}_output_intensity.png",intensity[i],f"{name} • output intensity",extent,circles=circles)
        plot(out/f"{slug}_output_phase.png",np.angle(field[i]),f"{name} • output phase",extent,phase=True,circles=circles)
        if c.target=="Logic detectors":
            expected=int(ctx["labels"][i]); hot=c.detector_one if expected else 1-c.detector_one
            contrast=10*np.log10((fractions[i,hot]+1e-16)/(fractions[i,1-hot]+1e-16))
            report.append(dict(sample=name,expected=expected,detector0=float(fractions[i,0]),
                               detector1=float(fractions[i,1]),outside=float(fractions[i,2]),
                               contrast_db=float(contrast),correct=bool(fractions[i,hot]>fractions[i,1-hot])))
    if report:
        with (out/"truth_table.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(report[0]));writer.writeheader();writer.writerows(report)
    plot(out/"target_intensity.png",array(ctx["target_intensity"]),"Target intensity (normalized)",extent,circles=circles)
    if result.classification is not None:
        from .digit_export import export_classification
        export_classification(result,out,prefs)
    custom={};plot_errors=[]
    for index,(name,value) in enumerate(result.plots.items()):
        try:
            a=array(value)
            if a.ndim not in (1,2) or not np.isfinite(a).all() or np.iscomplexobj(a):
                raise ValueError("Use finite real 1D / 2D arrays (e.g. abs(field)**2 or angle(field)).")
            key=f"{index+1}_{safe_name(name)}"
            custom[key]=a
            is_oam=name=="oam_spectrum" and a.ndim==1 and len(a)==17
            plot(out/f"custom_{key}.png",a,str(name),
                      extent if a.shape==c.output_shape else None,
                      phase="phase" in str(name).lower(),line_x=np.arange(-8,9) if is_oam else None)
        except Exception as exc:
            plot_errors.append(f"{name}: {exc}")
    if custom:
        np.savez_compressed(out/"custom_plots.npz",**custom)
    summary=dict(application="WaveForge 1.1.2",device=result.device,torch=torch.__version__,
                 seconds=result.seconds,epochs_completed=len(result.history),cancelled=result.cancelled,
                 metrics=result.metrics,truth_table=report,plot_errors=plot_errors,
                 note="Scalar phase-only design; exp(+iωt), outgoing exp(-ikR). No full-wave or polarization coupling simulation.")
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (out/"README.txt").write_text("Phases: best_meta_layer_N.txt = modulo 2pi radians (paper convention).\n"
        "_radians.csv phases are radians; _degrees.csv phases are degrees.\n"
        "phases.npz also retains the unwrapped optimization parameters.\n"
        "All arrays: rows=y, columns=x, and field stacks begin with sample or layer.\n"
        "Plots use origin=lower. fields.npz contains all sample fields and layer fields.\n"
        "training_loss.csv records pre-update loss; summary.json reports evaluated final loss.\n"
        "config.json + loss.py reproduce the experiment; initial_phases may point to phases.npz.\n"
        "Detector fractions are normalized over the sampled output window, not incident power.\n",encoding="utf-8")
    return out
