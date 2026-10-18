# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Eight real experiments through the packaged Qt app, with screenshots and receipts.

Change make_cases() to add your own repeatable design experiments. This does not
automate any other user's application: it invokes this app's verification mode.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
import numpy as np
from onnlab.presets import preset
from onnlab.objectives import EXAMPLES

ROOT=Path(__file__).resolve().parent


def at_frequency(c,frequency):
    c.frequency_thz=frequency
    c.dx_um=c.dy_um=c.wavelength*1e6/2
    return c


def make_cases():
    cases=[]
    c=at_frequency(preset("Gaussian focus"),.3);c.name="01 Gaussian focusing / 0.3 THz";c.epochs=150;cases.append(c)
    c=at_frequency(preset("Double-slit steering"),.6);c.name="02 Beam steering / 0.6 THz";c.propagation="FFT RS";c.epochs=250;cases.append(c)
    c=at_frequency(preset("Text hologram"),.8);c.name="03 Text hologram / 0.8 THz";c.propagation="ASM • evanescent decay";c.samples_per_cell=2;c.epochs=250;cases.append(c)
    c=at_frequency(preset("ASM • subcell OAM"),1.2);c.name="04 OAM conversion / 1.2 THz";c.target_charge=2;c.optimizer="AdamW";c.weight_decay=.0001;c.epochs=300;cases.append(c)
    c=at_frequency(preset("Gaussian focus"),473.6);c.name="05 Image aperture focus / visible light"
    c.aperture="Image / array";c.aperture_file=str(ROOT/"examples/target.png");c.beam="Bessel–Gauss"
    c.layers=2;c.distances_wl=[1.2,4.,10.8];c.propagation="ASM • evanescent decay";c.samples_per_cell=2;c.epochs=300;cases.append(c)
    c=at_frequency(preset("Two-plane focus"),.9);c.name="06 Two-plane focus / 0.9 THz";c.propagation="ASM • propagating only";c.samples_per_cell=2;c.epochs=250;cases.append(c)
    c=at_frequency(preset("Gaussian focus"),.1);c.name="07 Independent dual polarization / 0.1 THz"
    c.polarization_count=2;c.layers=2;c.distances_wl=[1.2,4.,10.8];c.propagation="ASM • evanescent decay";c.samples_per_cell=2
    c.phase_initialization="Small random"
    c.target_x_wl=-2;c.target_waist_wl=1.5;c.epochs=300
    c.channel2=dict(beam="OAM (LG p=0)",charge=-1,target="OAM mode",target_charge=1,target_x_wl=0.,target_waist_wl=2.,loss_code=EXAMPLES["OAM / complex mode overlap"])
    cases.append(c)
    c=preset("Paper • two polarizations");c.epochs=3000;cases.append(c)
    return cases


def check_receipt(path):
    receipt=json.loads(path.read_text(encoding="utf-8"));metrics=receipt["metrics"]
    if receipt["config"]["polarization_count"]==2:
        for index in [1,2]:assert metrics[f"P{index}_final_loss"]<metrics[f"P{index}_initial_loss"]
    else:assert metrics["final_loss"]<metrics["initial_loss"]
    assert all(device=="cuda" for device in receipt["device"].split(", "))
    folder=Path(receipt["result_folder"])
    for file in folder.rglob("phases.npz"):
        assert np.isfinite(np.load(file)["phases_rad"]).all()
    assert list(folder.rglob("CST_phase_layer_1.txt"))
    assert path.with_suffix(".png").is_file()
    assert path.with_name(path.stem+"_cst_popup.png").is_file()
    if receipt["config"]["name"].startswith("Paper"):
        for index in [1,2]:
            summary=json.loads((folder/f"Polarization_{index}/summary.json").read_text())
            assert len(summary["truth_table"])==14 and all(row["correct"] for row in summary["truth_table"])
    else:
        field_path=folder/("Polarization_1/fields.npz" if receipt["config"]["polarization_count"]==2 else "fields.npz")
        with np.load(field_path) as fields:
            a=fields["intensity"][0];b=fields["target_intensity"]
            a=a-a.mean();b=b-b.mean()
            ncc=float((a*b).sum()/max(float(np.sqrt((a*a).sum()*(b*b).sum())),1e-16))
        receipt["intensity_ncc"]=ncc
        name=receipt["config"]["name"]
        if name.startswith(("01","03","05","07")):assert ncc>.85
        if name.startswith("02"):assert abs(metrics["centroid_x_wavelengths"]-3)<.1
        if name.startswith("04"):assert metrics["target_angular_purity"]>.95 and metrics["final_loss"]<.05
        if name.startswith("06"):assert metrics["final_loss"]<.2
        if name.startswith("07"):assert metrics["P2_target_angular_purity"]>.95
    assert receipt["ui_heartbeat_ticks"]>0
    assert receipt["ui_gap_p95_ms"]<250
    receipt["passed"]=True
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--exe",type=Path,default=ROOT/"release/WaveForge/WaveForge.exe");args=parser.parse_args()
    qa=ROOT/"verification/eight_uses";qa.mkdir(exist_ok=True)
    cases=make_cases()
    for index,c in enumerate(cases,1):
        c.validate();(qa/f"case_{index:02d}.json").write_text(json.dumps(c.to_dict(),indent=2,ensure_ascii=False),encoding="utf-8")
    if args.prepare_only:return
    exe=args.exe.resolve();reports=[]
    env=os.environ.copy();env["QT_QPA_PLATFORM"]="offscreen"
    for index,c in enumerate(cases,1):
        print(f"GUI {index}/8: {c.name}",flush=True)
        image=qa/f"run_{index:02d}.png"
        arguments=[str(exe),"--config",str(qa/f"case_{index:02d}.json"),"--smoke-gui",str(image),
                   "--epochs",str(c.epochs),"--output",str(qa/"results")]
        start=time.perf_counter()
        completed=subprocess.run(arguments,cwd=ROOT,env=env,timeout=720,creationflags=subprocess.CREATE_NO_WINDOW)
        if completed.returncode:raise RuntimeError(f"Case {index} exited {completed.returncode}; see waveforge_error.log")
        receipt=check_receipt(image.with_suffix(".json"));receipt["wall_seconds"]=time.perf_counter()-start
        reports.append(receipt)
        (ROOT/"verification/eight_usecases.json").write_text(json.dumps(dict(passed=len(reports)==8,cases=reports),indent=2),encoding="utf-8")
        print(json.dumps(dict(case=index,frequency_thz=c.frequency_thz,metrics=receipt["metrics"],wall_seconds=receipt["wall_seconds"],ui_gap_p95_ms=receipt["ui_gap_p95_ms"])),flush=True)


if __name__=="__main__":main()
