# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Verify the frozen app, not an installed-Python substitute. Windows only."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import numpy as np
import argparse

ROOT=Path(__file__).resolve().parent
EXE=ROOT/"release/Waveforge/Waveforge.exe"


def launch(arguments,timeout=120):
    env=os.environ.copy();env["QT_QPA_PLATFORM"]="offscreen"
    completed=subprocess.run([str(EXE),*map(str,arguments)],cwd=ROOT,env=env,
                             stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,timeout=timeout,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    if completed.returncode:
        error=EXE.parent/"waveforge_error.log"
        raise RuntimeError(error.read_text() if error.exists() else f"Exit code {completed.returncode}")


def main():
    global EXE
    parser=argparse.ArgumentParser();parser.add_argument("--exe",type=Path,default=EXE)
    EXE=parser.parse_args().exe.resolve()
    checks=[]
    for name,extra in [("GPU",[]),("CPU",["--cpu"]),
                       ("Dual GPU",["--config",ROOT/"examples/04_experiment.json"]),
                       ("ASM GPU",["--config",ROOT/"examples/10_experiment.json"])]:
        base=ROOT/"verification"/("executable_"+name.lower().replace(" ","_"))
        print("Checking frozen",name,flush=True)
        launch(["--headless","--epochs","10" if name=="ASM GPU" else "3","--output",base,*extra])
        run=max(base.iterdir(),key=lambda p:p.stat().st_mtime)
        summary=json.loads((run/"summary.json").read_text())
        if name=="CPU":assert summary["device"]=="cpu"
        else:assert summary["device"].split(", ")==(["cuda","cuda"] if name=="Dual GPU" else ["cuda"])
        for phasefile in run.rglob("phases.npz"):
            assert np.isfinite(np.load(phasefile)["phases_rad"]).all()
        assert list(run.rglob("CST_phase_layer_1.txt"))
        metrics=summary["metrics"]
        if name=="Dual GPU":
            assert metrics["P1_final_loss"]<metrics["P1_initial_loss"]
            assert metrics["P2_final_loss"]<metrics["P2_initial_loss"]
        else:assert metrics["final_loss"]<metrics["initial_loss"]
        checks.append(dict(test=name,device=summary["device"],metrics=metrics,passed=True))
    screenshot=ROOT/"verification/frozen_ui.png"
    launch(["--smoke-gui",screenshot,"--epochs","30"])
    assert screenshot.stat().st_size>20000
    assert screenshot.with_name("frozen_ui_guide.png").stat().st_size>10000
    gui_receipt=json.loads(screenshot.with_suffix(".json").read_text())
    assert "app_source" in gui_receipt["application_module"]
    checks.append(dict(test="Windows launcher + Qt GUI: training + plots + export",passed=True,screenshot=str(screenshot.relative_to(ROOT))))
    source_hashes={str(p.relative_to(EXE.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (EXE.parent/"app_source").rglob("*.py")}
    report=dict(passed=True,executable_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest(),source_sha256=source_hashes,checks=checks)
    (ROOT/"verification/executable_checks.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
