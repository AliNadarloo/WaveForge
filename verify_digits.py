# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Train the complete handwritten-digit example through the delivered Windows EXE."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import numpy as np
from onnlab.presets import preset

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--exe",type=Path,default=ROOT/"release/WaveForge/WaveForge.exe")
    args=parser.parse_args();folder=ROOT/"verification/digit_use";folder.mkdir(exist_ok=True)
    config=preset("Advanced • four-layer digit router")
    config_file=folder/"experiment.json";config_file.write_text(json.dumps(config.to_dict(),indent=2,ensure_ascii=False),encoding="utf-8")
    (ROOT/"examples/11_experiment.json").write_text(config_file.read_text(encoding="utf-8"),encoding="utf-8")
    image=folder/"digit_router.png";env=os.environ.copy();env["QT_QPA_PLATFORM"]="offscreen"
    subprocess.run([str(args.exe.resolve()),"--config",str(config_file),"--smoke-gui",str(image),
                    "--epochs",str(config.epochs),"--output",str(folder/"results")],cwd=ROOT,env=env,
                    timeout=600,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    report=json.loads(image.with_suffix(".json").read_text(encoding="utf-8"));metrics=report["metrics"]
    assert report["device"]=="cuda"
    assert metrics["heldout_accuracy"]>=.90 and metrics["final_loss"]<metrics["initial_loss"]
    result=Path(report["result_folder"])
    for layer in range(1,5):assert (result/f"CST_phase_layer_{layer}.txt").is_file()
    with np.load(result/"phases.npz") as phases:assert phases["phases_rad"].shape==(4,32,32) and np.isfinite(phases["phases_rad"]).all()
    with np.load(result/"digit_predictions.npz") as predictions:
        assert len(predictions["train_indices"])==1438 and len(predictions["heldout_indices"])==359
        assert not set(predictions["train_indices"]) & set(predictions["heldout_indices"])
        assert predictions["confusion"].sum()==359
    assert (result/"digit_predictions.csv").is_file() and (result/"digit_accuracy.png").is_file()
    assert report["ui_gap_p95_ms"]<250
    report["passed"]=True
    (ROOT/"verification/digit_checks.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(dict(passed=True,metrics=metrics,ui_gap_p95_ms=report["ui_gap_p95_ms"],result_folder=str(result)),indent=2))


if __name__=="__main__":main()
