# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Run real design exercises and write their artifacts under application/results/."""
from pathlib import Path
import json
import sys
import time
import numpy as np
from onnlab.presets import preset,PRESET_NAMES
from onnlab.training import train
from onnlab.export import export_result
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent


def main():
    examples=ROOT/"examples";examples.mkdir(exist_ok=True)
    for old in examples.glob("*.json"):old.unlink()
    for i,name in enumerate(PRESET_NAMES,1):
        c=preset(name)
        (examples/f"{i:02d}_experiment.json").write_text(json.dumps(c.to_dict(),indent=2,ensure_ascii=False),encoding="utf-8")
    # Supply a small image so the image task can run immediately from the library.
    image=Image.new("L",(96,96));draw=ImageDraw.Draw(image)
    draw.ellipse((12,12,84,84),outline=255,width=10);draw.rectangle((42,30,54,66),fill=255)
    image.save(examples/"target.png")
    for path in examples.glob("*.json"):
        data=json.loads(path.read_text(encoding="utf-8"))
        if data["target"]=="Image intensity":
            data["target_file"]="target.png"
            path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    reports=[]
    for name,epochs in [("Gaussian focus",150),("OAM conversion",250),("Text hologram",250),
                        ("Paper • two polarizations",3000)]:
        c=preset(name);c.epochs=epochs
        print(f"Starting {name}: {epochs} epochs",flush=True)
        result=train(c)
        out=export_result(result,ROOT/"results")
        item=dict(name=name,epochs=epochs,seconds=result.seconds,device=result.device,
                  metrics=result.metrics,result_folder=str(out.relative_to(ROOT)))
        if hasattr(result,"channels"):
            item["truth_tables"]=[]
            for i in [1,2]:
                summary=json.loads((out/f"Polarization_{i}/summary.json").read_text())
                rows=summary["truth_table"]
                item["truth_tables"].append(dict(polarization=i,correct=sum(row["correct"] for row in rows),
                                                count=len(rows),mean_contrast_db=float(np.mean([row["contrast_db"] for row in rows]))))
        reports.append(item)
        (ROOT/"verification/scenarios.json").write_text(json.dumps(reports,indent=2))
        print(json.dumps(item,indent=2),flush=True)


if __name__=="__main__":main()
