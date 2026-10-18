# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Keep source attribution and portable example files consistent before a release."""
from pathlib import Path
import json
from onnlab.presets import PRESET_NAMES,preset
from onnlab.objectives import EXAMPLES,SOURCE_CREDIT

ROOT=Path(__file__).resolve().parent


def main():
    paths=list((ROOT/"onnlab").glob("*.py"))+list(ROOT.glob("*.py"))+list((ROOT/"tests").glob("*.py"))
    paths+=list((ROOT/"launcher").glob("*.cs"))+list((ROOT/"examples/loss_functions").glob("*.py"))+list((ROOT/"verification").glob("*.py"))
    for file in paths:
        text=file.read_text(encoding="utf-8-sig")
        if "Made by Ali Nadarloo" not in "\n".join(text.splitlines()[:3]):
            prefix="// Made by Ali Nadarloo.\n// SPDX-License-Identifier: MIT\n" if file.suffix==".cs" else SOURCE_CREDIT
            file.write_text(prefix+text,encoding="utf-8")
    # Local reference snapshots retain their original authorship and license scope.
    for file in (ROOT/"reference").rglob("*.py"):
        text=file.read_text(encoding="utf-8-sig")
        if "WaveForge integration: Made by Ali Nadarloo" not in text[:200]:
            file.write_text("# WaveForge integration: Made by Ali Nadarloo.\n# Original reference source and existing author notices follow.\n"+text,encoding="utf-8")
    for folder in [ROOT,ROOT/"examples/digits"]:
        for file in folder.glob("*.md"):
            text=file.read_text(encoding="utf-8-sig")
            if not text.startswith("<!-- Made by Ali Nadarloo."):
                file.write_text("<!-- Made by Ali Nadarloo. -->\n"+text,encoding="utf-8")
    for file in [ROOT/"requirements.txt",ROOT/"requirements-runtime.txt",ROOT/"requirements-lock.txt",ROOT/"pytest.ini"]:
        text=file.read_text(encoding="utf-8-sig")
        if not text.startswith("# Made by Ali Nadarloo."):file.write_text("# Made by Ali Nadarloo.\n"+text,encoding="utf-8")
    for index,name in enumerate(PRESET_NAMES,1):
        data=preset(name).to_dict()
        if data["target"]=="Image intensity":data["target_file"]="target.png"
        (ROOT/f"examples/{index:02d}_experiment.json").write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    for file in (ROOT/"examples/validated_uses").glob("*.json"):
        data=json.loads(file.read_text(encoding="utf-8"))
        for channel in [data,data.get("channel2",{})]:
            # Keep both channels portable, even when the second channel is inactive.
            for key in ("aperture_file","target_file"):
                if channel.get(key) and Path(channel[key]).name=="target.png":channel[key]="../target.png"
            code=channel.get("loss_code","")
            if code and "Made by Ali Nadarloo" not in code:channel["loss_code"]=SOURCE_CREDIT+code
        file.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    existing=list((ROOT/"examples/loss_functions").glob("*.py"))
    for file in existing:
        # Match recipe bodies; retain friendly existing filenames.
        text=file.read_text(encoding="utf-8")
        if "Made by Ali Nadarloo" not in text:file.write_text(SOURCE_CREDIT+text,encoding="utf-8")
    (ROOT/"examples/loss_functions/09_digit_recognition.py").write_text(EXAMPLES["Digit recognition / ten detectors"],encoding="utf-8")
    print("Attribution and portable example configurations prepared.")


if __name__=="__main__":main()
