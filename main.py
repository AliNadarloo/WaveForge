# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Launch the desktop app or a reproducible command-line experiment."""
import argparse
import json
from pathlib import Path
import sys

# The small Windows launcher starts this ordinary script with the selected
# Python environment and WAVEFORGE_HOME set to the folder beside the EXE.


def main():
    parser=argparse.ArgumentParser(description="Waveforge D2NN laboratory")
    parser.add_argument("--preset",default="Gaussian focus")
    parser.add_argument("--config",type=Path)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--cpu",action="store_true")
    parser.add_argument("--smoke-gui",type=Path,help="Render and validate the UI, save a screenshot, then exit.")
    args=parser.parse_args()
    if args.headless:
        from onnlab.presets import preset
        from onnlab.config import Config
        from onnlab.training import train
        from onnlab.export import export_result
        c=Config(**json.loads(args.config.read_text(encoding="utf-8"))).resolve_paths(args.config.resolve().parent) if args.config else preset(args.preset)
        if args.epochs:c.epochs=args.epochs
        if args.cpu:c.device="CPU"
        result=train(c)
        path=export_result(result,args.output)
        # Windowed executables have no stdout; always leave an on-disk receipt.
        receipt=dict(path=str(path),metrics=result.metrics,device=result.device)
        if sys.stdout:print(json.dumps(receipt,indent=2))
        return
    from onnlab.bootstrap import launch
    sys.exit(launch(args))


if __name__=="__main__":
    try:
        main()
    except Exception:
        import traceback
        from onnlab.paths import application_directory
        root=application_directory()
        (root/"waveforge_error.log").write_text(traceback.format_exc(),encoding="utf-8")
        if getattr(sys,"frozen",False):sys.exit(1)
        raise
