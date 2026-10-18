# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Build a small Windows launcher; Python, PyTorch and CUDA stay external."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parent
DOCS=["README.md","INSTALL.md","LOSS_GUIDE.md","DEVELOPER_GUIDE.md","VERIFICATION.md","requirements-runtime.txt",
      "LICENSE","THIRD_PARTY_NOTICES.md","CONTRIBUTING.md","GITHUB.md","CITATION.cff","CITATION.bib","CHANGELOG.md"]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=ROOT/"release_next/Waveforge")
    output=parser.parse_args().output.resolve();output.mkdir(parents=True,exist_ok=True)
    work=ROOT/"build_light";work.mkdir(exist_ok=True)
    payload={"app_source/main.py":ROOT/"main.py"}
    for file in (ROOT/"onnlab").glob("*.py"):payload["app_source/onnlab/"+file.name]=file
    for file in (ROOT/"examples").rglob("*"):
        if file.is_file() and "__pycache__" not in file.parts:payload[file.relative_to(ROOT).as_posix()]=file
    for file in (ROOT/"docs/images").glob("*.png"):payload[file.relative_to(ROOT).as_posix()]=file
    for name in DOCS:payload[name]=ROOT/name
    archive=work/"payload.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for relative,file in payload.items():z.write(file,relative)
    compiler=Path("C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe")
    command=[str(compiler),"/nologo","/target:winexe","/optimize+","/out:"+str(output/"Waveforge.exe"),
             "/resource:"+str(archive)+",Waveforge.payload.zip",
             "/reference:System.Windows.Forms.dll","/reference:System.Drawing.dll",
             "/reference:System.IO.Compression.dll","/reference:System.IO.Compression.FileSystem.dll",
             "/reference:System.Web.Extensions.dll",str(ROOT/"launcher/WaveforgeLauncher.cs")]
    subprocess.run(command,check=True)
    manifest={}
    for relative,file in payload.items():
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
        manifest[relative]=hashlib.sha256(file.read_bytes()).hexdigest()
    (output/"app_source/distribution.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(executable=str(output/"Waveforge.exe"),bytes=(output/"Waveforge.exe").stat().st_size,
                          payload_files=len(payload),bundled_python=False,bundled_cuda=False),indent=2))


if __name__=="__main__":main()
