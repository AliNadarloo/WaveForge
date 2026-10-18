# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Remove superseded app builds after verification, preserving user results/edits."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parent


def safe(path):
    resolved=Path(path).resolve()
    if resolved==ROOT or not resolved.is_relative_to(ROOT):raise ValueError(f"Outside application: {resolved}")
    return resolved


def size(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def main():
    for name in ["executable_checks","eight_usecases","digit_checks"]:
        assert json.loads((ROOT/f"verification/{name}.json").read_text(encoding="utf-8"))["passed"]
    old=safe(ROOT/"release/Waveforge");new=safe(ROOT/"release_next/Waveforge")
    assert (new/"Waveforge.exe").is_file()
    report=dict(removed=[],preserved=[],bytes_removed=0)
    baseline=ROOT/"delivery/Waveforge-source.zip"
    archive=zipfile.ZipFile(baseline) if baseline.exists() else None
    preserve=safe(ROOT/"preserved_user_files")

    def retain(file,relative):
        destination=safe(preserve/relative);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,destination);report["preserved"].append(str(destination.relative_to(ROOT)))

    if old.exists():
        # All historical results, including runs made after the previous delivery,
        # remain available in the new app's normal results directory.
        if (old/"results").exists():
            destination=safe(new/"results");destination.mkdir(exist_ok=True)
            for file in (old/"results").iterdir():
                target=safe(destination/file.name)
                if target.exists():target=target.with_name(target.name+"_previous")
                shutil.move(str(safe(file)),str(target));report["preserved"].append(str(target.relative_to(ROOT)))
        for folder in ["app_source","examples"]:
            for file in (old/folder).rglob("*"):
                if not file.is_file() or "__pycache__" in file.parts:continue
                relative=file.relative_to(old)
                original=Path(*relative.parts[1:]) if folder=="app_source" else relative
                entry="Waveforge-source/"+original.as_posix()
                try:expected=archive.read(entry) if archive else None
                except KeyError:expected=None
                if file.name=="distribution.json":continue
                if expected is None or hashlib.sha256(file.read_bytes()).digest()!=hashlib.sha256(expected).digest():retain(file,relative)
        known={"Waveforge.exe","preview.png","waveforge_error.log","launcher_error.log"}
        for file in old.iterdir():
            if file.is_file() and file.name not in known:
                try:expected=archive.read("Waveforge-source/"+file.name) if archive else None
                except KeyError:expected=None
                if expected is not None and file.read_bytes()==expected:continue
                target=safe(new/file.name)
                if not target.exists():shutil.move(str(safe(file)),str(target));report["preserved"].append(str(target.relative_to(ROOT)))
                else:retain(file,Path(file.name))
            elif file.is_dir() and file.name not in {"_internal","app_source","examples","results","__pycache__"}:
                target=safe(new/file.name)
                if target.exists():target=safe(preserve/file.name)
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(safe(file)),str(target));report["preserved"].append(str(target.relative_to(ROOT)))
        count=size(old);shutil.rmtree(safe(old));report["removed"].append(str(old.relative_to(ROOT)));report["bytes_removed"]+=count
    if archive:archive.close()
    final=safe(ROOT/"release/Waveforge");final.parent.mkdir(exist_ok=True)
    shutil.move(str(new),str(final))
    for name in ["build_release","build_light","hooks"]:
        target=safe(ROOT/name)
        if target.exists():
            report["bytes_removed"]+=size(target);shutil.rmtree(target);report["removed"].append(name)
    spec=safe(ROOT/"Waveforge.spec")
    if spec.exists():spec.unlink();report["removed"].append(spec.name)
    if (ROOT/"release_next").exists() and not any((ROOT/"release_next").iterdir()):(ROOT/"release_next").rmdir()
    (ROOT/"verification/cleanup.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
