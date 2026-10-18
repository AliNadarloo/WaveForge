# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Package the verified portable app and readable source using standard ZIP files."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import argparse

ROOT=Path(__file__).resolve().parent


def zip_paths(target,paths,prefix):
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as archive:
        for file,relative in paths:
            archive.write(file,str(Path(prefix)/relative))
    with zipfile.ZipFile(target) as archive:
        # CRC validation covers every delivered file.
        bad=archive.testzip()
        if bad:raise RuntimeError(f"ZIP verification failed: {bad}")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--release-root",type=Path,default=ROOT/"release/Waveforge")
    portable=parser.parse_args().release_root.resolve()
    checks=json.loads((ROOT/"verification/executable_checks.json").read_text())
    if not checks["passed"]:raise RuntimeError("Verify the executable before packaging.")
    uses=json.loads((ROOT/"verification/eight_usecases.json").read_text())
    if not uses["passed"] or len(uses["cases"])!=8:raise RuntimeError("Complete all eight GUI design runs before packaging.")
    digits=json.loads((ROOT/"verification/digit_checks.json").read_text(encoding="utf-8"))
    if not digits["passed"]:raise RuntimeError("Complete the four-layer digit GUI check before packaging.")
    if (portable/"_internal").exists():raise RuntimeError("This release must use external Python; bundled runtimes are obsolete.")
    if hashlib.sha256((portable/"Waveforge.exe").read_bytes()).hexdigest()!=checks["executable_sha256"]:
        raise RuntimeError("The executable changed after verification.")
    for relative,digest in checks["source_sha256"].items():
        if hashlib.sha256((portable/relative).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"Application source changed after verification: {relative}")
        source_file=ROOT/Path(relative).relative_to("app_source")
        if hashlib.sha256(source_file.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"Source ZIP would differ from the verified app: {source_file.name}")
    documents=["README.md","INSTALL.md","LOSS_GUIDE.md","DEVELOPER_GUIDE.md","VERIFICATION.md",
               "LICENSE","THIRD_PARTY_NOTICES.md","CONTRIBUTING.md","GITHUB.md","CITATION.cff","CITATION.bib","CHANGELOG.md","requirements-runtime.txt"]
    for name in documents:shutil.copyfile(ROOT/name,portable/name)
    shutil.copytree(ROOT/"examples",portable/"examples",dirs_exist_ok=True)
    shutil.copyfile(ROOT/"verification/frozen_ui.png",portable/"preview.png")
    delivery=ROOT/"delivery";delivery.mkdir(exist_ok=True)
    source=[]
    for folder in ["onnlab","tests","launcher","reference","examples","docs",".github"]:
        for file in (ROOT/folder).rglob("*"):
            if file.is_file() and "__pycache__" not in file.parts:source.append((file,file.relative_to(ROOT)))
    scripts=["main.py","build.py","package.py","prepare_source.py","cleanup_release.py","benchmark.py","validate_usecases.py","verify_scenarios.py","verify_executable.py","verify_digits.py","requirements.txt",
             "requirements-lock.txt","pytest.ini",".gitignore",".gitattributes"]
    for name in scripts+documents:
        source.append((ROOT/name,Path(name)))
    for name in ["environment.json","scenarios.json","paper_phase_agreement.json",
                 "original_files_check.json","tests_final.log","ui_tests_final.log",
                 "executable_checks.json","ui_training.png","frozen_ui.png","frozen_ui_guide.png","frozen_ui_inputs.png",
                 "performance.json","eight_usecases.json","digit_checks.json","launcher_setup.png","loss_workshop.png"]:
        source.append((ROOT/"verification"/name,Path("verification")/name))
    for file in (ROOT/"verification/eight_uses").glob("run_*.png"):
        source.append((file,file.relative_to(ROOT)))
    for file in (ROOT/"verification/digit_use").glob("*.png"):source.append((file,file.relative_to(ROOT)))
    for file in (ROOT/"verification").glob("theme_*.png"):source.append((file,file.relative_to(ROOT)))
    zip_paths(delivery/"Waveforge-source.zip",source,"Waveforge-source")
    print("Source ZIP verified",flush=True)
    files=[(p,p.relative_to(portable)) for p in portable.rglob("*")
           if p.is_file() and p.relative_to(portable).parts[0]!="results"
           and p.name not in {"waveforge_error.log","launcher_error.log","launcher_settings.json","ui_preferences.json"}
           and "__pycache__" not in p.parts]
    zip_paths(delivery/"Waveforge-Windows.zip",files,"Waveforge")
    print("Portable ZIP verified",flush=True)
    public=dict(application="Waveforge 1.1.1",author="Ali Nadarloo",license="MIT",tests=53,
                executable_bytes=(portable/"Waveforge.exe").stat().st_size,bundled_cuda=False,bundled_python=False,
                experiments=[dict(name=c["config"]["name"],frequency_thz=c["config"]["frequency_thz"],device=c["device"],metrics=c["metrics"],ui_gap_p95_ms=c["ui_gap_p95_ms"])
                             for c in uses["cases"]],digit_metrics=digits["metrics"],
                digit_ui_gap_p95_ms=digits["ui_gap_p95_ms"],
                paper_doi="10.1038/s41598-026-69471-w",
                affiliation="School of Electrical Engineering, Iran University of Science and Technology, Tehran, Iran",
                phase_comparison=json.loads((ROOT/"verification/paper_phase_agreement.json").read_text()),
                executable_checks=[dict(test=c["test"],passed=c["passed"],device=c.get("device")) for c in checks["checks"]])
    public_dir=ROOT/"verification";public_dir.mkdir(exist_ok=True)
    (public_dir/"public_summary.json").write_text(json.dumps(public,indent=2,ensure_ascii=False),encoding="utf-8")
    (public_dir/"README.md").write_text("<!-- Made by Ali Nadarloo. -->\n# Public verification\n\npublic_summary.json contains measured release outcomes without local machine paths.\nThe full source audit ZIP contains additional tests, receipts and screenshots.\n",encoding="utf-8")
    github=[item for item in source if item[1].parts[0] not in {"reference","verification"}]
    github.extend((public_dir/name,Path("verification")/name) for name in ["README.md","public_summary.json"])
    zip_paths(delivery/"Waveforge-GitHub.zip",github,"Waveforge")
    print("Clean GitHub ZIP verified",flush=True)
    manifest={}
    for p in delivery.glob("*.zip"):
        digest=hashlib.sha256()
        with p.open("rb") as file:
            for block in iter(lambda:file.read(8*1024*1024),b""):digest.update(block)
        manifest[p.name]=dict(bytes=p.stat().st_size,sha256=digest.hexdigest())
    (delivery/"SHA256.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__":main()
