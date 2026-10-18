# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Create Windows and source downloads from a built WaveForge checkout."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
DOCUMENTS = [
    "README.md", "INSTALL.md", "LOSS_GUIDE.md", "DEVELOPER_GUIDE.md",
    "VERIFICATION.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md",
    "GITHUB.md", "CITATION.cff", "CITATION.bib", "CHANGELOG.md",
]
SCRIPTS = [
    "main.py", "build.py", "package.py", "prepare_source.py", "benchmark.py",
    "validate_usecases.py", "verify_scenarios.py", "verify_executable.py",
    "verify_digits.py", "requirements.txt", "requirements-runtime.txt",
    "requirements-lock.txt", "pytest.ini", ".gitignore", ".gitattributes",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_zip(target, files):
    """Use one top-level folder and validate every archived file's CRC."""
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for source, relative in sorted(files, key=lambda item: str(item[1])):
            z.write(source, (Path("WaveForge") / relative).as_posix())
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:
            raise RuntimeError(f"Archive integrity check failed: {target.name}")


def source_files():
    files = [(ROOT / name, Path(name)) for name in DOCUMENTS + SCRIPTS]
    for folder in ["onnlab", "tests", "launcher", "examples", "docs", ".github"]:
        files.extend(
            (p, p.relative_to(ROOT)) for p in (ROOT / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
    for name in ["README.md", "public_summary.json"]:
        path = ROOT / "verification" / name
        files.append((path, path.relative_to(ROOT)))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=ROOT / "release/WaveForge")
    parser.add_argument("--output", type=Path, default=ROOT / "delivery")
    args = parser.parse_args()
    portable, output = args.release_root.resolve(), args.output.resolve()
    executable = portable / "WaveForge.exe"
    manifest = json.loads((portable / "app_source/distribution.json").read_text())
    if not executable.is_file():
        raise RuntimeError("Build WaveForge before packaging.")
    for relative, expected in manifest.items():
        distributed = portable / relative
        source_relative = relative.removeprefix("app_source/")
        if sha256(distributed) != expected or sha256(ROOT / source_relative) != expected:
            raise RuntimeError(f"Build is out of date: {relative}. Run build.py again.")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(executable, output / "WaveForge.exe")
    windows = [(executable, Path("WaveForge.exe"))]
    windows.extend((portable / name, Path(name)) for name in manifest)
    windows.append((portable / "app_source/distribution.json", Path("app_source/distribution.json")))
    write_zip(output / "WaveForge-Windows.zip", windows)
    write_zip(output / "WaveForge-source.zip", source_files())
    downloads = ["WaveForge.exe", "WaveForge-Windows.zip", "WaveForge-source.zip"]
    checksums = "".join(f"{sha256(output / name)}  {name}\n" for name in downloads)
    (output / "SHA256SUMS.txt").write_text(checksums, encoding="utf-8")
    print(f"Packages and SHA-256 checksums: {output}")


if __name__ == "__main__":
    main()
