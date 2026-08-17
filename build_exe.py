"""Build the standalone Windows application.

The point is a folder somebody can copy onto a reading-room machine and run,
with no Python and no pip. PyInstaller is a build-time tool only -- it is not
in requirements.txt, because nobody running the viewer needs it.

    python -m pip install pyinstaller
    python build_exe.py

The result is ``dist/MicrobleedReviewViewer/`` and a zip of it beside.

One-folder rather than one-file: a one-file build unpacks itself to a temp
directory on every launch, which for a Qt application is several seconds each
time and looks like a hang. The folder starts immediately.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "MicrobleedReviewViewer"

# Shipped beside the executable so a copied folder is self-contained: the
# configuration example, the input documentation and the demo generator are
# what a new user needs on the machine, not back on the web page.
ALONGSIDE = [
    ("config.example.json", "config.example.json"),
    ("README.md", "README.md"),
    ("LICENSE", "LICENSE"),
    ("examples/README.md", "examples/README.md"),
    ("examples/example_findings.xlsx", "examples/example_findings.xlsx"),
    ("examples/make_demo_data.py", "examples/make_demo_data.py"),
]


def build(clean: bool) -> Path:
    dist = HERE / "dist"
    work = HERE / "build"
    if clean:
        for folder in (dist, work):
            shutil.rmtree(folder, ignore_errors=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--name",
        NAME,
        "--windowed",  # no console window behind the application
        "--icon",
        str(HERE / "microbleed_review_icon.ico"),
        # dataset_config and config are imported by name at run time from the
        # application's own directory; PyInstaller sees them, but say so.
        "--hidden-import",
        "dataset_config",
        "--hidden-import",
        "config",
        # Qt modules the viewer never touches; leaving them out halves the
        # download.  Anything actually needed is pulled in by the imports.
        *[
            arg
            for module in (
                "PySide6.QtQml",
                "PySide6.QtQuick",
                "PySide6.QtQuick3D",
                "PySide6.Qt3DCore",
                "PySide6.Qt3DRender",
                "PySide6.QtCharts",
                "PySide6.QtDataVisualization",
                "PySide6.QtMultimedia",
                "PySide6.QtWebEngineCore",
                "PySide6.QtWebEngineWidgets",
                "PySide6.QtNetwork",
                "PySide6.QtPositioning",
                "PySide6.QtBluetooth",
                "PySide6.QtSensors",
                "PySide6.QtSerialPort",
                "PySide6.QtTest",
                "PySide6.QtDesigner",
                "PySide6.QtHelp",
                "PySide6.QtSql",
                "PySide6.QtPdf",
                "PySide6.QtPdfWidgets",
            )
            for arg in ("--exclude-module", module)
        ],
        str(HERE / "desktop_app.py"),
    ]
    print(" ".join(command[:6]), "…")
    subprocess.run(command, cwd=HERE, check=True)

    target = dist / NAME
    for source, destination in ALONGSIDE:
        origin = HERE / source
        if not origin.exists():
            print(f"  (skipping {source}: not here)")
            continue
        landing = target / destination
        landing.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, landing)
    return target


def package(folder: Path) -> Path:
    archive = folder.parent / f"{folder.name}-windows-x64.zip"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for item in sorted(folder.rglob("*")):
            if item.is_file():
                bundle.write(item, item.relative_to(folder.parent))
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-clean", action="store_true", help="reuse the previous build")
    parser.add_argument("--no-zip", action="store_true", help="stop after the folder")
    args = parser.parse_args()

    folder = build(clean=not args.no_clean)
    size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e6
    print(f"\n{folder}  ({size:.0f} MB)")
    if not args.no_zip:
        archive = package(folder)
        print(f"{archive}  ({archive.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
