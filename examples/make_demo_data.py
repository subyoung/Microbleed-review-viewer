"""Build a small synthetic dataset you can open the viewer on.

Nothing here comes from a scan of anybody. A phantom head, a few dark spots
standing in for microbleeds, and a findings workbook pointing at them -- enough
to click through the whole loop, and safe to publish, which is why the
screenshots in the README are taken from it.

    python examples/make_demo_data.py            # writes examples/demo/
    python examples/make_demo_data.py --open     # ... and prints how to run it

The geometry is deliberately unlike a real acquisition in one way: the voxels
are isotropic 0.8 mm and the matrix is small, so the whole thing is a couple of
megabytes rather than a couple of hundred.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from openpyxl import Workbook

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import dataset_config  # noqa: E402  (after sys.path)

SHAPE = (140, 176, 120)
VOXEL_MM = 0.8
# Where the demo's lesions are, in millimetres from the centre of the volume,
# and how wide. Two lobar and one deep, so the 3D view has something to say.
LESIONS = (
    ("SUBJ0001", (18.0, -26.0, 14.0), 3.4, "SC_front_R"),
    ("SUBJ0001", (-24.0, 30.0, -8.0), 2.6, "SC_occip_L"),
    ("SUBJ0001", (6.0, 4.0, -12.0), 4.0, "pons"),
    ("SUBJ0002", (-15.0, -18.0, 20.0), 3.0, "SC_front_L"),
    ("SUBJ0002", (11.0, 22.0, 6.0), 2.2, "thalamus_R"),
)


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = [(np.arange(n) - (n - 1) / 2.0) * VOXEL_MM for n in SHAPE]
    return np.meshgrid(*axes, indexing="ij")


def phantom(case_id: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A head-shaped object with a little structure, plus this case's lesions.

    Returned as (magnitude-like, susceptibility-like): a microbleed is dark on
    the first and bright on the second, which is the contrast the viewer's
    region growing is written against.
    """

    x, y, z = _grid()
    radius = np.sqrt((x / 62.0) ** 2 + (y / 76.0) ** 2 + (z / 52.0) ** 2)
    brain = np.clip((0.94 - radius) / 0.05, 0.0, 1.0)

    # Tissue levels rather than nudges.  The first attempt built the brain as
    # one value plus small perturbations, and the automatic window -- which is
    # computed over the whole volume, most of which is air -- pushed all of it
    # into the top of the ramp and rendered a flat white blob.  Real contrast
    # between the compartments is what makes the views look like anything.
    cortex = np.clip(1.0 - np.abs(radius - 0.88) / 0.10, 0.0, 1.0)
    ventricle = np.exp(
        -((((np.abs(x) - 10.0) / 4.5) ** 4) + ((y / 24.0) ** 4) + ((z / 7.0) ** 4))
    )
    nuclei = np.exp(-((((np.abs(x) - 17.0) / 7.0) ** 2) + ((y / 9.0) ** 2) + ((z / 8.0) ** 2)))
    texture = 0.04 * np.sin(x / 4.0) * np.sin(y / 5.5) * np.sin(z / 4.5)

    white_matter = 0.78
    grey_matter = 0.52
    deep_grey = 0.44
    csf = 0.10
    tissue = np.full(SHAPE, white_matter, dtype=np.float64)
    tissue = tissue * (1.0 - cortex) + grey_matter * cortex
    tissue = tissue * (1.0 - nuclei) + deep_grey * nuclei
    tissue = tissue * (1.0 - ventricle) + csf * ventricle
    magnitude = brain * (tissue + texture)
    # A dim rim: a skull that outshines the tissue takes the window with it.
    magnitude += np.clip(1.0 - np.abs(radius - 0.99) / 0.035, 0.0, 1.0) * 0.34
    susceptibility = 0.008 * texture + 0.002 * brain + 0.012 * nuclei * brain

    for owner, centre, diameter, _region in LESIONS:
        if owner != case_id:
            continue
        distance = np.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2 + (z - centre[2]) ** 2)
        blob = np.exp(-((distance / (diameter / 2.0)) ** 4))
        magnitude -= 0.70 * blob * brain
        susceptibility += 0.09 * blob * brain

    magnitude += rng.normal(0.0, 0.012, SHAPE)
    susceptibility += rng.normal(0.0, 0.0015, SHAPE)
    return np.clip(magnitude, 0.0, None).astype(np.float32), susceptibility.astype(np.float32)


def affine() -> np.ndarray:
    """RAS, isotropic, centred on the middle of the volume."""

    matrix = np.eye(4)
    matrix[0, 0] = -VOXEL_MM  # x decreases with index: an ordinary LAS-ish store
    matrix[1, 1] = VOXEL_MM
    matrix[2, 2] = VOXEL_MM
    centre = (np.asarray(SHAPE) - 1) / 2.0
    matrix[:3, 3] = -matrix[:3, :3] @ centre
    return matrix


def write_case(root: Path, case_id: str, rng: np.random.Generator) -> None:
    folder = root / case_id
    folder.mkdir(parents=True, exist_ok=True)
    magnitude, susceptibility = phantom(case_id, rng)
    geometry = affine()
    stamp = f"{case_id}-20240101_120000"
    sequences = dataset_config.DEFAULTS["sequences"]
    for data, key in ((magnitude, "swi"), (susceptibility, "qsm")):
        name = f"{stamp}{sequences[key]['suffix']}"
        nib.save(nib.Nifti1Image(data, geometry), str(folder / name))
    # A minimum-intensity projection over five slices, which is what the real
    # product is: useful for spotting, useless for measuring.
    window = 5
    projected = np.stack(
        [
            magnitude[:, :, max(0, index - window // 2): index + window // 2 + 1].min(axis=2)
            for index in range(SHAPE[2])
        ],
        axis=2,
    ).astype(np.float32)
    nib.save(
        nib.Nifti1Image(projected, geometry),
        str(folder / f"{stamp}{sequences['mip']['suffix']}"),
    )


def write_workbook(path: Path) -> None:
    book = Workbook()
    sheet = book.active
    sheet.title = dataset_config.DEFAULTS["workbook"]["sheet"]
    columns = dataset_config.DEFAULTS["workbook"]["columns"]
    headers = [
        columns["case_id"], columns["slice"], columns["region"],
        columns["ras_l"], columns["ras_p"], columns["ras_s"],
        columns["ras_verified"], columns["readers"], columns["verify"], columns["comments"],
    ]
    sheet.append(headers)
    geometry = affine()
    inverse = np.linalg.inv(geometry)
    for case_id, centre, _diameter, region in LESIONS:
        # The phantom places its blobs at millimetre offsets along the *array*
        # axes; the workbook wants physical RAS.  Going through the affine is
        # the only way those agree -- this store flips x, so writing the offset
        # straight out would put every finding on the wrong side of the head.
        offset = np.asarray(centre) / VOXEL_MM
        voxel = (np.asarray(SHAPE) - 1) / 2.0 + offset
        world = geometry[:3, :3] @ voxel + geometry[:3, 3]
        sheet.append(
            [
                case_id,
                int(round(float(voxel[2]))),
                region,
                round(float(world[0]), 3),
                round(float(world[1]), 3),
                round(float(world[2]), 3),
                1,
                "demo_reader",
                1,
                "",
            ]
        )
    for index, width in enumerate([12, 11, 16, 11, 11, 11, 13, 14, 14, 14], start=1):
        sheet.column_dimensions[chr(64 + index)].width = width
    book.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--into", default=str(HERE / "demo"), help="where to write it")
    parser.add_argument("--open", action="store_true", help="print how to run the viewer on it")
    args = parser.parse_args()

    root = Path(args.into)
    data_root = root / "Data"
    data_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20240101)
    cases = sorted({case for case, *_rest in LESIONS})
    for case_id in cases:
        write_case(data_root, case_id, rng)
    workbook = root / "demo_findings.xlsx"
    write_workbook(workbook)

    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6
    print(f"{len(cases)} cases, {len(LESIONS)} findings, {size:.1f} MB in {root}")
    if args.open:
        print()
        print("  $env:MICROBLEED_SOURCE_XLSX = '" + str(workbook) + "'")
        print("  $env:MICROBLEED_DATA_ROOT   = '" + str(data_root) + "'")
        print("  $env:MICROBLEED_REVIEW_DB   = '" + str(root / "demo_review.sqlite") + "'")
        print("  .\\run_app.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
