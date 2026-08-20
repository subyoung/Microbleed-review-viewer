"""Segmentations produced somewhere else, brought in to be looked at.

A detection model is trained and run outside this tool, and its output has to
be judged the same way a reader's own outlining is: on the images, in three
planes, against what a person marked. This module finds those files and says
what they are. It never writes to them.

The layout it expects is the one a batch run produces -- one folder per case,
the same case identifiers the review database uses:

    results/
      F101279/
        models/nnUNet_M0_baseline_prediction.nii.gz
        models/SHIVA_CPU_ensemble_probability.nii.gz
      J183708/
        models/...

``pattern`` says where inside a case folder to look: ``"."`` for the folder
itself, ``"./models"`` for a subfolder. Runs organise themselves differently
and the alternative is renaming somebody's output directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

VOLUME_SUFFIXES = (".nii", ".nii.gz")

# A map with more distinct values than this is a probability rather than a
# label. Real label files have a handful of integers; a probability map has
# thousands of levels, and the two want different treatment on screen.
MASK_MAX_LEVELS = 16


@dataclass(frozen=True)
class ExternalResult:
    """One file, and what is known about it without opening the data."""

    case_id: str
    name: str
    path: Path

    @property
    def label(self) -> str:
        """What to call it in a list: the filename without its extensions."""

        name = self.path.name
        for suffix in VOLUME_SUFFIXES:
            if name.lower().endswith(suffix):
                return name[: -len(suffix)]
        return self.path.stem


def resolve_pattern(case_folder: Path, pattern: str) -> Path:
    """Where inside a case folder the results are.

    Accepts ``.``, ``./models``, ``models``, and backslashes, because people
    type what their shell shows them.
    """

    cleaned = str(pattern or ".").strip().replace("\\", "/").strip("/")
    if cleaned in ("", "."):
        return case_folder
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return case_folder.joinpath(*[part for part in cleaned.split("/") if part not in ("", ".")])


def discover(root: Path | str, pattern: str = ".") -> dict[str, list[ExternalResult]]:
    """Every case under ``root`` that has at least one volume, by case id.

    Only the file names are read. Judging what is in them means loading
    hundreds of megabytes, and this runs to decide whether a case gets a mark
    in the queue -- which happens for every case, at start-up.
    """

    root = Path(root)
    if not root.is_dir():
        return {}
    found: dict[str, list[ExternalResult]] = {}
    for case_folder in sorted(item for item in root.iterdir() if item.is_dir()):
        folder = resolve_pattern(case_folder, pattern)
        if not folder.is_dir():
            continue
        results = [
            ExternalResult(case_folder.name, item.name, item)
            for item in sorted(folder.iterdir())
            if item.is_file() and item.name.lower().endswith(VOLUME_SUFFIXES)
        ]
        if results:
            found[case_folder.name] = results
    return found


def classify(data: np.ndarray) -> str:
    """``"mask"`` or ``"probability"``.

    Decided from the data rather than the filename: a run that calls
    everything "prediction" still has to be shown correctly, and a threshold
    slider on a binary mask is a control that does nothing.
    """

    array = np.asanyarray(data)
    if array.dtype.kind in "iub":
        return "mask"
    finite = array[np.isfinite(array)]
    if not finite.size:
        return "mask"
    sample = finite if finite.size <= 200000 else finite[:: max(1, finite.size // 200000)]
    levels = np.unique(sample)
    if len(levels) <= MASK_MAX_LEVELS and np.all(levels == np.rint(levels)):
        return "mask"
    return "probability"


def geometry_matches(image, reference) -> bool:
    """Same grid, so a voxel here is the voxel there.

    Anything else would need resampling, and a mask resampled without saying
    so is how a lesion moves a millimetre and nobody notices.
    """

    if tuple(image.shape[:3]) != tuple(reference.shape[:3]):
        return False
    return bool(np.allclose(np.asarray(image.affine), np.asarray(reference.affine), atol=1e-4))


def compare(ours: np.ndarray, theirs: np.ndarray, voxel_mm3: float) -> dict[str, float]:
    """How two binary masks of the same grid relate.

    Dice plus the three counts it hides: agreeing on nothing and agreeing on
    everything both give a number, and only the counts say which happened.
    """

    ours = np.asanyarray(ours).astype(bool)
    theirs = np.asanyarray(theirs).astype(bool)
    both = int(np.count_nonzero(ours & theirs))
    only_ours = int(np.count_nonzero(ours & ~theirs))
    only_theirs = int(np.count_nonzero(theirs & ~ours))
    total = 2 * both + only_ours + only_theirs
    return {
        "dice": (2.0 * both / total) if total else 0.0,
        "both_voxels": both,
        "only_ours_voxels": only_ours,
        "only_theirs_voxels": only_theirs,
        "both_mm3": both * voxel_mm3,
        "only_ours_mm3": only_ours * voxel_mm3,
        "only_theirs_mm3": only_theirs * voxel_mm3,
    }


def components(mask: np.ndarray) -> list[dict[str, object]]:
    """Separate blobs in a binary mask, by 6-connectivity.

    A detector's output is a set of candidate lesions, not one region, and
    "how many did it find and where" is the question being asked of it.
    Written out rather than imported: scipy is not a dependency of this tool.
    """

    mask = np.asanyarray(mask).astype(bool)
    if not mask.any():
        return []
    remaining = mask.copy()
    shape = mask.shape
    found: list[dict[str, object]] = []
    while True:
        seeds = np.argwhere(remaining)
        if not len(seeds):
            break
        stack = [tuple(int(v) for v in seeds[0])]
        remaining[stack[0]] = False
        voxels = []
        while stack:
            current = stack.pop()
            voxels.append(current)
            for axis in range(3):
                for step in (-1, 1):
                    neighbour = list(current)
                    neighbour[axis] += step
                    if not (0 <= neighbour[axis] < shape[axis]):
                        continue
                    spot = tuple(neighbour)
                    if remaining[spot]:
                        remaining[spot] = False
                        stack.append(spot)
        block = np.array(voxels, dtype=int)
        # The coordinates are kept, not just the count: drawing one detection
        # as its own surface needs them, and finding it again by flood-filling
        # from the centre fails for anything not shaped like a ball.
        found.append(
            {
                "voxels": len(voxels),
                "centre": block.mean(axis=0),
                "coords": block,
            }
        )
    found.sort(key=lambda item: -int(item["voxels"]))
    return found
