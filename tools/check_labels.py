"""Check that exported segmentations line up with the images they belong to.

Written for the question every training pipeline eventually asks: can I trust
that a mask voxel and an image voxel with the same index are the same piece of
brain? Headers can disagree in ways a viewer hides -- it may be resampling to
world coordinates, so a mask that overlays perfectly on screen can still be
stored against a different geometry.

    python tools/check_labels.py --database review.sqlite --data-root Data

It checks four things per stored mask:

  geometry   the label's affine and shape equal the image's, field by field
  scaling    no slope or intercept that a loader would apply to the integers
  content    the label value the database records is actually in the file
  position   the mask's centroid, in world coordinates, is near the finding
             it belongs to

Exits non-zero if anything fails, so it can sit in front of a training run.
"""
from __future__ import annotations

import argparse
import glob
import sqlite3
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import dataset_config  # noqa: E402
import review_store  # noqa: E402


def sequences_for(case_id: str, data_root: Path, config: dict) -> dict[str, Path]:
    found = {}
    for key, entry in config["sequences"].items():
        matches = sorted(Path(data_root, case_id).glob("*" + entry["suffix"]))
        if matches:
            found[key] = matches[0]
    return found


def target_position(db: sqlite3.Connection, row: sqlite3.Row) -> np.ndarray | None:
    own = db.execute(
        "SELECT ras_l, ras_p, ras_s FROM review_annotations "
        "WHERE target_id=? AND reader_id=? AND review_round=? AND ras_l IS NOT NULL",
        (row["target_id"], row["reader_id"], row["review_round"]),
    ).fetchone()
    if own is not None:
        return np.array([own[0], own[1], own[2]], dtype=float)
    for table in ("source_microbleeds", "manual_annotations"):
        try:
            hit = db.execute(
                "SELECT ras_l, ras_p, ras_s FROM " + table + " WHERE target_id=?",
                (row["target_id"],),
            ).fetchone()
        except sqlite3.OperationalError:
            continue
        if hit is not None:
            return np.array([hit[0], hit[1], hit[2]], dtype=float)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, help="the review database")
    parser.add_argument("--data-root", required=True, help="the folder of case folders")
    parser.add_argument("--config", default=None, help="config.json, if not the default")
    parser.add_argument(
        "--tolerance-mm",
        type=float,
        default=12.0,
        help="how far a mask centroid may sit from its finding before it is reported",
    )
    args = parser.parse_args()

    config = dataset_config.load(args.config)
    data_root = Path(args.data_root)
    db = sqlite3.connect(args.database)
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT * FROM roi_labels").fetchall()
    if not rows:
        print("No segmentations recorded in that database.")
        return 0

    problems: list[str] = []
    distances: list[float] = []
    missing = 0
    for row in rows:
        path = review_store.resolve_label_path(Path(args.database), row)
        if not path.exists():
            missing += 1
            problems.append(str(row["case_id"]) + " " + str(row["target_id"])
                            + ": no label file at " + str(row["path"]))
            continue
        label = nib.load(str(path))
        name = path.name + " value " + str(row["label_value"])

        slope, inter = label.header.get_slope_inter()
        if (slope not in (None, 1.0)) or (inter not in (None, 0.0)):
            problems.append(name + ": stored with slope " + str(slope) + " intercept "
                            + str(inter) + " -- a loader would rescale the labels")

        for key, image_path in sequences_for(row["case_id"], data_root, config).items():
            image = nib.load(str(image_path))
            if image.shape[:3] != label.shape[:3]:
                problems.append(name + ": shape " + str(label.shape[:3]) + " against "
                                + key + " " + str(image.shape[:3]))
                continue
            gap = float(np.abs(np.asarray(label.affine) - np.asarray(image.affine)).max())
            if gap > 1e-4:
                problems.append(name + ": affine differs from " + key + " by " + str(round(gap, 4)))

        data = np.asanyarray(label.dataobj)
        voxels = np.argwhere(data == int(row["label_value"]))
        if not len(voxels):
            problems.append(name + ": that label value is not present in the file")
            continue
        world = label.affine[:3, :3] @ voxels.mean(axis=0) + label.affine[:3, 3]
        target = target_position(db, row)
        if target is not None:
            distance = float(np.linalg.norm(world - target))
            distances.append(distance)
            if distance > args.tolerance_mm:
                problems.append(name + ": centroid is " + str(round(distance, 1))
                                + " mm from the finding it belongs to")

    print(str(len(rows)) + " segmentations recorded, " + str(missing) + " files missing")
    if distances:
        array = np.array(distances)
        print("centroid to finding: median " + str(round(float(np.median(array)), 2))
              + " mm, worst " + str(round(float(array.max()), 2)) + " mm")
    print(str(len(problems)) + " problems")
    for line in problems[:40]:
        print("   " + line)
    if len(problems) > 40:
        print("   ... and " + str(len(problems) - 40) + " more")
    db.close()
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
