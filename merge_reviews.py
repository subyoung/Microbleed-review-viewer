"""Combine per-reader review databases into one, and export the result.

The safe arrangement when the review folder is synchronised (OneDrive, a
network share) rather than served by a real database is: every reader works in
their own SQLite file, and the files are merged afterwards.  Concurrent writers
on one synchronised SQLite file can corrupt it or lose a reader's work; two
readers writing two files cannot.

Usage
-----

Merge two readers into a fresh aggregate database and export it::

    python viewer\\merge_reviews.py --target merged.sqlite ^
        --source reviews_ziyang.sqlite --source reviews_ella.sqlite ^
        --export merged_reviews.xlsx

The target may be an existing aggregate; merging is repeatable, and a review
round that would collide with one already in the target is renumbered instead
of overwriting anybody's work.  ``--export`` alone (no sources) just writes the
analysis table for a database that is already complete.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from review_store import export_reviews, initialize_store, merge_stores  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge per-reader review databases and export the combined results.",
    )
    parser.add_argument("--target", required=True, type=Path, help="Aggregate database to write into.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        type=Path,
        help="A reader's review database. Repeat for each reader.",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        help="Findings workbook, needed only when the target database does not exist yet.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="MRI data folder, used with --workbook when creating the target database.",
    )
    parser.add_argument("--export", type=Path, help="Write the combined results to this .xlsx or .csv.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = Path(args.target)

    if not target.exists():
        if not args.workbook:
            print(
                f"{target} does not exist yet. Pass --workbook (and optionally --data-root) "
                "so the findings can be imported into the new database first.",
                file=sys.stderr,
            )
            return 2
        data_root = args.data_root or Path(args.workbook).parent
        report = initialize_store(args.workbook, data_root, target)
        print(f"Created {target}: {report['source_count']} findings, {report['case_count']} cases.")

    missing = [str(path) for path in args.source if not Path(path).exists()]
    if missing:
        print("These review databases were not found:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2

    if args.source:
        summary = merge_stores(target, args.source)
        for entry in summary["sources"]:
            print(
                f"{entry['path']}: {entry['reviews_added']} reviews, "
                f"{entry['rois_added']} segmentations, "
                f"{entry['manual_added']} manual findings, {entry['sessions_added']} sessions"
                + (f", {entry['rounds_renumbered']} rounds renumbered" if entry["rounds_renumbered"] else "")
                + (f", {entry['reviews_skipped']} already newer in the target" if entry["reviews_skipped"] else "")
            )
        print(
            f"Merged into {target}: {summary['reviews_added']} reviews, "
            f"{summary['rois_added']} segmentations, "
            f"{summary['manual_added']} manual findings."
        )
        if summary["rois_added"]:
            print(
                "  Segmentation masks were copied into "
                f"{Path(target).parent / 'labels'}; the exported paths point there."
            )

    if args.export:
        result = export_reviews(target, args.export)
        print(
            f"Exported {result['findings']} findings and {result['reader_reports']} reader reports "
            f"from {result['readers']} readers to {result['path']} "
            f"({result['disagreements']} disagreements)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
