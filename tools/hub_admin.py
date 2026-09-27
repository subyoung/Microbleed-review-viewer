"""Set up and look after a shared review folder (a "hub") on the lab NAS.

    python tools/hub_admin.py init <hub> --from-db review.sqlite --workbook findings.xlsx --data-root Data
    python tools/hub_admin.py list <hub>
    python tools/hub_admin.py check <hub>
    python tools/hub_admin.py reset-password <hub> "Reader Name"
    python tools/hub_admin.py export <hub> results.xlsx [--keep-database combined.sqlite]

``init`` turns an existing single-database study into a hub: the workbook is
copied in, every reader in the database gets a profile, a published snapshot of
their own work and a copy of their masks.  Nobody has a password yet; each
reader sets one the first time they open the viewer.  The MRI folder is not
copied -- that is tens of gigabytes and a job for Explorer or robocopy -- but
it is checked, and the hub records where it is.  The source database is only
read.

``reset-password`` is for a reader who forgot theirs: their work is untouched
and they choose a new one at their next login.

``export`` builds the results table from everybody's published work.  It needs
nobody's password: it only reads.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import dataset_config  # noqa: E402
import hub  # noqa: E402
import review_store  # noqa: E402


# What counts as work.  Somebody who only opened a session -- a trial login,
# a test reader -- would otherwise become a reader everyone has to scroll past.
WORK_TABLES = (
    ("review_annotations", "reader_id"),
    ("roi_labels", "reader_id"),
    ("manual_annotations", "created_by"),
)


def readers_in(db_path: Path, tables=WORK_TABLES) -> list[str]:
    """Everybody who did any work in this database."""

    connection = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        names: set[str] = set()
        for table, column in tables:
            try:
                names.update(
                    str(row[0])
                    for row in connection.execute(
                        f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL"  # noqa: S608
                    )
                    if str(row[0]).strip()
                )
            except sqlite3.OperationalError:
                continue
        return sorted(names, key=str.casefold)
    finally:
        connection.close()


def shareable_config(path: Path | None) -> dict:
    """The dataset's shape without this PC's paths."""

    config = dict(dataset_config.load(path))
    config.pop("paths", None)
    return config


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.hub)
    source_db = Path(args.from_db)
    workbook = Path(args.workbook)
    data_root = Path(args.data_root)
    for label, path, check in (
        ("review database", source_db, Path.is_file),
        ("findings workbook", workbook, Path.is_file),
        ("MRI folder", data_root, Path.is_dir),
    ):
        if not check(path):
            print(f"The {label} was not found: {path}", file=sys.stderr)
            return 2
    if hub.is_hub(root):
        print(f"{root} is already a shared review folder; nothing was changed.", file=sys.stderr)
        return 2

    config = shareable_config(Path(args.config) if args.config else None)
    review_store.configure(config)
    root.mkdir(parents=True, exist_ok=True)
    shared_workbook = root / "findings.xlsx"
    hub.atomic_copy(workbook, shared_workbook)
    shared = hub.Hub.create(root, workbook=shared_workbook, data_root=data_root, dataset_config=config)
    try:
        data_root.resolve().relative_to(root.resolve())
    except ValueError:
        print(
            f"Note: the MRI folder {data_root} is outside the shared folder. Every PC must "
            "reach it at this same path; copying it into the shared folder avoids that."
        )

    names = readers_in(source_db)
    idle = sorted(set(readers_in(source_db, review_store.READER_TABLES)) - set(names), key=str.casefold)
    if idle:
        print(f"  left out (no reviews, masks or added findings): {', '.join(idle)}")
    with tempfile.TemporaryDirectory(prefix="microbleed_hub_init_") as scratch:
        for name in names:
            hub.create_reader(shared, name, None)
            snapshot = Path(scratch) / f"{review_store.safe_reader_name(name)}.sqlite"
            counts = review_store.export_reader_snapshot(source_db, name, snapshot, revision=1)
            masks = copy_masks(source_db, name, shared.labels_dir(name))
            hub.atomic_copy(snapshot, shared.snapshot_path(name))
            hub.atomic_write_json(
                shared.state_path(name),
                {"reader_id": name, "revision": 1, "published_at": hub._iso(hub.utc_now()), "machine": "hub_admin init"},
            )
            print(
                f"  {name}: {counts['reviews']} reviews, {counts['rois']} segmentations "
                f"({masks} mask files), {counts['manual']} added findings"
            )
    print(f"Shared folder ready at {root} with {len(names)} readers.")
    print("Each reader sets a password the first time they open it in the viewer.")
    return 0


def copy_masks(db_path: Path, reader: str, target: Path) -> int:
    """Copy a reader's mask files, under the names the hub expects."""

    connection = review_store.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT DISTINCT case_id, reader_id, review_round, path FROM roi_labels WHERE reader_id = ?",
            (reader,),
        ).fetchall()
    finally:
        connection.close()
    copied = 0
    for row in rows:
        source = review_store.resolve_label_path(db_path, dict(row))
        if not source.is_file():
            print(f"  warning: {reader}'s mask {source} is missing", file=sys.stderr)
            continue
        canonical = review_store.label_path(db_path, str(row["case_id"]), reader, int(row["review_round"]))
        hub.atomic_copy(source, target / canonical.name)
        copied += 1
    return copied


def cmd_list(args: argparse.Namespace) -> int:
    shared = hub.Hub.open(args.hub)
    print(f"Shared folder {shared.root}")
    print(f"  workbook  {shared.workbook}")
    print(f"  MRI       {shared.data_root}")
    for entry in hub.list_readers(shared):
        state = entry.get("state") or {}
        lock = entry.get("lock")
        in_use = ""
        if lock and hub.lock_is_fresh(lock):
            in_use = f"  open on {lock.get('machine')}"
        print(
            f"  {entry['reader_id']:<24} revision {state.get('revision', '-'):<4} "
            f"published {state.get('published_at', 'never')}"
            f"{'' if entry['has_password'] else '  (no password yet)'}{in_use}"
        )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Everything a sign-in will need from the shared folder, and whether it is there."""

    shared = hub.Hub.open(args.hub)
    problems: list[str] = []

    def report(ok: bool, what: str, where: Path | str) -> None:
        print(f"  {'OK     ' if ok else 'MISSING'}  {what}: {where}")
        if not ok:
            problems.append(f"{what}: {where}")

    print(f"Shared folder {shared.root}")
    report(shared.workbook.is_file(), "findings workbook", shared.workbook)
    report(shared.data_root.is_dir(), "MRI folder", shared.data_root)
    readers = hub.list_readers(shared)
    if not readers:
        print("  (no readers yet)")
    for entry in readers:
        name = str(entry["reader_id"])
        snapshot = shared.snapshot_path(name)
        if not entry.get("state"):
            print(f"  -        {name}: nothing published yet")
            continue
        report(snapshot.is_file(), f"{name}'s reviews", snapshot)
        if not snapshot.is_file():
            continue
        connection = sqlite3.connect(f"file:{snapshot.as_posix()}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT DISTINCT case_id, review_round FROM roi_labels WHERE reader_id = ?", (name,)
            ).fetchall()
        finally:
            connection.close()
        missing = [
            shared.labels_dir(name) / f"{case_id}_round{int(review_round)}.nii.gz"
            for case_id, review_round in rows
            if not (shared.labels_dir(name) / f"{case_id}_round{int(review_round)}.nii.gz").is_file()
        ]
        report(not missing, f"{name}'s masks ({len(rows) - len(missing)} of {len(rows)} files)", shared.labels_dir(name))
        for path in missing[:20]:
            print(f"             missing {path.name}")
        if len(missing) > 20:
            print(f"             ... and {len(missing) - 20} more")
    if problems:
        print(f"{len(problems)} problem(s). hub.json in {shared.root} says where each of these should be.")
        return 1
    print("Everything a sign-in needs is there.")
    return 0


def cmd_reset_password(args: argparse.Namespace) -> int:
    shared = hub.Hub.open(args.hub)
    hub.reset_password(shared, args.reader)
    print(f"{args.reader}'s password is cleared; they set a new one at their next login.")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    shared = hub.Hub.open(args.hub)
    if shared.dataset_config:
        review_store.configure(shared.dataset_config)
    report = hub.export_all(
        shared,
        Path(args.out),
        keep_database=Path(args.keep_database) if args.keep_database else None,
    )
    print(
        f"Wrote {args.out}: {report.get('readers', 0)} readers, "
        f"{report.get('reader_reports', 0)} reports, {report.get('disagreements', 0)} disagreements."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="make a shared folder from an existing study")
    init.add_argument("hub")
    init.add_argument("--from-db", required=True, help="the existing review database (only read)")
    init.add_argument("--workbook", required=True, help="the findings workbook (copied into the hub)")
    init.add_argument("--data-root", required=True, help="the MRI folder every PC will read")
    init.add_argument("--config", help="config.json describing the dataset (default: the viewer's)")
    init.set_defaults(run=cmd_init)

    listing = commands.add_parser("list", help="readers, revisions and who has them open")
    listing.add_argument("hub")
    listing.set_defaults(run=cmd_list)

    check = commands.add_parser("check", help="is everything a sign-in needs actually there")
    check.add_argument("hub")
    check.set_defaults(run=cmd_check)

    reset = commands.add_parser("reset-password", help="let a reader choose a new password")
    reset.add_argument("hub")
    reset.add_argument("reader")
    reset.set_defaults(run=cmd_reset_password)

    export = commands.add_parser("export", help="everybody's results in one table")
    export.add_argument("hub")
    export.add_argument("out", help=".xlsx or .csv")
    export.add_argument("--keep-database", help="also keep the combined SQLite here")
    export.set_defaults(run=cmd_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.run(args))
    except hub.HubError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
