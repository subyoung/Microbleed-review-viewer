"""Remove a reader from a review database.

Test readers accumulate.  A trial run, a prototype, a colleague who opened the
tool once to look -- each becomes a reader, and the export gives every reader
its own block of columns and counts it in the agreement between readers.  One
left over from an early prototype was fifteen of the fifty columns in the
findings sheet, and it agreed with the real reader on a finding it had never
seen.

    python tools/remove_reader.py --database review.sqlite --reader "Test QA"
    python tools/remove_reader.py --database review.sqlite --reader "Test QA" --apply

Nothing is written without --apply, and the database is copied beside itself
first.  Findings that reader added by hand are reported rather than removed:
somebody else may have reviewed them, and that is a judgement call rather than
a cleanup.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import review_store  # noqa: E402

# Every table that records who did something, and the column that says so.
OWNED = (
    ("review_annotations", "reader_id"),
    ("roi_labels", "reader_id"),
    ("reader_sessions", "reader_id"),
    ("operation_log", "reader_id"),
    ("reader_profiles", "reader_id"),
)


def counts(db: sqlite3.Connection, reader: str) -> list[tuple[str, int]]:
    found = []
    for table, column in OWNED:
        try:
            n = db.execute(
                "SELECT COUNT(*) FROM " + table + " WHERE " + column + " = ?", (reader,)
            ).fetchone()[0]
        except sqlite3.OperationalError:
            continue
        found.append((table, n))
    return found


def label_files(db: sqlite3.Connection, db_path: Path, reader: str) -> list[Path]:
    try:
        rows = db.execute("SELECT * FROM roi_labels WHERE reader_id = ?", (reader,)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [review_store.resolve_label_path(db_path, row) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--reader", help="the reader_id to remove")
    parser.add_argument("--apply", action="store_true", help="actually write the change")
    parser.add_argument(
        "--list", action="store_true", help="just show who is in this database"
    )
    args = parser.parse_args()

    db_path = Path(args.database)
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    readers = [
        (row[0], row[1])
        for row in db.execute(
            "SELECT reader_id, COUNT(*) FROM review_annotations GROUP BY reader_id"
        )
    ]
    if args.list:
        print("readers with reviews in " + str(db_path) + ":")
        for name, n in readers:
            print("   " + str(n).rjust(5) + "  " + name)
        return 0

    if not args.reader:
        parser.error("--reader is required unless you asked for --list")
    present = counts(db, args.reader)
    total = sum(n for _table, n in present)
    if not total:
        print("Nothing recorded for " + repr(args.reader) + ".")
        print("Readers in this database: " + ", ".join(name for name, _n in readers))
        return 1

    print("Rows belonging to " + repr(args.reader) + ":")
    for table, n in present:
        print("   " + str(n).rjust(5) + "  " + table)

    manual = db.execute(
        "SELECT target_id, case_id FROM manual_annotations WHERE created_by = ?",
        (args.reader,),
    ).fetchall()
    if manual:
        print()
        print("It also added " + str(len(manual)) + " finding(s) by hand, which are left alone:")
        for row in manual[:10]:
            others = db.execute(
                "SELECT COUNT(*) FROM review_annotations WHERE target_id = ? AND reader_id <> ?",
                (row["target_id"], args.reader),
            ).fetchone()[0]
            print("   " + row["case_id"] + " " + row["target_id"]
                  + " (" + str(others) + " other reader review(s))")
        print("   Remove those yourself if they are not real findings.")

    files = [path for path in label_files(db, db_path, args.reader) if path.is_file()]
    if files:
        print()
        print(str(len(files)) + " label file(s) would be deleted:")
        for path in files[:10]:
            print("   " + str(path))

    if not args.apply:
        print()
        print("Nothing written.  Add --apply to make the change.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = db_path.with_name(db_path.stem + ".before-" + stamp + db_path.suffix)
    shutil.copy2(db_path, backup)
    print()
    print("backed up to " + backup.name)

    removed = 0
    for table, column in OWNED:
        try:
            cursor = db.execute(
                "DELETE FROM " + table + " WHERE " + column + " = ?", (args.reader,)
            )
        except sqlite3.OperationalError:
            continue
        removed += cursor.rowcount
    db.commit()
    for path in files:
        path.unlink(missing_ok=True)
    db.close()
    print("deleted " + str(removed) + " row(s) and " + str(len(files)) + " label file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
