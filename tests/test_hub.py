"""Readers sharing one NAS folder, each writing only their own files.

None of this needs the study's data: the store only needs a findings
workbook, and ``examples/example_findings.xlsx`` is one.  The "NAS" is a
temporary folder.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIEWER_DIR))

import hub  # noqa: E402
import review_store  # noqa: E402
from review_store import (  # noqa: E402
    ReadOnlyError,
    add_manual_annotation,
    clear_write_guard,
    close_session,
    delete_manual_annotation,
    export_reader_snapshot,
    import_reader_snapshot,
    initialize_store,
    list_cases,
    list_targets,
    log_event,
    register_reader,
    resolve_label_path,
    save_review,
    save_roi,
    save_session_state,
    set_label_search_roots,
    set_write_guard,
    set_write_listener,
    start_new_session,
)

EXAMPLE_WORKBOOK = VIEWER_DIR / "examples" / "example_findings.xlsx"


class StoreFixture(unittest.TestCase):
    """A temporary study: one workbook, no images, any number of stores."""

    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="microbleed_hub_test_"))
        self.data_root = self.temp / "Data"
        self.data_root.mkdir()
        self.addCleanup(shutil.rmtree, self.temp, True)
        self.addCleanup(clear_write_guard)
        self.addCleanup(set_write_listener, None)
        self.addCleanup(set_label_search_roots, [])

    def new_store(self, name: str) -> Path:
        db_path = self.temp / name / "work.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        initialize_store(EXAMPLE_WORKBOOK, self.data_root, db_path)
        return db_path

    def first_case(self, db_path: Path) -> str:
        return str(list_cases(db_path, "x", 1)[0]["case_id"])

    def targets(self, db_path: Path, case_id: str) -> list[str]:
        return [str(item["target_id"]) for item in list_targets(db_path, case_id, "x", 1)]

    def rows(self, db_path: Path, sql: str, *args) -> list[sqlite3.Row]:
        connection = sqlite3.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        try:
            return connection.execute(sql, args).fetchall()
        finally:
            connection.close()

    def review(self, db_path: Path, reader: str, target: str, case: str, verify: int = 1) -> None:
        save_review(
            db_path, target_id=target, case_id=case, reader_id=reader,
            review_round=1, verify=verify, comment=f"{reader} says {verify}",
        )


class SnapshotStoreTests(StoreFixture):
    def test_snapshot_holds_only_that_reader(self) -> None:
        db = self.new_store("a")
        case = self.first_case(db)
        target = self.targets(db, case)[0]
        start_new_session(db, "Reader A")
        start_new_session(db, "Reader B")
        self.review(db, "Reader A", target, case, 1)
        self.review(db, "Reader B", target, case, 0)
        add_manual_annotation(db, case_id=case, ras=(1, 2, 3), reader_id="Reader B", review_round=1)
        snapshot = self.temp / "a.snapshot.sqlite"

        counts = export_reader_snapshot(db, "Reader A", snapshot, revision=4)

        self.assertEqual(counts["reviews"], 1)
        readers = {row["reader_id"] for row in self.rows(snapshot, "SELECT reader_id FROM review_annotations")}
        self.assertEqual(readers, {"Reader A"})
        self.assertEqual(self.rows(snapshot, "SELECT COUNT(*) AS n FROM manual_annotations")[0]["n"], 0)
        self.assertEqual(self.rows(snapshot, "SELECT COUNT(*) AS n FROM source_microbleeds")[0]["n"], 0)
        log_readers = {row["reader_id"] for row in self.rows(snapshot, "SELECT reader_id FROM operation_log")}
        self.assertEqual(log_readers, {"Reader A"})
        meta = {row["key"]: row["value"] for row in self.rows(snapshot, "SELECT key, value FROM meta")}
        self.assertEqual(meta["snapshot_reader"], "Reader A")
        self.assertEqual(meta["revision"], "4")

    def test_import_replaces_only_that_reader(self) -> None:
        mine = self.new_store("a")
        theirs = self.new_store("b")
        case = self.first_case(mine)
        first, second = self.targets(mine, case)[:2]
        start_new_session(mine, "Reader A")
        self.review(mine, "Reader A", first, case, 1)
        start_new_session(theirs, "Reader B")
        self.review(theirs, "Reader B", first, case, 0)
        snapshot = self.temp / "b.snapshot.sqlite"
        export_reader_snapshot(theirs, "Reader B", snapshot, revision=1)

        import_reader_snapshot(mine, snapshot)
        # B changes their mind and reviews a second finding; import again.
        self.review(theirs, "Reader B", first, case, 1)
        self.review(theirs, "Reader B", second, case, 1)
        export_reader_snapshot(theirs, "Reader B", snapshot, revision=2)
        result = import_reader_snapshot(mine, snapshot)

        self.assertEqual(result["reader_id"], "Reader B")
        self.assertEqual(result["revision"], 2)
        b_rows = self.rows(mine, "SELECT target_id, verify FROM review_annotations WHERE reader_id = 'Reader B'")
        self.assertEqual({(row["target_id"], row["verify"]) for row in b_rows}, {(first, 1), (second, 1)})
        a_rows = self.rows(mine, "SELECT target_id, verify FROM review_annotations WHERE reader_id = 'Reader A'")
        self.assertEqual([(row["target_id"], row["verify"]) for row in a_rows], [(first, 1)])
        # Importing twice must not double B's log.
        b_log = self.rows(theirs, "SELECT COUNT(*) AS n FROM operation_log WHERE reader_id = 'Reader B'")[0]["n"]
        imported = self.rows(mine, "SELECT COUNT(*) AS n FROM operation_log WHERE reader_id = 'Reader B'")[0]["n"]
        self.assertEqual(imported, b_log)
        # A's own log is still A's own (not tagged as imported).
        own = self.rows(
            mine, "SELECT COUNT(*) AS n FROM operation_log WHERE reader_id = 'Reader A' AND origin_store_id IS NULL"
        )[0]["n"]
        self.assertGreater(own, 0)

    def test_import_removes_rows_the_reader_deleted(self) -> None:
        mine = self.new_store("a")
        theirs = self.new_store("b")
        case = self.first_case(mine)
        target = add_manual_annotation(theirs, case_id=case, ras=(1, 2, 3), reader_id="Reader B", review_round=1)
        snapshot = self.temp / "b.snapshot.sqlite"
        export_reader_snapshot(theirs, "Reader B", snapshot, revision=1)
        import_reader_snapshot(mine, snapshot)
        self.assertIn(target, self.targets(mine, case))

        delete_manual_annotation(theirs, target_id=target, reader_id="Reader B")
        export_reader_snapshot(theirs, "Reader B", snapshot, revision=2)
        import_reader_snapshot(mine, snapshot)

        self.assertNotIn(target, self.targets(mine, case))

    def test_own_import_restores_rows_as_this_stores_own(self) -> None:
        old = self.new_store("old")
        new = self.new_store("new")
        case = self.first_case(old)
        target = self.targets(old, case)[0]
        start_new_session(old, "Reader A")
        self.review(old, "Reader A", target, case, 1)
        snapshot = self.temp / "a.snapshot.sqlite"
        export_reader_snapshot(old, "Reader A", snapshot, revision=3)

        import_reader_snapshot(new, snapshot, own=True)

        tagged = self.rows(new, "SELECT COUNT(*) AS n FROM operation_log WHERE origin_store_id IS NOT NULL")[0]["n"]
        self.assertEqual(tagged, 0)
        again = self.temp / "again.sqlite"
        counts = export_reader_snapshot(new, "Reader A", again, revision=4)
        self.assertEqual(counts["reviews"], 1)
        self.assertGreater(counts["log"], 0)

    def test_write_guard_refuses_other_names_and_read_only(self) -> None:
        db = self.new_store("a")
        case = self.first_case(db)
        target = self.targets(db, case)[0]
        session = start_new_session(db, "Reader A")
        set_write_guard("Reader A")

        # Own writes are fine.
        self.review(db, "Reader A", target, case, 1)
        attempts = {
            "register": lambda: register_reader(db, "Reader B"),
            "session": lambda: start_new_session(db, "Reader B"),
            "review": lambda: self.review(db, "Reader B", target, case, 1),
            "manual": lambda: add_manual_annotation(db, case_id=case, ras=(1, 2, 3), reader_id="Reader B", review_round=1),
            "roi": lambda: save_roi(
                db, target_id=target, case_id=case, reader_id="Reader B", review_round=1,
                label_value=1, path=self.temp / "x.nii.gz", voxel_count=3, volume_mm3=1.0, generated_from="swi",
            ),
            "log": lambda: log_event(db, "x", reader_id="Reader B"),
        }
        for name, attempt in attempts.items():
            with self.subTest(name=name), self.assertRaises(ReadOnlyError):
                attempt()

        set_write_guard(None)
        read_only_attempts = dict(attempts)
        read_only_attempts.update(
            {
                "own review": lambda: self.review(db, "Reader A", target, case, 1),
                "close": lambda: close_session(db, session["session_id"]),
                "state": lambda: save_session_state(db, session["session_id"], {}),
            }
        )
        for name, attempt in read_only_attempts.items():
            with self.subTest(read_only=name), self.assertRaises(ReadOnlyError):
                attempt()

    def test_deleting_a_manual_finding_is_guarded(self) -> None:
        db = self.new_store("a")
        case = self.first_case(db)
        target = add_manual_annotation(db, case_id=case, ras=(1, 2, 3), reader_id="Reader B", review_round=1)
        set_write_guard("Reader A")
        with self.assertRaises(ReadOnlyError):
            delete_manual_annotation(db, target_id=target, reader_id="Reader B")

    def test_listener_fires_after_saves(self) -> None:
        db = self.new_store("a")
        case = self.first_case(db)
        target = self.targets(db, case)[0]
        calls: list[int] = []
        set_write_listener(lambda: calls.append(1))

        start_new_session(db, "Reader A")
        after_session = len(calls)
        self.review(db, "Reader A", target, case, 1)

        self.assertGreater(after_session, 0)
        self.assertGreater(len(calls), after_session)

    def test_label_path_falls_back_to_search_root(self) -> None:
        db = self.new_store("a")
        hub = self.temp / "hub"
        stored = hub / "labels" / "Reader_B" / "CASE_round1.nii.gz"
        stored.parent.mkdir(parents=True)
        stored.write_bytes(b"x")
        row = {"path": "labels/Reader_B/CASE_round1.nii.gz", "case_id": "CASE", "reader_id": "Reader B", "review_round": 1}

        self.assertNotEqual(resolve_label_path(db, row), stored)
        set_label_search_roots([hub])
        self.assertEqual(resolve_label_path(db, row), stored)

    def test_safe_reader_name_is_the_label_folder_rule(self) -> None:
        self.assertEqual(review_store.safe_reader_name("Alex Smith"), "Alex_Smith")
        self.assertEqual(review_store.label_directory(Path("x/db.sqlite"), "Alex Smith").name, "Alex_Smith")


class InventoryCostTests(StoreFixture):
    """Scanning the MRI folder happens at every start, over SMB on a NAS."""

    def test_each_case_folder_is_listed_once_and_no_file_is_stat_ed(self) -> None:
        from unittest import mock

        db = self.new_store("a")
        cases = [str(row["case_id"]) for row in list_cases(db, "x", 1)]
        for case in cases:
            folder = self.data_root / case
            folder.mkdir()
            for suffix in [spec["suffixes"][0] for spec in review_store.MODALITY_SPECS.values()] + [
                f"_raw{i}.nii.gz" for i in range(15)
            ]:
                (folder / f"{case}{suffix}").write_bytes(b"")
        listings: list[str] = []
        real_scandir = os.scandir

        def counting_scandir(path="."):
            listings.append(str(path))
            return real_scandir(path)

        stats: list[str] = []
        real_is_file = Path.is_file

        def counting_is_file(self_path, *args, **kwargs):
            if str(self_path).startswith(str(self.data_root)) and self_path.parent != self.data_root:
                stats.append(str(self_path))
            return real_is_file(self_path, *args, **kwargs)

        connection = review_store.connect(db)
        try:
            with mock.patch("os.scandir", counting_scandir), mock.patch.object(Path, "is_file", counting_is_file),                     mock.patch.object(Path, "iterdir", side_effect=AssertionError("use one scandir per folder")):
                counts = review_store.refresh_inventory(connection, self.data_root)
        finally:
            connection.close()

        self.assertEqual(counts["complete"], len(cases))
        self.assertEqual(stats, [])
        per_folder = {path: listings.count(path) for path in listings}
        self.assertTrue(all(n == 1 for n in per_folder.values()), per_folder)


class HubLayoutTests(StoreFixture):
    def test_create_and_open_round_trip_with_relative_paths(self) -> None:
        root = self.temp / "nas"
        root.mkdir()
        (root / "Data").mkdir()
        shutil.copy2(EXAMPLE_WORKBOOK, root / "findings.xlsx")

        created = hub.Hub.create(root, workbook=root / "findings.xlsx", data_root=root / "Data")
        # Moved to another drive letter: relative paths still resolve.
        moved = self.temp / "moved"
        shutil.move(str(root), str(moved))
        opened = hub.Hub.open(moved)

        self.assertEqual(opened.hub_id, created.hub_id)
        self.assertEqual(opened.workbook, moved / "findings.xlsx")
        self.assertEqual(opened.data_root, moved / "Data")
        self.assertTrue(hub.is_hub(moved))
        self.assertTrue((moved / "readers").is_dir())

    def test_a_folder_that_is_not_a_hub_is_refused(self) -> None:
        with self.assertRaises(hub.HubError) as caught:
            hub.Hub.open(self.temp)
        self.assertNotIsInstance(caught.exception, hub.HubUnavailable)

    def test_a_missing_folder_is_unavailable(self) -> None:
        with self.assertRaises(hub.HubUnavailable):
            hub.Hub.open(self.temp / "unplugged")

    def test_creating_over_an_existing_hub_is_refused(self) -> None:
        root = make_hub(self.temp)
        with self.assertRaises(hub.HubError):
            hub.Hub.create(root.root, workbook=root.workbook, data_root=root.data_root)


def make_hub(base: Path) -> "hub.Hub":
    root = base / "nas"
    root.mkdir(exist_ok=True)
    (root / "Data").mkdir(exist_ok=True)
    shutil.copy2(EXAMPLE_WORKBOOK, root / "findings.xlsx")
    return hub.Hub.create(root, workbook=root / "findings.xlsx", data_root=root / "Data")


class PasswordTests(StoreFixture):
    def setUp(self) -> None:
        super().setUp()
        self.hub = make_hub(self.temp)

    def test_password_is_checked_and_never_stored_in_clear(self) -> None:
        hub.create_reader(self.hub, "Alex Smith", "s3cret-pw")

        self.assertTrue(hub.verify_reader(self.hub, "Alex Smith", "s3cret-pw"))
        self.assertFalse(hub.verify_reader(self.hub, "Alex Smith", "wrong"))
        self.assertFalse(hub.verify_reader(self.hub, "Nobody", "s3cret-pw"))
        text = self.hub.profile_path("Alex Smith").read_text(encoding="utf-8")
        self.assertNotIn("s3cret-pw", text)
        self.assertIn("pbkdf2_sha256", text)

    def test_a_reader_without_a_password_must_set_one(self) -> None:
        hub.create_reader(self.hub, "Sam", None)
        self.assertFalse(hub.has_password(self.hub, "Sam"))
        self.assertFalse(hub.verify_reader(self.hub, "Sam", ""))
        hub.set_password(self.hub, "Sam", "pw-1234")
        self.assertTrue(hub.verify_reader(self.hub, "Sam", "pw-1234"))

    def test_reset_clears_only_the_password(self) -> None:
        hub.create_reader(self.hub, "Sam", "pw-1234")
        created = hub.get_profile(self.hub, "Sam")["created_at"]
        hub.reset_password(self.hub, "Sam")
        self.assertFalse(hub.has_password(self.hub, "Sam"))
        self.assertEqual(hub.get_profile(self.hub, "Sam")["created_at"], created)

    def test_reader_names_that_map_to_one_folder_are_refused(self) -> None:
        hub.create_reader(self.hub, "Alex Smith", "pw-1234")
        for clash in ("alex smith", "Alex_Smith", "ALEX SMITH"):
            with self.subTest(name=clash), self.assertRaises(hub.HubError):
                hub.create_reader(self.hub, clash, "pw-1234")
        with self.assertRaises(hub.HubError):
            hub.create_reader(self.hub, "   ", "pw-1234")

    def test_list_readers_names_everybody(self) -> None:
        hub.create_reader(self.hub, "Alex Smith", "pw-1234")
        hub.create_reader(self.hub, "Sam", None)
        names = [item["reader_id"] for item in hub.list_readers(self.hub)]
        self.assertEqual(sorted(names), ["Alex Smith", "Sam"])


class LockTests(StoreFixture):
    def setUp(self) -> None:
        super().setUp()
        self.hub = make_hub(self.temp)
        hub.create_reader(self.hub, "Sam", "pw-1234")
        from datetime import datetime, timedelta, timezone

        self.t0 = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
        self.later = lambda seconds: self.t0 + timedelta(seconds=seconds)

    def test_a_fresh_lock_on_another_machine_is_refused(self) -> None:
        hub.acquire_lock(self.hub, "Sam", machine="LAB-1", now=self.t0)
        with self.assertRaises(hub.LockHeld) as caught:
            hub.acquire_lock(self.hub, "Sam", machine="LAB-2", now=self.later(60))
        self.assertEqual(caught.exception.holder["machine"], "LAB-1")

    def test_a_stale_lock_can_be_taken_over_and_the_old_owner_learns_it(self) -> None:
        old = hub.acquire_lock(self.hub, "Sam", machine="LAB-1", now=self.t0)
        stale = self.later(hub.STALE_LOCK_SECONDS + 1)
        with self.assertRaises(hub.LockHeld):
            hub.acquire_lock(self.hub, "Sam", machine="LAB-2", now=stale)
        new = hub.acquire_lock(self.hub, "Sam", machine="LAB-2", force=True, now=stale)

        self.assertNotEqual(old, new)
        self.assertFalse(hub.heartbeat(self.hub, "Sam", old, now=stale))
        self.assertTrue(hub.heartbeat(self.hub, "Sam", new, now=stale))

    def test_force_does_not_take_a_fresh_lock(self) -> None:
        hub.acquire_lock(self.hub, "Sam", machine="LAB-1", now=self.t0)
        with self.assertRaises(hub.LockHeld):
            hub.acquire_lock(self.hub, "Sam", machine="LAB-2", force=True, now=self.later(30))

    def test_the_same_machine_may_reopen_after_a_crash(self) -> None:
        hub.acquire_lock(self.hub, "Sam", machine="LAB-1", now=self.t0)
        token = hub.acquire_lock(self.hub, "Sam", machine="LAB-1", now=self.later(10))
        self.assertTrue(hub.heartbeat(self.hub, "Sam", token, now=self.later(20)))

    def test_heartbeat_keeps_a_lock_fresh(self) -> None:
        token = hub.acquire_lock(self.hub, "Sam", machine="LAB-1", now=self.t0)
        hub.heartbeat(self.hub, "Sam", token, now=self.later(150))
        with self.assertRaises(hub.LockHeld):
            hub.acquire_lock(self.hub, "Sam", machine="LAB-2", now=self.later(300))

    def test_an_unplugged_share_is_offline_not_a_lost_lock(self) -> None:
        token = hub.acquire_lock(self.hub, "Sam", machine="LAB-1", now=self.t0)
        away = self.temp / "unplugged"
        self.hub.root.rename(away)
        try:
            with self.assertRaises(hub.HubUnavailable):
                hub.heartbeat(self.hub, "Sam", token, now=self.later(60))
            self.assertFalse(self.hub.root.exists())
        finally:
            away.rename(self.hub.root)

    def test_release_only_by_the_owner(self) -> None:
        token = hub.acquire_lock(self.hub, "Sam", machine="LAB-1", now=self.t0)
        hub.release_lock(self.hub, "Sam", "not-the-token")
        self.assertIsNotNone(hub.read_lock(self.hub, "Sam"))
        hub.release_lock(self.hub, "Sam", token)
        self.assertIsNone(hub.read_lock(self.hub, "Sam"))


class SyncFixture(StoreFixture):
    """A hub and a local folder per PC, both temporary."""

    def setUp(self) -> None:
        super().setUp()
        self.hub = make_hub(self.temp)
        previous = os.environ.get("MICROBLEED_LOCAL_ROOT")
        os.environ["MICROBLEED_LOCAL_ROOT"] = str(self.temp / "pc")
        self.addCleanup(self._restore_env, previous)
        for name in ("Reader A", "Reader B"):
            hub.create_reader(self.hub, name, "pw-1234")

    @staticmethod
    def _restore_env(previous: str | None) -> None:
        if previous is None:
            os.environ.pop("MICROBLEED_LOCAL_ROOT", None)
        else:
            os.environ["MICROBLEED_LOCAL_ROOT"] = previous

    def workspace(self, reader: str | None, pc: str = "pc1") -> "hub.Workspace":
        os.environ["MICROBLEED_LOCAL_ROOT"] = str(self.temp / pc)
        ws = hub.prepare_workspace(self.hub, reader)
        if reader:
            start_new_session(ws.work_db, reader)
        return ws

    def verdicts(self, ws: "hub.Workspace", reader: str) -> dict[str, int]:
        return {
            str(row["target_id"]): row["verify"]
            for row in self.rows(ws.work_db, "SELECT target_id, verify FROM review_annotations WHERE reader_id = ?", reader)
        }

    def write_mask(self, ws: "hub.Workspace", case: str, content: bytes = b"mask") -> Path:
        path = review_store.label_path(ws.work_db, case, ws.reader_id, 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path


class SyncTests(SyncFixture):
    def test_two_readers_see_each_other_and_only_replace_the_other(self) -> None:
        a = self.workspace("Reader A", "pc1")
        b = self.workspace("Reader B", "pc2")
        case = self.first_case(a.work_db)
        first, second = self.targets(a.work_db, case)[:2]

        self.review(a.work_db, "Reader A", first, case, 1)
        hub.publish(a)
        self.assertEqual(hub.pull_others(b), ["Reader A"])
        self.assertEqual(self.verdicts(b, "Reader A"), {first: 1})

        self.review(b.work_db, "Reader B", second, case, 0)
        hub.publish(b)
        self.assertEqual(hub.pull_others(a), ["Reader B"])
        self.assertEqual(self.verdicts(a, "Reader B"), {second: 0})
        self.assertEqual(self.verdicts(a, "Reader A"), {first: 1})
        # Nothing new: nothing imported.
        self.assertEqual(hub.pull_others(a), [])
        self.assertEqual(hub.pull_others(b), [])

    def test_publishing_marks_clean_and_bumps_the_revision(self) -> None:
        a = self.workspace("Reader A")
        case = self.first_case(a.work_db)
        target = self.targets(a.work_db, case)[0]
        self.review(a.work_db, "Reader A", target, case, 1)
        hub.mark_dirty(a)
        self.assertTrue(hub.is_dirty(a))

        first = hub.publish(a)
        second = hub.publish(a)

        self.assertFalse(hub.is_dirty(a))
        self.assertEqual(second, first + 1)
        self.assertEqual(hub.read_json(self.hub.state_path("Reader A"))["revision"], second)

    def test_a_write_during_publish_stays_dirty(self) -> None:
        a = self.workspace("Reader A")
        hub.mark_dirty(a)
        original = review_store.export_reader_snapshot

        def export_then_write(*args, **kwargs):
            counts = original(*args, **kwargs)
            hub.mark_dirty(a)  # the reader saved again while this was copying
            return counts

        with unittest.mock.patch("hub.export_reader_snapshot", export_then_write):
            hub.publish(a)
        self.assertTrue(hub.is_dirty(a))

    def test_publish_failure_keeps_dirty_across_restart(self) -> None:
        a = self.workspace("Reader A")
        hub.mark_dirty(a)
        away = self.temp / "unplugged"
        self.hub.root.rename(away)
        try:
            with self.assertRaises(hub.HubUnavailable):
                hub.publish(a)
            restarted = hub.Workspace(a.hub, "Reader A", a.root)
            self.assertTrue(hub.is_dirty(restarted))
        finally:
            away.rename(self.hub.root)
        hub.publish(restarted)
        self.assertFalse(hub.is_dirty(restarted))

    def test_a_snapshot_claiming_another_reader_is_ignored(self) -> None:
        a = self.workspace("Reader A", "pc1")
        b = self.workspace("Reader B", "pc2")
        case = self.first_case(a.work_db)
        target = self.targets(a.work_db, case)[0]
        self.review(a.work_db, "Reader A", target, case, 1)
        hub.publish(a)
        # Reader A's snapshot copied into B's folder: it must not arrive as B.
        shutil.copy2(self.hub.snapshot_path("Reader A"), self.hub.snapshot_path("Reader B"))
        hub.atomic_write_json(self.hub.state_path("Reader B"), {"revision": 9})
        c = self.workspace(None, "pc3")

        changed = hub.pull_others(c)

        self.assertEqual(changed, ["Reader A"])
        self.assertIn("Reader_B", c.last_pull_errors)

    def test_state_behind_snapshot_still_imports_newest(self) -> None:
        a = self.workspace("Reader A", "pc1")
        b = self.workspace("Reader B", "pc2")
        case = self.first_case(a.work_db)
        target = self.targets(a.work_db, case)[0]
        self.review(a.work_db, "Reader A", target, case, 1)
        hub.publish(a)
        hub.pull_others(b)
        # A crashes after replacing the snapshot but before writing state.json.
        self.review(a.work_db, "Reader A", target, case, 0)
        review_store.export_reader_snapshot(a.work_db, "Reader A", a.root / "x.sqlite", revision=2)
        hub.atomic_copy(a.root / "x.sqlite", self.hub.snapshot_path("Reader A"))
        # A comes back and saves again; its next publish must reach B.
        hub.publish(a)

        self.assertEqual(hub.pull_others(b), ["Reader A"])
        self.assertEqual(self.verdicts(b, "Reader A"), {target: 0})

    def test_masks_are_mirrored_both_ways_including_deletions(self) -> None:
        a = self.workspace("Reader A", "pc1")
        case = self.first_case(a.work_db)
        mask = self.write_mask(a, case)
        hub.publish(a)
        shared = self.hub.labels_dir("Reader A") / mask.name
        self.assertEqual(shared.read_bytes(), b"mask")

        mask.write_bytes(b"mask v2, longer")
        hub.publish(a)
        self.assertEqual(shared.read_bytes(), b"mask v2, longer")

        mask.unlink()
        hub.publish(a)
        self.assertFalse(shared.exists())

    def test_another_readers_mask_resolves_to_the_shared_copy(self) -> None:
        a = self.workspace("Reader A", "pc1")
        b = self.workspace("Reader B", "pc2")
        case = self.first_case(a.work_db)
        target = self.targets(a.work_db, case)[0]
        mask = self.write_mask(a, case)
        save_roi(
            a.work_db, target_id=target, case_id=case, reader_id="Reader A", review_round=1,
            label_value=1, path=mask, voxel_count=3, volume_mm3=1.0, generated_from="swi",
        )
        hub.publish(a)
        hub.pull_others(b)

        row = self.rows(b.work_db, "SELECT * FROM roi_labels WHERE reader_id = 'Reader A'")[0]
        self.assertEqual(resolve_label_path(b.work_db, dict(row)), self.hub.labels_dir("Reader A") / mask.name)

    def test_restore_pulls_newer_snapshot_and_masks(self) -> None:
        office = self.workspace("Reader A", "pc1")
        case = self.first_case(office.work_db)
        target = self.targets(office.work_db, case)[0]
        self.review(office.work_db, "Reader A", target, case, 1)
        hub.publish(office)
        lab = self.workspace("Reader A", "pc2")
        self.assertEqual(hub.restore_own(lab), {"restored": True, "backup": None})
        self.assertEqual(self.verdicts(lab, "Reader A"), {target: 1})

        # Work at the lab PC, then go back to the office PC.
        self.review(lab.work_db, "Reader A", target, case, 0)
        self.write_mask(lab, case, b"drawn at the lab")
        hub.publish(lab)
        result = hub.restore_own(office)

        self.assertTrue(result["restored"])
        self.assertEqual(self.verdicts(office, "Reader A"), {target: 0})
        office_mask = review_store.label_path(office.work_db, case, "Reader A", 1)
        self.assertEqual(office_mask.read_bytes(), b"drawn at the lab")
        # Up to date now: a second restore does nothing.
        self.assertFalse(hub.restore_own(office)["restored"])

    def test_restore_backs_up_unpublished_local_work_before_replacing_it(self) -> None:
        office = self.workspace("Reader A", "pc1")
        case = self.first_case(office.work_db)
        target = self.targets(office.work_db, case)[0]
        hub.publish(office)
        lab = self.workspace("Reader A", "pc2")
        hub.restore_own(lab)
        self.review(lab.work_db, "Reader A", target, case, 1)
        hub.publish(lab)
        # Meanwhile the office PC saved something it never managed to publish.
        self.review(office.work_db, "Reader A", target, case, 0)
        hub.mark_dirty(office)

        result = hub.restore_own(office)

        self.assertTrue(result["restored"])
        self.assertTrue(result["backup"].is_file())
        backup_verdicts = {
            row["target_id"]: row["verify"]
            for row in self.rows(result["backup"], "SELECT target_id, verify FROM review_annotations")
        }
        self.assertEqual(backup_verdicts, {target: 0})
        self.assertEqual(self.verdicts(office, "Reader A"), {target: 1})
        self.assertFalse(hub.is_dirty(office))

    def test_publish_clears_temporary_files_a_crash_left_behind(self) -> None:
        a = self.workspace("Reader A")
        hub.publish(a)
        stale = self.hub.reader_dir("Reader A") / ".reviews.sqlite.dead.tmp"
        fresh = self.hub.reader_dir("Reader A") / ".reviews.sqlite.busy.tmp"
        stale_mask = self.hub.labels_dir("Reader A") / ".x.nii.gz.dead.tmp"
        for path in (stale, fresh, stale_mask):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"half")
        old = time.time() - 2 * 3600
        os.utime(stale, (old, old))
        os.utime(stale_mask, (old, old))

        hub.publish(a)

        self.assertFalse(stale.exists())
        self.assertFalse(stale_mask.exists())
        self.assertTrue(fresh.exists())  # may be another copy still in flight

    def test_restore_keeps_unpublished_local_masks_with_the_backup(self) -> None:
        office = self.workspace("Reader A", "pc1")
        case = self.first_case(office.work_db)
        hub.publish(office)
        lab = self.workspace("Reader A", "pc2")
        hub.restore_own(lab)
        hub.publish(lab)
        # The office PC drew a mask it never managed to publish.
        drawn = self.write_mask(office, case, b"drawn offline at the office")
        hub.mark_dirty(office)

        result = hub.restore_own(office)

        kept = result["backup"].with_name(result["backup"].stem + "-labels") / drawn.name
        self.assertEqual(kept.read_bytes(), b"drawn offline at the office")

    def test_read_only_workspace_cannot_publish(self) -> None:
        viewer = self.workspace(None)
        self.assertTrue(viewer.read_only)
        with self.assertRaises(hub.HubError):
            hub.publish(viewer)


class MissingFileTests(SyncFixture):
    """A shared folder with something missing must say what, and where."""

    def test_a_missing_workbook_is_named(self) -> None:
        self.hub.workbook.unlink()
        os.environ["MICROBLEED_LOCAL_ROOT"] = str(self.temp / "pc")
        with self.assertRaises(hub.HubError) as caught:
            hub.prepare_workspace(self.hub, "Reader A")
        self.assertNotIsInstance(caught.exception, hub.HubUnavailable)
        self.assertIn(str(self.hub.workbook), str(caught.exception))
        self.assertIn("workbook", str(caught.exception).lower())

    def test_a_failed_copy_names_what_it_was_copying(self) -> None:
        missing = self.temp / "gone.sqlite"
        with self.assertRaises(hub.HubUnavailable) as caught:
            hub.atomic_copy(missing, self.temp / "copy.sqlite")
        self.assertIn(str(missing), str(caught.exception))


class ExportTests(SyncFixture):
    def test_export_contains_every_reader(self) -> None:
        a = self.workspace("Reader A", "pc1")
        b = self.workspace("Reader B", "pc2")
        case = self.first_case(a.work_db)
        target = self.targets(a.work_db, case)[0]
        self.review(a.work_db, "Reader A", target, case, 1)
        self.review(b.work_db, "Reader B", target, case, 0)
        hub.publish(a)
        hub.publish(b)
        out = self.temp / "combined.xlsx"

        report = hub.export_all(self.hub, out, keep_database=self.temp / "combined.sqlite")

        self.assertTrue(out.is_file())
        self.assertEqual(report["readers"], 2)
        self.assertEqual(report["disagreements"], 1)
        self.assertTrue((self.temp / "combined.sqlite").is_file())


class AdminToolTests(StoreFixture):
    """``tools/hub_admin.py``, run the way a person runs it."""

    def run_tool(self, *arguments: str) -> int:
        import subprocess

        script = VIEWER_DIR / "tools" / "hub_admin.py"
        result = subprocess.run(
            [sys.executable, str(script), *arguments], capture_output=True, text=True,
        )
        self.output = result.stdout + result.stderr
        return result.returncode

    def existing_study(self) -> Path:
        """A single-database study of the kind that exists today."""

        db = self.new_store("study")
        case = self.first_case(db)
        first, second = self.targets(db, case)[:2]
        start_new_session(db, "Alex Smith")
        start_new_session(db, "Sam")
        self.review(db, "Alex Smith", first, case, 1)
        self.review(db, "Alex Smith", second, case, 0)
        self.review(db, "Sam", first, case, 1)
        add_manual_annotation(db, case_id=case, ras=(1, 2, 3), reader_id="Sam", review_round=1)
        mask = review_store.label_path(db, case, "Alex Smith", 1)
        mask.parent.mkdir(parents=True)
        mask.write_bytes(b"alex's mask")
        save_roi(
            db, target_id=first, case_id=case, reader_id="Alex Smith", review_round=1,
            label_value=1, path=mask, voxel_count=4, volume_mm3=1.0, generated_from="swi",
        )
        self.case = case
        return db

    def init_hub(self, db: Path) -> Path:
        root = self.temp / "nas"
        code = self.run_tool(
            "init", str(root), "--from-db", str(db),
            "--workbook", str(EXAMPLE_WORKBOOK), "--data-root", str(self.data_root),
        )
        self.assertEqual(code, 0, self.output)
        return root

    def test_init_moves_every_reader_into_their_own_snapshot(self) -> None:
        db = self.existing_study()
        root = self.init_hub(db)
        shared = hub.Hub.open(root)

        self.assertEqual(shared.workbook, root / "findings.xlsx")
        self.assertTrue(shared.workbook.is_file())
        names = sorted(item["reader_id"] for item in hub.list_readers(shared))
        self.assertEqual(names, ["Alex Smith", "Sam"])
        for name, reviews in (("Alex Smith", 2), ("Sam", 1)):
            snapshot = shared.snapshot_path(name)
            rows = self.rows(snapshot, "SELECT reader_id FROM review_annotations")
            self.assertEqual(len(rows), reviews, name)
            self.assertFalse(hub.has_password(shared, name))
            self.assertEqual(hub.read_json(shared.state_path(name))["revision"], 1)
        manual = self.rows(shared.snapshot_path("Sam"), "SELECT COUNT(*) AS n FROM manual_annotations")[0]["n"]
        self.assertEqual(manual, 1)
        mask = shared.labels_dir("Alex Smith") / f"{self.case}_round1.nii.gz"
        self.assertEqual(mask.read_bytes(), b"alex's mask")
        # The original is not touched.
        self.assertEqual(len(self.rows(db, "SELECT * FROM review_annotations")), 3)

    def test_init_records_an_mri_folder_outside_the_hub_absolutely(self) -> None:
        import subprocess

        db = self.existing_study()
        root = self.temp / "nas"
        result = subprocess.run(
            [
                sys.executable, str(VIEWER_DIR / "tools" / "hub_admin.py"), "init", str(root),
                "--from-db", str(db), "--workbook", str(EXAMPLE_WORKBOOK), "--data-root", "Data",
            ],
            capture_output=True, text=True, cwd=str(self.temp),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(hub.Hub.open(root).data_root.resolve(), self.data_root.resolve())

    def test_init_leaves_out_readers_who_never_did_any_work(self) -> None:
        db = self.existing_study()
        log_event(db, "session_opened", reader_id="Ghost QA")
        root = self.init_hub(db)

        names = [item["reader_id"] for item in hub.list_readers(hub.Hub.open(root))]
        self.assertNotIn("Ghost QA", names)
        self.assertIn("Ghost QA", self.output)

    def test_check_reports_what_is_missing(self) -> None:
        db = self.existing_study()
        root = self.init_hub(db)
        self.assertEqual(self.run_tool("check", str(root)), 0, self.output)
        self.assertIn("OK", self.output)

        shared = hub.Hub.open(root)
        mask = next(shared.labels_dir("Alex Smith").glob("*.nii.gz"))
        mask.unlink()
        shared.workbook.unlink()
        self.assertNotEqual(self.run_tool("check", str(root)), 0)
        self.assertIn("findings.xlsx", self.output)
        self.assertIn(mask.name, self.output)

    def test_init_refuses_an_existing_hub(self) -> None:
        db = self.existing_study()
        root = self.init_hub(db)
        code = self.run_tool(
            "init", str(root), "--from-db", str(db),
            "--workbook", str(EXAMPLE_WORKBOOK), "--data-root", str(self.data_root),
        )
        self.assertNotEqual(code, 0)
        self.assertIn("already", self.output)

    def test_list_reset_and_export(self) -> None:
        db = self.existing_study()
        root = self.init_hub(db)
        shared = hub.Hub.open(root)
        hub.set_password(shared, "Sam", "pw-1234")

        self.assertEqual(self.run_tool("list", str(root)), 0, self.output)
        self.assertIn("Alex Smith", self.output)
        self.assertIn("Sam", self.output)

        self.assertEqual(self.run_tool("reset-password", str(root), "Sam"), 0, self.output)
        self.assertFalse(hub.has_password(shared, "Sam"))

        out = self.temp / "all.xlsx"
        self.assertEqual(self.run_tool("export", str(root), str(out)), 0, self.output)
        self.assertTrue(out.is_file())
        self.assertIn("2 readers", self.output)


if __name__ == "__main__":
    unittest.main()
