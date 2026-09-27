"""Readers sharing one folder on a NAS, without a server.

The folder (the "hub") is only storage: nothing runs on it.  SQLite cannot be
shared safely over SMB -- WAL does not work across machines and SMB file locks
are not reliable -- so no SQLite file here is ever opened in place, and every
file has exactly one writer::

    <hub>/
      hub.json                   what this hub is, where its workbook and MRI are
      labels/<reader>/           that reader's masks            (written by them)
      readers/<reader>/
        profile.json             name + password hash           (written by them)
        reviews.sqlite           that reader's published work   (written by them)
        state.json               revision of reviews.sqlite     (written by them)
        lock.json                which PC is using this reader  (written by them)
      exports/

Each reader works in a store on their own PC and publishes a snapshot of their
own rows after every save; everybody else copies the snapshot home and
imports it.  Every write here goes to a temporary name first and is renamed
over the target, so a reader of the folder sees the old file or the new one,
never half of one.

A password here stops somebody picking the wrong name and overwriting a
colleague's work.  It is not access control: anyone who can write to the
share can edit the files directly.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import socket
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from review_store import (
    connect,
    export_reader_snapshot,
    export_reviews,
    get_meta,
    import_reader_snapshot,
    initialize_store,
    label_directory,
    read_snapshot_meta,
    safe_reader_name,
    set_label_search_roots,
    set_meta,
)

HUB_FILE = "hub.json"
HUB_FORMAT = 1
PBKDF2_ITERATIONS = 200_000
# A PC that has not renewed its claim on a reader for this long is presumed
# gone (crashed, unplugged), and another PC may take the reader over.
STALE_LOCK_SECONDS = 180
HEARTBEAT_SECONDS = 60


class HubError(RuntimeError):
    """Something about the shared folder that the reader has to act on."""


class HubUnavailable(HubError):
    """The shared folder cannot be reached right now."""


class LockHeld(HubError):
    """Another PC is using this reader."""

    def __init__(self, message: str, holder: dict[str, Any]) -> None:
        super().__init__(message)
        self.holder = holder


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(text: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


@contextmanager
def _io(what: str) -> Iterator[None]:
    """Turn "the share went away" into one exception the caller can handle."""

    try:
        yield
    except HubError:
        raise
    except OSError as exc:
        raise HubUnavailable(f"Could not {what}: {exc}") from exc


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with _io(f"write {path}"):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)


def atomic_copy(source: Path, target: Path) -> None:
    target = Path(target)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    with _io(f"copy {source} to {target}"):
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()


def read_json(path: Path) -> Any | None:
    """The file's content, or None when it does not exist or is unreadable JSON."""

    path = Path(path)
    with _io(f"read {path}"):
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None


def _stored_path(root: Path, path: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        # Outside the hub: only an absolute path means the same thing on
        # every PC (a relative one would be read relative to the hub).
        return str(path.resolve())


def require_reachable(hub: "Hub") -> None:
    """Refuse to touch a shared folder that is not there.

    Every write below creates missing parent folders, which is right inside
    a hub and wrong for the hub itself: an unplugged share must fail, not be
    quietly re-created as an empty folder where it used to be.
    """

    if not is_hub(hub.root):
        raise HubUnavailable(f"The shared folder cannot be reached: {hub.root}")


def machine_name() -> str:
    return os.environ.get("COMPUTERNAME") or socket.gethostname() or "this PC"


def is_hub(path: Path | str) -> bool:
    try:
        return (Path(path) / HUB_FILE).is_file()
    except OSError:
        return False


@dataclass(frozen=True)
class Hub:
    root: Path
    info: dict[str, Any] = field(compare=False)

    @classmethod
    def create(
        cls,
        root: Path | str,
        *,
        workbook: Path,
        data_root: Path,
        dataset_config: dict[str, Any] | None = None,
    ) -> "Hub":
        root = Path(root)
        with _io(f"create a shared folder at {root}"):
            root.mkdir(parents=True, exist_ok=True)
            if (root / HUB_FILE).exists():
                raise HubError(f"{root} is already a shared review folder.")
            for name in ("readers", "labels", "exports"):
                (root / name).mkdir(exist_ok=True)
        info = {
            "format": HUB_FORMAT,
            "hub_id": uuid.uuid4().hex,
            "created_at": _iso(utc_now()),
            "workbook": _stored_path(root, workbook),
            "data_root": _stored_path(root, data_root),
            "dataset_config": dataset_config,
        }
        atomic_write_json(root / HUB_FILE, info)
        return cls(root, info)

    @classmethod
    def open(cls, root: Path | str) -> "Hub":
        root = Path(root)
        with _io(f"open {root}"):
            if not root.is_dir():
                raise HubUnavailable(f"The shared folder cannot be reached: {root}")
        info = read_json(root / HUB_FILE)
        if not isinstance(info, dict) or "hub_id" not in info:
            raise HubError(f"{root} is not a shared review folder (no {HUB_FILE}).")
        if int(info.get("format") or 0) > HUB_FORMAT:
            raise HubError(f"{root} was made by a newer version of the viewer; update it first.")
        return cls(root, info)

    def _resolve(self, stored: str) -> Path:
        path = Path(stored)
        return path if path.is_absolute() else self.root / path

    @property
    def hub_id(self) -> str:
        return str(self.info["hub_id"])

    @property
    def workbook(self) -> Path:
        return self._resolve(str(self.info["workbook"]))

    @property
    def data_root(self) -> Path:
        return self._resolve(str(self.info["data_root"]))

    @property
    def dataset_config(self) -> dict[str, Any] | None:
        value = self.info.get("dataset_config")
        return value if isinstance(value, dict) else None

    @property
    def readers_dir(self) -> Path:
        return self.root / "readers"

    def reader_dir(self, reader_id: str) -> Path:
        return self.readers_dir / safe_reader_name(reader_id)

    def profile_path(self, reader_id: str) -> Path:
        return self.reader_dir(reader_id) / "profile.json"

    def snapshot_path(self, reader_id: str) -> Path:
        return self.reader_dir(reader_id) / "reviews.sqlite"

    def state_path(self, reader_id: str) -> Path:
        return self.reader_dir(reader_id) / "state.json"

    def lock_path(self, reader_id: str) -> Path:
        return self.reader_dir(reader_id) / "lock.json"

    def labels_dir(self, reader_id: str) -> Path:
        return self.root / "labels" / safe_reader_name(reader_id)

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"


# -------------------------------------------------------------- passwords --
def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> dict[str, Any]:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return {
        "algorithm": "pbkdf2_sha256",
        "iterations": iterations,
        "salt": salt.hex(),
        "hash": digest.hex(),
    }


def check_password(record: Any, password: str) -> bool:
    if not isinstance(record, dict) or record.get("algorithm") != "pbkdf2_sha256":
        return False
    try:
        salt = bytes.fromhex(str(record["salt"]))
        expected = bytes.fromhex(str(record["hash"]))
        iterations = int(record["iterations"])
    except (KeyError, TypeError, ValueError):
        return False
    digest = hashlib.pbkdf2_hmac("sha256", str(password).encode("utf-8"), salt, iterations)
    return hmac.compare_digest(digest, expected)


# --------------------------------------------------------------- profiles --
def get_profile(hub: Hub, reader_id: str) -> dict[str, Any] | None:
    profile = read_json(hub.profile_path(reader_id))
    if not isinstance(profile, dict):
        return None
    # The folder name is lossy ("Alex Smith" and "Alex_Smith" share one), so
    # the profile has to name the same reader, not merely live in its folder.
    if str(profile.get("reader_id", "")).casefold() != str(reader_id).strip().casefold():
        return None
    return profile


def list_readers(hub: Hub) -> list[dict[str, Any]]:
    readers: list[dict[str, Any]] = []
    with _io(f"list the readers in {hub.root}"):
        folders = sorted(path for path in hub.readers_dir.iterdir() if path.is_dir()) if hub.readers_dir.is_dir() else []
    for folder in folders:
        profile = read_json(folder / "profile.json")
        if not isinstance(profile, dict) or not profile.get("reader_id"):
            continue
        entry = dict(profile)
        entry.pop("password", None)
        entry["has_password"] = isinstance(profile.get("password"), dict)
        entry["state"] = read_json(folder / "state.json") or {}
        entry["lock"] = read_json(folder / "lock.json")
        entry["folder"] = folder.name
        readers.append(entry)
    return sorted(readers, key=lambda item: str(item["reader_id"]).casefold())


def create_reader(hub: Hub, reader_id: str, password: str | None) -> dict[str, Any]:
    name = " ".join(str(reader_id).split())
    if not name:
        raise HubError("A reader needs a name.")
    folder = safe_reader_name(name)
    for existing in list_readers(hub):
        if existing["folder"].casefold() == folder.casefold() or str(existing["reader_id"]).casefold() == name.casefold():
            raise HubError(
                f"“{name}” is too close to the existing reader “{existing['reader_id']}”. "
                "Pick that reader, or a clearly different name."
            )
    profile = {
        "reader_id": name,
        "display_name": name,
        "created_at": _iso(utc_now()),
        "password": hash_password(password) if password else None,
    }
    atomic_write_json(hub.profile_path(name), profile)
    return profile


def has_password(hub: Hub, reader_id: str) -> bool:
    profile = get_profile(hub, reader_id)
    return bool(profile and isinstance(profile.get("password"), dict))


def verify_reader(hub: Hub, reader_id: str, password: str) -> bool:
    profile = get_profile(hub, reader_id)
    return bool(profile) and check_password(profile.get("password"), password)


def _update_profile(hub: Hub, reader_id: str, **changes: Any) -> None:
    profile = get_profile(hub, reader_id)
    if profile is None:
        raise HubError(f"There is no reader called “{reader_id}”.")
    profile.update(changes)
    atomic_write_json(hub.profile_path(reader_id), profile)


def set_password(hub: Hub, reader_id: str, password: str) -> None:
    if not password:
        raise HubError("The password cannot be empty.")
    _update_profile(hub, reader_id, password=hash_password(password))


def reset_password(hub: Hub, reader_id: str) -> None:
    """Forget the password; the reader sets a new one at their next login."""

    _update_profile(hub, reader_id, password=None)


# ------------------------------------------------------------------ locks --
def read_lock(hub: Hub, reader_id: str) -> dict[str, Any] | None:
    lock = read_json(hub.lock_path(reader_id))
    return lock if isinstance(lock, dict) else None


def lock_is_fresh(lock: dict[str, Any] | None, now: datetime | None = None) -> bool:
    if not lock:
        return False
    beat = _parse(lock.get("heartbeat_at"))
    if beat is None:
        return False
    return ((now or utc_now()) - beat).total_seconds() <= STALE_LOCK_SECONDS


def acquire_lock(
    hub: Hub,
    reader_id: str,
    *,
    machine: str,
    force: bool = False,
    now: datetime | None = None,
) -> str:
    """Claim this reader for this PC, and return the claim's token.

    A fresh claim from another PC is refused even with ``force``: two PCs
    publishing one reader would overwrite each other.  A stale one may be
    taken over with ``force``.  The same PC may always reclaim -- that is the
    viewer being reopened after a crash.
    """

    require_reachable(hub)
    now = now or utc_now()
    current = read_lock(hub, reader_id)
    if current and str(current.get("machine")) != machine:
        fresh = lock_is_fresh(current, now)
        if fresh or not force:
            raise LockHeld(
                f"{reader_id} is open on {current.get('machine')}"
                + ("" if fresh else " (not seen for several minutes)"),
                current,
            )
    token = uuid.uuid4().hex
    atomic_write_json(
        hub.lock_path(reader_id),
        {
            "reader_id": reader_id,
            "machine": machine,
            "pid": os.getpid(),
            "token": token,
            "acquired_at": _iso(now),
            "heartbeat_at": _iso(now),
        },
    )
    return token


def heartbeat(hub: Hub, reader_id: str, token: str, now: datetime | None = None) -> bool:
    """Renew the claim.  False when another PC has taken the reader over."""

    require_reachable(hub)
    current = read_lock(hub, reader_id)
    if not current or current.get("token") != token:
        return False
    current["heartbeat_at"] = _iso(now or utc_now())
    atomic_write_json(hub.lock_path(reader_id), current)
    return True


def release_lock(hub: Hub, reader_id: str, token: str) -> None:
    current = read_lock(hub, reader_id)
    if current and current.get("token") == token:
        with _io(f"release {reader_id}"):
            hub.lock_path(reader_id).unlink(missing_ok=True)


# ------------------------------------------------------------- workspace --
def local_base() -> Path:
    """Where this PC keeps its working copies, outside any synced folder."""

    override = os.environ.get("MICROBLEED_LOCAL_ROOT")
    if override:
        return Path(override)
    appdata = os.environ.get("LOCALAPPDATA")
    base = Path(appdata) if appdata else Path.home() / ".local" / "share"
    return base / "MicrobleedReview" / "hubs"


@dataclass
class Workspace:
    """This PC's working copy of one reader's work (or a read-only view)."""

    hub: Hub
    reader_id: str | None
    root: Path
    # Folder name -> the state revision last imported from it.  In memory:
    # the first pull after a start imports everybody once, which is cheap.
    seen: dict[str, int] = field(default_factory=dict)
    last_pull_errors: dict[str, str] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def read_only(self) -> bool:
        return self.reader_id is None

    @property
    def work_db(self) -> Path:
        return self.root / "work.sqlite"

    @property
    def others_dir(self) -> Path:
        return self.root / "others"

    @property
    def source_xlsx(self) -> Path:
        return self.root / "source.xlsx"

    @property
    def own_labels(self) -> Path:
        return label_directory(self.work_db, str(self.reader_id))


def workspace_root(hub: Hub, reader_id: str | None) -> Path:
    return local_base() / hub.hub_id / (safe_reader_name(reader_id) if reader_id else "_readonly")


def _same_file(first: Path, second: Path) -> bool:
    try:
        a, b = first.stat(), second.stat()
    except OSError:
        return False
    # copy2 carries the modification time; two seconds covers file systems
    # that store it coarsely.
    return a.st_size == b.st_size and abs(a.st_mtime - b.st_mtime) < 2.0


def prepare_workspace(hub: Hub, reader_id: str | None) -> Workspace:
    """Make this PC's working copy ready: workbook, store, where masks are."""

    require_reachable(hub)
    with _io(f"read {hub.workbook}"):
        present = hub.workbook.is_file()
    if not present:
        raise HubError(
            f"The shared folder has no findings workbook at {hub.workbook}. "
            "Put the workbook back there (hub.json names it), or run "
            "tools/hub_admin.py check to see what else is missing."
        )
    workspace = Workspace(hub, reader_id, workspace_root(hub, reader_id))
    workspace.root.mkdir(parents=True, exist_ok=True)
    workspace.others_dir.mkdir(exist_ok=True)
    # The store reads a local copy: a workbook open in Excel on another PC
    # would otherwise lock every reader out of starting.
    if not _same_file(hub.workbook, workspace.source_xlsx):
        atomic_copy(hub.workbook, workspace.source_xlsx)
    workspace.report = initialize_store(workspace.source_xlsx, hub.data_root, workspace.work_db)
    set_label_search_roots([hub.root])
    return workspace


def _meta_int(workspace: Workspace, key: str) -> int:
    connection = connect(workspace.work_db)
    try:
        return int(get_meta(connection, key) or 0)
    finally:
        connection.close()


def _set_meta(workspace: Workspace, **values: Any) -> None:
    connection = connect(workspace.work_db)
    try:
        for key, value in values.items():
            set_meta(connection, key, value)
        connection.commit()
    finally:
        connection.close()


def local_revision(workspace: Workspace) -> int:
    return _meta_int(workspace, "hub_revision")


def mark_dirty(workspace: Workspace) -> None:
    """Record that there is local work the shared folder has not seen.

    A counter rather than a flag: a save that lands while a publish is being
    copied must still count as unpublished afterwards.
    """

    connection = connect(workspace.work_db)
    try:
        current = int(get_meta(connection, "hub_dirty_seq") or 0)
        set_meta(connection, "hub_dirty_seq", current + 1)
        connection.commit()
    finally:
        connection.close()


def is_dirty(workspace: Workspace) -> bool:
    return _meta_int(workspace, "hub_dirty_seq") > _meta_int(workspace, "hub_published_seq")


def remote_state(hub: Hub, reader_id: str) -> dict[str, Any]:
    state = read_json(hub.state_path(reader_id))
    return state if isinstance(state, dict) else {}


def mirror_folder(source: Path, target: Path) -> int:
    """Make ``target``'s masks match ``source``'s.  Returns files changed."""

    changed = 0
    with _io(f"mirror {source} to {target}"):
        wanted = {path.name: path for path in source.glob("*.nii.gz")} if source.is_dir() else {}
        for name, path in wanted.items():
            destination = target / name
            if not _same_file(path, destination):
                atomic_copy(path, destination)
                changed += 1
        if target.is_dir():
            for path in target.glob("*.nii.gz"):
                if path.name not in wanted:
                    path.unlink()
                    changed += 1
    return changed


def publish(workspace: Workspace, *, machine: str | None = None) -> int:
    """Put this reader's work in the shared folder.  Returns the new revision."""

    if workspace.read_only:
        raise HubError("A read-only session has nothing to publish.")
    reader = str(workspace.reader_id)
    hub = workspace.hub
    require_reachable(hub)
    pending = _meta_int(workspace, "hub_dirty_seq")
    revision = max(local_revision(workspace), int(remote_state(hub, reader).get("revision") or 0)) + 1
    outbox = workspace.root / "outbox.sqlite"
    export_reader_snapshot(workspace.work_db, reader, outbox, revision=revision)
    # Masks first: a snapshot must never name a mask the folder lacks yet.
    mirror_folder(workspace.own_labels, hub.labels_dir(reader))
    atomic_copy(outbox, hub.snapshot_path(reader))
    atomic_write_json(
        hub.state_path(reader),
        {
            "reader_id": reader,
            "revision": revision,
            "published_at": _iso(utc_now()),
            "machine": machine or os.environ.get("COMPUTERNAME") or "",
        },
    )
    _set_meta(workspace, hub_revision=revision, hub_published_seq=pending)
    _clear_leftovers(hub.reader_dir(reader), hub.labels_dir(reader))
    return revision


# A copy still in flight is seconds old; one this old was abandoned by a crash.
LEFTOVER_SECONDS = 3600


def _clear_leftovers(*folders: Path) -> None:
    """Remove temporary files an interrupted publish of this reader left behind."""

    cutoff = utc_now().timestamp() - LEFTOVER_SECONDS
    for folder in folders:
        try:
            for path in folder.glob(".*.tmp"):
                if path.stat().st_mtime < cutoff:
                    path.unlink()
        except OSError:
            pass  # tidying up is never worth failing a publish over


def pull_others(workspace: Workspace) -> list[str]:
    """Import every other reader whose published work changed.  Returns who."""

    hub = workspace.hub
    require_reachable(hub)
    own = safe_reader_name(workspace.reader_id) if workspace.reader_id else None
    changed: list[str] = []
    with _io(f"read {hub.readers_dir}"):
        folders = sorted(path for path in hub.readers_dir.iterdir() if path.is_dir())
    for folder in folders:
        if folder.name == own:
            continue
        state = read_json(folder / "state.json")
        with _io(f"read {folder}"):
            present = (folder / "reviews.sqlite").is_file()
        if not isinstance(state, dict) or not present:
            continue
        revision = int(state.get("revision") or 0)
        if workspace.seen.get(folder.name) == revision:
            continue
        cache = workspace.others_dir / f"{folder.name}.sqlite"
        atomic_copy(folder / "reviews.sqlite", cache)
        try:
            claimed = read_snapshot_meta(cache).get("snapshot_reader", "")
            # A snapshot speaks for the reader whose folder it is in, or for
            # nobody: otherwise a file in the wrong folder would put rows
            # under somebody else's name.
            if safe_reader_name(claimed) != folder.name:
                raise HubError(f"the snapshot in {folder.name} is {claimed!r}'s")
            result = import_reader_snapshot(workspace.work_db, cache)
        except Exception as exc:  # one broken reader must not stop the rest
            workspace.last_pull_errors[folder.name] = f"{type(exc).__name__}: {exc}"
            workspace.seen[folder.name] = revision
            continue
        workspace.last_pull_errors.pop(folder.name, None)
        workspace.seen[folder.name] = revision
        changed.append(str(result["reader_id"]))
    return changed


def restore_own(workspace: Workspace) -> dict[str, Any]:
    """Bring this reader's newer shared work into this PC's copy.

    Needed when the reader last worked on another PC.  If this PC also has
    work it never published, that is copied aside first and the shared
    version wins -- the reader is told where the copy is.
    """

    if workspace.read_only:
        return {"restored": False, "backup": None}
    reader = str(workspace.reader_id)
    hub = workspace.hub
    require_reachable(hub)
    remote = int(remote_state(hub, reader).get("revision") or 0)
    with _io(f"read {hub.snapshot_path(reader)}"):
        exists = hub.snapshot_path(reader).is_file()
    if not exists or remote <= local_revision(workspace):
        return {"restored": False, "backup": None}
    backup = None
    if is_dirty(workspace):
        stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        backup = workspace.root / f"work.conflict-{stamp}.sqlite"
        source = connect(workspace.work_db)
        target = sqlite3.connect(str(backup))
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        # The masks go with it: the mirror below makes this PC's masks match
        # the shared ones, which deletes any drawn here and never sent.
        if workspace.own_labels.is_dir():
            shutil.copytree(workspace.own_labels, backup.with_name(backup.stem + "-labels"))
    cache = workspace.others_dir / "_own.sqlite"
    atomic_copy(hub.snapshot_path(reader), cache)
    import_reader_snapshot(workspace.work_db, cache, own=True)
    mirror_folder(hub.labels_dir(reader), workspace.own_labels)
    _set_meta(
        workspace,
        hub_revision=remote,
        hub_published_seq=_meta_int(workspace, "hub_dirty_seq"),
    )
    return {"restored": True, "backup": backup}


# ---------------------------------------------------------------- export --
def export_all(hub: Hub, out_path: Path, *, keep_database: Path | None = None) -> dict[str, Any]:
    """Every reader's published work in one results table.

    Built from scratch each time in a temporary store, so it never depends on
    anybody's working copy and cannot disturb one.
    """

    with tempfile.TemporaryDirectory(prefix="microbleed_export_") as scratch_name:
        scratch = Path(scratch_name)
        workbook = scratch / "source.xlsx"
        atomic_copy(hub.workbook, workbook)
        combined = scratch / "combined.sqlite"
        initialize_store(workbook, hub.data_root, combined)
        for entry in list_readers(hub):
            snapshot = hub.snapshot_path(str(entry["reader_id"]))
            if not snapshot.is_file():
                continue
            cache = scratch / f"{entry['folder']}.sqlite"
            atomic_copy(snapshot, cache)
            import_reader_snapshot(combined, cache)
        report = export_reviews(combined, Path(out_path))
        if keep_database is not None:
            source = connect(combined)
            target = sqlite3.connect(str(keep_database))
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
    return report
