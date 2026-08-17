from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def install_directory() -> Path:
    """Where this installation keeps the things that belong to it.

    Running from source that is the folder holding these modules.  In a
    PyInstaller build ``__file__`` points inside the bundle -- so the
    configuration would be looked for, and the review database written, into
    an internal directory the reader never sees and a reinstall replaces.
    Beside the executable is where they belong.
    """

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = install_directory()
PROJECT_DIR = BASE_DIR.parent


def configured_path(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser() if value else default


SOURCE_XLSX = configured_path(
    "MICROBLEED_SOURCE_XLSX",
    BASE_DIR / ".source_snapshot.xlsx"
    if (BASE_DIR / ".source_snapshot.xlsx").exists()
    else PROJECT_DIR / "microbleeds.xlsx",
)
DATA_ROOT = configured_path(
    "MICROBLEED_DATA_ROOT",
    PROJECT_DIR / "Data",
)
REVIEW_DB = configured_path(
    "MICROBLEED_REVIEW_DB",
    BASE_DIR / "microbleed_review_data.sqlite",
)


# The window, the taskbar, every dialog and the installed shortcut all show
# the same mark. ".ico" first: it carries several resolutions in one file, so
# Windows picks a crisp one for the 16px taskbar instead of downscaling.
ICON_CANDIDATES = ("microbleed_review_icon.ico", "microbleed_review_icon.png")


def icon_file() -> Path | None:
    """The application icon, or None if it is missing.

    A missing icon is not worth refusing to start over, so every caller treats
    this as optional.
    """

    for name in ICON_CANDIDATES:
        candidate = BASE_DIR / name
        if candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True)
class Dataset:
    """One study: its findings workbook, its MRI files and its review store.

    The three travel together.  Reviews are rows in one specific database, so
    pointing the viewer at another workbook without also moving the review
    store would mix the reviews of two studies.
    """

    workbook: Path
    data_root: Path
    review_db: Path

    @classmethod
    def create(cls, workbook, data_root, review_db) -> "Dataset":
        return cls(Path(workbook), Path(data_root), Path(review_db))

    @property
    def name(self) -> str:
        return f"{self.workbook.stem} · {self.data_root.name}"

    def as_dict(self) -> dict[str, str]:
        return {
            "workbook": str(self.workbook),
            "data_root": str(self.data_root),
            "review_db": str(self.review_db),
        }

    @classmethod
    def from_dict(cls, values: dict) -> "Dataset | None":
        try:
            return cls.create(values["workbook"], values["data_root"], values["review_db"])
        except (KeyError, TypeError):
            return None

    def problems(self) -> list[str]:
        """Human-readable reasons this dataset cannot be opened."""

        issues: list[str] = []
        if not self.workbook.is_file():
            issues.append(f"The findings workbook was not found: {self.workbook}")
        if not self.data_root.is_dir():
            issues.append(f"The MRI data folder was not found: {self.data_root}")
        parent = self.review_db.parent
        if not parent.exists() and not parent.parent.exists():
            issues.append(f"The review database folder does not exist: {parent}")
        return issues


def default_dataset(config: dict | None = None) -> Dataset:
    """Where this installation's dataset is.

    Three sources, most specific first: an environment variable set by the
    launcher, then ``config.json`` written by the dataset dialog, then the
    conventional layout beside the code.  The environment wins so a launcher
    can open a second study without editing anyone's saved configuration.
    """

    paths = (config or {}).get("paths") or {}

    def resolve(env_name: str, key: str, fallback: Path) -> Path:
        given = os.environ.get(env_name)
        if given:
            return Path(given).expanduser()
        stored = str(paths.get(key) or "").strip()
        if stored:
            return Path(stored).expanduser()
        return fallback

    return Dataset.create(
        resolve("MICROBLEED_SOURCE_XLSX", "workbook", SOURCE_XLSX),
        resolve("MICROBLEED_DATA_ROOT", "data_root", DATA_ROOT),
        resolve("MICROBLEED_REVIEW_DB", "review_database", REVIEW_DB),
    )
