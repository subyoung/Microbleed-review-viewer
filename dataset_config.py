"""What a dataset looks like, in one editable file.

Everything about *this* study -- which sheet the findings are on, what its
columns are called, how the NIfTI files are named, which sequences a case must
have -- used to be constants in the source. That is fine for one study and
useless for anyone else's, so it lives in ``config.json`` instead.

Nothing here reads or writes image data. It answers three questions:

* where the dataset is (workbook, MRI folder, review database),
* how to read the workbook (sheet name, column names),
* what counts as a sequence (label, filename suffix, whether it is required).

``DEFAULTS`` is the configuration the tool was originally written against, so
a file that omits a key still works, and a study that matches it needs no file
at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CONFIG_NAME = "config.json"

# The three sequence slots. Fixed in number -- a fourth would need a fourth
# toolbar button and a fourth shortcut -- but everything about each of them is
# yours to set, including whether a case may be read without it.
DEFAULTS: dict[str, Any] = {
    "dataset_name": "",
    "paths": {
        # Empty means "ask", or "take it from the MICROBLEED_* variables".
        "workbook": "",
        "data_root": "",
        "review_database": "",
    },
    "workbook": {
        "sheet": "MCH-microhemorrage",
        "columns": {
            "case_id": "subjectid",
            "ras_l": "RAS L-R L",
            "ras_p": "RAS P-A A",
            "ras_s": "RAS I-S S",
            "verify": "verify (yes=1)",
            "slice": "slicecount",
            "region": "atlasregions",
            "ras_verified": "RAS verified",
            "flag_2d_qsm": "2DQSM",
            "flag_swi": "SWI",
            "flag_3d_qsm": "3DQSM",
            "readers": "readers",
            "adjudicate": "need adjudicate",
            "comments": "comments",
        },
    },
    "sequences": {
        "swi": {
            "label": "SWI",
            "short": "SWI",
            "suffix": "_GRE4_SWI_AffineRestored.nii.gz",
            "required": True,
            "segmentable": True,
        },
        "qsm": {
            "label": "QSM",
            "short": "QSM",
            "suffix": "_chi_nSFCR+0_Avg_AffineRestored.nii.gz",
            "required": True,
            "segmentable": True,
        },
        "mip": {
            "label": "SWI MIP",
            "short": "MIP",
            "suffix": "_GRE4_SWI_axiMIP2_AffineRestored.nii.gz",
            # Optional by default: a projection is useful for spotting a
            # lesion and for nothing after that, so a study without one should
            # not have every case marked incomplete.
            "required": False,
            # A projection smears a microbleed along the projection direction
            # -- about seven times on the data this was written for -- so a
            # mask drawn on it would be a mask of an artefact.
            "segmentable": False,
        },
    },
}

# The keys, in the order they appear as toolbar buttons and shortcuts 1/2/3.
SEQUENCE_ORDER = ("swi", "qsm", "mip")

# Which sequence's grid a case's label file uses, most preferred first. The
# first one present wins, so a study without SWI still gets a stable choice.
LABEL_REFERENCE_ORDER = ("swi", "qsm", "mip")


class ConfigError(ValueError):
    """A configuration file that cannot be used, with the reason."""


def default_path(base_dir: Path | str | None = None) -> Path:
    """Where the configuration lives: beside the code, unless told otherwise."""

    if base_dir:
        return Path(base_dir) / CONFIG_NAME
    # Beside the executable in a packaged build, beside the code otherwise --
    # see config.install_directory for why that distinction matters.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / CONFIG_NAME
    return Path(__file__).resolve().parent / CONFIG_NAME


def _merge(defaults: Any, given: Any) -> Any:
    """Fill in what the file left out, one level at a time."""

    if not isinstance(defaults, dict) or not isinstance(given, dict):
        return given
    merged = dict(defaults)
    for key, value in given.items():
        merged[key] = _merge(defaults.get(key), value)
    return merged


def _without_comments(value: Any) -> Any:
    """Drop ``_comment`` keys.

    JSON has no comments and the example file needs them: a configuration
    nobody can annotate is one people copy without understanding.
    """

    if isinstance(value, dict):
        return {
            key: _without_comments(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    return value


def validate(config: dict[str, Any]) -> dict[str, Any]:
    """Fill in the gaps and refuse what cannot work.

    Refusing early matters more than it looks: a mistyped suffix means every
    case reads as missing, and a mistyped column name means an import that
    fails on row 2 with no clue why.
    """

    merged = _merge(DEFAULTS, _without_comments(config or {}))

    sheet = str(merged["workbook"].get("sheet") or "").strip()
    if not sheet:
        raise ConfigError("workbook.sheet is empty; name the sheet the findings are on.")
    merged["workbook"]["sheet"] = sheet

    columns = merged["workbook"]["columns"]
    for key in DEFAULTS["workbook"]["columns"]:
        name = str(columns.get(key) or "").strip()
        if key in {"case_id", "ras_l", "ras_p", "ras_s", "verify"} and not name:
            raise ConfigError(f"workbook.columns.{key} is required and cannot be empty.")
        columns[key] = name

    sequences = merged["sequences"]
    unknown = sorted(set(sequences) - set(SEQUENCE_ORDER))
    if unknown:
        raise ConfigError(
            f"Unknown sequence key(s): {', '.join(unknown)}. "
            f"The three slots are {', '.join(SEQUENCE_ORDER)}."
        )
    for key in SEQUENCE_ORDER:
        entry = sequences.setdefault(key, dict(DEFAULTS["sequences"][key]))
        suffix = str(entry.get("suffix") or "").strip()
        if not suffix:
            raise ConfigError(f"sequences.{key}.suffix is empty; give the filename ending.")
        entry["suffix"] = suffix
        entry["label"] = str(entry.get("label") or key.upper()).strip()
        entry["short"] = str(entry.get("short") or entry["label"]).strip()
        entry["required"] = bool(entry.get("required", False))
        entry["segmentable"] = bool(entry.get("segmentable", True))

    if not any(sequences[key]["required"] for key in SEQUENCE_ORDER):
        raise ConfigError(
            "At least one sequence must be required, or no case could ever be readable."
        )
    if not any(
        sequences[key]["required"] and sequences[key]["segmentable"] for key in SEQUENCE_ORDER
    ):
        raise ConfigError(
            "At least one required sequence must be segmentable, or a mask could "
            "never be drawn on a case that has only the required ones."
        )
    labels = [sequences[key]["label"] for key in SEQUENCE_ORDER]
    if len(set(labels)) != len(labels):
        raise ConfigError(f"Two sequences share a label: {labels}.")
    return merged


def load(path: Path | str | None = None) -> dict[str, Any]:
    """Read the configuration, or return the defaults when there is no file."""

    target = Path(path) if path else default_path()
    if not target.exists():
        return validate({})
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{target.name} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{target.name} should contain a JSON object.")
    return validate(raw)


def save(config: dict[str, Any], path: Path | str | None = None) -> Path:
    """Write the configuration, validated first so a bad one never lands."""

    target = Path(path) if path else default_path()
    checked = validate(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written beside and renamed, so an interrupted save cannot leave the tool
    # unable to start.
    pending = target.with_name(target.name + ".partial")
    pending.write_text(json.dumps(checked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(target)
    return target


def describe(config: dict[str, Any]) -> str:
    """One line per sequence, for a status bar or a dialog."""

    parts = []
    for key in SEQUENCE_ORDER:
        entry = config["sequences"][key]
        parts.append(f"{entry['label']} ({'required' if entry['required'] else 'optional'})")
    return " · ".join(parts)


def suggest_suffixes(data_root: Path | str, sample: int = 12) -> list[str]:
    """Work out the filename endings this data folder actually uses.

    Nobody arriving at this tool knows that a "sequence" is identified by the
    tail of its filename, and asking them to type
    ``_chi_nSFCR+0_Avg_AffineRestored.nii.gz`` from memory is a poor welcome.

    The tail is what the files have in common across cases: names differ by
    subject and by acquisition date, so a candidate is the longest ending of
    some file that also ends a file in most of the other sampled cases.  Most,
    not all, because real folders are uneven -- in the study this was written
    for, protocol names differ between scans, and demanding unanimity threw
    away every ending worth offering.

    Returned longest first, since the longest is the most specific and a
    shorter one risks matching two sequences at once.
    """

    root = Path(data_root)
    if not root.is_dir():
        return []

    def volumes(folder: Path) -> list[str]:
        try:
            return sorted(
                item.name
                for item in folder.iterdir()
                if item.is_file() and item.name.lower().endswith((".nii", ".nii.gz"))
            )
        except OSError:
            return []

    folders = []
    for item in sorted(root.iterdir()):
        if item.is_dir() and volumes(item):
            folders.append(item)
        if len(folders) >= sample:
            break
    if not folders:
        return []

    groups = [volumes(folder) for folder in folders]
    needed = max(2, (len(groups) + 1) // 2)
    candidates: dict[str, int] = {}
    for index, group in enumerate(groups):
        rest = groups[:index] + groups[index + 1:]
        for name in group:
            for start in range(len(name)):
                ending = name[start:]
                if len(ending) < 8:
                    break
                shared = sum(
                    1 for other in rest if any(item.endswith(ending) for item in other)
                )
                if shared + 1 >= needed:
                    candidates[ending] = max(candidates.get(ending, 0), shared + 1)
                    break
        if index >= 2:
            break
    # Drop an ending that is only the tail of a longer one we already have.
    ordered = sorted(candidates, key=len, reverse=True)
    kept: list[str] = []
    for ending in ordered:
        if not any(longer.endswith(ending) for longer in kept):
            kept.append(ending)
    return kept
