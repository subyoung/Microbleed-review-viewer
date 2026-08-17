"""The documentation must agree with the code it describes.

Docs drift silently: a default changes, a control is renamed, and the manual
keeps saying the old thing until somebody is misled by it. Every claim in
``docs/FEATURES.md`` that can be read straight out of the source is checked
here, so a stale number fails the suite instead of a reader's expectations.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = VIEWER_DIR / "docs"
sys.path.insert(0, str(VIEWER_DIR))

try:
    from PySide6.QtWidgets import QApplication  # noqa: F401
except ImportError:  # pragma: no cover - allows core-only environments to run
    QApplication = None


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
@unittest.skipUnless(
    (DOCS_DIR / "FEATURES.md").exists(),
    "docs/ is not part of a published checkout; these tests guard the working copy",
)
class DocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.features = _read(DOCS_DIR / "FEATURES.md")
        cls.changelog = _read(DOCS_DIR / "CHANGELOG.md")
        cls.app_src = _read(VIEWER_DIR / "desktop_app.py")
        cls.store_src = _read(VIEWER_DIR / "review_store.py")

    def test_every_shortcut_is_documented_with_its_real_default(self) -> None:
        import desktop_app

        for action, label, default, _group in desktop_app.SHORTCUT_ACTIONS:
            pattern = re.escape(label) + r"\s*\|\s*`" + re.escape(default) + r"`"
            with self.subTest(action=action):
                self.assertRegex(
                    self.features,
                    pattern,
                    f"FEATURES.md should list '{label}' with key '{default}'",
                )

    def test_the_documented_defaults_match_the_widgets(self) -> None:
        import desktop_app

        for name, value in (
            ("lesion FOV", desktop_app.DEFAULT_LESION_FOV_MM),
            ("minimum FOV", desktop_app.MIN_LESION_FOV_MM),
            ("maximum FOV", desktop_app.MAX_LESION_FOV_MM),
        ):
            with self.subTest(setting=name):
                self.assertIn(str(int(value)), self.features)

        for name, pattern, template in (
            ("sensitivity", r"sensitivity_spin\.setRange\(([\d.]+), ([\d.]+)\)", "{0}–{1}"),
            ("brush", r"brush_spin\.setRange\(([\d.]+), ([\d.]+)\)", "{0}–{1} mm"),
        ):
            match = re.search(pattern, self.app_src)
            self.assertIsNotNone(match, f"{name} spin not found in the source")
            assert match is not None
            with self.subTest(setting=name):
                self.assertIn(template.format(match.group(1), match.group(2)), self.features)

        radius = re.search(r"roi_radius_spin\.setRange\(([\d.]+), ([\d.]+)\)", self.app_src)
        self.assertIsNotNone(radius)
        assert radius is not None
        self.assertIn(
            f"{int(float(radius.group(1)))}–{int(float(radius.group(2)))} mm",
            self.features,
        )

    def test_the_growth_cap_is_not_labelled_as_a_diameter(self) -> None:
        """The spin's value is passed as a radius, so a "ø" label would lie.

        A reader capping growth at 6 mm expects a 6 mm lesion, not a 12 mm one.
        """

        self.assertIn(
            "radius_mm = float(self.roi_radius_spin.value())",
            self.app_src,
            "the growth cap is no longer applied as a radius; re-check the label",
        )
        self.assertNotIn('_label("max ø"', self.app_src)

    def test_the_declared_minimum_window_size_is_the_one_in_the_code(self) -> None:
        """Both minimums, because the case queue can be folded away."""

        import desktop_app

        self.assertIn(
            "setMinimumSize(MINIMUM_WINDOW_WIDTH, MINIMUM_WINDOW_HEIGHT)", self.app_src
        )
        for value in (
            desktop_app.MINIMUM_WINDOW_WIDTH,
            desktop_app.MINIMUM_WINDOW_HEIGHT,
            desktop_app.QUEUE_PINNED_MIN_WIDTH,
        ):
            with self.subTest(value=value):
                self.assertIn(str(value), self.features)

    def test_orientation_presets_are_documented(self) -> None:
        from imaging import DEFAULT_ORIENTATION, ORIENTATION_PRESETS

        for key, preset in ORIENTATION_PRESETS.items():
            with self.subTest(preset=key):
                self.assertIn(str(preset["label"]), self.features)
                self.assertIn(", ".join(preset["axcodes"]), self.features)
        self.assertIn(str(ORIENTATION_PRESETS[DEFAULT_ORIENTATION]["label"]), self.features)

    def test_every_sequence_and_filter_is_documented(self) -> None:
        import desktop_app

        for key, label in desktop_app.MODALITY_LABELS.items():
            with self.subTest(sequence=key):
                self.assertIn(label, self.features)

        # Read the filter labels out of the source and require each one.
        filters = re.findall(r"_only_cb = QCheckBox\(\"([^\"]+)\"\)", self.app_src)
        filters += re.findall(r"(?:hide_missing|source_unverified|adjudication)_cb = "
                              r"QCheckBox\(\"([^\"]+)\"\)", self.app_src)
        self.assertTrue(filters, "no filter checkboxes found in the source")
        for label in filters:
            with self.subTest(filter=label):
                self.assertIn(label, self.features)

    def test_every_database_table_is_documented(self) -> None:
        tables = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", self.store_src))
        self.assertTrue(tables)
        for table in sorted(tables):
            with self.subTest(table=table):
                self.assertIn(table, self.features)

    def test_every_exported_segmentation_column_is_documented(self) -> None:
        from review_store import SEGMENTATION_COLUMNS

        for column in SEGMENTATION_COLUMNS:
            grouped = f"{column.rsplit('_', 1)[0]}_l/p/s"
            with self.subTest(column=column):
                self.assertTrue(
                    column in self.features or grouped in self.features,
                    f"{column} is exported but not documented",
                )

    def test_every_preference_is_in_the_manual(self) -> None:
        """A preference the manual does not mention may as well not exist.

        "Smooth the image when magnified" had been there for weeks, was
        checked by a test, and still got reported as missing -- because the
        one place anyone would look for it did not say so.
        """

        dialog = self.app_src[
            self.app_src.index("class SettingsDialog") : self.app_src.index("class DatabaseWriter")
        ]
        labels = re.findall(r"self\.\w+_cb = QCheckBox\(\"([^\"]+)\"\)", dialog)
        self.assertGreaterEqual(len(labels), 6, "the preference dialog was not found")
        for label in labels:
            with self.subTest(preference=label):
                self.assertIn(
                    label,
                    self.features,
                    f"Preferences offers '{label}', and FEATURES.md never mentions it",
                )

    def test_the_test_counts_are_current(self) -> None:
        # Parsed rather than grepped: a substring search counts the search
        # string on this very line, so the documented figure was one higher
        # than the number of tests that actually run.
        import ast

        counts = {
            path.name: sum(
                1
                for node in ast.parse(_read(path)).body
                if isinstance(node, ast.ClassDef)
                for method in node.body
                if isinstance(method, ast.FunctionDef) and method.name.startswith("test_")
            )
            for path in sorted((VIEWER_DIR / "tests").glob("test_*.py"))
        }
        total = sum(counts.values())
        # Matched where the figure is actually written, not anywhere in the
        # file: a plain substring search passed on a stale count because 220
        # also happens to be a pixel constant elsewhere in the same document.
        self.assertRegex(
            self.features,
            r"tests/\s+" + str(total) + r" 个测试",
            f"FEATURES.md should show the suite at {total}",
        )
        self.assertRegex(
            self.changelog,
            r"测试数量 \| " + str(total) + r"（core "
            + str(counts["test_core.py"]) + r" \+ desktop "
            + str(counts["test_desktop_app.py"]) + r" \+ docs "
            + str(counts["test_docs.py"]) + r"）",
            f"CHANGELOG.md should show the suite at {total}",
        )
        # The overview quotes the per-module figures as its QA record, and a
        # stale one there is worse than none: it said 15 and 8 for weeks.
        overview = _read(DOCS_DIR / "PROJECT_OVERVIEW.md")
        self.assertIn(str(total), overview, f"the suite now has {total} tests")
        for name, count in counts.items():
            with self.subTest(module=name):
                stem = re.escape(name.removesuffix(".py"))
                self.assertRegex(
                    overview,
                    r"`" + stem + r"`[^\n]*" + str(count),
                    f"PROJECT_OVERVIEW.md should record {name} at {count}",
                )

    def test_the_files_the_docs_name_all_exist(self) -> None:
        named = set(re.findall(r"`([\w./]+\.(?:py|bat))`", self.features))
        for name in sorted(named):
            with self.subTest(file=name):
                # Test modules are named without their directory in prose.
                self.assertTrue(
                    (VIEWER_DIR / name).exists() or (VIEWER_DIR / "tests" / name).exists(),
                    f"FEATURES.md names {name}, which does not exist",
                )

    def test_what_the_docs_call_ignored_really_is(self) -> None:
        ignore = _read(VIEWER_DIR / ".gitignore")
        for pattern in ("labels/", "*.sqlite", "*.log"):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, ignore)

    def test_the_cross_document_links_resolve(self) -> None:
        for name in ("FEATURES.md", "CHANGELOG.md", "README.md", "PROJECT_OVERVIEW.md"):
            text = _read(DOCS_DIR / name)
            for link in re.findall(r"\]\((\w+\.md)\)", text):
                with self.subTest(document=name, link=link):
                    self.assertTrue((DOCS_DIR / link).exists(), f"{name} links to a missing {link}")

    def test_the_process_record_says_it_is_not_the_manual(self) -> None:
        """``PROJECT_OVERVIEW.md`` records decisions as they were made.

        Its dated sections are not rewritten when behaviour changes later --
        the ROI tools were one tool with a right-button eraser in the section
        that introduced them, and are now two tools -- so it has to say up
        front which document is authoritative, or a reader takes a two-round-old
        description for the current one.
        """

        overview = _read(DOCS_DIR / "PROJECT_OVERVIEW.md")
        preamble = overview.split("## 1.", 1)[0]
        self.assertIn("FEATURES.md", preamble, "the overview must point at the manual")
        self.assertIn(
            "为准",
            preamble,
            "the overview must say the manual is authoritative for current behaviour",
        )

    def test_the_manual_describes_the_tools_the_code_actually_has(self) -> None:
        """Each tool's real name and real default key, from the one binding table."""

        import desktop_app

        tools = {
            action: (label, default)
            for action, label, default, _group in desktop_app.SHORTCUT_ACTIONS
            if action.startswith("tool_")
        }
        self.assertIn("tool_brush", tools)
        self.assertIn("tool_eraser", tools)
        for action, (label, default) in tools.items():
            name = label.split(" on / off")[0].split(" tool")[0]
            # The manual introduces a tool either in the toolbar table or in
            # prose; either way its name and its key belong on one line.
            introduced = [
                line
                for line in self.features.splitlines()
                if f"**{name}**" in line and f"`{default}`" in line
            ]
            with self.subTest(tool=action):
                self.assertTrue(
                    introduced,
                    f"FEATURES.md should introduce the {name} tool with key {default}",
                )


if __name__ == "__main__":
    unittest.main()
