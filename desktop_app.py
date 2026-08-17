from __future__ import annotations

"""Desktop microbleed review viewer.

This module is the application.  It follows the interaction model of the supplied QSM QC reviewer: a dark, case-centred
workspace, large linked image views, wheel-based slice scrolling, fit/zoom
controls, and a compact review sidebar.

The image display has one explicit coordinate contract:

* every NIfTI is reoriented by :func:`imaging.load_volume` to L-P-I display
  orientation while retaining the reoriented affine;
* workbook RAS values are converted through that exact affine;
* all three planes use the same physical RAS target, even when their native
  grids differ;
* the displayed directions are radiological: Axial R/L/A/P, Coronal R/L/S/I,
  and Sagittal A/P/S/I.

No source workbook writes happen here.  Reader reports, annotations, session
state and logs use the shared SQLite store from ``review_store.py``.
"""

import faulthandler
import json
import math
import os
import sys
from datetime import datetime
from collections import OrderedDict
from pathlib import Path
from queue import Queue
from typing import Any, Iterable

import numpy as np

try:  # Keep the import error useful when somebody launches before installing.
    from PySide6.QtCore import (
        QEvent,
        QEventLoop,
        QObject,
        QPointF,
        QRectF,
        QSettings,
        QSize,
        Qt,
        QThread,
        QTimer,
        QtMsgType,
        Signal,
        qInstallMessageHandler,
    )
    from PySide6.QtGui import (
        QColor,
        QFont,
        QIcon,
        QImage,
        QKeySequence,
        QPainter,
        QPen,
        QPixmap,
        QPolygonF,
        QShortcut,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QDockWidget,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QKeySequenceEdit,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSplitter,
        QStatusBar,
        QTabWidget,
        QTextBrowser,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QWidgetAction,
    )
except ImportError as exc:  # pragma: no cover - exercised by launchers
    raise RuntimeError(
        "The desktop viewer requires PySide6. Run viewer\\install.ps1 or "
        "install PySide6 into the viewer virtual environment."
    ) from exc

try:
    from config import (
        BASE_DIR,
        DATA_ROOT,
        REVIEW_DB,
        SOURCE_XLSX,
        Dataset,
        default_dataset,
        icon_file,
    )
    import dataset_config
    from imaging import (
        DEFAULT_ORIENTATION,
        DISPLAY_AXCODES,
        ORIENTATION_PRESETS,
        Volume,
        clamp_voxel,
        extract_plane,
        grow_lesion,
        isotropic_context,
        lesion_shape,
        lesion_surface,
        load_volume,
        project_context,
        plane_direction_labels,
        preset_axcodes,
        save_label_volume,
        segment_lesion,
        snap_to_extremum,
        ras_to_voxel,
        robust_window,
        voxel_in_bounds,
        voxel_to_ras,
    )
    from review_store import (
        CERTAINTY_CHOICES,
        NEARBY_FINDING_MM,
        SAME_FINDING_MM,
        MIMIC_CHOICES,
        MODALITY_SPECS,
        configure as review_store_configure,
        SourceReadError,
        add_manual_annotation,
        delete_manual_annotation,
        export_reviews,
        findings_near,
        manual_deletion_blockers,
        get_case,
        get_resume_candidate,
        initialize_store,
        list_reader_rounds,
        reimport_source,
        list_cases,
        distance_mm as _distance_mm,
        label_path,
        list_targets,
        log_event,
        refresh_inventory_store,
        resume_session,
        save_review,
        save_roi,
        save_session_state,
        set_busy_timeout_ms,
        start_new_session,
        close_session,
    )
except ImportError as exc:  # pragma: no cover - launch-path guard
    raise RuntimeError(
        "Could not import the viewer data layer. Launch this file via "
        "viewer\\run_app.bat from the project directory."
    ) from exc


APP_TITLE = "Microbleed Review"
PLANE_ORDER = ("axial", "coronal", "sagittal")
PLANE_TITLES = {"axial": "Axial", "coronal": "Coronal", "sagittal": "Sagittal"}
# Both orientation presets keep the same axis roles, so the slice axis and the
# displayed axes of a plane are preset-independent.
PLANE_AXES = {"axial": 2, "coronal": 1, "sagittal": 0}
# Each tuple is (column voxel axis, row voxel axis) in the display array.
PLANE_IMAGE_AXES = {
    "axial": (0, 1),
    "coronal": (0, 2),
    "sagittal": (1, 2),
}


def plane_directions(plane: str, axcodes: tuple[str, ...]) -> tuple[str, str, str, str]:
    """Direction labels for a plane, derived from the array on screen."""

    column_axis, row_axis = PLANE_IMAGE_AXES[plane]
    return plane_direction_labels(axcodes, column_axis, row_axis)


def orientation_summary(preset: str) -> str:
    """Short description of a display preset for the case header."""

    entry = ORIENTATION_PRESETS.get(preset, ORIENTATION_PRESETS[DEFAULT_ORIENTATION])
    return f"{entry['label']} ({''.join(preset_axcodes(preset))})"
MODALITY_ORDER = ("qsm", "swi", "mip")
# Order of the toolbar segments and of shortcuts 1/2/3.  On the data this was
# written for: SWI is the conventional first read for microbleeds, QSM
# confirms the paramagnetic signal, the MIP is the overview.
MODALITY_BUTTON_ORDER = dataset_config.SEQUENCE_ORDER
# Which sequence's geometry a case's label file uses, most preferred first.
# SWI leads because that is what a microbleed is read and drawn on; the first
# one a case actually has wins, so a study without SWI still gets one answer.
LABEL_REFERENCE_ORDER = dataset_config.LABEL_REFERENCE_ORDER
# Names, not identities: the keys are fixed, what they are called is not.
# ``apply_dataset_config`` refills these in place at start-up.
MODALITY_LABELS = {key: MODALITY_SPECS[key]["label"] for key in MODALITY_ORDER}
MODALITY_SHORT_LABELS = {
    key: MODALITY_SPECS[key].get("short", MODALITY_SPECS[key]["label"])
    for key in MODALITY_ORDER
}


def can_segment(modality: str) -> bool:
    """Whether a mask may be drawn on this sequence.

    A projection smears a microbleed along the projection direction -- about
    seven times, measured on the data this was written for -- so a mask drawn
    there is a mask of an artefact.  Which sequences that applies to is the
    study's to say.
    """

    return bool(MODALITY_SPECS.get(modality, {}).get("segmentable", True))


def apply_dataset_config(config: dict | None = None) -> dict:
    """Load a dataset's shape into the store and into these tables."""

    checked = review_store_configure(config)
    for key in MODALITY_ORDER:
        MODALITY_LABELS[key] = MODALITY_SPECS[key]["label"]
        MODALITY_SHORT_LABELS[key] = MODALITY_SPECS[key].get(
            "short", MODALITY_SPECS[key]["label"]
        )
    return checked
ZOOM_PRESETS = (50, 75, 100, 125, 150, 200, 300, 400)
WHEEL_ANGLE_UNITS_PER_SLICE = 120.0
WHEEL_PIXEL_UNITS_PER_SLICE = 80.0
WHEEL_ZOOM_FACTOR = 1.10
LESION_ZOOM_LABEL = "Lesion"
# One table drives the bindings, the settings editor and the sidebar legend, so
# a rebound key cannot disagree with the printed hint.
SHORTCUT_ACTIONS: tuple[tuple[str, str, str, str], ...] = (
    ("sequence_1", f"Sequence: {MODALITY_LABELS[MODALITY_BUTTON_ORDER[0]]}", "1", "Sequences"),
    ("sequence_2", f"Sequence: {MODALITY_LABELS[MODALITY_BUTTON_ORDER[1]]}", "2", "Sequences"),
    ("sequence_3", f"Sequence: {MODALITY_LABELS[MODALITY_BUTTON_ORDER[2]]}", "3", "Sequences"),
    ("verdict_yes", "Verdict: yes", "Y", "Review"),
    ("verdict_no", "Verdict: no", "N", "Review"),
    ("verdict_unset", "Verdict: not set", "0", "Review"),
    ("save_review", "Save review", "Ctrl+S", "Review"),
    ("prev_finding", "Previous finding", "[", "Navigation"),
    ("next_finding", "Next finding", "]", "Navigation"),
    ("prev_case", "Previous case", "PgUp", "Navigation"),
    ("next_case", "Next case", "PgDown", "Navigation"),
    ("tool_point", "Point tool on / off", "P", "View"),
    ("cancel_pick", "Cancel the picked position", "Esc", "View"),
    ("tool_brush", "Brush on / off", "B", "Review"),
    ("tool_eraser", "Eraser on / off", "E", "Review"),
    ("toggle_roi_overlay", "Show / hide the segmentation", "S", "Review"),
    ("brush_smaller", "Smaller brush", "-", "Review"),
    ("brush_larger", "Larger brush", "=", "Review"),
    ("undo_roi", "Undo the last brush stroke", "Ctrl+Z", "Review"),
    ("overlay_target", "Finding crosshair on / off", "X", "View"),
    ("overlay_mouse", "Mouse crosshair on / off", "C", "View"),
    ("overlay_labels", "Direction labels on / off", "D", "View"),
    ("peek_versions", "Hold: show every recorded position", "V", "View"),
    ("lesion_zoom", "Toggle lesion zoom", "Z", "View"),
    ("contrast_dialog", "Contrast (window/level)", "Ctrl+L", "View"),
    ("reset_contrast", "Contrast back to automatic", "Ctrl+Shift+L", "View"),
    ("fit_views", "Fit every view", "R", "View"),
    ("maximize_view", "Maximise the view under the mouse", "F", "View"),
    ("refresh_files", "Rescan the MRI files", "Ctrl+R", "View"),
)
SHORTCUT_DEFAULTS = {action: default for action, _label, default, _group in SHORTCUT_ACTIONS}
SHORTCUT_LABELS = {action: label for action, label, _default, _group in SHORTCUT_ACTIONS}
DEFAULT_LESION_FOV_MM = 60.0
MIN_LESION_FOV_MM = 15.0
MAX_LESION_FOV_MM = 200.0
MAX_ZOOM_MULTIPLIER = 20.0
# A finding this close to the displayed slice is drawn as a neighbour marker.
NEIGHBOUR_SLICE_TOLERANCE = 2.5
# Undo steps kept for the segmentation.  Each holds only the voxels it changed,
# so twenty of them cost kilobytes rather than the 461 MB that twenty copies of
# a 256x256x176 label volume did.
ROI_UNDO_STEPS = 20
# Laplacian passes behind the 3D view's Smooth box.  Two rounds off the voxel
# steps without turning a five-voxel lesion into a bead: measured shrinkage is
# 1.4% on a 5 mm ball and 8.4% on a one-voxel-thick sheet, and nothing shown
# in that window is measured from the smoothed mesh.
LESION_SMOOTH_PASSES = 2
# Where the 3D view opens: a three-quarter turn from straight ahead, tilted a
# little from above.  Straight on reads as a flat disc, and the three-quarter
# view is the one that shows depth without hiding either hemisphere.
_DEFAULT_YAW = 0.6
_DEFAULT_PITCH = 0.2
# Width of the case queue column, pinned open or hovering over the images.
QUEUE_COLUMN_WIDTH = 300
# What the window needs, measured by shrinking it until something clips.
# Two numbers because the case queue is a column that can be folded away, and
# a single minimum would either be a lie while it is open or forbid a size
# that works perfectly well while it is shut.  Both floors are set by the
# toolbar rather than the panels: it is the one row that cannot wrap.
MINIMUM_WINDOW_WIDTH = 1160
MINIMUM_WINDOW_HEIGHT = 760
QUEUE_PINNED_MIN_WIDTH = 1440
# What the two reading panels need before their contents start being cut off.
# A QScrollArea shrinks happily and grows a scrollbar instead, so the layout
# has to be told, or the grid divides the middle evenly and clips the panel.
LOCATION_PANEL_MIN_WIDTH = 358
SEGMENT_PANEL_MIN_WIDTH = 322
# The reference column is draggable between these two.  The floor is what
# its contents actually need (measured: 262 for the widest row of controls,
# rounded up); the ceiling is generous because widening it is the reader's
# call -- what the window must not do is decide to be wide on its own.
RIGHT_COLUMN_MIN_WIDTH = 300
RIGHT_COLUMN_MAX_WIDTH = 560
# Narrow by default.  A findings row folds to two lines when it has to (see
# _fit_finding_rows), so the column does not have to be wide to be readable.
RIGHT_COLUMN_DEFAULT_WIDTH = 330
# Four rows of the taller, folded kind.
FINDING_LIST_MAX_HEIGHT = 220
# How long a background write waits for a locked database before giving up.
# The GUI thread keeps the full default: a reader saving a verdict has to know
# it was stored, so that write waits.  Nobody waits for an operation log entry.
BACKGROUND_BUSY_TIMEOUT_MS = 2000
# How many robust deviations past the local background a voxel has to be to
# join a grown mask.  Chosen by sweeping thirty real findings: at 3.0 the
# median mask is 3.6 mm across, which is where a microbleed on this data sits,
# and the fewest masks either overrun the safety cap or collapse to the seed.
DEFAULT_GROW_SENSITIVITY = 3.0
# Distinguishable on greyscale MRI, and distinct from the red finding marker.
VARIANT_COLORS = ("#32d9e8", "#f2b000", "#7ee081", "#c58cff", "#4a9eff", "#ff8f6b")


def quantize_wheel_delta(
    delta: float,
    *,
    pixel: bool,
    remainder: float = 0.0,
) -> tuple[int, float]:
    """Convert a raw Qt wheel delta into at most one controlled step.

    Standard mouse wheels report 120 angle units per notch. Touchpads report
    smaller pixel deltas across many events. Accumulating both forms prevents
    one touchpad gesture from being interpreted as one slice per low-level
    event, while capping each event at one visible step avoids abrupt jumps.
    """

    try:
        raw_delta = float(delta)
        accumulated = float(remainder)
    except (TypeError, ValueError):
        return 0, 0.0
    if not math.isfinite(raw_delta) or not math.isfinite(accumulated) or raw_delta == 0:
        return 0, accumulated
    units_per_step = WHEEL_PIXEL_UNITS_PER_SLICE if pixel else WHEEL_ANGLE_UNITS_PER_SLICE
    total = accumulated + raw_delta / units_per_step
    if abs(total) < 1.0:
        return 0, total
    direction = 1 if total > 0 else -1
    # Discard excess whole steps from a single high-resolution event. A
    # subsequent event can still generate the next controlled step.
    return direction, 0.0

COLORS = {
    "background": "#171a20",
    "panel": "#1f232b",
    "header": "#272c36",
    "field": "#1b1f27",
    "canvas": "#07090c",
    "border": "#333945",
    "border_soft": "#2a2f39",
    "text": "#dde2ec",
    "dim": "#8892a3",
    "faint": "#69717f",
    "accent": "#4a9eff",
    "success": "#4fb96c",
    "warn": "#e0913a",
    "danger": "#e05c5c",
    "target": "#ff4e58",
    "neighbour": "#ffa2a6",
    "variant_source": "#e8edf5",
    "roi": "#ffd24a",
    "cursor": "#32d9e8",
    "direction": "#f2b000",
}

GLOBAL_STYLE = f"""
QWidget {{
    background: {COLORS['background']};
    color: {COLORS['text']};
    font-size: 10pt;
}}
QMainWindow {{ background: {COLORS['background']}; }}
QLabel {{ background: transparent; }}
QFrame#Card {{
    background: {COLORS['panel']};
    border: 1px solid {COLORS['border_soft']};
    border-radius: 6px;
}}
QFrame#Toolbar {{
    background: {COLORS['panel']};
    border: 1px solid {COLORS['border_soft']};
    border-radius: 6px;
}}
QFrame#Separator {{ background: {COLORS['border_soft']}; border: 0; max-height: 1px; min-height: 1px; }}
QFrame#ToolbarDivider {{ background: {COLORS['border']}; border: 0; max-width: 1px; min-width: 1px; }}
QGroupBox {{
    background: transparent;
    border: 1px solid {COLORS['border_soft']};
    border-radius: 6px;
    margin-top: 8px;
    padding: 10px 7px 7px 7px;
    font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 9px; padding: 0 4px; color: {COLORS['accent']}; }}
QLabel#SectionTitle {{
    color: {COLORS['faint']};
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 1px;
}}
QLineEdit, QComboBox, QDoubleSpinBox, QTextEdit, QTextBrowser {{
    background: {COLORS['field']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    padding: 4px 7px;
    selection-background-color: {COLORS['accent']};
}}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{ border-color: {COLORS['accent']}; }}
QComboBox QAbstractItemView {{
    background: {COLORS['field']};
    color: {COLORS['text']};
    selection-background-color: #315d91;
}}
QPushButton {{
    background: #2c323d;
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    padding: 6px 11px;
}}
QPushButton:hover {{ border-color: {COLORS['accent']}; background: #353d4b; }}
QPushButton:pressed {{ background: #232b36; }}
QPushButton:disabled {{ color: {COLORS['faint']}; background: #23272f; border-color: {COLORS['border_soft']}; }}
QPushButton#PrimaryButton {{
    background: {COLORS['accent']};
    color: #06121f;
    border-color: {COLORS['accent']};
    font-weight: 700;
}}
QPushButton#PrimaryButton:hover {{ background: #66adff; }}
QPushButton#PrimaryButton:disabled {{ background: #2f4560; color: #7f93a8; border-color: #2f4560; }}
QPushButton#DangerButton {{ color: #ffd5d5; border-color: #7e4147; }}
/* Three of these share one row and take their width from it, so the
   padding only ever decides how narrow the row is allowed to get. */
QPushButton#SaveButton {{ padding: 6px 5px; }}
QPushButton#IconButton {{ padding: 4px 8px; background: transparent; border-color: transparent; color: {COLORS['text']}; }}
QPushButton#IconButton:hover {{ background: #333a47; color: {COLORS['text']}; }}
QPushButton#IconButton:checked {{ background: #2b4258; color: {COLORS['accent']}; }}
QPushButton#Segment {{
    background: {COLORS['field']};
    color: {COLORS['dim']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    padding: 5px 9px;
    font-weight: 600;
}}
QPushButton#Segment:hover {{ color: {COLORS['text']}; border-color: {COLORS['accent']}; }}
QPushButton#Segment:checked {{
    background: {COLORS['accent']};
    color: #06121f;
    border-color: {COLORS['accent']};
}}
QPushButton#Segment:disabled {{ color: #545c6a; background: #1a1d24; border-color: {COLORS['border_soft']}; }}
QPushButton#Segment[tone="yes"]:checked {{ background: {COLORS['success']}; border-color: {COLORS['success']}; color: #06180d; }}
QPushButton#Segment[tone="no"]:checked {{ background: {COLORS['danger']}; border-color: {COLORS['danger']}; color: #200707; }}
QPushButton#Segment[tone="neutral"]:checked {{ background: #545c6b; border-color: #545c6b; color: {COLORS['text']}; }}
QPushButton#SectionHeader {{
    background: transparent;
    border: 0;
    padding: 5px 2px;
    color: {COLORS['faint']};
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 1px;
    text-align: left;
}}
QPushButton#SectionHeader:hover {{ color: {COLORS['text']}; }}
QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; }}
QListWidget {{
    background: {COLORS['field']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    padding: 3px;
    outline: 0;
}}
QListWidget::item {{ padding: 6px 7px; border-radius: 4px; }}
QListWidget::item:selected {{ background: #2f5c8f; color: #ffffff; }}
QListWidget::item:hover {{ background: #2a3240; }}
QTextBrowser, QTextEdit {{ line-height: 1.25; }}
QTabWidget::pane {{ border: 1px solid {COLORS['border_soft']}; border-radius: 5px; top: -1px; }}
QTabBar::tab {{
    background: transparent;
    color: {COLORS['faint']};
    border: 1px solid transparent;
    border-bottom: 0;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    padding: 4px 11px;
    margin-right: 2px;
    font-size: 9pt;
}}
QTabBar::tab:selected {{
    background: {COLORS['field']};
    color: {COLORS['text']};
    border-color: {COLORS['border_soft']};
}}
QTabBar::tab:hover {{ color: {COLORS['text']}; }}
QScrollBar:vertical {{ width: 10px; background: transparent; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #414958; border-radius: 5px; min-height: 26px; }}
QScrollBar::handle:vertical:hover {{ background: #4f5867; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: transparent; width: 6px; height: 6px; }}
QSlider::groove:horizontal {{ height: 4px; background: #333a47; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {COLORS['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #e8edf5; width: 11px; margin: -5px 0; border-radius: 6px; }}
QSlider::handle:horizontal:disabled {{ background: #4a5160; }}
QStatusBar {{ background: {COLORS['background']}; color: {COLORS['dim']}; border-top: 1px solid {COLORS['border_soft']}; }}
QStatusBar::item {{ border: 0; }}
QMenu {{
    background: {COLORS['panel']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    padding: 4px;
}}
QMenu::item {{ padding: 5px 22px 5px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: #2f5c8f; color: #ffffff; }}
QMenu::item:disabled {{ color: {COLORS['faint']}; }}
QMenu::item:disabled:selected {{ background: transparent; color: {COLORS['faint']}; }}
QMenu::separator {{ height: 1px; background: {COLORS['border_soft']}; margin: 4px 6px; }}
QToolTip {{
    background: #10131a;
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    padding: 5px 7px;
}}
"""


def _label(
    text: str,
    *,
    color: str | None = None,
    bold: bool = False,
    size: int | None = None,
    wrap: bool = True,
) -> QLabel:
    # Wrapping by default matters for layout, not just looks: a QLabel that
    # cannot wrap reports its whole text as its minimum width, and one long
    # help sentence is enough to stop a whole panel from ever shrinking.
    widget = QLabel(text)
    widget.setWordWrap(wrap)
    style: list[str] = []
    if color:
        style.append(f"color:{color};")
    if bold:
        style.append("font-weight:600;")
    if size:
        style.append(f"font-size:{size}pt;")
    if style:
        widget.setStyleSheet("".join(style))
    return widget


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _verification_text(value: Any) -> str:
    if value == 1 or value == "1":
        return "Yes (1)"
    if value == 0 or value == "0":
        return "No (0)"
    return "Not set"


def _short_status(verify: Any, comment: Any = None) -> str:
    """A verdict in as few characters as a list row can spare.

    The long forms ("Verified", "Not verified", "Yes (1)") are what This
    finding and the reports print.  In a row they cost real width for no
    information: "Yes (1)" is 19px more than "Yes", and an untouched finding
    reading "me: Not set  ·  source: Not set" is 170px against 112 for two
    dashes.  Yes and No are the words on the verdict buttons; a dash is
    nothing recorded, which the row's tooltip spells out.
    """

    if verify in (1, "1"):
        base = "Yes"
    elif verify in (0, "0"):
        base = "No"
    else:
        base = "—"
    return f"{base} · note" if str(comment or "").strip() else base


def _report_status(verify: Any, comment: Any) -> str:
    if verify in (1, "1"):
        base = "Verified"
    elif verify in (0, "0"):
        base = "Not verified"
    else:
        base = "Not set"
    return f"{base} · comment" if str(comment or "").strip() else base


def _human_count(count: int, singular: str, plural: str | None = None) -> str:
    plural = plural or f"{singular}s"
    return f"{count} {singular if count == 1 else plural}"


class ElidedLabel(QLabel):
    """A single-line label that shrinks by eliding rather than by demanding space.

    A plain QLabel reports its whole text as its minimum width, so one long
    status string in a toolbar squeezes every control next to it.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = text
        # Preferred, not Ignored: the label asks for its full width and gets it
        # whenever the row has room, and only gives way -- by eliding -- when
        # the controls beside it need the space.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._apply()

    def sizeHint(self):  # noqa: N802 - Qt API
        hint = super().sizeHint()
        hint.setWidth(self.fontMetrics().horizontalAdvance(self._full_text) + 4)
        return hint

    def minimumSizeHint(self):  # noqa: N802 - Qt API
        hint = super().minimumSizeHint()
        hint.setWidth(48)
        return hint

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._full_text = str(text)
        self._apply()

    def text(self) -> str:  # noqa: N802 - Qt API
        return self._full_text

    def _apply(self) -> None:
        metrics = self.fontMetrics()
        available = max(0, self.width() - 2)
        elided = metrics.elidedText(self._full_text, Qt.TextElideMode.ElideRight, available)
        super().setText(elided)
        super().setToolTip(self._full_text if elided != self._full_text else "")

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._apply()
        super().resizeEvent(event)


class VerticalLabel(QWidget):
    """Text turned on its side, for a rail too narrow to read across.

    Stacking the letters one per line is what a 26px strip forces on an
    ordinary label, and it reads as a column of unrelated capitals.  Rotating
    the baseline keeps it a word.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = text
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._text = str(text)
        self.updateGeometry()
        self.update()

    def text(self) -> str:
        return self._text

    def sizeHint(self):  # noqa: N802 - Qt API
        metrics = self.fontMetrics()
        return QSize(metrics.height() + 2, metrics.horizontalAdvance(self._text) + 12)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setPen(QColor(COLORS["dim"]))
        # Bottom-to-top, the way a spine is read.
        painter.translate(0, self.height())
        painter.rotate(-90)
        painter.drawText(
            QRectF(0, 0, self.height(), self.width()),
            int(Qt.AlignmentFlag.AlignCenter),
            self._text,
        )
        painter.end()


def _stroke_icon(kind: str, size: int = 15) -> QIcon:
    """A line drawing of a tool, in both the colours the button can wear.

    Drawn rather than shipped: three glyphs are not worth a binary asset that
    can go missing from an install, and drawing them means they follow the
    palette.  A checked tool button turns light with dark text, so the icon
    carries a dark variant under ``State.On`` and Qt swaps it for us instead
    of the icon disappearing into the highlight.
    """

    icon = QIcon()
    for state, colour in (
        (QIcon.State.Off, QColor(COLORS["dim"])),
        (QIcon.State.On, QColor("#06121f")),
    ):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(colour, 1.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        middle = size / 2.0
        if kind == "point":
            # A crosshair with a gap, the same shape it leaves in the view.
            gap = size * 0.17
            painter.drawLine(QPointF(middle, 1.0), QPointF(middle, middle - gap))
            painter.drawLine(QPointF(middle, middle + gap), QPointF(middle, size - 1.0))
            painter.drawLine(QPointF(1.0, middle), QPointF(middle - gap, middle))
            painter.drawLine(QPointF(middle + gap, middle), QPointF(size - 1.0, middle))
            painter.drawEllipse(QPointF(middle, middle), size * 0.13, size * 0.13)
        elif kind == "brush":
            # A handle running to a round tip, tip towards the lower left.
            painter.drawLine(QPointF(size - 2.0, 2.0), QPointF(size * 0.42, size * 0.58))
            painter.drawEllipse(QPointF(size * 0.34, size * 0.66), size * 0.2, size * 0.2)
        elif kind in ("collapse-left", "expand-right"):
            # The panel-and-chevron mark every editor uses for "hide this
            # side": a bare chevron on its own is too quiet to find, which is
            # exactly what happened to the one that was here.
            bar = size * 0.22 if kind == "collapse-left" else size * 0.78
            painter.drawLine(QPointF(bar, 1.5), QPointF(bar, size - 1.5))
            painter.drawRect(QRectF(1.0, 1.5, size - 2.0, size - 3.0))
            tip = size * 0.42 if kind == "collapse-left" else size * 0.58
            back = size * 0.62 if kind == "collapse-left" else size * 0.38
            painter.drawLine(QPointF(back, middle - size * 0.18), QPointF(tip, middle))
            painter.drawLine(QPointF(tip, middle), QPointF(back, middle + size * 0.18))
        elif kind == "eraser":
            # A slanted block, wiping along its lower edge.
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(size * 0.30, size * 0.62),
                        QPointF(size * 0.62, size * 0.20),
                        QPointF(size * 0.88, size * 0.42),
                        QPointF(size * 0.56, size * 0.84),
                    ]
                )
            )
            painter.drawLine(QPointF(1.5, size - 1.5), QPointF(size * 0.62, size - 1.5))
        painter.end()
        icon.addPixmap(pixmap, QIcon.Mode.Normal, state)
        icon.addPixmap(pixmap, QIcon.Mode.Active, state)
    return icon


def _section_title(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("SectionTitle")
    return label


def _separator() -> QFrame:
    line = QFrame()
    line.setObjectName("Separator")
    line.setFrameShape(QFrame.Shape.HLine)
    return line


class ViewerSettings:
    """Reader preferences that shape the reading loop.

    These are workstation preferences rather than review data, so they live in
    a per-user ini file instead of the shared review database.
    """

    def __init__(self, store: QSettings | None = None) -> None:
        self.store = store or QSettings(
            QSettings.Format.IniFormat,
            QSettings.Scope.UserScope,
            "MicrobleedReview",
            "Viewer",
        )

    def _bool(self, key: str, default: bool) -> bool:
        value = self.store.value(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @property
    def auto_zoom(self) -> bool:
        return self._bool("reading/auto_zoom", True)

    @property
    def lesion_fov_mm(self) -> float:
        value = _safe_float(self.store.value("reading/lesion_fov_mm", DEFAULT_LESION_FOV_MM))
        if value is None:
            return DEFAULT_LESION_FOV_MM
        return float(min(max(value, MIN_LESION_FOV_MM), MAX_LESION_FOV_MM))

    @property
    def save_advances(self) -> bool:
        return self._bool("reading/save_advances", True)

    @property
    def keep_tool_on_switch(self) -> bool:
        """Leave the tab and the tool alone when the finding changes.

        Off by default: arriving at a finding nobody has judged yet with a
        brush in hand and the Segment tab open offers the one thing that
        cannot be done there yet.  A reader who is going through their own
        segmentations rather than judging can turn it on.
        """

        return self._bool("reading/keep_tool_on_switch", False)

    @property
    def default_modality(self) -> str:
        value = str(self.store.value("reading/default_modality", "swi") or "swi").lower()
        return value if value in set(MODALITY_ORDER) | {"last"} else "swi"

    @property
    def prefetch_enabled(self) -> bool:
        return self._bool("reading/prefetch", True)

    @property
    def sticky_window(self) -> bool:
        return self._bool("display/sticky_window", False)

    def set_sticky_window(self, sticky: bool) -> None:
        self.store.setValue("display/sticky_window", bool(sticky))
        self.store.sync()

    @property
    def snap_to_lesion(self) -> bool:
        """Let ``Move here`` settle onto the focus instead of the click."""

        return self._bool("reading/snap_to_lesion", True)

    @property
    def snap_radius_mm(self) -> float:
        value = _safe_float(self.store.value("reading/snap_radius_mm", 2.0))
        if value is None:
            return 2.0
        return float(min(max(value, 0.5), 5.0))

    @property
    def scroll_moves_cursor(self) -> bool:
        """Whether scrolling one view moves the shared cursor in the others.

        Off by default, which is what this viewer has always done and what
        makes scrolling a way to check whether a focus persists across slices.
        On is the ITK-SNAP habit, where the 3D cursor follows the wheel.
        """

        return self._bool("reading/scroll_moves_cursor", False)

    @property
    def smooth_zoom(self) -> bool:
        """Interpolate the image when magnified, instead of showing squares."""

        return self._bool("display/smooth_zoom", False)

    @property
    def roi_outline(self) -> bool:
        """Draw segmentations as an edge rather than a filled wash."""

        return self._bool("display/roi_outline", False)

    def set_roi_outline(self, outline: bool) -> None:
        self.store.setValue("display/roi_outline", bool(outline))
        self.store.sync()

    @property
    def orientation(self) -> str:
        value = str(self.store.value("display/orientation", DEFAULT_ORIENTATION) or "").lower()
        return value if value in ORIENTATION_PRESETS else DEFAULT_ORIENTATION

    @property
    def axcodes(self) -> tuple[str, str, str]:
        return preset_axcodes(self.orientation)

    def update(
        self,
        *,
        auto_zoom: bool,
        lesion_fov_mm: float,
        save_advances: bool,
        default_modality: str,
        keep_tool_on_switch: bool | None = None,
        orientation: str | None = None,
        prefetch: bool | None = None,
        snap_to_lesion: bool | None = None,
        scroll_moves_cursor: bool | None = None,
        smooth_zoom: bool | None = None,
    ) -> None:
        self.store.setValue("reading/auto_zoom", bool(auto_zoom))
        self.store.setValue("reading/lesion_fov_mm", float(lesion_fov_mm))
        self.store.setValue("reading/save_advances", bool(save_advances))
        if keep_tool_on_switch is not None:
            self.store.setValue("reading/keep_tool_on_switch", bool(keep_tool_on_switch))
        self.store.setValue("reading/default_modality", str(default_modality))
        if prefetch is not None:
            self.store.setValue("reading/prefetch", bool(prefetch))
        if snap_to_lesion is not None:
            self.store.setValue("reading/snap_to_lesion", bool(snap_to_lesion))
        if scroll_moves_cursor is not None:
            self.store.setValue("reading/scroll_moves_cursor", bool(scroll_moves_cursor))
        if smooth_zoom is not None:
            self.store.setValue("display/smooth_zoom", bool(smooth_zoom))
        if orientation is not None:
            self.store.setValue("display/orientation", str(orientation))
        self.store.sync()

    # ------------------------------------------------------------ shortcuts --
    def shortcut(self, action: str) -> str:
        stored = self.store.value(f"shortcuts/{action}", None)
        if stored is None:
            return SHORTCUT_DEFAULTS.get(action, "")
        return str(stored)

    def shortcuts(self) -> dict[str, str]:
        return {action: self.shortcut(action) for action in SHORTCUT_DEFAULTS}

    def set_shortcuts(self, bindings: dict[str, str]) -> None:
        for action, sequence in bindings.items():
            if action not in SHORTCUT_DEFAULTS:
                continue
            self.store.setValue(f"shortcuts/{action}", str(sequence))
        self.store.sync()

    def reset_shortcuts(self) -> None:
        for action in SHORTCUT_DEFAULTS:
            self.store.remove(f"shortcuts/{action}")
        self.store.sync()

    def last_round(self, db_path: Path, reader_id: str) -> int | None:
        """The round this reader opened last time, for this database."""

        value = self.store.value(self._round_key(db_path, reader_id), None)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def set_last_round(self, db_path: Path, reader_id: str, review_round: int) -> None:
        self.store.setValue(self._round_key(db_path, reader_id), int(review_round))
        self.store.sync()

    @staticmethod
    def _round_key(db_path: Path, reader_id: str) -> str:
        return f"rounds/{Path(db_path).resolve()}|{reader_id}"

    # ------------------------------------------------------------- datasets --
    def recent_datasets(self) -> list[Dataset]:
        raw = self.store.value("datasets/recent", "")
        try:
            entries = json.loads(str(raw)) if raw else []
        except (TypeError, ValueError):
            entries = []
        datasets: list[Dataset] = []
        for entry in entries if isinstance(entries, list) else []:
            dataset = Dataset.from_dict(entry) if isinstance(entry, dict) else None
            if dataset is not None and dataset not in datasets:
                datasets.append(dataset)
        return datasets

    def remember_dataset(self, dataset: Dataset, limit: int = 8) -> None:
        entries = [dataset] + [item for item in self.recent_datasets() if item != dataset]
        self.store.setValue(
            "datasets/recent",
            json.dumps([item.as_dict() for item in entries[:limit]], ensure_ascii=False),
        )
        self.store.sync()

    @property
    def queue_pinned(self) -> bool:
        return self._bool("display/queue_pinned", True)

    def set_queue_pinned(self, pinned: bool) -> None:
        self.store.setValue("display/queue_pinned", bool(pinned))
        self.store.sync()

    @property
    def right_column_width(self) -> int:
        raw = self.store.value("display/right_column_width", RIGHT_COLUMN_DEFAULT_WIDTH)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return RIGHT_COLUMN_DEFAULT_WIDTH

    def set_right_column_width(self, width: int) -> None:
        self.store.setValue("display/right_column_width", int(width))

    @property
    def sidebar_split(self) -> list[int] | None:
        raw = self.store.value("display/sidebar_split", None)
        try:
            sizes = [int(value) for value in raw]
        except (TypeError, ValueError):
            return None
        return sizes if len(sizes) == 2 and sum(sizes) > 0 else None

    def set_sidebar_split(self, sizes: Iterable[int]) -> None:
        self.store.setValue("display/sidebar_split", [int(value) for value in sizes])
        self.store.sync()

    def section_expanded(self, key: str, default: bool) -> bool:
        return self._bool(f"sections/{key}", default)

    def set_section_expanded(self, key: str, expanded: bool) -> None:
        self.store.setValue(f"sections/{key}", bool(expanded))


class SegmentedControl(QWidget):
    """A compact exclusive button row used for modality and verdict."""

    selected = Signal(str)

    def __init__(
        self,
        options: list[tuple[str, str]],
        *,
        tones: dict[str, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._buttons: dict[str, QPushButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._emitting = True
        for key, text in options:
            button = QPushButton(text)
            button.setObjectName("Segment")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            if tones and key in tones:
                button.setProperty("tone", tones[key])
            self._group.addButton(button)
            # "Minimum" means the size hint *is* the floor, so the row can
            # stretch a button but never squeeze one below its own label.  A
            # width snapshotted here would be the unstyled one -- the
            # application stylesheet has not been applied yet.
            button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            layout.addWidget(button, 1)
            self._buttons[key] = button
            button.clicked.connect(lambda _checked=False, name=key: self._on_clicked(name))

    def minimumSizeHint(self):  # noqa: N802 - Qt API
        """Never narrower than the labels it carries.

        The layout caches a minimum computed before the application stylesheet
        is applied, and setting a size policy afterwards does not always
        invalidate it -- so the row would hand each button 52px for a label
        that needs 59 and elide "SWI" to "SW…".  Asking the buttons directly
        cannot go stale.
        """

        hint = super().minimumSizeHint()
        spacing = self.layout().spacing() * max(0, len(self._buttons) - 1)
        needed = sum(button.sizeHint().width() for button in self._buttons.values()) + spacing
        return QSize(max(hint.width(), needed), hint.height())

    def _on_clicked(self, key: str) -> None:
        if self._emitting:
            self.selected.emit(key)

    def button(self, key: str) -> QPushButton | None:
        return self._buttons.get(key)

    def current_key(self) -> str | None:
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return None

    def set_current_key(self, key: str | None) -> None:
        """Check a segment without emitting ``selected``."""

        self._emitting = False
        try:
            self._group.setExclusive(False)
            for name, button in self._buttons.items():
                button.setChecked(name == key)
            self._group.setExclusive(True)
        finally:
            self._emitting = True

    def set_key_enabled(self, key: str, enabled: bool, *, tooltip: str = "") -> None:
        button = self._buttons.get(key)
        if button is None:
            return
        button.setEnabled(bool(enabled))
        button.setToolTip(tooltip)


class CollapsibleSection(QWidget):
    """A titled section that folds away controls used only occasionally."""

    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        *,
        expanded: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.header = QPushButton()
        self.header.setObjectName("SectionHeader")
        self.header.setCheckable(True)
        self.header.setChecked(bool(expanded))
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.toggled.connect(self._on_toggled)
        layout.addWidget(self.header)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 4)
        self.content_layout.setSpacing(6)
        self.content.setVisible(bool(expanded))
        layout.addWidget(self.content)
        self._badge = ""
        self._refresh_header()

    def _on_toggled(self, checked: bool) -> None:
        self.content.setVisible(bool(checked))
        self._refresh_header()
        self.toggled.emit(bool(checked))

    def _refresh_header(self) -> None:
        arrow = "▾" if self.header.isChecked() else "▸"
        badge = f"   {self._badge}" if self._badge else ""
        self.header.setText(f"{arrow}  {self._title.upper()}{badge}")

    def set_badge(self, text: str) -> None:
        self._badge = str(text or "")
        self._refresh_header()

    def add_widget(self, widget: QWidget) -> None:
        self.content_layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self.content_layout.addLayout(layout)

    def is_expanded(self) -> bool:
        return self.header.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.header.setChecked(bool(expanded))


class SettingsDialog(QDialog):
    """Reading, display and keyboard preferences."""

    def __init__(self, settings: ViewerSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Microbleed Review · Preferences")
        self.setMinimumWidth(520)
        self.setMinimumHeight(430)
        root = QVBoxLayout(self)
        root.setSpacing(11)
        root.setContentsMargins(14, 12, 14, 12)
        tabs = QTabWidget()
        tabs.addTab(self._build_reading_tab(), "Reading")
        tabs.addTab(self._build_display_tab(), "Display")
        tabs.addTab(self._build_shortcut_tab(), "Shortcuts")
        root.addWidget(tabs, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.fov_spin.setEnabled(self.auto_zoom_cb.isChecked())

    def _build_reading_tab(self) -> QWidget:
        settings = self.settings
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        self.auto_zoom_cb = QCheckBox("Zoom to the lesion when a finding is selected")
        self.auto_zoom_cb.setChecked(settings.auto_zoom)
        layout.addWidget(self.auto_zoom_cb)

        fov_row = QHBoxLayout()
        fov_row.setContentsMargins(20, 0, 0, 0)
        fov_row.addWidget(_label("Lesion field of view:", color=COLORS["dim"], size=9))
        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(MIN_LESION_FOV_MM, MAX_LESION_FOV_MM)
        self.fov_spin.setDecimals(0)
        self.fov_spin.setSingleStep(5.0)
        self.fov_spin.setSuffix(" mm")
        self.fov_spin.setValue(settings.lesion_fov_mm)
        fov_row.addWidget(self.fov_spin)
        fov_row.addStretch(1)
        layout.addLayout(fov_row)
        self.auto_zoom_cb.toggled.connect(self.fov_spin.setEnabled)

        layout.addWidget(_separator())
        self.advance_cb = QCheckBox("Saving a review moves to the next finding or case")
        self.advance_cb.setChecked(settings.save_advances)
        layout.addWidget(self.advance_cb)
        self.keep_tool_cb = QCheckBox("Switching finding or case keeps the current tab and tool")
        self.keep_tool_cb.setChecked(settings.keep_tool_on_switch)
        self.keep_tool_cb.setToolTip(
            "Off: a finding nobody has judged yet opens on Review with the Point\n"
            "tool, which is what deciding where it is needs.  A finding already\n"
            "judged never moves you -- going back over your own segmentations\n"
            "keeps the Segment tab either way.\n"
            "On: nothing moves, ever."
        )
        layout.addWidget(self.keep_tool_cb)

        layout.addWidget(_separator())
        modality_row = QHBoxLayout()
        modality_row.addWidget(_label("Open cases on:", color=COLORS["dim"], size=9))
        self.modality_combo = QComboBox()
        for key in MODALITY_BUTTON_ORDER:
            self.modality_combo.addItem(MODALITY_LABELS[key], key)
        self.modality_combo.addItem("Last used", "last")
        index = self.modality_combo.findData(settings.default_modality)
        self.modality_combo.setCurrentIndex(max(0, index))
        modality_row.addWidget(self.modality_combo)
        modality_row.addStretch(1)
        layout.addLayout(modality_row)
        note = _label(
            "A case that is missing the preferred sequence opens on the first "
            "sequence it does have.",
            color=COLORS["dim"],
            size=8,
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addWidget(_separator())
        self.snap_cb = QCheckBox("Move here settles onto the focus under the click")
        self.snap_cb.setChecked(settings.snap_to_lesion)
        self.snap_cb.setToolTip(
            "Nobody can click the centre of a 3 mm lesion by eye, and the\n"
            "coordinate recorded is the one the analysis uses. The darkest\n"
            "voxel on SWI (brightest on QSM) within a couple of millimetres is\n"
            "both a better answer and the same answer between readers."
        )
        layout.addWidget(self.snap_cb)

        self.scroll_cursor_cb = QCheckBox("Scrolling moves the cursor in the other views")
        self.scroll_cursor_cb.setChecked(settings.scroll_moves_cursor)
        self.scroll_cursor_cb.setToolTip(
            "Off: scrolling is local, so a view can be scrolled off the finding\n"
            "to check whether a focus persists across slices.\n"
            "On: the shared cursor follows the wheel, as it does in ITK-SNAP."
        )
        layout.addWidget(self.scroll_cursor_cb)

        self.prefetch_cb = QCheckBox("Read the next case in the background")
        self.prefetch_cb.setChecked(settings.prefetch_enabled)
        self.prefetch_cb.setToolTip(
            "Makes moving to the next case almost instant.\n"
            "Turn it off to rule the background reader out if the viewer misbehaves."
        )
        layout.addWidget(self.prefetch_cb)
        layout.addStretch(1)
        return page

    def _build_display_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(9)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(_label("Left-right display convention", bold=True, size=10))
        self.orientation_combo = QComboBox()
        for key, entry in ORIENTATION_PRESETS.items():
            axcodes = "".join(preset_axcodes(key))
            self.orientation_combo.addItem(f"{entry['label']}  ·  {axcodes}", key)
        index = self.orientation_combo.findData(self.settings.orientation)
        self.orientation_combo.setCurrentIndex(max(0, index))
        layout.addWidget(self.orientation_combo)
        self.orientation_note = _label("", color=COLORS["dim"], size=9)
        self.orientation_note.setWordWrap(True)
        layout.addWidget(self.orientation_note)
        self.orientation_combo.currentIndexChanged.connect(lambda _index: self._describe_orientation())
        self._describe_orientation()
        layout.addWidget(_separator())
        layout.addWidget(
            _label(
                "The preset only mirrors the display. Findings, manual coordinates and "
                "everything stored in the review database stay in physical RAS, and the "
                "direction labels are derived from the preset, so they always match the "
                "pixels on screen.\n\n"
                "Changing this reloads the current case.",
                color=COLORS["dim"],
                size=9,
            )
        )
        layout.addWidget(_separator())
        self.smooth_cb = QCheckBox("Smooth the image when magnified")
        self.smooth_cb.setChecked(self.settings.smooth_zoom)
        self.smooth_cb.setToolTip(
            "Off shows the voxels as they are, which is the honest default.\n"
            "On interpolates them, which some readers find easier for judging\n"
            "whether a 3-voxel focus is round. It changes the picture, not the\n"
            "data: measurements and segmentation are unaffected."
        )
        layout.addWidget(self.smooth_cb)
        for widget in page.findChildren(QLabel):
            widget.setWordWrap(True)
        layout.addStretch(1)
        return page

    def _describe_orientation(self) -> None:
        key = str(self.orientation_combo.currentData() or DEFAULT_ORIENTATION)
        entry = ORIENTATION_PRESETS.get(key, ORIENTATION_PRESETS[DEFAULT_ORIENTATION])
        axial = plane_directions("axial", preset_axcodes(key))
        self.orientation_note.setText(
            f"{entry['summary']}. Axial view: {axial[0]} on the left, {axial[1]} on the right."
        )

    def _build_shortcut_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(7)
        layout.setContentsMargins(12, 12, 12, 12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        form = QVBoxLayout(holder)
        form.setContentsMargins(0, 0, 8, 0)
        form.setSpacing(5)
        self.shortcut_edits: dict[str, QKeySequenceEdit] = {}
        current = self.settings.shortcuts()
        group_seen: set[str] = set()
        for action, label, _default, group in SHORTCUT_ACTIONS:
            if group not in group_seen:
                group_seen.add(group)
                if group_seen != {group}:
                    form.addSpacing(4)
                form.addWidget(_section_title(group))
            row = QHBoxLayout()
            row.setSpacing(7)
            name = _label(label, size=9)
            name.setMinimumWidth(210)
            row.addWidget(name, 1)
            edit = QKeySequenceEdit(QKeySequence(current.get(action, "")))
            edit.setMaximumWidth(150)
            row.addWidget(edit)
            reset = QPushButton("Default")
            reset.setObjectName("IconButton")
            reset.setToolTip(f"Restore {SHORTCUT_DEFAULTS.get(action) or 'no shortcut'}")
            reset.clicked.connect(
                lambda _checked=False, name=action: self.shortcut_edits[name].setKeySequence(
                    QKeySequence(SHORTCUT_DEFAULTS.get(name, ""))
                )
            )
            row.addWidget(reset)
            form.addLayout(row)
            self.shortcut_edits[action] = edit
        form.addStretch(1)
        scroll.setWidget(holder)
        layout.addWidget(scroll, 1)
        footer = QHBoxLayout()
        self.shortcut_warning = _label("", color=COLORS["danger"], size=9)
        self.shortcut_warning.setWordWrap(True)
        footer.addWidget(self.shortcut_warning, 1)
        reset_all = QPushButton("Restore all defaults")
        reset_all.clicked.connect(self._reset_all_shortcuts)
        footer.addWidget(reset_all)
        layout.addLayout(footer)
        hint = _label(
            "Click a field and press the combination. Single letters are safe: text "
            "fields keep receiving them while they have focus.",
            color=COLORS["dim"],
            size=8,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return page

    def _reset_all_shortcuts(self) -> None:
        for action, edit in self.shortcut_edits.items():
            edit.setKeySequence(QKeySequence(SHORTCUT_DEFAULTS.get(action, "")))
        self.shortcut_warning.setText("")

    def _collect_shortcuts(self) -> dict[str, str] | None:
        """Read the editors back, refusing a binding used twice."""

        bindings: dict[str, str] = {}
        used: dict[str, str] = {}
        for action, edit in self.shortcut_edits.items():
            sequence = edit.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
            bindings[action] = sequence
            if not sequence:
                continue
            key = sequence.lower()
            if key in used:
                self.shortcut_warning.setText(
                    f"“{sequence}” is assigned to both {SHORTCUT_LABELS[used[key]]} and "
                    f"{SHORTCUT_LABELS[action]}. Give one of them a different key."
                )
                return None
            used[key] = action
        self.shortcut_warning.setText("")
        return bindings

    def _apply(self) -> None:
        bindings = self._collect_shortcuts()
        if bindings is None:
            return
        self.settings.update(
            auto_zoom=self.auto_zoom_cb.isChecked(),
            lesion_fov_mm=float(self.fov_spin.value()),
            save_advances=self.advance_cb.isChecked(),
            keep_tool_on_switch=self.keep_tool_cb.isChecked(),
            default_modality=str(self.modality_combo.currentData() or "swi"),
            orientation=str(self.orientation_combo.currentData() or DEFAULT_ORIENTATION),
            prefetch=self.prefetch_cb.isChecked(),
            snap_to_lesion=self.snap_cb.isChecked(),
            scroll_moves_cursor=self.scroll_cursor_cb.isChecked(),
            smooth_zoom=self.smooth_cb.isChecked(),
        )
        self.settings.set_shortcuts(bindings)
        self.accept()


class DatabaseWriter(QThread):
    """Runs the frequent, non-critical database writes off the GUI thread.

    The review store often lives in a synchronised folder (OneDrive here), and
    a sync client or virus scanner can hold the file for seconds.  SQLite is
    configured to wait for the lock, so doing these writes on the GUI thread
    freezes the window until the lock clears -- which is what "not responding"
    looked like.  Reviews are still saved synchronously: the reader has to know
    a verdict was stored.  Only the operation log and the session state, which
    nobody waits for, are queued here.
    """

    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._queue: "Queue[tuple[str, Any] | None]" = Queue()
        self._abandon = False

    def submit(self, description: str, work: Any) -> None:
        self._queue.put((str(description), work))

    def pending(self) -> int:
        return self._queue.qsize()

    def stop(self, timeout_ms: int = 8000) -> None:
        """Drain what is queued, then end the thread -- without killing it.

        ``QThread::terminate`` unwinds a thread at an arbitrary instruction,
        and this one spends its time inside SQLite commits, so killing it is
        the one way this design could actually damage the database.  If the
        graceful drain runs out of time the queue is abandoned instead: these
        writes are the ones nobody waits for, and the thread only has to
        finish the single call already in flight.  That call is bounded by
        ``BACKGROUND_BUSY_TIMEOUT_MS`` rather than the GUI's thirty seconds.
        """

        if not self.isRunning():
            return
        self._queue.put(None)
        if self.wait(timeout_ms):
            return
        # Out of patience: stop starting new work and wait out the one running.
        self._abandon = True
        self._queue.put(None)
        if not self.wait(max(BACKGROUND_BUSY_TIMEOUT_MS + 2000, timeout_ms)):
            # Nothing safe is left to do; say so rather than kill it.
            self.failed.emit(
                "The background database writer did not finish. Some operation "
                "log entries may be missing; reviews are unaffected."
            )

    def run(self) -> None:
        # This thread's writes are discardable, so it gives up on a locked file
        # quickly instead of holding the shutdown open.
        set_busy_timeout_ms(BACKGROUND_BUSY_TIMEOUT_MS)
        while True:
            item = self._queue.get()
            if item is None or self._abandon:
                return
            description, work = item
            try:
                work()
            except Exception as exc:
                self.failed.emit(f"{description}: {type(exc).__name__}: {exc}")


class VolumeCache:
    """A small LRU of loaded volumes, keyed by file path and orientation.

    Three volumes are about 140 MB, so the limit is deliberately low: enough
    for the case on screen plus the one being prefetched behind it.
    """

    def __init__(self, limit: int = 6) -> None:
        self.limit = int(limit)
        self._entries: "OrderedDict[tuple[str, tuple[str, ...]], Volume]" = OrderedDict()

    @staticmethod
    def key(path: str, axcodes: Iterable[str]) -> tuple[str, tuple[str, ...]]:
        return str(path), tuple(str(code).upper() for code in axcodes)

    def get(self, path: str, axcodes: Iterable[str]) -> Volume | None:
        key = self.key(path, axcodes)
        volume = self._entries.get(key)
        if volume is not None:
            self._entries.move_to_end(key)
        return volume

    def put(self, path: str, axcodes: Iterable[str], volume: Volume) -> None:
        key = self.key(path, axcodes)
        self._entries[key] = volume
        self._entries.move_to_end(key)
        while len(self._entries) > self.limit:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def __contains__(self, item: tuple[str, Iterable[str]]) -> bool:
        path, axcodes = item
        return self.key(path, axcodes) in self._entries

    def __len__(self) -> int:
        return len(self._entries)


class PrefetchWorker(QObject):
    """Reads NIfTI files off the GUI thread.

    It touches no Qt widgets and no viewer state: it reads files and emits the
    result, which the window accepts only if its generation still matches.
    """

    loaded = Signal(int, str, object)
    finished = Signal(int)

    def __init__(self, generation: int, paths: list[str], axcodes: tuple[str, ...]) -> None:
        super().__init__()
        self.generation = int(generation)
        self.paths = list(paths)
        self.axcodes = tuple(axcodes)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        for path in self.paths:
            if self._cancelled:
                break
            try:
                volume = load_volume(path, self.axcodes)
            except Exception:
                # A prefetch failure is not an error the reader needs to see;
                # the synchronous load will report it if the case is opened.
                continue
            if self._cancelled:
                break
            self.loaded.emit(self.generation, path, volume)
        self.finished.emit(self.generation)


class ContrastDialog(QDialog):
    """Numeric window/level for the sequence on screen.

    Modeless on purpose: readers adjust contrast while looking at the image,
    and the spin boxes have to track the value the drag gesture produces.
    """

    valuesChanged = Signal(float, float)
    resetRequested = Signal()
    stickyChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Microbleed Review · Contrast")
        self.setMinimumWidth(330)
        self._updating = False
        layout = QVBoxLayout(self)
        layout.setSpacing(9)
        layout.setContentsMargins(14, 12, 14, 12)
        self.sequence_label = _label("—", color=COLORS["accent"], bold=True, size=10)
        layout.addWidget(self.sequence_label)

        form = QFormLayout()
        self.level_spin = QDoubleSpinBox()
        self.level_spin.setDecimals(4)
        self.level_spin.setKeyboardTracking(False)
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setDecimals(4)
        self.window_spin.setKeyboardTracking(False)
        form.addRow("Level:", self.level_spin)
        form.addRow("Window:", self.window_spin)
        layout.addLayout(form)
        self.level_spin.valueChanged.connect(self._emit_values)
        self.window_spin.valueChanged.connect(self._emit_values)

        self.sticky_cb = QCheckBox("Keep this window when changing case")
        self.sticky_cb.setToolTip(
            "Off: every case starts from its own automatic window.\n"
            "On: your manual window is reused for this sequence in the next case."
        )
        self.sticky_cb.toggled.connect(self.stickyChanged)
        layout.addWidget(self.sticky_cb)

        layout.addWidget(
            _label(
                "In a view: right-drag, or Alt + left-drag.\n"
                "Up and down changes brightness, left and right changes contrast.",
                color=COLORS["dim"],
                size=8,
            )
        )
        buttons = QHBoxLayout()
        reset = QPushButton("Reset to automatic")
        reset.clicked.connect(lambda _checked=False: self.resetRequested.emit())
        buttons.addWidget(reset)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def configure(self, sequence: str, level: float, window: float, auto_window: float, sticky: bool) -> None:
        """Point the dialog at a sequence and scale the spin boxes to it."""

        self._updating = True
        try:
            self.sequence_label.setText(sequence)
            span = max(abs(auto_window), 1e-6) * 40.0
            decimals = 4 if abs(auto_window) < 10 else 2
            step = max(abs(auto_window) / 40.0, 10 ** (-decimals))
            for spin in (self.level_spin, self.window_spin):
                spin.setDecimals(decimals)
                spin.setSingleStep(step)
            self.level_spin.setRange(-span, span)
            self.window_spin.setRange(10 ** (-decimals), span)
            self.level_spin.setValue(float(level))
            self.window_spin.setValue(float(window))
            self.sticky_cb.setChecked(bool(sticky))
        finally:
            self._updating = False

    def show_values(self, level: float, window: float) -> None:
        self._updating = True
        try:
            self.level_spin.setValue(float(level))
            self.window_spin.setValue(float(window))
        finally:
            self._updating = False

    def _emit_values(self, _value: float) -> None:
        if not self._updating:
            self.valuesChanged.emit(float(self.level_spin.value()), float(self.window_spin.value()))


class DatasetDialog(QDialog):
    """Choose the study to review: workbook, MRI folder and review database."""

    def __init__(
        self,
        settings: ViewerSettings,
        current: Dataset,
        parent: QWidget | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.dataset: Dataset | None = None
        self.config = dataset_config.validate(config or dataset_config.load())
        self.setWindowTitle("Microbleed Review · Dataset")
        self.setMinimumWidth(640)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.addWidget(_label("Dataset", color=COLORS["accent"], bold=True, size=13))
        layout.addWidget(
            _label(
                "A dataset is one findings workbook, the MRI folder its case IDs live in, "
                "and the review database its reviews are written to. They move together so "
                "reviews of different studies never mix.",
                color=COLORS["dim"],
                size=9,
            )
        )

        self.workbook_edit = QLineEdit(str(current.workbook))
        self.data_edit = QLineEdit(str(current.data_root))
        self.db_edit = QLineEdit(str(current.review_db))
        for label_text, edit, chooser in (
            ("Findings workbook", self.workbook_edit, self._browse_workbook),
            ("MRI data folder", self.data_edit, self._browse_folder),
            ("Review database", self.db_edit, self._browse_database),
        ):
            row = QVBoxLayout()
            row.setSpacing(3)
            row.addWidget(_label(label_text, color=COLORS["dim"], size=8))
            line = QHBoxLayout()
            line.setSpacing(6)
            line.addWidget(edit, 1)
            button = QPushButton("Browse…")
            button.clicked.connect(chooser)
            line.addWidget(button)
            row.addLayout(line)
            layout.addLayout(row)

        layout.addWidget(_separator())
        layout.addLayout(self._build_format_section())

        recent = [item for item in settings.recent_datasets() if item != current]
        if recent:
            layout.addWidget(_separator())
            layout.addWidget(_section_title("Recent"))
            self.recent_list = QListWidget()
            self.recent_list.setMaximumHeight(112)
            for item in recent:
                entry = QListWidgetItem(f"{item.name}      {item.workbook.parent}")
                entry.setData(Qt.ItemDataRole.UserRole, item.as_dict())
                entry.setToolTip(
                    f"Workbook: {item.workbook}\nData: {item.data_root}\nReviews: {item.review_db}"
                )
                self.recent_list.addItem(entry)
            self.recent_list.itemDoubleClicked.connect(self._use_recent)
            self.recent_list.currentRowChanged.connect(self._preview_recent)
            layout.addWidget(self.recent_list)

        self.problem_label = _label("", color=COLORS["danger"], size=9)
        self.problem_label.setWordWrap(True)
        layout.addWidget(self.problem_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Open).setText("Open dataset")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_format_section(self):
        """How to read this dataset: the sheet, and what the files are called.

        Folded away, because a study that matches the defaults never needs it
        and a dialog that opens with eight fields nobody has to fill in reads
        as eight decisions.  It opens by itself when the current settings do
        not fit what is on disk.
        """

        self.format_section = CollapsibleSection("Format of this dataset", expanded=False)

        self.sheet_edit = QLineEdit(self.config["workbook"]["sheet"])
        self.sheet_edit.setToolTip("The worksheet the findings are listed on")
        sheet_row = QHBoxLayout()
        sheet_row.setSpacing(6)
        sheet_row.addWidget(_label("Sheet", color=COLORS["dim"], size=8, wrap=False))
        sheet_row.addWidget(self.sheet_edit, 1)
        holder = QWidget()
        holder.setLayout(sheet_row)
        self.format_section.add_widget(holder)

        self.format_section.add_widget(
            _label(
                "A sequence is recognised by the end of its filename — everything before "
                "it (subject, date, protocol) is free. Detect reads your data folder and "
                "offers what it finds.",
                color=COLORS["faint"],
                size=8,
            )
        )

        self.sequence_rows: dict[str, tuple] = {}
        for key in MODALITY_BUTTON_ORDER:
            entry = self.config["sequences"][key]
            row = QHBoxLayout()
            row.setSpacing(6)
            name = QLineEdit(entry["label"])
            name.setFixedWidth(78)
            name.setToolTip("What this sequence is called in the toolbar")
            suffix = QComboBox()
            suffix.setEditable(True)
            suffix.addItem(entry["suffix"])
            suffix.setCurrentText(entry["suffix"])
            suffix.setToolTip("The ending every file of this sequence shares")
            required = QCheckBox("required")
            required.setChecked(entry["required"])
            required.setToolTip(
                "A case without a required sequence cannot be read.\n"
                "Leave it off for a sequence some cases simply do not have."
            )
            row.addWidget(name)
            row.addWidget(suffix, 1)
            row.addWidget(required)
            holder = QWidget()
            holder.setLayout(row)
            self.format_section.add_widget(holder)
            self.sequence_rows[key] = (name, suffix, required)

        detect = QPushButton("Detect from the data folder")
        detect.clicked.connect(self._detect_suffixes)
        self.format_section.add_widget(detect)
        self.detect_label = _label("", color=COLORS["dim"], size=8)
        self.detect_label.setWordWrap(True)
        self.format_section.add_widget(self.detect_label)

        wrapper = QVBoxLayout()
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(self.format_section)
        return wrapper

    def _detect_suffixes(self) -> None:
        found = dataset_config.suggest_suffixes(self.data_edit.text().strip())
        if not found:
            self.detect_label.setText(
                "No NIfTI files found under that folder. Check the MRI data folder above."
            )
            return
        for key, (_name, suffix, _required) in self.sequence_rows.items():
            current = suffix.currentText().strip()
            suffix.clear()
            suffix.addItems(found)
            # Keep what is already set if the data agrees with it.
            match = next((item for item in found if item == current), None)
            if match is None:
                match = next((item for item in found if item.endswith(current)), None)
            suffix.setCurrentText(match or current)
        self.detect_label.setText(
            f"Found {len(found)} filename ending{'' if len(found) == 1 else 's'}. "
            "Pick one per sequence; the longest that matches only that sequence is the "
            "safest."
        )

    def _collected_config(self) -> dict:
        config = json.loads(json.dumps(self.config))
        config["workbook"]["sheet"] = self.sheet_edit.text().strip()
        for key, (name, suffix, required) in self.sequence_rows.items():
            entry = config["sequences"][key]
            entry["label"] = name.text().strip()
            entry["suffix"] = suffix.currentText().strip()
            entry["required"] = required.isChecked()
        return config

    def _current_dataset(self) -> Dataset:
        return Dataset.create(
            self.workbook_edit.text().strip(),
            self.data_edit.text().strip(),
            self.db_edit.text().strip(),
        )

    def _browse_workbook(self) -> None:
        start = str(Path(self.workbook_edit.text().strip() or ".").parent)
        path, _filter = QFileDialog.getOpenFileName(
            self, "Select the findings workbook", start, "Excel workbook (*.xlsx *.xlsm);;All files (*)"
        )
        if path:
            self.workbook_edit.setText(path)

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select the MRI data folder", self.data_edit.text().strip() or "."
        )
        if path:
            self.data_edit.setText(path)

    def _browse_database(self) -> None:
        start = self.db_edit.text().strip() or "review.sqlite"
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Select or create the review database",
            start,
            "SQLite database (*.sqlite *.db);;All files (*)",
        )
        if path:
            self.db_edit.setText(path)

    def _preview_recent(self, row: int) -> None:
        if row < 0:
            return
        values = self.recent_list.item(row).data(Qt.ItemDataRole.UserRole)
        dataset = Dataset.from_dict(values) if isinstance(values, dict) else None
        if dataset is None:
            return
        self.workbook_edit.setText(str(dataset.workbook))
        self.data_edit.setText(str(dataset.data_root))
        self.db_edit.setText(str(dataset.review_db))

    def _use_recent(self, _item: QListWidgetItem) -> None:
        self._accept()

    def _accept(self) -> None:
        dataset = self._current_dataset()
        problems = dataset.problems()
        if problems:
            self.problem_label.setText("\n".join(problems))
            return
        try:
            self.config = dataset_config.validate(self._collected_config())
        except dataset_config.ConfigError as exc:
            self.problem_label.setText(str(exc))
            self.format_section.set_expanded(True)
            return
        self.dataset = dataset
        self.accept()


class SliceCanvas(QWidget):
    """A lightweight traditional image canvas with continuous wheel scroll."""

    sliceScrollRequested = Signal(str, int)
    targetClicked = Signal(str, object)
    pickCleared = Signal()
    roiPainted = Signal(str)
    roiStrokeStarted = Signal()
    # (flat voxel indices, the values they held) -- emitted just before the
    # stamp is applied, which is what makes undo cheap.
    roiChanging = Signal(object, object)
    mouseVoxelMoved = Signal(str, object)
    sliceChanged = Signal(str, int, int)
    zoomChanged = Signal(str)
    windowChanged = Signal(str, float, float)

    def __init__(self, plane: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plane = plane
        self.volume: Volume | None = None
        self._window: tuple[float, float] = (-1.0, 1.0)
        self._auto_window: tuple[float, float] = (-1.0, 1.0)
        self._wl_dragging = False
        self._wl_origin = QPointF()
        self._wl_start: tuple[float, float] = (-1.0, 1.0)
        self._slice_index = 0
        # ``_target_voxel`` is the navigation cursor: it decides which slice is
        # shown and what lesion zoom frames.  ``_marker_voxel`` is the selected
        # finding, which stays where it is while the reader scrolls away.
        self._target_voxel: np.ndarray | None = None
        self._target_in_bounds = True
        self._marker_voxel: np.ndarray | None = None
        self._marker_in_bounds = True
        self._ghost_voxel: np.ndarray | None = None
        self._variant_markers: list[tuple[np.ndarray, str, str]] = []
        self._show_variants = False
        self._secondary_voxels: list[np.ndarray] = []
        self._lesion_fov_mm = DEFAULT_LESION_FOV_MM
        self._missing_message = "No loadable MRI volume for this view"
        # "missing" or "loading": a sequence that exists but is still being read
        # must not be announced as unavailable.
        self._empty_state = "missing"
        self._show_target = True
        self._show_mouse = True
        self._show_directions = True
        self._mouse_image: tuple[float, float] | None = None
        self._mouse_voxel: np.ndarray | None = None
        # Clicking only places the crosshair while the Point tool is active, so
        # an ordinary click in a view cannot move it by accident.
        self._pick_enabled = False
        # Segmentation: the label volume is shared with the window, so a stroke
        # here is immediately visible in the other two planes.
        self._label_volume: np.ndarray | None = None
        self._label_value = 1
        self._show_labels = True
        # Outline rather than fill: a 40% wash over a 3 mm lesion at lesion
        # zoom covers the very signal that says whether the mask is right.
        self._label_outline = False
        self._smooth_zoom = False
        self._paint_mode: str | None = None
        self._brush_radius_mm = 1.5
        self._brush_3d = True
        self._painting: str | None = None
        self._last_paint: QPointF | None = None
        self._zoom_multiplier = 1.0
        self._zoom_mode = "fit"
        self._wheel_remainder = 0.0
        self._pan = QPointF(0, 0)
        self._panning = False
        self._pan_start = QPointF()
        self._pan_origin = QPointF()
        self._pixmap_cache: dict[tuple[int, float, float], QPixmap] = {}
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    @property
    def slice_axis(self) -> int:
        return PLANE_AXES[self.plane]

    @property
    def slice_index(self) -> int:
        return int(self._slice_index)

    @property
    def max_slice(self) -> int:
        if self.volume is None:
            return 0
        return max(0, int(self.volume.shape[self.slice_axis]) - 1)

    @property
    def zoom_mode(self) -> str:
        return self._zoom_mode

    def _display_array(self) -> np.ndarray | None:
        if self.volume is None:
            return None
        data = self.volume.data
        index = int(np.clip(self._slice_index, 0, self.max_slice))
        if self.plane == "axial":
            # LPI data with order (2, 1, 0): rows A -> P, columns R -> L.
            return extract_plane(data, "axial", index)[0]
        if self.plane == "coronal":
            # LPI data with order (1, 2, 0): rows S -> I, columns R -> L.
            return extract_plane(data, "coronal", index)[0]
        # LPI data with order (0, 2, 1): rows S -> I, columns A -> P.
        return extract_plane(data, "sagittal", index)[0]

    def _image_shape(self) -> tuple[int, int]:
        image = self._display_array()
        if image is None:
            return 1, 1
        return int(image.shape[1]), int(image.shape[0])

    def _pixmap(self, image: np.ndarray) -> QPixmap:
        low, high = self._window
        key = (self._slice_index, round(low, 6), round(high, 6))
        cached = self._pixmap_cache.get(key)
        if cached is not None:
            return cached
        values = np.asarray(image, dtype=np.float32)
        finite = np.isfinite(values)
        normalized = np.zeros(values.shape, dtype=np.uint8)
        if np.any(finite):
            spread = max(float(high - low), 1e-8)
            normalized[finite] = np.clip((values[finite] - low) / spread * 255.0, 0, 255).astype(np.uint8)
        qimage = QImage(
            normalized.data,
            int(normalized.shape[1]),
            int(normalized.shape[0]),
            int(normalized.strides[0]),
            QImage.Format.Format_Grayscale8,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        self._pixmap_cache[key] = pixmap
        if len(self._pixmap_cache) > 18:
            self._pixmap_cache.pop(next(iter(self._pixmap_cache)))
        return pixmap

    def _plane_spacing(self) -> tuple[float, float]:
        if self.volume is None:
            return 1.0, 1.0
        column_axis, row_axis = PLANE_IMAGE_AXES[self.plane]
        sizes = self.volume.voxel_sizes
        column_spacing = float(sizes[column_axis]) if len(sizes) > column_axis else 1.0
        row_spacing = float(sizes[row_axis]) if len(sizes) > row_axis else 1.0
        return max(column_spacing, 1e-6), max(row_spacing, 1e-6)

    def _fit_scale(self, image_w: int, image_h: int) -> float:
        margin = 28.0
        column_spacing, row_spacing = self._plane_spacing()
        physical_w = max(1e-6, image_w * column_spacing)
        physical_h = max(1e-6, image_h * row_spacing)
        return max(
            0.01,
            min(
                (self.width() - margin * 2) / physical_w,
                (self.height() - margin * 2) / physical_h,
            ),
        )

    def _draw_geometry(self, image_w: int, image_h: int) -> tuple[QRectF, tuple[float, float]]:
        fit = self._fit_scale(image_w, image_h)
        physical_scale = fit * self._zoom_multiplier
        column_spacing, row_spacing = self._plane_spacing()
        scale_x = physical_scale * column_spacing
        scale_y = physical_scale * row_spacing
        draw_w, draw_h = image_w * scale_x, image_h * scale_y
        center = QPointF(self.width() / 2.0 + self._pan.x(), self.height() / 2.0 + self._pan.y())
        rect = QRectF(center.x() - draw_w / 2.0, center.y() - draw_h / 2.0, draw_w, draw_h)
        return rect, (scale_x, scale_y)

    def _image_point_of(self, voxel: np.ndarray | None) -> tuple[float, float] | None:
        """Where a voxel sits in the pixmap that is drawn on screen.

        ``drawPixmap`` spreads pixmap pixel ``j`` over the half-open interval
        ``[j, j+1)`` of the target rectangle, so the voxel it stands for is
        centred half a pixel further on.  Without that half pixel every marker
        is drawn towards the top-left corner of its own voxel -- 0.43 mm on this
        data, which is a fifth of a small microbleed.
        """

        if voxel is None:
            return None
        x, y, z = (float(value) for value in voxel)
        if self.plane == "axial":
            column, row = x, y
        elif self.plane == "coronal":
            column, row = x, z
        else:
            column, row = y, z
        return column + 0.5, row + 0.5

    def _target_image_point(self) -> tuple[float, float] | None:
        return self._image_point_of(self._target_voxel)

    def _marker_image_point(self) -> tuple[float, float] | None:
        return self._image_point_of(self._marker_voxel)

    def _voxel_from_image_point(self, x: float, y: float) -> np.ndarray | None:
        """The voxel a pixmap coordinate falls on -- the inverse of the above.

        ``x`` and ``y`` are continuous pixmap coordinates, so the middle of
        pixel ``j`` is ``j + 0.5``; subtracting the half pixel is what makes
        clicking the centre of a voxel return that voxel and not the next one.
        """

        if self.volume is None:
            return None
        column, row = float(x) - 0.5, float(y) - 0.5
        if self.plane == "axial":
            values = (column, row, float(self._slice_index))
        elif self.plane == "coronal":
            values = (column, float(self._slice_index), row)
        else:
            values = (float(self._slice_index), column, row)
        return np.asarray(values, dtype=np.float64)

    def _image_point_at(self, position: QPointF) -> tuple[float, float] | None:
        image = self._display_array()
        if image is None:
            return None
        rect, (scale_x, scale_y) = self._draw_geometry(int(image.shape[1]), int(image.shape[0]))
        if scale_x <= 0 or scale_y <= 0 or not rect.contains(position):
            return None
        return (
            float((position.x() - rect.left()) / scale_x),
            float((position.y() - rect.top()) / scale_y),
        )

    def set_volume(self, volume: Volume | None, *, reset_view: bool = True) -> None:
        self.volume = volume
        self._missing_message = "No loadable MRI volume for this view"
        self._pixmap_cache.clear()
        self._wheel_remainder = 0.0
        if volume is None:
            self._auto_window = (-1.0, 1.0)
        else:
            # ``Volume.window`` is precomputed once per file; the fallback keeps
            # volumes built directly in tests or scripts working.
            self._auto_window = volume.window or robust_window(volume.data, fallback=volume.data)
        self._window = self._auto_window
        self._wl_dragging = False
        if volume is None:
            self._target_voxel = None
            self._secondary_voxels = []
            self._slice_index = 0
        elif reset_view:
            self._slice_index = min(self.max_slice, self.max_slice // 2)
            self.reset_view()
        else:
            self._slice_index = min(self._slice_index, self.max_slice)
            if self.lesion_focus:
                # A different sequence has a different grid; re-frame the same
                # physical field of view instead of keeping stale pan values.
                self._apply_lesion_focus()
        self.sliceChanged.emit(self.plane, self._slice_index, self.max_slice)
        self.update()

    # ------------------------------------------------------ window / level --
    @property
    def window_limits(self) -> tuple[float, float]:
        return self._window

    @property
    def window_level(self) -> tuple[float, float]:
        """Grayscale mapping as the (level, window) pair readers think in."""

        low, high = self._window
        return (low + high) / 2.0, max(high - low, 1e-9)

    @property
    def auto_window_level(self) -> tuple[float, float]:
        low, high = self._auto_window
        return (low + high) / 2.0, max(high - low, 1e-9)

    def set_window_limits(self, low: float, high: float, *, notify: bool = True) -> None:
        low_value = _safe_float(low)
        high_value = _safe_float(high)
        if low_value is None or high_value is None:
            return
        if high_value <= low_value:
            # A collapsed window renders a flat panel; keep a hair of range.
            high_value = low_value + max(abs(low_value), 1.0) * 1e-6
        if (low_value, high_value) == self._window:
            return
        self._window = (low_value, high_value)
        if notify:
            self.windowChanged.emit(self.plane, low_value, high_value)
        self.update()

    def set_window_level(self, level: float, window: float, *, notify: bool = True) -> None:
        level_value = _safe_float(level)
        window_value = _safe_float(window)
        if level_value is None or window_value is None:
            return
        window_value = max(window_value, 1e-9)
        self.set_window_limits(
            level_value - window_value / 2.0,
            level_value + window_value / 2.0,
            notify=notify,
        )

    def reset_window(self, *, notify: bool = True) -> None:
        """Return to the percentile window computed from the volume."""

        self.set_window_limits(*self._auto_window, notify=notify)

    def _adjust_window_by_drag(self, position: QPointF) -> None:
        """Standard radiology drag: up/down brightness, left/right contrast."""

        start_level, start_window = (
            (self._wl_start[0] + self._wl_start[1]) / 2.0,
            max(self._wl_start[1] - self._wl_start[0], 1e-9),
        )
        dx = float(position.x() - self._wl_origin.x())
        dy = float(position.y() - self._wl_origin.y())
        _auto_level, auto_window = self.auto_window_level
        # Sensitivity follows the data range so the same gesture behaves the
        # same on QSM (values around 0.1) and SWI (values in the hundreds).
        # Dragging up brightens, which means lowering the window centre.
        level = start_level + dy * (auto_window / 250.0)
        window = start_window * float(math.exp(dx / 180.0))
        window = max(window, auto_window * 1e-3)
        self.set_window_level(level, window)

    def set_secondary_targets(self, voxels: list[np.ndarray] | None) -> None:
        """Other findings of the same case, drawn as faint markers."""

        self._secondary_voxels = [np.asarray(voxel, dtype=np.float64).reshape(3) for voxel in (voxels or [])]
        self.update()

    def set_missing(self, message: str | None = None) -> None:
        self.set_volume(None)
        self._empty_state = "missing"
        if message:
            self._missing_message = str(message)
        self.update()

    def set_loading(self, message: str | None = None) -> None:
        """This view has a file; it is being read."""

        self.set_volume(None)
        self._empty_state = "loading"
        self._missing_message = str(message or "Reading the MRI volume…")
        self.update()

    def set_target_voxel(self, voxel: np.ndarray | None, *, recenter: bool = True) -> None:
        if voxel is None or self.volume is None:
            self._target_voxel = None if voxel is None else np.asarray(voxel, dtype=np.float64)
            self._target_in_bounds = True
            self.update()
            return
        point = np.asarray(voxel, dtype=np.float64).reshape(3)
        self._target_in_bounds = voxel_in_bounds(point, self.volume.shape)
        self._target_voxel = np.asarray(clamp_voxel(point, self.volume.shape), dtype=np.float64)
        if recenter:
            self._slice_index = int(np.clip(round(point[self.slice_axis]), 0, self.max_slice))
            self.sliceChanged.emit(self.plane, self._slice_index, self.max_slice)
        if self.lesion_focus:
            self._apply_lesion_focus()
        self.update()

    def set_marker_voxel(self, voxel: np.ndarray | None) -> None:
        """Pin the selected finding in place.

        The marker does not follow the slice: scrolling away simply stops it
        being drawn, which is how a reader tells a round focus from a vessel
        running through several slices.
        """

        if voxel is None or self.volume is None:
            self._marker_voxel = None if voxel is None else np.asarray(voxel, dtype=np.float64)
            self._marker_in_bounds = True
            self.update()
            return
        point = np.asarray(voxel, dtype=np.float64).reshape(3)
        self._marker_in_bounds = voxel_in_bounds(point, self.volume.shape)
        self._marker_voxel = np.asarray(clamp_voxel(point, self.volume.shape), dtype=np.float64)
        self.update()

    def set_ghost_voxel(self, voxel: np.ndarray | None) -> None:
        """The source position, shown faintly while a correction is displayed."""

        self._ghost_voxel = None if voxel is None else np.asarray(voxel, dtype=np.float64).reshape(3)
        self.update()

    def set_variant_markers(self, markers: list[tuple[np.ndarray, str, str]] | None) -> None:
        """Every recorded position of the selected finding, for the peek key."""

        self._variant_markers = [
            (np.asarray(voxel, dtype=np.float64).reshape(3), str(colour), str(label))
            for voxel, colour, label in (markers or [])
        ]
        if self._show_variants:
            self.update()

    def set_show_variants(self, visible: bool) -> None:
        visible = bool(visible)
        if visible == self._show_variants:
            return
        self._show_variants = visible
        self.update()

    def marker_on_slice(self) -> bool:
        return bool(
            self._marker_voxel is not None
            and self._marker_in_bounds
            and abs(float(self._marker_voxel[self.slice_axis]) - float(self._slice_index)) < 0.51
        )

    @property
    def marker_in_bounds(self) -> bool:
        """False when the finding's coordinate falls outside this volume.

        The voxel is clamped to the edge so the rest of the drawing code has
        something valid to work with, which is exactly why this has to be
        visible: an edge crosshair otherwise looks like a real position.
        """

        return bool(self._marker_voxel is None or self._marker_in_bounds)

    # ---------------------------------------------------------- segmentation --
    def set_label_volume(self, volume: np.ndarray | None, label_value: int = 1) -> None:
        """Share the case's label array; painting writes into it in place."""

        self._label_volume = volume
        self._label_value = int(label_value)
        self.update()

    def set_show_labels(self, visible: bool) -> None:
        self._show_labels = bool(visible)
        self.update()

    def set_paint_mode(self, mode: str | None) -> None:
        """``"paint"``, ``"erase"`` or ``None`` for no drawing."""

        self._paint_mode = mode if mode in ("paint", "erase") else None
        self.setCursor(
            Qt.CursorShape.CrossCursor
            if (self._paint_mode or self._pick_enabled)
            else Qt.CursorShape.ArrowCursor
        )
        self.update()

    def set_brush_radius(self, millimetres: float) -> None:
        value = _safe_float(millimetres)
        if value is None:
            return
        self._brush_radius_mm = float(max(0.2, value))
        self.update()

    def set_brush_3d(self, spherical: bool) -> None:
        """Paint a ball through the slices, or a disc on the one on screen."""

        self._brush_3d = bool(spherical)
        self.update()

    def _paint_at(self, position: QPointF, *, erase: bool) -> bool:
        """Stamp the brush into the label volume around the cursor.

        The stamp is a ball measured in millimetres, so it stays round on
        anisotropic voxels.  In flat mode it is the single-slice cross-section
        of that ball, which is what a reader wants when correcting one voxel;
        in spherical mode it reaches through the neighbouring slices, which is
        what they want when painting a lesion that is 2-12 voxels across in
        every direction and would otherwise take five separate strokes.
        """

        if self._label_volume is None or self.volume is None:
            return False
        image_point = self._image_point_at(position)
        if image_point is None:
            return False
        voxel = self._voxel_from_image_point(*image_point)
        if voxel is None:
            return False

        shape = self._label_volume.shape
        sizes = [float(size) for size in self.volume.voxel_sizes[:3]]
        radius = self._brush_radius_mm
        slice_axis = self.slice_axis
        # Continuous voxel coordinates of the cursor, so the stamp is centred
        # on what is under the pointer and not on the corner of the next voxel.
        centre = [float(value) for value in voxel]
        centre[slice_axis] = float(np.clip(self._slice_index, 0, self.max_slice))

        lows: list[int] = []
        highs: list[int] = []
        for axis in range(3):
            flat_axis = axis == slice_axis and not self._brush_3d
            reach = 0 if flat_axis else max(0, int(np.ceil(radius / max(sizes[axis], 1e-6))))
            middle = int(round(centre[axis]))
            lows.append(max(0, middle - reach))
            highs.append(min(shape[axis], middle + reach + 1))
            if lows[axis] >= highs[axis]:
                return False

        offsets = [
            (np.arange(lows[axis], highs[axis]) - centre[axis]) * sizes[axis] for axis in range(3)
        ]
        distance_sq = (
            offsets[0][:, None, None] ** 2
            + offsets[1][None, :, None] ** 2
            + offsets[2][None, None, :] ** 2
        )
        ball = distance_sq <= radius**2
        # A radius under half a voxel covers no voxel centre, so the brush would
        # silently do nothing at the settings a reader reaches for to correct a
        # single voxel.  The voxel under the cursor is always part of the stamp.
        middle = tuple(int(round(centre[axis])) - lows[axis] for axis in range(3))
        if all(0 <= middle[axis] < ball.shape[axis] for axis in range(3)):
            ball[middle] = True
        if not ball.any():
            return False

        patch = self._label_volume[lows[0]:highs[0], lows[1]:highs[1], lows[2]:highs[2]]
        if erase:
            # Erasing only removes this finding's label, never a neighbour's.
            target = ball & (patch == self._label_value)
        else:
            target = ball & (patch != self._label_value)
        if not target.any():
            return False

        # Hand the window the voxels this stamp is about to change, and what
        # they held, so undo can be recorded without copying the whole volume.
        local = np.nonzero(target)
        coordinates = tuple(local[axis] + lows[axis] for axis in range(3))
        self.roiChanging.emit(
            np.ravel_multi_index(coordinates, shape),
            np.asarray(patch[target], dtype=self._label_volume.dtype).copy(),
        )

        patch[target] = 0 if erase else self._label_value
        self.update()
        return True

    def _label_slice(self) -> np.ndarray | None:
        if self._label_volume is None:
            return None
        index = int(np.clip(self._slice_index, 0, self.max_slice))
        return extract_plane(self._label_volume, self.plane, index)[0]

    def set_smooth_zoom(self, smooth: bool) -> None:
        """Interpolate the image when magnified, instead of showing squares."""

        self._smooth_zoom = bool(smooth)
        self.update()

    def set_label_outline(self, outline: bool) -> None:
        """Draw the segmentation as an edge rather than a filled wash."""

        self._label_outline = bool(outline)
        self.update()

    @staticmethod
    def _edge_of(mask: np.ndarray) -> np.ndarray:
        """The voxels of a mask that touch something outside it.

        A voxel is on the edge unless all four of its neighbours are also in
        the mask, so a single voxel -- most of a small microbleed -- is all
        edge and stays visible.
        """

        interior = np.ones(mask.shape, dtype=bool)
        interior[:-1, :] &= mask[1:, :]
        interior[1:, :] &= mask[:-1, :]
        interior[:, :-1] &= mask[:, 1:]
        interior[:, 1:] &= mask[:, :-1]
        # The image border cannot be interior: nothing is known beyond it.
        interior[0, :] = interior[-1, :] = False
        interior[:, 0] = interior[:, -1] = False
        return mask & ~interior

    def _label_overlay_mask(self) -> np.ndarray:
        """The voxels of this finding's label that the overlay would paint."""

        labels = self._label_slice()
        if labels is None:
            return np.zeros((1, 1), dtype=bool)
        mine = labels == self._label_value
        return self._edge_of(mine) if self._label_outline else mine

    def _draw_label_overlay(self, painter: QPainter, rect: QRectF) -> None:
        labels = self._label_slice()
        if labels is None or not np.any(labels):
            return
        height, width = labels.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        others = (labels > 0) & (labels != self._label_value)
        if np.any(others):
            if self._label_outline:
                others = self._edge_of(others)
            colour = QColor(COLORS["dim"])
            rgba[others] = (colour.blue(), colour.green(), colour.red(), 70)
        mine = labels == self._label_value
        if np.any(mine):
            colour = QColor(COLORS["roi"])
            if self._label_outline:
                # Opaque, because there is nothing underneath it to see now.
                rgba[self._edge_of(mine)] = (colour.blue(), colour.green(), colour.red(), 235)
            else:
                rgba[mine] = (colour.blue(), colour.green(), colour.red(), 105)
        image = QImage(
            rgba.data, int(width), int(height), int(rgba.strides[0]), QImage.Format.Format_ARGB32
        ).copy()
        painter.drawPixmap(rect.toRect(), QPixmap.fromImage(image))

    def _draw_brush_outline(self, painter: QPainter, rect: QRectF, scales: tuple[float, float]) -> None:
        if self._mouse_image is None or self.volume is None:
            return
        scale_x, scale_y = scales
        column_axis, row_axis = PLANE_IMAGE_AXES[self.plane]
        sizes = self.volume.voxel_sizes
        radius_x = self._brush_radius_mm / max(float(sizes[column_axis]), 1e-6) * scale_x
        radius_y = self._brush_radius_mm / max(float(sizes[row_axis]), 1e-6) * scale_y
        centre = QPointF(
            rect.left() + self._mouse_image[0] * scale_x,
            rect.top() + self._mouse_image[1] * scale_y,
        )
        colour = COLORS["danger"] if self._paint_mode == "erase" else COLORS["roi"]
        painter.setPen(QPen(QColor(colour), 1.2, Qt.PenStyle.DashLine if self._paint_mode == "erase" else Qt.PenStyle.SolidLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(centre, radius_x, radius_y)

    def set_pick_enabled(self, enabled: bool) -> None:
        """Allow clicks to place the crosshair, and show it in the cursor."""

        self._pick_enabled = bool(enabled)
        self.setCursor(
            Qt.CursorShape.CrossCursor if self._pick_enabled else Qt.CursorShape.ArrowCursor
        )

    def set_show_target(self, visible: bool) -> None:
        self._show_target = bool(visible)
        self.update()

    def set_show_mouse(self, visible: bool) -> None:
        self._show_mouse = bool(visible)
        self.update()

    def set_show_directions(self, visible: bool) -> None:
        self._show_directions = bool(visible)
        self.update()

    @property
    def zoom_text(self) -> str:
        """Zoom as it should appear in the header combo."""

        if self._zoom_mode == "fit":
            return "Autofit"
        if self._zoom_mode == "lesion":
            return LESION_ZOOM_LABEL
        return self._zoom_mode

    @property
    def lesion_focus(self) -> bool:
        return self._zoom_mode == "lesion"

    def set_lesion_fov(self, millimetres: float) -> None:
        """Set the physical field of view used by lesion focus."""

        value = _safe_float(millimetres)
        if value is None:
            return
        self._lesion_fov_mm = float(min(max(value, MIN_LESION_FOV_MM), MAX_LESION_FOV_MM))
        if self.lesion_focus:
            self._apply_lesion_focus()
            self.update()

    def _apply_lesion_focus(self) -> bool:
        """Frame a fixed physical field of view centred on the target.

        A microbleed is a 2-10 mm focus, so a whole-brain fit is the wrong
        magnification for judging it.  Driving the zoom from millimetres rather
        than a percentage keeps every lesion, and every sequence, at the same
        apparent size regardless of matrix size or voxel spacing.
        """

        image = self._display_array()
        target_point = self._target_image_point()
        if image is None or target_point is None:
            return False
        image_h, image_w = int(image.shape[0]), int(image.shape[1])
        margin = 28.0
        usable = min(self.width() - margin * 2.0, self.height() - margin * 2.0)
        if usable <= 1.0:
            return False
        fit = self._fit_scale(image_w, image_h)
        if fit <= 0:
            return False
        wanted = usable / max(self._lesion_fov_mm, 1.0)
        self._zoom_multiplier = float(np.clip(wanted / fit, 0.1, MAX_ZOOM_MULTIPLIER))
        column_spacing, row_spacing = self._plane_spacing()
        scale_x = fit * self._zoom_multiplier * column_spacing
        scale_y = fit * self._zoom_multiplier * row_spacing
        # Place the target under the middle of the viewport.
        self._pan = QPointF(
            image_w * scale_x / 2.0 - target_point[0] * scale_x,
            image_h * scale_y / 2.0 - target_point[1] * scale_y,
        )
        return True

    def set_zoom_mode(self, mode: str, *, anchor: QPointF | None = None) -> None:
        parsed = (mode or "Autofit").strip().lower()
        if parsed.startswith("lesion"):
            self._zoom_mode = "lesion"
            if not self._apply_lesion_focus():
                # Without a target there is nothing to frame; stay on autofit
                # rather than leaving the view at an arbitrary magnification.
                self._zoom_mode = "fit"
                self._zoom_multiplier = 1.0
                self._pan = QPointF(0, 0)
        elif parsed in {"autofit", "fit", "auto"}:
            self._zoom_mode = "fit"
            self._zoom_multiplier = 1.0
            self._pan = QPointF(0, 0)
        else:
            try:
                percentage = float(parsed.rstrip("%"))
                if percentage <= 0:
                    raise ValueError
                new_multiplier = percentage / 100.0
            except ValueError:
                return
            self._zoom_mode = f"{percentage:g}%"
            self._set_zoom_multiplier(new_multiplier, anchor=anchor)
        self.zoomChanged.emit(self.zoom_text)
        self.update()

    def _set_zoom_multiplier(self, multiplier: float, *, anchor: QPointF | None = None) -> None:
        old_multiplier = max(self._zoom_multiplier, 1e-8)
        if anchor is None:
            anchor = QPointF(self.width() / 2.0, self.height() / 2.0)
        image = self._display_array()
        if image is not None:
            old_rect, (old_scale_x, old_scale_y) = self._draw_geometry(
                int(image.shape[1]),
                int(image.shape[0]),
            )
            image_point = (
                (anchor.x() - old_rect.left()) / max(old_scale_x, 1e-8),
                (anchor.y() - old_rect.top()) / max(old_scale_y, 1e-8),
            )
        else:
            image_point = None
        self._zoom_multiplier = float(np.clip(multiplier, 0.1, 12.0))
        if image_point is not None:
            new_rect, (new_scale_x, new_scale_y) = self._draw_geometry(
                int(image.shape[1]),
                int(image.shape[0]),
            )
            desired_left = anchor.x() - image_point[0] * new_scale_x
            desired_top = anchor.y() - image_point[1] * new_scale_y
            new_center_x = desired_left + new_rect.width() / 2.0
            new_center_y = desired_top + new_rect.height() / 2.0
            self._pan = QPointF(new_center_x - self.width() / 2.0, new_center_y - self.height() / 2.0)
        elif old_multiplier:
            self._pan = QPointF(self._pan.x(), self._pan.y())

    def step_zoom(self, step: int, *, anchor: QPointF | None = None) -> None:
        current_multiplier = 1.0 if self._zoom_mode == "fit" else self._zoom_multiplier
        new_multiplier = float(np.clip(
            current_multiplier * (WHEEL_ZOOM_FACTOR ** int(step)),
            0.1,
            12.0,
        ))
        percentage = round(new_multiplier * 100.0, 1)
        self.set_zoom_mode(f"{percentage}%", anchor=anchor)

    def reset_view(self) -> None:
        self._zoom_mode = "fit"
        self._zoom_multiplier = 1.0
        self._pan = QPointF(0, 0)
        self.zoomChanged.emit(self.zoom_text)
        self.update()

    def set_slice(self, index: int) -> None:
        new_index = int(np.clip(int(index), 0, self.max_slice))
        if new_index == self._slice_index:
            self.update()
            return
        # The cache key contains the slice index, so slices already rendered
        # for this volume stay reusable while scrolling back and forth.
        self._slice_index = new_index
        self.sliceChanged.emit(self.plane, self._slice_index, self.max_slice)
        self.update()

    def step_slice_independent(self, delta: int) -> None:
        self.set_slice(self._slice_index + int(delta))

    def _update_mouse(self, position: QPointF) -> None:
        image_point = self._image_point_at(position)
        self._mouse_image = image_point
        self._mouse_voxel = self._voxel_from_image_point(*image_point) if image_point is not None else None
        self.mouseVoxelMoved.emit(self.plane, self._mouse_voxel)
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        angle_delta = event.angleDelta().y()
        pixel_delta = event.pixelDelta().y()
        if angle_delta:
            steps, self._wheel_remainder = quantize_wheel_delta(
                angle_delta,
                pixel=False,
                remainder=self._wheel_remainder,
            )
        elif pixel_delta:
            steps, self._wheel_remainder = quantize_wheel_delta(
                pixel_delta,
                pixel=True,
                remainder=self._wheel_remainder,
            )
        else:
            event.accept()
            return
        modifiers = event.modifiers()
        if not steps:
            event.accept()
            return
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self.step_zoom(steps, anchor=event.position())
        else:
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                steps *= 5
            self.sliceScrollRequested.emit(self.plane, steps)
        event.accept()

    def _is_window_level_gesture(self, event) -> bool:
        # Right-drag is the convention in most radiology viewers; Alt+left-drag
        # is the same gesture for readers who prefer to stay on the left button.
        if self.volume is None:
            return False
        if event.button() == Qt.MouseButton.RightButton:
            return True
        return bool(
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.AltModifier
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        # Brush and eraser are separate tools, so the left button is the only
        # one that draws and the right button keeps one meaning everywhere.
        if self._paint_mode and event.button() == Qt.MouseButton.LeftButton and not (
            event.modifiers() & Qt.KeyboardModifier.AltModifier
        ):
            self._painting = self._paint_mode
            self._last_paint = event.position()
            self.roiStrokeStarted.emit()
            if self._paint_at(event.position(), erase=self._painting == "erase"):
                self.roiPainted.emit(self.plane)
            event.accept()
            return
        if self._is_window_level_gesture(event):
            self._wl_dragging = True
            self._wl_origin = event.position()
            self._wl_start = self._window
            self.setCursor(Qt.CursorShape.SizeVerCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self._wl_origin = event.position()
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self._panning = True
            self._pan_start = event.position()
            self._pan_origin = QPointF(self._pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if not self._pick_enabled:
                # No tool selected: a click is just a click.
                event.accept()
                return
            image_point = self._image_point_at(event.position())
            if image_point is not None:
                voxel = self._voxel_from_image_point(*image_point)
                if voxel is not None:
                    self.targetClicked.emit(self.plane, voxel)
            event.accept()
            return
        super().mousePressEvent(event)

    def _paint_stroke(self, position: QPointF) -> bool:
        """Stamp along the whole way from the last point, not just at this one.

        The mouse reports positions, not the path between them, so a quick
        stroke used to come out as separate blobs -- measured with 24 px
        between reports, a stroke broke into six.  Anything that slows the
        event loop widens the gaps, which is exactly when a reader is least
        likely to notice.

        Stepping every two pixels overlaps any brush wider than a pixel on
        screen, and the cap keeps a jump across the window from stamping
        thousands of times.
        """

        erase = self._painting == "erase"
        previous = self._last_paint
        self._last_paint = position
        painted = False
        if previous is not None:
            delta = position - previous
            distance = math.hypot(delta.x(), delta.y())
            steps = min(int(distance / 2.0), 96)
            for step in range(1, steps):
                fraction = step / steps
                between = QPointF(
                    previous.x() + delta.x() * fraction,
                    previous.y() + delta.y() * fraction,
                )
                painted = self._paint_at(between, erase=erase) or painted
        return self._paint_at(position, erase=erase) or painted

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._painting is not None:
            if self._paint_stroke(event.position()):
                self.roiPainted.emit(self.plane)
            self._update_mouse(event.position())
            event.accept()
            return
        if self._wl_dragging:
            self._adjust_window_by_drag(event.position())
            event.accept()
            return
        if self._panning:
            movement = event.position() - self._pan_start
            self._pan = self._pan_origin + movement
            self.update()
        self._update_mouse(event.position())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._painting is not None:
            self._painting = None
            self._last_paint = None
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            moved = (event.position() - self._wl_origin).manhattanLength()
            was_dragging = self._wl_dragging
            self._wl_dragging = False
            self.setCursor(
                Qt.CursorShape.CrossCursor
                if (self._pick_enabled or self._paint_mode)
                else Qt.CursorShape.ArrowCursor
            )
            # A right *click* clears the picked position; a right *drag* is a
            # window/level adjustment and must not clear anything.
            if moved <= 3 and (not was_dragging or self._window == self._wl_start):
                self.pickCleared.emit()
            event.accept()
            return
        if self._wl_dragging and event.button() == Qt.MouseButton.LeftButton:
            self._wl_dragging = False
            self.setCursor(
                Qt.CursorShape.CrossCursor if self._pick_enabled else Qt.CursorShape.ArrowCursor
            )
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        # Double-click used to restore Autofit, which fired by accident while
        # picking a position. A second click now simply picks again.
        if event.button() == Qt.MouseButton.LeftButton and not (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.mousePressEvent(event)
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._mouse_image = None
        self._mouse_voxel = None
        self.mouseVoxelMoved.emit(self.plane, None)
        self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.lesion_focus:
            # The lesion field of view is defined in millimetres, so it has to
            # be recomputed whenever the viewport size changes.
            self._apply_lesion_focus()
        self.update()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COLORS["canvas"]))
        # Nearest-neighbour by default: at lesion zoom a microbleed is a
        # handful of voxels, and interpolation invents edges that are not in
        # the data.  The preference exists because some readers judge
        # roundness better with it on, and it changes only the picture.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._smooth_zoom)
        if self.volume is None:
            loading = self._empty_state == "loading"
            painter.setPen(QColor(COLORS["accent"] if loading else COLORS["warn"]))
            painter.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
            painter.drawText(
                self.rect().adjusted(14, 14, -14, -14),
                Qt.AlignmentFlag.AlignCenter,
                "Loading…" if loading else "Not available",
            )
            painter.setPen(QColor(COLORS["dim"]))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(self.rect().adjusted(20, 90, -20, -20), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, self._missing_message)
            painter.end()
            return

        image = self._display_array()
        if image is None:
            painter.end()
            return
        image_h, image_w = int(image.shape[0]), int(image.shape[1])
        rect, (scale_x, scale_y) = self._draw_geometry(image_w, image_h)
        painter.drawPixmap(rect.toRect(), self._pixmap(image))

        if self._show_labels:
            self._draw_label_overlay(painter, rect)

        # Thin frame makes the actual image bounds clear when zoomed out.
        painter.setPen(QPen(QColor("#525b6b"), 1))
        painter.drawRect(rect)
        if self._paint_mode:
            self._draw_brush_outline(painter, rect, (scale_x, scale_y))

        target_point = self._marker_image_point()
        target_is_current_slice = self.marker_on_slice()
        if self._show_target and self._secondary_voxels:
            self._draw_neighbour_markers(painter, rect, (scale_x, scale_y))
        if self._show_target and self._ghost_voxel is not None and not self._show_variants:
            self._draw_ghost_marker(painter, rect, (scale_x, scale_y))
        if self._show_variants and self._variant_markers:
            self._draw_variant_markers(painter, rect, (scale_x, scale_y))
        if self._show_target and target_point is not None and target_is_current_slice:
            x = rect.left() + target_point[0] * scale_x
            y = rect.top() + target_point[1] * scale_y
            # The long arms only need to lead the eye, so they are drawn faint;
            # the bright marker is the part that identifies the lesion.  A gap
            # keeps the lesion itself uncovered at lesion zoom.
            arm_colour = QColor(COLORS["target"])
            arm_colour.setAlphaF(0.45)
            painter.setPen(QPen(arm_colour, 1.2))
            gap = 11.0
            painter.drawLine(QPointF(rect.left(), y), QPointF(x - gap, y))
            painter.drawLine(QPointF(x + gap, y), QPointF(rect.right(), y))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, y - gap))
            painter.drawLine(QPointF(x, y + gap), QPointF(x, rect.bottom()))
            painter.setPen(QPen(QColor(COLORS["target"]), 1.6))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(x, y), 3.5, 3.5)

        if self._show_mouse and self._target_voxel is not None:
            self._draw_cursor_crosshair(painter, rect, (scale_x, scale_y))

        if self._show_mouse and self._mouse_image is not None:
            mx = rect.left() + self._mouse_image[0] * scale_x
            my = rect.top() + self._mouse_image[1] * scale_y
            if rect.contains(QPointF(mx, my)):
                pen = QPen(QColor(COLORS["cursor"]), 1.0, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(QPointF(rect.left(), my), QPointF(rect.right(), my))
                painter.drawLine(QPointF(mx, rect.top()), QPointF(mx, rect.bottom()))

        if self._show_directions:
            self._draw_direction_labels(painter, rect)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        # The plane name is already in the panel header; this overlay carries
        # the position and, when it applies, the off-slice warning.
        slice_text = f"{self._slice_index + 1} / {self.max_slice + 1}"
        self._draw_hud_text(painter, slice_text, x=7.0, y=6.0, colour=COLORS["text"])
        if not self.marker_in_bounds:
            # Nothing on screen is the finding, so say that rather than let the
            # clamped edge crosshair pass for a position.
            self._draw_hud_text(
                painter,
                "finding is outside this volume",
                x=7.0,
                y=27.0,
                colour=COLORS["danger"],
            )
        elif self._marker_voxel is not None and not target_is_current_slice:
            offset = float(self._marker_voxel[self.slice_axis]) - float(self._slice_index)
            plural = "" if abs(offset) < 1.5 else "s"
            direction = "back" if offset < 0 else "ahead"
            self._draw_hud_text(
                painter,
                f"finding is {abs(offset):.0f} slice{plural} {direction}",
                x=7.0,
                y=27.0,
                colour=COLORS["warn"],
            )

        level, window = self.window_level
        magnitude = max(abs(level), window, 1e-9)
        digits = 3 if magnitude >= 1.0 else 4
        manual = self._window != self._auto_window
        wl_text = f"L {level:.{digits}f}   W {window:.{digits}f}"
        if manual:
            wl_text += "   manual"
        self._draw_hud_text(
            painter,
            wl_text,
            x=7.0,
            y=self.height() - 24.0,
            colour=COLORS["warn"] if manual else COLORS["text"],
        )
        painter.end()

    def _draw_neighbour_markers(
        self,
        painter: QPainter,
        rect: QRectF,
        scales: tuple[float, float],
    ) -> None:
        """Mark the other findings of this case that lie near this slice.

        A case can hold up to 25 findings; showing the neighbours makes it
        obvious when two lesions sit next to each other and which one the
        current verdict applies to.
        """

        scale_x, scale_y = scales
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for voxel in self._secondary_voxels:
            distance = abs(float(voxel[self.slice_axis]) - float(self._slice_index))
            if distance > NEIGHBOUR_SLICE_TOLERANCE:
                continue
            point = self._image_point_of(voxel)
            if point is None:
                continue
            x = rect.left() + point[0] * scale_x
            y = rect.top() + point[1] * scale_y
            if not rect.contains(QPointF(x, y)):
                continue
            colour = QColor(COLORS["neighbour"])
            # Fade with distance so the in-plane neighbours read as nearest.
            colour.setAlphaF(0.85 if distance < 0.51 else 0.4)
            painter.setPen(QPen(colour, 1.2, Qt.PenStyle.DashLine if distance >= 0.51 else Qt.PenStyle.SolidLine))
            painter.drawEllipse(QPointF(x, y), 7.0, 7.0)

    def _draw_cursor_crosshair(
        self,
        painter: QPainter,
        rect: QRectF,
        scales: tuple[float, float],
    ) -> None:
        """Show where the reader last clicked, in every view.

        Without this, clicking to inspect a position gives no feedback in the
        other two planes, so there is no way to tell what was actually picked
        before pressing ``Move here``.
        """

        point = self._target_image_point()
        if point is None:
            return
        # When the cursor sits on the finding, the red marker already says so.
        if self._marker_voxel is not None and np.allclose(
            self._target_voxel, self._marker_voxel, atol=0.5
        ):
            return
        scale_x, scale_y = scales
        x = rect.left() + point[0] * scale_x
        y = rect.top() + point[1] * scale_y
        if not rect.contains(QPointF(x, y)):
            return
        on_slice = abs(float(self._target_voxel[self.slice_axis]) - float(self._slice_index)) < 0.51
        colour = QColor(COLORS["cursor"])
        colour.setAlphaF(0.85 if on_slice else 0.4)
        painter.setPen(QPen(colour, 1.1, Qt.PenStyle.SolidLine if on_slice else Qt.PenStyle.DashLine))
        gap = 9.0
        painter.drawLine(QPointF(rect.left(), y), QPointF(x - gap, y))
        painter.drawLine(QPointF(x + gap, y), QPointF(rect.right(), y))
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, y - gap))
        painter.drawLine(QPointF(x, y + gap), QPointF(x, rect.bottom()))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(x - 3.0, y - 3.0, 6.0, 6.0))

    def _draw_hud_text(
        self,
        painter: QPainter,
        text: str,
        *,
        x: float,
        y: float,
        colour: str,
        align_right: bool = False,
    ) -> QRectF:
        """Overlay text on a backdrop.

        Plain text on an MRI is unreadable wherever the tissue underneath is
        bright, which for SWI is most of the image.
        """

        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 9.0
        height = metrics.height() + 3.0
        box = QRectF(x - width if align_right else x, y, width, height)
        backdrop = QColor("#04060a")
        backdrop.setAlphaF(0.68)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(backdrop)
        painter.drawRoundedRect(box, 3.0, 3.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QColor(colour))
        painter.drawText(
            box.adjusted(4.5, 0, -4.5, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            text,
        )
        return box

    def _draw_ghost_marker(
        self,
        painter: QPainter,
        rect: QRectF,
        scales: tuple[float, float],
    ) -> None:
        """Show where the source put the finding, while a correction is shown."""

        distance = abs(float(self._ghost_voxel[self.slice_axis]) - float(self._slice_index))
        if distance > NEIGHBOUR_SLICE_TOLERANCE:
            return
        point = self._image_point_of(self._ghost_voxel)
        if point is None:
            return
        scale_x, scale_y = scales
        x = rect.left() + point[0] * scale_x
        y = rect.top() + point[1] * scale_y
        if not rect.contains(QPointF(x, y)):
            return
        colour = QColor(COLORS["dim"])
        colour.setAlphaF(0.9 if distance < 0.51 else 0.45)
        painter.setPen(QPen(colour, 1.3, Qt.PenStyle.DotLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(x, y), 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 8))
        self._draw_hud_text(painter, "source", x=x + 8, y=y - 8, colour=COLORS["text"])

    def _draw_variant_markers(
        self,
        painter: QPainter,
        rect: QRectF,
        scales: tuple[float, float],
    ) -> None:
        """Every reader's position for this finding, held open for comparison."""

        scale_x, scale_y = scales
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        drawable: list[tuple[float, float, QColor, str]] = []
        for voxel, colour_name, label in self._variant_markers:
            distance = abs(float(voxel[self.slice_axis]) - float(self._slice_index))
            if distance > NEIGHBOUR_SLICE_TOLERANCE:
                continue
            point = self._image_point_of(voxel)
            if point is None:
                continue
            x = rect.left() + point[0] * scale_x
            y = rect.top() + point[1] * scale_y
            if not rect.contains(QPointF(x, y)):
                continue
            colour = QColor(colour_name)
            colour.setAlphaF(1.0 if distance < 0.51 else 0.5)
            drawable.append((x, y, colour, label))

        for x, y, colour, _label in drawable:
            painter.setPen(QPen(colour, 1.6))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(x, y), 6.0, 6.0)
            painter.drawLine(QPointF(x - 10, y), QPointF(x - 7, y))
            painter.drawLine(QPointF(x + 7, y), QPointF(x + 10, y))

        # Corrections are usually a couple of millimetres apart, so at lesion
        # zoom the labels would land on top of each other: stack them instead
        # and run a leader line back to the marker each one belongs to.
        drawable.sort(key=lambda item: item[1])
        for index, (x, y, colour, label) in enumerate(drawable):
            label_y = y - 14.0 - index * 14.0
            to_left = x > self.width() * 0.6
            label_x = x - 12.0 if to_left else x + 12.0
            leader = QColor(colour)
            leader.setAlphaF(0.55)
            painter.setPen(QPen(leader, 1.0, Qt.PenStyle.DotLine))
            painter.drawLine(
                QPointF(x + (-8.0 if to_left else 8.0), y),
                QPointF(x + (-11.0 if to_left else 11.0), label_y + 7.0),
            )
            self._draw_hud_text(
                painter, label, x=label_x, y=label_y, colour=colour, align_right=to_left
            )

    def _draw_direction_labels(self, painter: QPainter, rect: QRectF) -> None:
        # Derived from the orientation of the array being displayed, so the
        # labels follow the display preset automatically.
        axcodes = self.volume.orientation if self.volume is not None else preset_axcodes(None)
        left, right, top, bottom = plane_directions(self.plane, tuple(axcodes))
        # Pin the labels to the visible area: when the view is zoomed into a
        # lesion the image rectangle extends far outside the widget, and labels
        # drawn on its edges would be invisible exactly when orientation
        # matters most.
        visible = rect.intersected(QRectF(self.rect()))
        if visible.width() < 40 or visible.height() < 40:
            visible = QRectF(self.rect())
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        metrics = painter.fontMetrics()
        half = metrics.height() / 2.0
        colour = COLORS["direction"]
        self._draw_hud_text(painter, left, x=visible.left() + 3, y=visible.center().y() - half, colour=colour)
        self._draw_hud_text(
            painter, right, x=visible.right() - 3, y=visible.center().y() - half, colour=colour, align_right=True
        )
        centre = visible.center().x() - metrics.horizontalAdvance(top) / 2.0 - 4.5
        self._draw_hud_text(painter, top, x=centre, y=visible.top() + 3, colour=colour)
        centre = visible.center().x() - metrics.horizontalAdvance(bottom) / 2.0 - 4.5
        self._draw_hud_text(
            painter, bottom, x=centre, y=visible.bottom() - metrics.height() - 6, colour=colour
        )


class ViewPanel(QFrame):
    """Header + canvas + precise slider, modelled after the supplied viewer."""

    # ``sliceRequested`` carries an absolute slice index (slider), while
    # ``sliceStepRequested`` carries a relative wheel step.  Forwarding a wheel
    # step onto the absolute signal made one wheel notch jump to slice 1.
    sliceRequested = Signal(str, int)
    sliceStepRequested = Signal(str, int)
    maximizeRequested = Signal(str)

    def __init__(self, plane: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.plane = plane
        self.canvas = SliceCanvas(plane, self)
        self.canvas.sliceScrollRequested.connect(self.sliceStepRequested)
        self.canvas.zoomChanged.connect(self.set_zoom_combo)
        self._updating_slider = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        header = QFrame()
        header.setStyleSheet(
            f"background:{COLORS['header']}; border:0;"
            "border-top-left-radius:6px; border-top-right-radius:6px;"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(9, 3, 5, 3)
        hl.setSpacing(6)
        title = _label(PLANE_TITLES[plane], bold=True, size=9)
        hl.addWidget(title)
        hl.addStretch(1)
        self.zoom_combo = QComboBox()
        self.zoom_combo.setEditable(True)
        self.zoom_combo.addItem("Autofit")
        self.zoom_combo.addItem(LESION_ZOOM_LABEL)
        self.zoom_combo.addItems([f"{value}%" for value in ZOOM_PRESETS])
        self.zoom_combo.setCurrentText("Autofit")
        self.zoom_combo.setMinimumWidth(84)
        self.zoom_combo.setMaximumWidth(104)
        self.zoom_combo.setToolTip("Zoom for this view · Lesion frames a fixed field of view around the target")
        self.zoom_combo.currentTextChanged.connect(self.canvas.set_zoom_mode)
        zoom_line_edit = self.zoom_combo.lineEdit()
        if zoom_line_edit is not None:
            zoom_line_edit.editingFinished.connect(self._restore_zoom_text)
        hl.addWidget(self.zoom_combo)
        self.sync_cb = QCheckBox("Link")
        self.sync_cb.setChecked(True)
        self.sync_cb.setToolTip(
            "When enabled, clicking in another view moves this one to the same point.\n"
            "Scrolling is always local to the view under the mouse."
        )
        hl.addWidget(self.sync_cb)
        self.maximize_btn = QPushButton("Full")
        self.maximize_btn.setObjectName("IconButton")
        self.maximize_btn.setCheckable(True)
        self.maximize_btn.setToolTip("Show only this view  (F)")
        self.maximize_btn.clicked.connect(lambda _checked=False: self.maximizeRequested.emit(self.plane))
        hl.addWidget(self.maximize_btn)
        root.addWidget(header)
        root.addWidget(self.canvas, 1)

        footer = QFrame()
        footer.setStyleSheet(
            f"background:{COLORS['header']}; border:0;"
            "border-bottom-left-radius:6px; border-bottom-right-radius:6px;"
        )
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(9, 2, 9, 3)
        fl.setSpacing(7)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._slider_changed)
        self.slice_label = _label("— / —", color=COLORS["dim"], size=8)
        self.slice_label.setMinimumWidth(58)
        self.slice_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        fl.addWidget(self.slider, 1)
        fl.addWidget(self.slice_label)
        root.addWidget(footer)
        self.canvas.sliceChanged.connect(self._canvas_slice_changed)

    @property
    def sync_enabled(self) -> bool:
        return self.sync_cb.isChecked()

    def _slider_changed(self, value: int) -> None:
        if self._updating_slider:
            return
        self.sliceRequested.emit(self.plane, int(value))

    def _canvas_slice_changed(self, _plane: str, current: int, maximum: int) -> None:
        self._updating_slider = True
        self.slider.setRange(0, max(0, maximum))
        self.slider.setValue(int(current))
        self._updating_slider = False
        # Without a volume the canvas reports 0/0; showing "1 / 1" there would
        # suggest a one-slice image instead of an unavailable view.
        has_volume = self.canvas.volume is not None
        self.slider.setEnabled(has_volume)
        self.slice_label.setText(f"{current + 1} / {maximum + 1}" if has_volume else "— / —")

    def set_zoom_combo(self, text: str) -> None:
        self.zoom_combo.blockSignals(True)
        self.zoom_combo.setCurrentText(text)
        self.zoom_combo.blockSignals(False)

    def _restore_zoom_text(self) -> None:
        """Undo unparsable text typed into the editable zoom combo."""

        if self.zoom_combo.currentText().strip() != self.canvas.zoom_text:
            self.set_zoom_combo(self.canvas.zoom_text)


class RoundDialog(QDialog):
    """Choose which of this reader's rounds to open, or start another.

    Starting a new round by accident used to be hard to undo, because only the
    most recent one could be resumed. Every round is listed here, and the one
    used last time is selected by default.
    """

    def __init__(
        self,
        reader_id: str,
        rounds: list[dict[str, Any]],
        *,
        preferred_round: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.rounds = list(rounds)
        self.chosen_round: int | None = None
        self.chosen_session_id: str | None = None
        self.start_new = False
        self.setWindowTitle("Microbleed Review · Review round")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.addWidget(_label(f"Rounds for {reader_id}", color=COLORS["accent"], bold=True, size=13))
        layout.addWidget(
            _label(
                "Pick up an earlier round, or start another one. A new round keeps every "
                "earlier round intact; reviews are recorded per round.",
                color=COLORS["dim"],
                size=9,
            )
        )

        self.list = QListWidget()
        self.list.setMinimumHeight(150)
        next_round = (max(int(item["review_round"]) for item in self.rounds) + 1) if self.rounds else 1
        for entry in self.rounds:
            review_round = int(entry["review_round"])
            reviewed = int(entry.get("reviewed_count") or 0)
            rois = int(entry.get("roi_count") or 0)
            parts = [f"Round {review_round}"]
            if reviewed or rois:
                parts.append(_human_count(reviewed, "finding") + " reviewed")
                if rois:
                    parts.append(_human_count(rois, "segmentation"))
                if entry.get("last_case_id"):
                    parts.append(f"last case {entry['last_case_id']}")
            else:
                # Usually a round started by accident; easy to spot and reuse.
                parts.append("nothing recorded yet")
            item = QListWidgetItem("   ·   ".join(parts))
            if not (reviewed or rois):
                item.setForeground(QColor(COLORS["faint"]))
            item.setData(Qt.ItemDataRole.UserRole, review_round)
            item.setToolTip(
                f"Started {entry.get('started_at') or 'unknown'}\n"
                f"Last opened {entry.get('last_seen_at') or 'unknown'}"
            )
            self.list.addItem(item)
        new_item = QListWidgetItem(f"Start a new round  (round {next_round})")
        new_item.setData(Qt.ItemDataRole.UserRole, "new")
        new_item.setForeground(QColor(COLORS["accent"]))
        self.list.addItem(new_item)

        # Default to the round this reader used last time, if it still exists.
        row = next(
            (
                index
                for index in range(self.list.count())
                if self.list.item(index).data(Qt.ItemDataRole.UserRole) == preferred_round
            ),
            0,
        )
        self.list.setCurrentRow(row)
        self.list.itemDoubleClicked.connect(lambda _item: self._accept())
        layout.addWidget(self.list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Open).setText("Open round")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        value = item.data(Qt.ItemDataRole.UserRole)
        if value == "new":
            self.start_new = True
        else:
            self.chosen_round = int(value)
            entry = next(
                item for item in self.rounds if int(item["review_round"]) == self.chosen_round
            )
            self.chosen_session_id = str(entry.get("session_id") or "")
        self.accept()


class ReaderDialog(QDialog):
    """Register a reader and implement resume versus new-round behavior."""

    datasetChangeRequested = Signal()

    def __init__(
        self,
        db_path: Path,
        parent: QWidget | None = None,
        *,
        initial_reader: str = "",
        dataset: Dataset | None = None,
        allow_dataset_change: bool = False,
        settings: ViewerSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.db_path = Path(db_path)
        self.settings = settings
        self.session: dict[str, Any] | None = None
        self.reader_id = ""
        self.change_dataset_requested = False
        self.setWindowTitle("Microbleed Review · Review session")
        self.setMinimumWidth(470)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(_label("Review session", color=COLORS["accent"], bold=True, size=15))
        layout.addWidget(_label("Enter the reader name used for this review. Your work is stored in the shared datasheet."))
        form = QFormLayout()
        self.reader_edit = QLineEdit(str(initial_reader or ""))
        self.reader_edit.setPlaceholderText("your name, as it should appear in the results")
        form.addRow("Reader:", self.reader_edit)
        layout.addLayout(form)
        if dataset is not None:
            layout.addWidget(_separator())
            dataset_row = QHBoxLayout()
            dataset_row.setSpacing(7)
            info = _label(f"Dataset: {dataset.name}", color=COLORS["dim"], size=9)
            info.setToolTip(
                f"Workbook: {dataset.workbook}\nMRI folder: {dataset.data_root}\n"
                f"Reviews: {dataset.review_db}"
            )
            info.setWordWrap(True)
            dataset_row.addWidget(info, 1)
            if allow_dataset_change:
                change_btn = QPushButton("Change…")
                change_btn.clicked.connect(self._request_dataset_change)
                dataset_row.addWidget(change_btn)
            layout.addLayout(dataset_row)
        self.note = _label(
            "If you have reviewed before, the next step lets you pick which round to open.",
            color=COLORS["dim"],
            size=9,
        )
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Open review")
        buttons.accepted.connect(self._open)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.reader_edit.returnPressed.connect(self._open)

    def _request_dataset_change(self) -> None:
        """Close with a marker so the caller can offer another dataset."""

        self.change_dataset_requested = True
        self.reject()

    def _open(self) -> None:
        reader = self.reader_edit.text().strip()
        if not reader:
            QMessageBox.warning(self, "Reader required", "Please enter a reader name before opening the viewer.")
            return
        rounds = list_reader_rounds(self.db_path, reader)
        if rounds:
            preferred = self.settings.last_round(self.db_path, reader) if self.settings else None
            if preferred is None:
                preferred = int(rounds[0]["review_round"])
            chooser = RoundDialog(reader, rounds, preferred_round=preferred, parent=self)
            if chooser.exec() != QDialog.DialogCode.Accepted:
                return
            if chooser.start_new:
                session = start_new_session(self.db_path, reader, str(rounds[0].get("session_id") or ""))
            elif chooser.chosen_session_id:
                session = resume_session(self.db_path, chooser.chosen_session_id)
            else:
                return
        else:
            session = start_new_session(self.db_path, reader)
        self.reader_id = reader
        self.session = session
        if self.settings is not None:
            self.settings.set_last_round(self.db_path, reader, int(session["review_round"]))
        # The dialog has no background writer; this one write happens before
        # the window exists and is not on a hot path.
        try:
            log_event(
                self.db_path,
                "session_opened",
                session_id=session["session_id"],
                reader_id=reader,
                review_round=int(session["review_round"]),
                details={"resumed": bool(candidate and session["review_round"] == candidate["review_round"])},
            )
        except Exception:
            pass
        self.accept()


class LesionCanvas(QWidget):
    """A mask you can turn over in your hand.

    Software rendering, on purpose.  The alternative was Qt3D or a raw
    QOpenGLWidget, both of which add an OpenGL requirement to a tool that has
    so far needed nothing but numpy, nibabel and widgets -- and reading rooms
    are exactly where remote desktops and software GL live.  It costs nothing
    to avoid: measured on this machine, a 3 mm lesion is 174 faces and 0.8 ms
    a frame, a 5 mm one 486 faces and 2.1 ms, an 8 mm one 5.1 ms.  A whole
    brain surface would not fit in this budget; a microbleed is not close to
    the edge of it.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(280, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._quads = np.zeros((0, 4, 3))
        self._normals = np.zeros((0, 3))
        self._tints = np.zeros((0, 3))
        self._radius_mm = 1.0
        self._yaw = _DEFAULT_YAW
        self._pitch = _DEFAULT_PITCH
        self._zoom = 1.0
        self._drag: QPointF | None = None
        self._caption = ""
        # The brain behind the lesion: two resampled cubes, coarse for while
        # the mouse is down and fine for when it stops.
        self._cube_fine: np.ndarray | None = None
        self._cube_coarse: np.ndarray | None = None
        self._cube_mm = (1.0, 1.0)
        self._cube_window = (0.0, 1.0)
        self._lesion_centre_mm = np.zeros(3)
        self._brain_alpha = 0.55
        self._buffers: list[np.ndarray] = []

    # -- contents ---------------------------------------------------------
    def set_surface(
        self,
        quads,
        normals,
        caption: str = "",
        radius_mm: float | None = None,
        tints=None,
    ) -> None:
        self._quads = np.asarray(quads, dtype=np.float64)
        self._normals = np.asarray(normals, dtype=np.float64)
        if tints is None:
            base = QColor(COLORS["roi"])
            tints = np.repeat(
                np.array([[base.red(), base.green(), base.blue()]], dtype=np.float64),
                len(self._quads),
                axis=0,
            )
        self._tints = np.asarray(tints, dtype=np.float64)
        self._caption = caption
        if radius_mm is not None:
            self._radius_mm = max(float(radius_mm), 0.5)
        elif len(self._quads):
            corners = self._quads.reshape(-1, 3)
            self._radius_mm = float(max(np.abs(corners).max(), 0.5))
        else:
            self._radius_mm = 1.0
        self.update()

    def set_context(self, fine, coarse, mm_per_voxel, window, lesion_centre_mm) -> None:
        """The volume the lesion sits in, already resampled to cubes.

        ``None`` for the cubes turns the brain off; the lesion is then drawn
        on its own, which is the default and the faster of the two.
        """

        self._cube_fine = fine
        self._cube_coarse = coarse if coarse is not None else fine
        self._cube_mm = tuple(float(value) for value in mm_per_voxel)
        self._cube_window = (float(window[0]), float(window[1]))
        self._lesion_centre_mm = np.asarray(lesion_centre_mm, dtype=np.float64)
        self.update()

    def set_brain_alpha(self, alpha: float) -> None:
        self._brain_alpha = max(0.0, min(1.0, float(alpha)))
        self.update()

    def has_surface(self) -> bool:
        return bool(len(self._quads))

    def has_context(self) -> bool:
        return self._cube_fine is not None and bool(self._cube_fine.size)

    def reset_view(self) -> None:
        self._yaw, self._pitch, self._zoom = _DEFAULT_YAW, _DEFAULT_PITCH, 1.0
        self.update()

    @property
    def angles(self) -> tuple[float, float, float]:
        """Yaw, pitch and zoom, so a test can drive the view without a mouse."""

        return self._yaw, self._pitch, self._zoom

    # -- interaction ------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._drag is None:
            return
        delta = event.position() - self._drag
        self._drag = event.position()
        # Dragging right pushes the near side right, the way a globe turns
        # under a finger.  Adding the delta sent the far side that way, which
        # reads as the model rotating backwards.
        self._yaw -= delta.x() * 0.012
        # Clamped rather than free: rolling past the pole flips the lesion
        # upside down mid-drag, and there is no up-vector to recover it from.
        self._pitch = max(-1.5, min(1.5, self._pitch + delta.y() * 0.012))
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._drag = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        # The coarse cube was for the drag; redraw the fine one now.
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.reset_view()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        self._zoom = max(0.4, min(6.0, self._zoom * (1.12 ** steps)))
        self.update()

    def step_zoom(self, steps: float) -> None:
        self._zoom = max(0.4, min(6.0, self._zoom * (1.12 ** steps)))
        self.update()

    def turn(self, yaw: float, pitch: float) -> None:
        self._yaw += yaw
        self._pitch = max(-1.5, min(1.5, self._pitch + pitch))
        self.update()

    # -- drawing ----------------------------------------------------------
    # The display array is ordered (L, P, I), so an unrotated view of it looks
    # straight down the I axis -- an axial slice, which is a head lying on its
    # back.  This maps the volume's axes onto the screen's before any turning:
    # screen right is L, screen down is I (so superior is up), and depth is P
    # (so the face is towards the reader).  Yaw is then a turntable about the
    # patient's own vertical, which is the rotation someone reaches for first.
    _BASE_VIEW = np.array(
        [
            [1.0, 0.0, 0.0],   # screen x  <- L
            [0.0, 0.0, 1.0],   # screen y  <- I
            [0.0, 1.0, 0.0],   # depth     <- P
        ]
    )

    def _rotation(self) -> np.ndarray:
        cy, sy = math.cos(self._yaw), math.sin(self._yaw)
        cp, sp = math.cos(self._pitch), math.sin(self._pitch)
        turn = np.array(
            [[cy, 0.0, sy], [sp * sy, cp, -sp * cy], [-cp * sy, sp, cp * cy]]
        )
        return turn @ self._BASE_VIEW

    def _scale(self) -> float:
        """Screen pixels per millimetre at the current zoom."""

        span = min(self.width(), self.height()) * 0.42
        return span * self._zoom / self._radius_mm

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor(COLORS["canvas"]))
        if not len(self._quads) and not self.has_context():
            painter.setPen(QColor(COLORS["dim"]))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No segmentation on this finding yet",
            )
            painter.end()
            return

        rotation = self._rotation()
        scale = self._scale()
        self._buffers.clear()
        # The half of the head behind the lesion, then the lesion, then the
        # half in front of it: three layers get the occlusion right without
        # sorting a surface against a volume.
        if self.has_context():
            depth_mm = float(self._lesion_centre_mm @ rotation.T[:, 2])
            self._draw_brain(painter, rotation, scale, near=depth_mm)
        if not len(self._quads):
            if self.has_context():
                self._draw_brain(painter, rotation, scale, far=float(
                    self._lesion_centre_mm @ rotation.T[:, 2]
                ))
            self._draw_scale_bar(painter, scale)
            self._draw_axes(painter, rotation, scale)
            painter.end()
            return

        rotated = self._quads @ rotation.T
        facing = (self._normals @ rotation.T)[:, 2]
        # Screen y grows downwards, so a face pointing at the viewer has a
        # negative rotated z; the rest are hidden behind it.
        visible = facing < 0
        if not visible.any():
            painter.end()
            return
        centre = np.array([self.width() / 2.0, self.height() / 2.0])
        screen = rotated[visible][:, :, :2] * scale + centre
        depth = rotated[visible][:, :, 2].mean(axis=1)
        order = np.argsort(depth)          # painter's algorithm, far to near
        screen = screen[order]
        # Straight Lambert from a headlight, floored so a face edge-on is dark
        # but not black: an unlit facet reads as a hole in the surface.
        lit = np.clip(-facing[visible][order], 0.18, 1.0)
        tints = self._tints[visible][order]
        painter.setPen(Qt.PenStyle.NoPen)
        for quad, shade, tint in zip(screen, lit, tints):
            painter.setBrush(
                QColor(
                    int(tint[0] * shade),
                    int(tint[1] * shade),
                    int(tint[2] * shade * 0.85 + 40 * (1.0 - shade)),
                )
            )
            painter.drawPolygon(
                QPolygonF([QPointF(float(x), float(y)) for x, y in quad])
            )
        if self.has_context():
            depth_mm = float(self._lesion_centre_mm @ rotation.T[:, 2])
            self._draw_brain(painter, rotation, scale, far=depth_mm)
            # A 3 mm lesion in a 220 mm head is five pixels across, and half
            # of those are behind the glass.  The ring is how you find it;
            # what it circles is still the lesion, drawn to scale.
            self._draw_lesion_marker(painter, rotation, scale)
        self._draw_scale_bar(painter, scale)
        self._draw_axes(painter, rotation, scale)
        if self._caption:
            painter.setPen(QColor(COLORS["dim"]))
            painter.drawText(8, 16, self._caption)
        painter.end()

    def _draw_brain(
        self,
        painter: QPainter,
        rotation: np.ndarray,
        scale: float,
        *,
        near: float | None = None,
        far: float | None = None,
    ) -> None:
        """One slab of the head, as translucent glass.

        Alpha follows the tissue rather than being flat, so the air around the
        head does not fog the lesion: a mean projection of SWI is nearly zero
        outside the scalp, and multiplying that into the alpha is what makes
        the thing read as a head with something inside it rather than as a
        grey rectangle.
        """

        cube = self._cube_coarse if self._drag is not None else self._cube_fine
        if cube is None or not cube.size:
            return
        mm_per_voxel = self._cube_mm[0 if self._drag is None else 1]
        bound = None if near is None else near / mm_per_voxel
        bound_far = None if far is None else far / mm_per_voxel
        projection = project_context(cube, rotation, near=bound, far=bound_far)
        if not projection.size:
            return
        low, high = self._cube_window
        norm = np.clip((projection - low) / max(high - low, 1e-6), 0.0, 1.0)

        size = projection.shape[0]
        grey = (norm * 235.0).astype(np.uint32)
        alpha = (norm * (255.0 * self._brain_alpha)).astype(np.uint32)
        # Slightly cool, so tissue never competes with the lesion's yellow.
        pixels = (alpha << 24) | (grey << 16) | (grey << 8) | np.minimum(grey + 12, 255)
        buffer = np.ascontiguousarray(pixels, dtype=np.uint32)
        self._buffers.append(buffer)          # QImage does not copy
        image = QImage(
            buffer.data, size, size, size * 4, QImage.Format.Format_ARGB32
        )
        span = size * mm_per_voxel * scale
        # The projection is 96 or 128 px landing on about 400, so the stretch
        # is where the blockiness comes from -- and a smooth one is free
        # (0.18 ms measured).  Interpolating the volume sampling instead is
        # not: trilinear per frame costs 24.6 ms -> 156.2, and a trilinear
        # cube costs 8 ms -> 218 at build time, for a projection that differs
        # by 1.4% of its range.  A mean through a hundred samples has already
        # done the anti-aliasing; only the last stretch had not.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(
            QRectF(
                self.width() / 2.0 - span / 2.0,
                self.height() / 2.0 - span / 2.0,
                span,
                span,
            ),
            image,
        )

    def _draw_lesion_marker(self, painter: QPainter, rotation: np.ndarray, scale: float) -> None:
        centre = self._lesion_centre_mm @ rotation.T
        x = self.width() / 2.0 + centre[0] * scale
        y = self.height() / 2.0 + centre[1] * scale
        radius = max(9.0, self._radius_mm * 0.02 * scale)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(COLORS["target"]), 1.4))
        painter.drawEllipse(QPointF(x, y), radius, radius)
        gap = radius + 3.0
        arm = radius + 9.0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            painter.drawLine(
                QPointF(x + dx * gap, y + dy * gap), QPointF(x + dx * arm, y + dy * arm)
            )

    def _draw_scale_bar(self, painter: QPainter, scale: float) -> None:
        """Without one, a blob on a screen has no size."""

        target_px = self.width() * 0.28
        for millimetres in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
            length = millimetres * scale
            if length >= target_px:
                break
        y = self.height() - 14
        x = 10
        painter.setPen(QPen(QColor(COLORS["text"]), 1))
        painter.drawLine(int(x), int(y), int(x + length), int(y))
        painter.drawLine(int(x), int(y - 3), int(x), int(y + 3))
        painter.drawLine(int(x + length), int(y - 3), int(x + length), int(y + 3))
        painter.drawText(int(x), int(y - 6), f"{millimetres:g} mm")

    def _draw_axes(self, painter: QPainter, rotation: np.ndarray, scale: float) -> None:
        """Which way is left, posterior, inferior -- the display axes.

        A lesion turned in the hand loses its orientation within one drag, and
        "is that spur pointing inferiorly or anteriorly" is a question about
        the anatomy, not about the mask.
        """

        origin = np.array([self.width() - 46.0, self.height() - 46.0])
        arm = 22.0
        painter.setFont(QFont(painter.font().family(), 7))
        for axis, code in enumerate(DISPLAY_AXCODES):
            direction = np.zeros(3)
            direction[axis] = 1.0
            end = (direction @ rotation.T)[:2] * arm
            painter.setPen(QPen(QColor(COLORS["dim"]), 1))
            painter.drawLine(
                int(origin[0]), int(origin[1]), int(origin[0] + end[0]), int(origin[1] + end[1])
            )
            painter.setPen(QColor(COLORS["direction"]))
            painter.drawText(
                int(origin[0] + end[0] * 1.28) - 3, int(origin[1] + end[1] * 1.28) + 3, code
            )


class Lesion3DDialog(QDialog):
    """The selected finding's mask, on its own, turnable.

    A separate window rather than a fourth cell in the grid: it is a check you
    run on a mask you have just drawn, not something to read every finding
    against, and it is worth more left open beside the images while painting
    -- it redraws as the mask changes.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Segmentation in 3D")
        self.setModal(False)
        self.resize(430, 470)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(7)
        self.heading = _label("No finding", color=COLORS["text"], size=10, bold=True, wrap=False)
        layout.addWidget(self.heading)
        self.measurements = _label("", color=COLORS["dim"], size=8)
        self.measurements.setWordWrap(True)
        layout.addWidget(self.measurements)
        self.canvas = LesionCanvas()
        layout.addWidget(self.canvas, 1)

        options = QHBoxLayout()
        options.setSpacing(8)
        self.brain_cb = QCheckBox("In the brain")
        self.brain_cb.setToolTip(
            "Put the lesion back where it sits, inside a see-through head.\n"
            "The head is a mean projection of the sequence on screen, split at\n"
            "the lesion's depth so what is behind it stays behind it.\n"
            "No brain extraction is involved, so nothing here is a boundary\n"
            "anyone should measure against."
        )
        self.brain_cb.toggled.connect(self._options_changed)
        options.addWidget(self.brain_cb)
        # Which sequence the head is made of.  QSM and SWI show a microbleed's
        # surroundings differently, and which one reads better is a matter of
        # the case rather than something this window should decide.
        self.brain_source = QComboBox()
        self.brain_source.setToolTip("Which sequence the see-through head is drawn from")
        self.brain_source.setEnabled(False)
        self.brain_source.currentIndexChanged.connect(
            lambda _index: self._options_changed()
        )
        self.brain_cb.toggled.connect(self.brain_source.setEnabled)
        options.addWidget(self.brain_source)
        self.smooth_cb = QCheckBox("Smooth")
        self.smooth_cb.setToolTip(
            "Round off the voxel steps.  This smooths the surface that is\n"
            "stored, it does not re-derive a different one: no voxel is added,\n"
            "removed or re-thresholded, and the numbers above are counted from\n"
            "the voxels either way.  It does pull the surface in slightly --\n"
            "0.7% on a 5 mm ball, 4.5% on a one-voxel-thick sheet."
        )
        self.smooth_cb.toggled.connect(self._options_changed)
        options.addWidget(self.smooth_cb)
        options.addStretch(1)
        self.brain_alpha = QSlider(Qt.Orientation.Horizontal)
        self.brain_alpha.setRange(10, 100)
        self.brain_alpha.setValue(55)
        self.brain_alpha.setFixedWidth(96)
        self.brain_alpha.setToolTip("How solid the head looks")
        self.brain_alpha.valueChanged.connect(
            lambda value: self.canvas.set_brain_alpha(value / 100.0)
        )
        self.brain_alpha.setEnabled(False)
        self.brain_cb.toggled.connect(self.brain_alpha.setEnabled)
        options.addWidget(_label("glass", color=COLORS["faint"], size=8, wrap=False))
        options.addWidget(self.brain_alpha)
        layout.addLayout(options)

        hint = _label(
            "Drag to turn · wheel to zoom · double-click to reset",
            color=COLORS["faint"],
            size=8,
            wrap=False,
        )
        layout.addWidget(hint)
        # Set by the window: the surface has to be rebuilt when either box
        # changes, because both change what is asked of lesion_surface.
        self.options_changed: Any = None

    def _options_changed(self, _checked: bool = False) -> None:
        if callable(self.options_changed):
            self.options_changed()

    @property
    def wants_brain(self) -> bool:
        return self.brain_cb.isChecked()

    @property
    def brain_modality(self) -> str:
        return str(self.brain_source.currentData() or "")

    def offer_brain_sources(self, keys: list[str], labels: dict[str, str], prefer: str) -> None:
        """Rebuild the sequence list for the case that is loaded.

        Only what the case actually has: offering a sequence that is missing
        and then quietly drawing another one is worse than not offering it.
        """

        wanted = self.brain_modality or prefer
        if [self.brain_source.itemData(i) for i in range(self.brain_source.count())] == keys:
            return
        self.brain_source.blockSignals(True)
        self.brain_source.clear()
        for key in keys:
            self.brain_source.addItem(labels.get(key, key), key)
        index = self.brain_source.findData(wanted)
        self.brain_source.setCurrentIndex(index if index >= 0 else 0)
        self.brain_source.blockSignals(False)

    @property
    def wants_smooth(self) -> bool:
        return self.smooth_cb.isChecked()

    def show_lesion(
        self,
        title: str,
        measurements: str,
        quads,
        normals,
        *,
        radius_mm: float | None = None,
        tints=None,
    ) -> None:
        self.heading.setText(title)
        self.measurements.setText(measurements)
        self.canvas.set_surface(quads, normals, radius_mm=radius_mm, tints=tints)


class MicrobleedViewer(QMainWindow):
    def __init__(
        self,
        db_path: Path,
        data_root: Path,
        session: dict[str, Any],
        parent: QWidget | None = None,
        *,
        settings: ViewerSettings | None = None,
        dataset: Dataset | None = None,
    ) -> None:
        super().__init__(parent)
        self.db_path = Path(db_path)
        self.data_root = Path(data_root)
        self.settings = settings or ViewerSettings()
        # The workbook is only needed to describe the dataset; the review store
        # already holds the imported findings.
        self.dataset = dataset or Dataset.create(SOURCE_XLSX, data_root, db_path)
        self.session = dict(session)
        self.reader_id = str(session["reader_id"])
        self.review_round = int(session["review_round"])
        self.session_id = str(session["session_id"])
        self.setWindowTitle(f"{APP_TITLE} · {self.reader_id} · round {self.review_round}")
        # The window carries the icon itself, so it is right even when the
        # viewer is constructed directly rather than through ``main``.
        window_icon = icon_file()
        if window_icon is not None:
            self.setWindowIcon(QIcon(str(window_icon)))
        self.resize(1540, 960)
        # Measured, not guessed: the smallest size at which every control in
        # the review panel still fits with all sections expanded and a vertical
        # scrollbar present. Promising anything narrower would be a lie.
        self.setMinimumSize(MINIMUM_WINDOW_WIDTH, MINIMUM_WINDOW_HEIGHT)
        self.setStyleSheet(GLOBAL_STYLE)

        self.all_cases: list[dict[str, Any]] = []
        self.visible_cases: list[dict[str, Any]] = []
        self.current_case: dict[str, Any] | None = None
        self.current_case_id: str | None = None
        self.targets: list[dict[str, Any]] = []
        self.selected_target: dict[str, Any] | None = None
        # ``target_ras`` is the navigation cursor; ``marker_ras`` is where the
        # selected finding sits and does not move when the reader scrolls.
        self.target_ras: tuple[float, float, float] | None = None
        self.marker_ras: tuple[float, float, float] | None = None
        # Recorded positions of the selected finding, and which one is shown.
        # Segmentation state for the case on screen. One label volume per case
        # and reader; the value picks the finding out of it.
        self.label_volume: np.ndarray | None = None
        self._lesion_dialog: Lesion3DDialog | None = None
        self._context_cache: tuple[Any, Any] | None = None
        # Collapses a burst of brush events into one readout; see
        # _queue_roi_readout.
        self._readout_timer = QTimer(self)
        self._readout_timer.setSingleShot(True)
        self._readout_timer.setInterval(120)
        self._readout_timer.timeout.connect(self._update_roi_readout)
        self._others_cache: tuple[Any, Any] | None = None
        self.label_values: dict[str, int] = {}
        self.label_sources: dict[str, str | None] = {}
        # How each finding's mask was made, and under what settings, so the
        # exported volume can be reproduced or knowingly excluded.
        self.label_methods: dict[str, str] = {}
        self.label_settings: dict[str, dict[str, float]] = {}
        # The measurement from the last Generate, for the panel and the tests.
        self.last_segmentation: dict[str, Any] | None = None
        # Findings that already had a segmentation row when the case opened, so
        # clearing one still writes the deletion while an untouched finding
        # costs nothing.
        self._stored_roi_targets: set[str] = set()
        # Why this case cannot be segmented, when its sequences disagree.
        self._grid_problem: str | None = None
        # Panel tab keys in the order they appear, filled in while building.
        # Whether the window was maximised when the queue was pulled out, so
        # re-docking can put it back the way it was.
        # Whether the case queue stays open or folds down to its strip, and
        # whether it was the window that folded it rather than the reader.
        self.queue_pinned = True
        self._queue_auto_folded = False
        self._checked_screen_width = False
        self._roi_dirty = False
        # One entry per undoable step: the label shape it applies to, and the
        # (flat indices, previous values) pairs that reverse it.
        self._roi_undo: list[dict[str, Any]] = []
        self.position_variants: list[dict[str, Any]] = []
        self.selected_variant: str = "source"
        self.pending_ras: tuple[float, float, float] | None = None
        self.volumes: dict[str, Volume | None] = {modality: None for modality in MODALITY_ORDER}
        self.load_errors: dict[str, str] = {}
        preferred = self.settings.default_modality
        self.current_modality = preferred if preferred in MODALITY_ORDER else "swi"
        # Loading is intentionally synchronous. Keeping this marker makes
        # the state explicit without allowing a worker to outlive the window.
        self._load_thread = None
        self._load_generation = 0
        # Set while a case is being read on the GUI thread; that read pumps the
        # event loop, so timers must not start work on half-updated state.
        self._loading = False
        self._updating_form = False
        self._review_dirty = False
        self._closing = False
        self._verdict: int | None = None
        self.active_tool: str | None = None
        self._active_plane = "axial"
        self._maximized_plane: str | None = None
        # Window/level is per sequence: QSM and SWI have unrelated intensity
        # ranges, so one shared window would be meaningless.
        self._window_levels: dict[str, tuple[float, float]] = {}
        self._applying_window = False
        self.contrast_dialog: ContrastDialog | None = None
        # Room for all three sequences of the case on screen and all three of
        # the prefetched case, with two spare so a step back is still warm.
        self.volume_cache = VolumeCache(limit=8)
        self._prefetch_thread: QThread | None = None
        self._prefetch_worker: PrefetchWorker | None = None
        self._prefetch_generation = 0
        self._prefetch_timer = QTimer(self)
        self._prefetch_timer.setSingleShot(True)
        self._prefetch_timer.setInterval(400)
        self._prefetch_timer.timeout.connect(self._start_prefetch)
        self._session_timer = QTimer(self)
        self._session_timer.setSingleShot(True)
        self._session_timer.setInterval(400)
        self._session_timer.timeout.connect(self._save_session_state)
        self._writer = DatabaseWriter(self)
        self._writer.failed.connect(self._on_write_failed)
        self._writer.start()
        self._pending_slice_log: dict[str, Any] | None = None
        self._slice_log_timer = QTimer(self)
        self._slice_log_timer.setSingleShot(True)
        self._slice_log_timer.setInterval(900)
        self._slice_log_timer.timeout.connect(self._flush_slice_log)

        self._build_ui()
        self._bind_shortcuts()
        self._restore_session_preferences()
        self._update_save_buttons()
        self.set_tool(None)
        self.more_btn.setToolTip(self._dataset_tooltip())
        self._reload_case_list()
        self._set_status("Select a case from the queue.")

        last_case = self.session.get("last_case_id")
        initial = last_case if any(item["case_id"] == last_case for item in self.all_cases) else (self.all_cases[0]["case_id"] if self.all_cases else None)
        if initial:
            QTimer.singleShot(0, lambda case_id=initial: self.load_case(case_id))

    # ------------------------------------------------------------------ UI --
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 4)
        outer.setSpacing(7)

        # The case queue runs the full height of the window, with the toolbar
        # only over the two columns it belongs to: a heading above a list of
        # cases would be a heading about something else.
        self.queue_panel = self._build_sidebar()
        self.queue_panel.installEventFilter(self)
        self.queue_panel.setMinimumWidth(QUEUE_COLUMN_WIDTH)
        self.queue_panel.setMaximumWidth(QUEUE_COLUMN_WIDTH + 60)
        self.queue_rail = self._build_queue_rail()
        outer.addWidget(self.queue_rail)
        outer.addWidget(self.queue_panel)

        working = QWidget()
        working_layout = QVBoxLayout(working)
        working_layout.setContentsMargins(0, 0, 0, 0)
        working_layout.setSpacing(7)
        working_layout.addWidget(self._build_toolbar())

        # A splitter, not a fixed column: how much room the reference
        # material deserves depends on the reader and the screen, and it was
        # draggable before the three-column rebuild took the handle away.
        body = QSplitter(Qt.Orientation.Horizontal)
        body.setObjectName("BodySplit")
        body.setChildrenCollapsible(False)
        body.setHandleWidth(7)
        body.addWidget(self._build_view_area())

        # Everything that is reference or drawing, in one column of full
        # height: what the sheet and the other readers say on top, the mask
        # below.  Deciding where the finding is stays next to the images.
        right = QFrame()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._build_findings_panel())
        right_layout.addWidget(self._build_details_panel(), 1)
        right_layout.addWidget(self._build_shortcut_section())
        right.setMinimumWidth(RIGHT_COLUMN_MIN_WIDTH)
        right.setMaximumWidth(RIGHT_COLUMN_MAX_WIDTH)
        body.addWidget(right)
        self.right_column = right
        self.body_split = body
        # All the give goes to the images: dragging sets the reference column's
        # width, and resizing the window afterwards leaves it where it was put.
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)
        body.splitterMoved.connect(
            lambda _pos, _index: self.settings.set_right_column_width(right.width())
        )

        working_layout.addWidget(body, 1)
        outer.addWidget(working, 1)

        self._status_bar = QStatusBar()
        self._status_bar.setSizeGripEnabled(False)
        self.setStatusBar(self._status_bar)
        self._status_label = _label("Ready", color=COLORS["dim"], size=9, wrap=False)
        self._status_bar.addWidget(self._status_label, 1)
        # Permanent widgets survive ``showMessage``; the live readout is added
        # on the right so a status message never hides the mouse coordinate.
        self._coord_label = _label("Move the mouse over a view for voxel / RAS", color=COLORS["dim"], size=9, wrap=False)
        self._status_bar.addPermanentWidget(self._coord_label)

    def _toolbar_divider(self) -> QFrame:
        divider = QFrame()
        divider.setObjectName("ToolbarDivider")
        divider.setFrameShape(QFrame.Shape.VLine)
        return divider

    def _build_toolbar(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Toolbar")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 6, 8, 6)
        layout.setSpacing(11)

        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        self.case_title = ElidedLabel("No case loaded")
        self.case_title.setStyleSheet("font-weight:600; font-size:13pt;")
        title_col.addWidget(self.case_title)
        self.case_status = ElidedLabel("Select a case from the sidebar")
        self.case_status.setStyleSheet(f"color:{COLORS['dim']}; font-size:8pt;")
        title_col.addWidget(self.case_status)
        # Wide enough for the whole title when the row has room; it elides
        # only when the controls beside it need the space.
        title_holder = QWidget()
        title_holder.setLayout(title_col)
        title_holder.setMinimumWidth(90)
        title_holder.setMaximumWidth(460)
        layout.addWidget(title_holder)
        layout.addStretch(1)

        # Tools. Only one can be active, and clicking the active one turns it
        # off, so the default state is "no tool" and a click in a view does
        # nothing. Future tools (an ROI pen) join this row.
        self.tool_buttons: dict[str, QPushButton] = {}
        for key, text, tooltip in (
            (
                "point",
                "Point",
                "Place the crosshair where you click, to inspect a location or to "
                "correct where a finding sits",
            ),
            (
                "brush",
                "Brush",
                "Paint the segmentation of this finding with the left button",
            ),
            (
                "eraser",
                "Eraser",
                "Rub out part of this finding's segmentation with the left button",
            ),
        ):
            button = QPushButton(text)
            button.setObjectName("Segment")
            button.setCheckable(True)
            button.setIcon(_stroke_icon(key))
            button.setIconSize(QSize(15, 15))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(tooltip)
            button.clicked.connect(lambda checked, name=key: self.set_tool(name if checked else None))
            layout.addWidget(button)
            self.tool_buttons[key] = button
        layout.addWidget(self._toolbar_divider())

        # Sequence switching is the most frequent action during a read, so it
        # is a one-click segmented control with 1/2/3 shortcuts.
        self.modality_segments = SegmentedControl(
            [(key, MODALITY_SHORT_LABELS[key]) for key in MODALITY_BUTTON_ORDER]
        )
        for position, key in enumerate(MODALITY_BUTTON_ORDER, start=1):
            button = self.modality_segments.button(key)
            if button is not None:
                button.setToolTip(f"{MODALITY_LABELS[key]}  ({position})")
            button.setMinimumWidth(52)
        self.modality_segments.set_current_key(self.current_modality)
        self.modality_segments.selected.connect(self.set_modality)
        layout.addWidget(self.modality_segments)
        layout.addWidget(self._toolbar_divider())

        self.lesion_zoom_btn = QPushButton("Lesion")
        self.lesion_zoom_btn.setObjectName("Segment")
        self.lesion_zoom_btn.setCheckable(True)
        self.lesion_zoom_btn.setToolTip("Frame a fixed field of view around the target in all views  (Z)")
        self.lesion_zoom_btn.clicked.connect(lambda checked: self.set_lesion_focus(bool(checked)))
        layout.addWidget(self.lesion_zoom_btn)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setObjectName("Segment")
        self.fit_btn.setToolTip("Fit the whole image in every view  (R)")
        self.fit_btn.clicked.connect(lambda _checked=False: self.reset_views())
        layout.addWidget(self.fit_btn)
        self.contrast_btn = QPushButton("Contrast")
        self.contrast_btn.setObjectName("Segment")
        self.contrast_btn.setToolTip(
            "Window and level for this sequence  (Ctrl+L)\n"
            "In a view: right-drag, or Alt + left-drag"
        )
        self.contrast_btn.clicked.connect(lambda _checked=False: self.open_contrast_dialog())
        layout.addWidget(self.contrast_btn)
        layout.addWidget(self._toolbar_divider())

        # The overlay toggles are set once in a while, so they live in the menu
        # rather than spending toolbar width the case title and the reading
        # controls need. They stay checkboxes, just inside a menu.
        self.target_crosshair_cb = QCheckBox("Target")
        self.target_crosshair_cb.setChecked(True)
        self.target_crosshair_cb.setToolTip("Show the finding crosshair and the other findings of this case")
        self.target_crosshair_cb.toggled.connect(self._toggle_crosshair)
        self.mouse_crosshair_cb = QCheckBox("Cursor")
        self.mouse_crosshair_cb.setChecked(True)
        self.mouse_crosshair_cb.setToolTip("Show a crosshair that follows the mouse")
        self.mouse_crosshair_cb.toggled.connect(self._toggle_crosshair)
        self.direction_cb = QCheckBox("Labels")
        self.direction_cb.setChecked(True)
        self.direction_cb.setToolTip("Show the anatomical direction labels")
        self.direction_cb.toggled.connect(self._toggle_crosshair)

        # Export, Dataset, Settings and Rescan are used once in a while, not
        # once per finding, and four labelled buttons cost more toolbar width
        # than the reading controls can spare. One menu keeps them named.
        self.more_btn = QPushButton("More")
        self.more_btn.setObjectName("IconButton")
        self.more_btn.setToolTip("Export, dataset, preferences and rescanning")
        menu = QMenu(self.more_btn)
        overlays = menu.addMenu("Overlays")
        # Named here, keyed in the same table as everything else, and printed
        # from that table -- so a rebound key shows up in the menu too.  The
        # text carries the key rather than setShortcut(): these actions are
        # already bound as window shortcuts, and Qt would call that ambiguous.
        self.overlay_actions: list[tuple[str, Any, str]] = []
        for checkbox, name, action_key in (
            (self.target_crosshair_cb, "Finding crosshair", "overlay_target"),
            (self.mouse_crosshair_cb, "Mouse crosshair", "overlay_mouse"),
            (self.direction_cb, "Direction labels", "overlay_labels"),
        ):
            action = overlays.addAction(name)
            action.setCheckable(True)
            action.setChecked(checkbox.isChecked())
            action.toggled.connect(checkbox.setChecked)
            checkbox.toggled.connect(action.setChecked)
            self.overlay_actions.append((action_key, action, name))
        menu.addSeparator()
        self.export_action = menu.addAction("Export results…")
        self.export_action.triggered.connect(lambda _checked=False: self.export_reviews())
        self.dataset_action = menu.addAction("Dataset…")
        self.dataset_action.triggered.connect(lambda _checked=False: self.change_dataset())
        menu.addSeparator()
        self.settings_action = menu.addAction("Preferences…")
        self.settings_action.triggered.connect(lambda _checked=False: self.open_settings())
        self.refresh_action = menu.addAction("Rescan MRI files")
        self.refresh_action.triggered.connect(lambda _checked=False: self.refresh_inventory())
        self.more_btn.setMenu(menu)
        layout.addWidget(self.more_btn)
        return card

    def _make_coordinate_spin(self, value: float = 0.0) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1000.0, 1000.0)
        # Full workbook precision: the explicit width bounds below are what
        # keep three of these fitting a narrow panel, not the decimal count,
        # so there is no reason to quantise the stored coordinate.
        spin.setDecimals(5)
        spin.setSingleStep(0.5)
        spin.setValue(float(value))
        spin.setKeyboardTracking(False)
        spin.setMinimumWidth(64)
        spin.setMaximumWidth(120)
        return spin

    def _build_view_area(self) -> QWidget:
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(7)
        self.view_panels = {plane: ViewPanel(plane) for plane in PLANE_ORDER}
        for plane, panel in self.view_panels.items():
            panel.sliceRequested.connect(self._on_slice_request)
            panel.sliceStepRequested.connect(self._on_slice_step)
            panel.maximizeRequested.connect(self.toggle_maximized_view)
            panel.canvas.targetClicked.connect(self._on_canvas_target_clicked)
            panel.canvas.mouseVoxelMoved.connect(self._on_mouse_voxel_moved)
            panel.canvas.windowChanged.connect(self._on_window_changed)
            panel.canvas.pickCleared.connect(self.clear_picked_position)
            panel.canvas.roiPainted.connect(self._on_roi_painted)
            panel.canvas.roiStrokeStarted.connect(self._on_roi_stroke_started)
            panel.canvas.roiChanging.connect(self._record_roi_change)
        # Axial top-left, sagittal top-right, coronal bottom-right, and the
        # panel that decides where the finding is bottom-left: during a read
        # the eye goes between the image and that decision, so it sits with
        # the images rather than in a column of its own.
        self.location_panel = self._build_location_panel()
        grid.addWidget(self.view_panels["axial"], 0, 0)
        grid.addWidget(self.view_panels["sagittal"], 0, 1)
        grid.addWidget(self.location_panel, 1, 0)
        grid.addWidget(self.view_panels["coronal"], 1, 1)
        # Hiding a widget does not release its row or column: the stretch
        # factor keeps the space reserved, so maximising also has to move the
        # stretch onto the cell that stays visible.
        self.view_grid = grid
        self.view_cells = {"axial": (0, 0), "sagittal": (0, 1), "coronal": (1, 1)}
        self._set_grid_stretch(None)
        return container

    def _set_grid_stretch(self, plane: str | None) -> None:
        row, column = self.view_cells.get(plane or "", (None, None))
        for index in (0, 1):
            self.view_grid.setRowStretch(index, 1 if row is None or index == row else 0)
            self.view_grid.setColumnStretch(index, 1 if column is None or index == column else 0)

    def _build_location_panel(self) -> QWidget:
        """Where the finding is, and what this reader makes of it.

        This is the bottom-left cell of the image grid rather than a column of
        its own: during a read the eye travels between the picture and this
        decision, and the far side of the window is the worst place to put it.
        The reference material went to the right column instead -- that is
        what a reader consults, not what they answer with -- and the list of
        findings went with it, on top of the panel that describes them.
        """

        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        header = QWidget()
        layout = QVBoxLayout(header)
        layout.setContentsMargins(8, 5, 8, 3)
        layout.setSpacing(5)

        # One line above the tabs: the source context on the left, and on the
        # right this finding's verdict and whether it is saved.  It reports on
        # the decision being made in this card, so it stays with the buttons
        # that make it rather than following the list into the right column --
        # pressing Y while drawing has to be visible, an unsaved verdict must
        # not hide behind a tab, and neither has room in a 400px column.
        # Elided rather than wrapping: a wrapping label grew the header by up
        # to two rows depending on how much the source sheet had to say, and
        # that came straight out of the tab below.  The full text is in This
        # finding, and in this label's tooltip.
        self.source_summary = ElidedLabel("—")
        self.source_summary.setStyleSheet(f"color:{COLORS['dim']}; font-size:8pt;")
        self.source_summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        state_row = QHBoxLayout()
        state_row.setSpacing(6)
        state_row.addWidget(self.source_summary, 1)
        self.verdict_keys_label = _label("", color=COLORS["faint"], size=8, wrap=False)
        state_row.addWidget(self.verdict_keys_label)
        self.verdict_summary = _label("Not set", color=COLORS["dim"], size=8, wrap=False)
        self.verdict_summary.setToolTip("This finding's verdict, from any tab")
        state_row.addWidget(self.verdict_summary)
        self.dirty_label = _label("", color=COLORS["warn"], size=8, wrap=False)
        state_row.addWidget(self.dirty_label)
        layout.addLayout(state_row)

        card_layout.addWidget(header)

        self.panel_tabs = QTabWidget()
        self.panel_tabs.setDocumentMode(True)
        self.panel_pages: dict[str, QScrollArea] = {}
        self._panel_tab_keys: list[str] = []
        card_layout.addWidget(self.panel_tabs, 1)
        card.setMinimumWidth(LOCATION_PANEL_MIN_WIDTH)
        layout = self._add_panel_tab(
            "review", "Review", "Where this finding is and what you make of it"
        )

        # Which recorded position of this finding is being shown. Saving
        # records the position that is selected here, so adopting another
        # reader's correction is simply a matter of looking at it.
        position_row = QHBoxLayout()
        position_row.setSpacing(6)
        position_row.addWidget(_section_title("Position"))
        self.position_combo = QComboBox()
        self.position_combo.setToolTip(
            "Which recorded position of this finding to show.\n"
            "Saving stores the position shown here as yours."
        )
        self.position_combo.currentIndexChanged.connect(self._on_variant_selected)
        position_row.addWidget(self.position_combo, 1)
        self.position_hint = _label("", color=COLORS["warn"], size=8, wrap=False)
        self.position_hint.setToolTip("What saving this review would do to the position")
        position_row.addWidget(self.position_hint)
        self.move_here_btn = QPushButton("Move")
        self.move_here_btn.setToolTip(
            "Record the crosshair position as your correction of this finding.\n"
            "Click the true location in a view first."
        )
        self.move_here_btn.clicked.connect(lambda _checked=False: self.move_finding_here())
        position_row.addWidget(self.move_here_btn)
        layout.addLayout(position_row)
        layout.addWidget(_separator())

        # The shortcut is part of the label: it is the fastest way to teach the
        # keyboard flow, and it avoids symbol glyphs that depend on font
        # fallback being available on the workstation.
        self.verdict_segments = SegmentedControl(
            [("yes", "Yes"), ("no", "No"), ("unset", "Not set")],
            tones={"yes": "yes", "no": "no", "unset": "neutral"},
        )
        for key, hint in (
            ("yes", "This is a microbleed"),
            ("no", "This is not a microbleed"),
            ("unset", "No decision recorded"),
        ):
            button = self.verdict_segments.button(key)
            if button is not None:
                button.setToolTip(hint)
        self.verdict_segments.selected.connect(self._on_verdict_selected)
        layout.addWidget(self.verdict_segments)

        # Yes/no alone cannot be tabulated: an adjudication needs to know how
        # sure each reader was, and when they said no, what they thought it was
        # instead.  Both are optional, and both are one dropdown rather than a
        # sentence somebody has to read.
        # The panel is narrow, so the two dropdowns carry their own prompt as
        # their empty entry rather than a label beside them, and neither is
        # allowed to demand the width of its longest option.
        detail_row = QHBoxLayout()
        detail_row.setSpacing(5)
        self.certainty_combo = QComboBox()
        self.certainty_combo.addItem("How sure?", "")
        for choice in CERTAINTY_CHOICES:
            self.certainty_combo.addItem(choice.capitalize(), choice)
        self.certainty_combo.setToolTip("How confident this verdict is (optional)")
        self.certainty_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.certainty_combo.setMinimumContentsLength(6)
        self.certainty_combo.currentIndexChanged.connect(
            lambda _index: self._mark_review_dirty()
        )
        detail_row.addWidget(self.certainty_combo, 1)
        self.mimic_combo = QComboBox()
        self.mimic_combo.addItem("If not, what?", "")
        for choice in MIMIC_CHOICES:
            self.mimic_combo.addItem(choice.capitalize(), choice)
        self.mimic_combo.setToolTip(
            "What it is instead, when this is not a microbleed (optional).\n"
            "These are the mimics the rating scales ask a reader to exclude,\n"
            "so a 'no' becomes something the analysis can count."
        )
        self.mimic_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.mimic_combo.setMinimumContentsLength(8)
        self.mimic_combo.currentIndexChanged.connect(lambda _index: self._mark_review_dirty())
        detail_row.addWidget(self.mimic_combo, 1)
        layout.addLayout(detail_row)

        self.comment_edit = QTextEdit()
        self.comment_edit.setPlaceholderText("Comment (optional)…")
        self.comment_edit.setMinimumHeight(40)
        self.comment_edit.setMaximumHeight(50)
        self.comment_edit.textChanged.connect(self._mark_review_dirty)
        layout.addWidget(self.comment_edit)

        # Three destinations, because after a verdict there are exactly three
        # things a reader wants: stay and look again, draw this one, or move
        # on.  Labels are short because the panel is 409px wide at its
        # narrowest and all three have to fit on one row.
        save_row = QHBoxLayout()
        save_row.setSpacing(5)
        self.save_review_btn = QPushButton("Save")
        self.save_review_btn.setToolTip("Save this review and stay on the finding")
        self.save_review_btn.clicked.connect(lambda _checked=False: self.save_current_review(advance=False))
        # Equal thirds.  "+ Next" had every spare pixel of the row, which
        # left the other two reading as afterthoughts and cost "+ Segment" the
        # width for its own name.
        self.save_review_btn.setObjectName("SaveButton")
        save_row.addWidget(self.save_review_btn, 1)
        self.save_segment_btn = QPushButton("+ Segment")
        self.save_segment_btn.setToolTip(
            "Save, then open the Segment tab on this same finding.\n"
            "The order of work is decide, draw, move on."
        )
        self.save_segment_btn.clicked.connect(lambda _checked=False: self.save_and_segment())
        self.save_segment_btn.setObjectName("SaveButton")
        save_row.addWidget(self.save_segment_btn, 1)
        self.save_next_btn = QPushButton("+ Next  ›")
        self.save_next_btn.setObjectName("PrimaryButton")
        self.save_next_btn.setToolTip("Save and move to the next finding, or the next case  (Ctrl+S)")
        self.save_next_btn.clicked.connect(lambda _checked=False: self.save_current_review(advance=True))
        save_row.addWidget(self.save_next_btn, 1)
        layout.addLayout(save_row)
        layout.addStretch(1)

        self._build_segment_tab()

        return card

    def _add_panel_tab(self, key: str, title: str, tooltip: str) -> QVBoxLayout:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        body = QWidget()
        box = QVBoxLayout(body)
        box.setContentsMargins(6, 6, 6, 6)
        box.setSpacing(5)
        scroll.setWidget(body)
        index = self.panel_tabs.addTab(scroll, title)
        self.panel_tabs.setTabToolTip(index, tooltip)
        self.panel_pages[key] = scroll
        self._panel_tab_keys.append(key)
        return box

    def _build_segment_tab(self) -> None:
        """The mask for the selected finding, beside the verdict.

        A column of its own gave it room but put Generate two panels away from
        the button that says "save and start drawing" -- so having pressed
        that, a reader had nowhere obvious to go.  Back here it is one tab
        from the verdict, and the finding list above the tabs never moves.
        """

        layout = self._add_panel_tab(
            "segment", "Segment", "Draw or grow this finding's mask"
        )

        # Visible only when something is in the way, and it says which thing.
        # Disabled buttons with no reason beside them are the worst of both.
        self.segment_block_label = _label("", color=COLORS["warn"], size=8)
        self.segment_block_label.setWordWrap(True)
        self.segment_block_label.setVisible(False)
        layout.addWidget(self.segment_block_label)

        roi_top = QHBoxLayout()
        roi_top.setSpacing(6)
        self.roi_label = ElidedLabel("no segmentation")
        self.roi_label.setStyleSheet(f"color:{COLORS['roi']}; font-size:9pt;")
        self.roi_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        roi_top.addWidget(self.roi_label, 1)
        self.outline_roi_cb = QCheckBox("Outline")
        self.outline_roi_cb.setChecked(self.settings.roi_outline)
        self.outline_roi_cb.setToolTip(
            "Draw the segmentation as an edge instead of a filled wash.\n"
            "At lesion zoom a fill covers the signal that decides whether\n"
            "the mask is right."
        )
        self.outline_roi_cb.toggled.connect(self._set_roi_outline)
        roi_top.addWidget(self.outline_roi_cb)
        self.show_roi_cb = QCheckBox("Show")
        self.show_roi_cb.setChecked(True)
        self.show_roi_cb.setToolTip("Show the segmentation overlay in the views")
        self.show_roi_cb.toggled.connect(self._toggle_roi_overlay)
        roi_top.addWidget(self.show_roi_cb)
        layout.addLayout(roi_top)

        auto_row = QHBoxLayout()
        auto_row.setSpacing(5)
        self.auto_roi_btn = QPushButton("Generate")
        self.auto_roi_btn.setToolTip(
            "Grow the mask from the finding, using the sequence on screen.\n"
            "SWI and QSM can disagree; whichever you generate from is recorded."
        )
        self.auto_roi_btn.clicked.connect(lambda _checked=False: self.auto_segment())
        auto_row.addWidget(self.auto_roi_btn)
        self.grow_stroke_btn = QPushButton("Grow stroke")
        self.grow_stroke_btn.setToolTip(
            "Use what you have already painted as the seed and expand it to the\n"
            "lesion boundary. Better than a single point for irregular lesions."
        )
        self.grow_stroke_btn.clicked.connect(lambda _checked=False: self.grow_from_stroke())
        auto_row.addWidget(self.grow_stroke_btn)
        layout.addLayout(auto_row)

        # A grid, not two rows of boxes: the spin boxes are capped in width, so
        # a stretch factor could only put the slack *between* them -- which is
        # where the empty middle of these two rows came from.  Here the slack
        # is one spare column on the right, and the two rows line up.
        fields = QGridLayout()
        fields.setHorizontalSpacing(6)
        fields.setVerticalSpacing(5)
        fields.setColumnStretch(5, 1)
        fields.setColumnMinimumWidth(2, 14)
        fields.addWidget(_label("sens.", color=COLORS["faint"], size=8, wrap=False), 0, 0)
        self.sensitivity_spin = QDoubleSpinBox()
        self.sensitivity_spin.setRange(0.5, 8.0)
        self.sensitivity_spin.setSingleStep(0.25)
        self.sensitivity_spin.setDecimals(2)
        # 3.0 measured over thirty real findings: median mask diameter 3.6 mm,
        # which is where a microbleed on this data sits, with the fewest masks
        # that either overrun the cap or collapse to the seed.
        self.sensitivity_spin.setValue(DEFAULT_GROW_SENSITIVITY)
        self.sensitivity_spin.setToolTip(
            "How far past the local background a voxel must be to join the mask.\n"
            "Lower includes more; raise it if the mask runs into nearby tissue."
        )
        self.sensitivity_spin.setMinimumWidth(58)
        self.sensitivity_spin.setMaximumWidth(74)
        fields.addWidget(self.sensitivity_spin, 0, 1)
        # "max r", not "max ø": the value is passed to the grower as a radius,
        # so a diameter label would quietly let the mask reach twice as far as
        # the reader asked for.
        fields.addWidget(_label("max r", color=COLORS["faint"], size=8, wrap=False), 0, 3)
        self.roi_radius_spin = QDoubleSpinBox()
        self.roi_radius_spin.setRange(1.0, 25.0)
        self.roi_radius_spin.setSingleStep(1.0)
        self.roi_radius_spin.setDecimals(1)
        self.roi_radius_spin.setSuffix(" mm")
        self.roi_radius_spin.setValue(6.0)
        self.roi_radius_spin.setToolTip("Growth cannot leave this radius around the finding")
        self.roi_radius_spin.setMinimumWidth(62)
        self.roi_radius_spin.setMaximumWidth(84)
        fields.addWidget(self.roi_radius_spin, 0, 4)

        fields.addWidget(_label("brush", color=COLORS["faint"], size=8, wrap=False), 1, 0)
        self.brush_spin = QDoubleSpinBox()
        self.brush_spin.setRange(0.3, 10.0)
        self.brush_spin.setSingleStep(0.5)
        self.brush_spin.setDecimals(1)
        self.brush_spin.setSuffix(" mm")
        self.brush_spin.setValue(1.5)
        self.brush_spin.setToolTip(
            "Brush radius · the left button paints, the Eraser tool erases\n"
            "Smaller / larger without leaving the image: the shortcut keys in Preferences"
        )
        self.brush_spin.setMinimumWidth(62)
        self.brush_spin.setMaximumWidth(84)
        self.brush_spin.valueChanged.connect(self._set_brush_radius)
        fields.addWidget(self.brush_spin, 1, 1)
        self.brush_3d_cb = QCheckBox("3D")
        self.brush_3d_cb.setChecked(True)
        self.brush_3d_cb.setToolTip(
            "Paint a ball through the neighbouring slices instead of a disc on\n"
            "this one. A microbleed is a few voxels across in every direction,\n"
            "so a flat brush makes one lesion five separate strokes.\n"
            "Turn it off to correct a single slice."
        )
        self.brush_3d_cb.toggled.connect(self._set_brush_3d)
        fields.addWidget(self.brush_3d_cb, 1, 3, 1, 2)
        layout.addLayout(fields)

        edit_row = QHBoxLayout()
        edit_row.setSpacing(5)
        self.undo_roi_btn = QPushButton("Undo")
        self.undo_roi_btn.setToolTip("Undo the last brush stroke  (Ctrl+Z)")
        self.undo_roi_btn.clicked.connect(lambda _checked=False: self.undo_roi())
        edit_row.addWidget(self.undo_roi_btn, 1)
        self.clear_roi_btn = QPushButton("Clear")
        self.clear_roi_btn.setToolTip("Remove this finding's segmentation")
        self.clear_roi_btn.clicked.connect(lambda _checked=False: self.clear_roi())
        edit_row.addWidget(self.clear_roi_btn, 1)
        self.lesion_3d_btn = QPushButton("3D")
        self.lesion_3d_btn.setObjectName("SaveButton")
        self.lesion_3d_btn.setToolTip(
            "Turn this mask over in a window of its own.\n"
            "Whether the grower slipped down a vessel, or a stroke is one\n"
            "slice thick, is one drag rather than a scroll through the stack."
        )
        self.lesion_3d_btn.clicked.connect(lambda _checked=False: self.open_lesion_3d())
        edit_row.addWidget(self.lesion_3d_btn, 1)
        layout.addLayout(edit_row)
        segment_save_row = QHBoxLayout()
        segment_save_row.setSpacing(5)
        self.segment_save_btn = QPushButton("Save")
        self.segment_save_btn.setToolTip("Save the review and this mask, and stay here")
        self.segment_save_btn.clicked.connect(
            lambda _checked=False: self.save_current_review(advance=False)
        )
        self.segment_save_btn.setObjectName("SaveButton")
        segment_save_row.addWidget(self.segment_save_btn, 1)
        # The same destination as on the Review tab, so finishing a mask does
        # not mean going back a tab to find the button that moves on.
        self.segment_next_btn = QPushButton("+ Next  ›")
        self.segment_next_btn.setObjectName("PrimaryButton")
        self.segment_next_btn.setToolTip(
            "Save and move to the next finding, or the next case  (Ctrl+S)"
        )
        self.segment_next_btn.clicked.connect(
            lambda _checked=False: self.save_current_review(advance=True)
        )
        segment_save_row.addWidget(self.segment_next_btn, 1)
        layout.addLayout(segment_save_row)
        layout.addStretch(1)

    def _build_shortcut_section(self) -> QWidget:
        """The key legend, folded away under the reference column.

        It sat above the case queue, where it was neither reference nor
        navigation and took height from the list.  Here it is next to the
        other thing a reader consults rather than answers with.
        """

        self.shortcut_section = CollapsibleSection(
            "Shortcuts",
            expanded=self.settings.section_expanded("shortcuts", False),
        )
        self.shortcut_section.toggled.connect(
            lambda expanded: self.settings.set_section_expanded("shortcuts", expanded)
        )
        # Rendered from the binding table, so a rebound key updates the legend.
        self.shortcut_legend = _label("", color=COLORS["dim"], size=8)
        self.shortcut_legend.setWordWrap(True)
        self.shortcut_legend.setFont(QFont("Consolas", 8))
        self.shortcut_section.add_widget(self.shortcut_legend)
        edit_shortcuts = QPushButton("Edit…")
        edit_shortcuts.setObjectName("IconButton")
        edit_shortcuts.clicked.connect(lambda _checked=False: self.open_settings(tab=2))
        self.shortcut_section.add_widget(edit_shortcuts)
        return self.shortcut_section

    def _build_findings_panel(self) -> QWidget:
        """Which finding, at the top of the column that describes it.

        It used to sit above the tabs in the middle card.  The panel
        directly under it now, This finding, is the long form of the row you
        just picked, so the selector belongs on top of it.

        A row still fits on one line here, and stays on one line whenever it
        fits -- see _fit_finding_rows.  What made that possible in a 330px
        column is the shorter wording: 238px at the median across the 434
        findings in this data and 270 at the worst, against 333 and 371 for
        the long form.
        """

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(9, 6, 9, 4)
        layout.setSpacing(5)

        heading = QHBoxLayout()
        heading.setSpacing(6)
        heading.addWidget(_section_title("Findings"))
        heading.addStretch(1)
        self.target_count_label = _label("No case loaded", color=COLORS["dim"], size=8, wrap=False)
        heading.addWidget(self.target_count_label)
        self.prev_finding_btn = QPushButton("‹")
        self.prev_finding_btn.setObjectName("IconButton")
        self.prev_finding_btn.setToolTip("Previous finding  ([)")
        self.prev_finding_btn.clicked.connect(lambda _checked=False: self.step_finding(-1))
        self.next_finding_btn = QPushButton("›")
        self.next_finding_btn.setObjectName("IconButton")
        self.next_finding_btn.setToolTip("Next finding  (])")
        self.next_finding_btn.clicked.connect(lambda _checked=False: self.step_finding(1))
        heading.addWidget(self.prev_finding_btn)
        heading.addWidget(self.next_finding_btn)
        layout.addLayout(heading)

        self.target_list = QListWidget()
        self.target_list.setMinimumHeight(46)
        # Taller than it was: the list no longer takes this height from the
        # images, only from the report below it, and a two-line row costs
        # twice the height of the one it replaced.
        self.target_list.setMaximumHeight(FINDING_LIST_MAX_HEIGHT)
        self.target_list.currentRowChanged.connect(self._on_target_row_changed)
        self.target_list.itemClicked.connect(self._on_target_item_clicked)
        self.target_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.target_list.customContextMenuRequested.connect(self._show_finding_menu)
        self.target_list.installEventFilter(self)
        layout.addWidget(self.target_list)

        self.findings_panel = panel
        return panel

    def _build_details_panel(self) -> QWidget:
        """What is known about the selected finding, plus the rare tools.

        Not a third tab: adjudication means weighing what another reader wrote
        *while* deciding, and a tab would put the two on opposite sides of a
        click.  It sits above the case queue instead, which had 471px at the
        smallest window and 791px at 1920x1080 -- thirty-one rows of a list
        the reader walks with Page Up and Page Down.
        """

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(9, 2, 9, 6)
        layout.setSpacing(6)

        layout.addWidget(_section_title("This finding"))
        reports_tabs = QTabWidget()
        reports_tabs.setDocumentMode(True)
        self.source_report = QTextBrowser()
        self.source_report.setOpenExternalLinks(False)
        self.source_report.setMinimumHeight(72)
        reports_tabs.addTab(self.source_report, "Source")
        self.reports_browser = QTextBrowser()
        self.reports_browser.setMinimumHeight(72)
        reports_tabs.addTab(self.reports_browser, "All readers")
        layout.addWidget(reports_tabs, 1)
        layout.addWidget(_separator())
        layout.addWidget(self._build_jump_tool())
        layout.addWidget(self._build_manual_tool())
        scroll.setWidget(body)
        self.details_panel = scroll
        return scroll

    def _build_jump_tool(self) -> QWidget:
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(_label("Jump to RAS coordinate", color=COLORS["dim"], size=9, bold=True))
        spins = QHBoxLayout()
        spins.setSpacing(5)
        self.ras_l_spin = self._make_coordinate_spin()
        self.ras_p_spin = self._make_coordinate_spin()
        self.ras_s_spin = self._make_coordinate_spin()
        for label_text, spin in (
            ("L-R (L)", self.ras_l_spin),
            ("P-A (A)", self.ras_p_spin),
            ("I-S (S)", self.ras_s_spin),
        ):
            column = QVBoxLayout()
            column.setSpacing(2)
            column.addWidget(_label(label_text, color=COLORS["faint"], size=8, wrap=False))
            column.addWidget(spin)
            spins.addLayout(column, 1)
        layout.addLayout(spins)
        buttons = QHBoxLayout()
        buttons.setSpacing(5)
        self.jump_btn = QPushButton("Jump")
        self.jump_btn.setToolTip("Move the views to the coordinate above")
        self.jump_btn.clicked.connect(lambda _checked=False: self.jump_to_ras())
        # Equal halves: Return had every spare pixel and Jump had none, which
        # read as though going back were the main thing you do here.
        buttons.addWidget(self.jump_btn, 1)
        self.return_btn = QPushButton("Return")
        self.return_btn.setToolTip("Bring the views back to the selected finding")
        self.return_btn.setEnabled(False)
        self.return_btn.clicked.connect(lambda _checked=False: self.return_to_finding())
        buttons.addWidget(self.return_btn, 1)
        layout.addLayout(buttons)
        return group

    def _build_manual_tool(self) -> QWidget:
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(
            _label("Add a microbleed that is not in the list", color=COLORS["dim"], size=9, bold=True)
        )
        layout.addWidget(
            _label(
                "Uses the coordinate the crosshair is on. To correct where an "
                "existing finding sits, use Position above instead — that keeps "
                "it linked to the source finding.",
                color=COLORS["faint"],
                size=8,
            )
        )
        spins = QHBoxLayout()
        spins.setSpacing(5)
        self.manual_l_spin = self._make_coordinate_spin()
        self.manual_p_spin = self._make_coordinate_spin()
        self.manual_s_spin = self._make_coordinate_spin()
        for label_text, spin in (
            ("L-R (L)", self.manual_l_spin),
            ("P-A (A)", self.manual_p_spin),
            ("I-S (S)", self.manual_s_spin),
        ):
            column = QVBoxLayout()
            column.setSpacing(2)
            column.addWidget(_label(label_text, color=COLORS["faint"], size=8, wrap=False))
            column.addWidget(spin)
            spins.addLayout(column, 1)
        layout.addLayout(spins)
        self.manual_region_edit = QLineEdit()
        self.manual_region_edit.setPlaceholderText("Region (optional)")
        layout.addWidget(self.manual_region_edit)
        self.manual_note_edit = QTextEdit()
        self.manual_note_edit.setPlaceholderText("Note (optional)")
        self.manual_note_edit.setMaximumHeight(50)
        layout.addWidget(self.manual_note_edit)
        manual_row = QHBoxLayout()
        manual_row.setSpacing(5)
        self.add_manual_btn = QPushButton("Add here")
        self.add_manual_btn.setToolTip("Record a new finding at the crosshair")
        self.add_manual_btn.clicked.connect(lambda _checked=False: self.add_manual_microbleed())
        manual_row.addWidget(self.add_manual_btn, 1)
        # Removing is the counterpart of adding, so it belongs beside it.  It
        # only ever applies to the selected finding, and only when that finding
        # is one this reader added.
        self.remove_manual_btn = QPushButton("Remove")
        self.remove_manual_btn.setToolTip("Remove the selected finding, if you added it")
        self.remove_manual_btn.clicked.connect(lambda _checked=False: self.remove_manual_microbleed())
        manual_row.addWidget(self.remove_manual_btn, 1)
        layout.addLayout(manual_row)
        return group

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Card")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(7, 8, 7, 8)
        layout.setSpacing(6)

        session_row = QHBoxLayout()
        session_row.setSpacing(6)
        # Elided, not plain: the queue column is a fixed 300px, so a reader
        # name of any length used to be cut off mid-word with no ellipsis to
        # say so.  "Desktop Test Reader · round 1" wanted 264px of the 185
        # left beside the Switch button.
        self.session_label = ElidedLabel(f"{self.reader_id} · round {self.review_round}")
        self.session_label.setStyleSheet(
            f"color:{COLORS['accent']}; font-size:9pt; font-weight:600;"
        )
        self.session_label.setToolTip(
            f"{self.reader_id} · round {self.review_round}\n"
            "Reader reports are shared with every reader of this datasheet"
        )
        session_row.addWidget(self.session_label, 1)
        # Switching, not quitting: reading a second round, or handing the
        # workstation to the other reader, used to mean closing the
        # application and starting it again.
        self.switch_session_btn = QPushButton("Switch…")
        self.switch_session_btn.setObjectName("IconButton")
        self.switch_session_btn.setToolTip("Change reader or review round without restarting")
        self.switch_session_btn.clicked.connect(lambda _checked=False: self.switch_session())
        session_row.addWidget(self.switch_session_btn)
        self.collapse_queue_btn = QPushButton()
        self.collapse_queue_btn.setObjectName("IconButton")
        self.collapse_queue_btn.setIcon(_stroke_icon("collapse-left", 16))
        self.collapse_queue_btn.setIconSize(QSize(16, 16))
        self.collapse_queue_btn.setToolTip(
            "Fold the case queue away and give the width to the images.\n"
            "The strip that is left expands when the mouse is over it."
        )
        self.collapse_queue_btn.clicked.connect(lambda _checked=False: self.set_queue_pinned(False))
        session_row.addWidget(self.collapse_queue_btn)
        layout.addLayout(session_row)

        # Filters are set once per sitting and the search box does not need a
        # row to itself, so the two share one: a dropdown of checkable entries
        # beside the box.  A fold-out panel cost a row when shut and five when
        # open, out of a queue that has the height left over.
        self.filter_menu = QMenu(self)
        # Short labels with the full sentence in the tooltip: a QCheckBox does
        # not wrap, so its label sets the sidebar's minimum width, and a wide
        # sidebar takes the space away from the images.
        self.hide_missing_cb = QCheckBox("Hide cases without MRI")
        self.hide_missing_cb.setToolTip("Hide cases whose required AffineRestored files are missing")
        self.hide_missing_cb.setChecked(True)
        self.complete_only_cb = QCheckBox("Only complete cases")
        self.complete_only_cb.setToolTip("Show only cases that have all three sequences")
        self.unverified_only_cb = QCheckBox("Not yet reviewed by me")
        self.unverified_only_cb.setToolTip("Show cases that still have findings without a verification or comment from me")
        self.verified_only_cb = QCheckBox("Fully reviewed by me")
        self.verified_only_cb.setToolTip("Show only cases where I have decided on every finding")
        self.source_unverified_cb = QCheckBox("Source-unverified")
        self.source_unverified_cb.setToolTip("Show cases that contain findings the source workbook did not verify")
        self.adjudication_cb = QCheckBox("Has adjudication notes")
        self.adjudication_cb.setToolTip("Show cases whose source findings carry an adjudication note")
        # The note above is the source sheet's opinion.  This one is about the
        # work in this database: where the readers have actually disagreed, so
        # a second round has a queue to work from.
        self.disagreement_cb = QCheckBox("Readers disagree")
        self.disagreement_cb.setToolTip(
            "Show cases with a finding two readers decided differently.\n"
            "This is the adjudication queue for the reviews in this database,\n"
            "as opposed to the notes that came with the source sheet."
        )
        self.filter_checkboxes = (
            self.hide_missing_cb,
            self.complete_only_cb,
            self.unverified_only_cb,
            self.verified_only_cb,
            self.source_unverified_cb,
            self.adjudication_cb,
            self.disagreement_cb,
        )
        for checkbox in self.filter_checkboxes:
            checkbox.toggled.connect(self._apply_case_filters)
            checkbox.toggled.connect(lambda _checked: self._update_filter_badge())
            # The checkbox itself goes in the menu, so the filter code and the
            # saved session state keep reading the same object they always did.
            holder = QWidgetAction(self.filter_menu)
            checkbox.setContentsMargins(8, 3, 8, 3)
            holder.setDefaultWidget(checkbox)
            self.filter_menu.addAction(holder)
        self.filter_btn = QPushButton("Filters")
        self.filter_btn.setObjectName("IconButton")
        self.filter_btn.setToolTip("Narrow the case queue")
        self.filter_btn.setMenu(self.filter_menu)

        queue_layout = layout
        queue_row = QHBoxLayout()
        queue_row.setSpacing(6)
        queue_row.addWidget(_section_title("Case queue"))
        queue_row.addStretch(1)
        self.visible_label = _label("—", color=COLORS["dim"], size=8, wrap=False)
        queue_row.addWidget(self.visible_label)
        queue_layout.addLayout(queue_row)
        find_row = QHBoxLayout()
        find_row.setSpacing(5)
        self.case_search = QLineEdit()
        self.case_search.setPlaceholderText("Search case")
        self.case_search.setClearButtonEnabled(True)
        self.case_search.textChanged.connect(self._apply_case_filters)
        find_row.addWidget(self.case_search, 1)
        find_row.addWidget(self.filter_btn)
        queue_layout.addLayout(find_row)

        self.case_list = QListWidget()
        self.case_list.setMinimumWidth(240)
        self.case_list.currentRowChanged.connect(self._on_case_row_changed)
        queue_layout.addWidget(self.case_list, 1)

        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(6)
        # Short words: side by side in a 300px column the two buttons wanted
        # 180px each and had 169, so both were quietly cut off.  The list they
        # sit under already says these are cases.
        self.prev_btn = QPushButton("‹  Prev")
        self.prev_btn.setToolTip("Previous case  (Page Up)")
        self.next_btn = QPushButton("Next  ›")
        self.next_btn.setToolTip("Next case  (Page Down)")
        self.prev_btn.clicked.connect(lambda _checked=False: self.previous_case())
        self.next_btn.clicked.connect(lambda _checked=False: self.next_case())
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        layout.addLayout(nav_layout)

        return sidebar

    def _restore_session_preferences(self) -> None:
        """Restore the last reader view when the user chose Resume."""
        raw_filters = self.session.get("filters_json")
        try:
            filters = json.loads(raw_filters) if raw_filters else {}
        except (TypeError, ValueError):
            filters = {}
        controls = {
            "search": self.case_search,
            "hide_missing": self.hide_missing_cb,
            "complete_only": self.complete_only_cb,
            "unverified_only": self.unverified_only_cb,
            "verified_only": self.verified_only_cb,
            "source_unverified": self.source_unverified_cb,
            "adjudication": self.adjudication_cb,
            "disagreement": self.disagreement_cb,
        }
        for key, control in controls.items():
            if key not in filters:
                continue
            control.blockSignals(True)
            if isinstance(control, QLineEdit):
                control.setText(str(filters[key] or ""))
            else:
                control.setChecked(bool(filters[key]))
            control.blockSignals(False)
        self._update_filter_badge()
        # Push the segmentation controls' starting state into the canvases, so
        # what the panel shows is what the brush and the overlay actually do.
        self.set_queue_pinned(self.settings.queue_pinned)
        self.set_right_column_width(self.settings.right_column_width)
        self._set_brush_radius(self.brush_spin.value())
        self._set_brush_3d(self.brush_3d_cb.isChecked())
        self._set_roi_outline(self.outline_roi_cb.isChecked())
        for panel in self.view_panels.values():
            panel.canvas.set_smooth_zoom(self.settings.smooth_zoom)
        # The reading preference decides which sequence a case opens on; the
        # session's last sequence only matters when that preference is "last".
        last_modality = str(self.session.get("last_modality") or "").lower()
        if self.settings.default_modality == "last" and last_modality in MODALITY_ORDER:
            self.current_modality = last_modality
        self.modality_segments.set_current_key(self.current_modality)

    def _update_filter_badge(self) -> None:
        """Say how many filters are on without opening the menu.

        A hidden filter that silently shortens the queue is how a reader comes
        to believe a case is missing.
        """

        active = sum(1 for checkbox in self.filter_checkboxes if checkbox.isChecked())
        self.filter_btn.setText(f"Filters · {active}" if active else "Filters")
        names = [
            checkbox.text() for checkbox in self.filter_checkboxes if checkbox.isChecked()
        ]
        hint = "Narrow the case queue"
        if names:
            hint += "\n\nOn: " + ", ".join(names)
        self.filter_btn.setToolTip(hint)

    def _shortcut_callbacks(self) -> dict[str, Any]:
        # The verdict keys are the reason case navigation defaults away from
        # N/P: during a read, "no" is pressed far more often than "next case".
        callbacks: dict[str, Any] = {
            "save_review": lambda: self.save_current_review(),
            "verdict_yes": lambda: self.set_verdict(1),
            "verdict_no": lambda: self.set_verdict(0),
            "verdict_unset": lambda: self.set_verdict(None),
            "prev_finding": lambda: self.step_finding(-1),
            "next_finding": lambda: self.step_finding(1),
            "prev_case": self.previous_case,
            "next_case": self.next_case,
            "tool_point": self.toggle_point_tool,
            # Straight at the checkbox: the menu entry follows it, not the
            # other way round, so one path sets the state however it is asked
            # for.
            "overlay_target": lambda: self.target_crosshair_cb.toggle(),
            "overlay_mouse": lambda: self.mouse_crosshair_cb.toggle(),
            "overlay_labels": lambda: self.direction_cb.toggle(),
            "cancel_pick": self.clear_picked_position,
            "tool_brush": lambda: self.set_tool(None if self.active_tool == "brush" else "brush"),
            "tool_eraser": lambda: self.set_tool(None if self.active_tool == "eraser" else "eraser"),
            "toggle_roi_overlay": lambda: self.show_roi_cb.toggle(),
            "brush_smaller": lambda: self.step_brush_radius(-self.brush_spin.singleStep()),
            "brush_larger": lambda: self.step_brush_radius(self.brush_spin.singleStep()),
            "undo_roi": self.undo_roi,
            "lesion_zoom": self.toggle_lesion_focus,
            "contrast_dialog": self.open_contrast_dialog,
            "reset_contrast": self.reset_window_level,
            "fit_views": self.reset_views,
            "maximize_view": lambda: self.toggle_maximized_view(None),
            "refresh_files": self.refresh_inventory,
        }
        for position, modality in enumerate(MODALITY_BUTTON_ORDER, start=1):
            callbacks[f"sequence_{position}"] = lambda name=modality: self.set_modality(name)
        return callbacks

    def _bind_shortcuts(self) -> None:
        """(Re)create every shortcut from the reader's key bindings."""

        for shortcut in getattr(self, "_shortcuts", {}).values():
            shortcut.setEnabled(False)
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._shortcuts: dict[str, QShortcut] = {}
        callbacks = self._shortcut_callbacks()
        for action, sequence in self.settings.shortcuts().items():
            callback = callbacks.get(action)
            if callback is None or not sequence:
                continue
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self._shortcuts[action] = shortcut
        self._refresh_shortcut_hints()

    def _shortcut_text(self, action: str) -> str:
        """The binding for an action as the reader should see it."""

        sequence = self.settings.shortcut(action)
        if not sequence:
            return "unbound"
        return QKeySequence(sequence).toString(QKeySequence.SequenceFormat.NativeText)

    # ``peek_versions`` is deliberately absent from the callbacks above: it is
    # a hold-to-preview key, so it must reach keyPressEvent/keyReleaseEvent
    # instead of being swallowed by a QShortcut that only fires on press.
    def _peek_combination(self) -> tuple[int, int] | None:
        sequence = QKeySequence(self.settings.shortcut("peek_versions"))
        if sequence.isEmpty():
            return None
        combination = sequence[0]
        try:
            return int(combination.key().value), int(combination.keyboardModifiers().value)
        except AttributeError:  # pragma: no cover - older binding shapes
            raw = int(combination)
            return raw & 0x01FFFFFF, raw & ~0x01FFFFFF

    def _is_peek_key(self, event) -> bool:
        combination = self._peek_combination()
        if combination is None:
            return False
        key, modifiers = combination
        if int(event.key()) != key:
            return False
        # Modifiers held for other reasons must not block the preview.
        return not modifiers or int(event.modifiers().value) & modifiers == modifiers

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if not event.isAutoRepeat() and self._is_peek_key(event):
            self.set_variant_peek(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if not event.isAutoRepeat() and self._is_peek_key(event):
            self.set_variant_peek(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt API
        # Alt-tabbing away while holding the key would never deliver a release.
        if event.type() == QEvent.Type.WindowDeactivate:
            self.set_variant_peek(False)
        super().changeEvent(event)

    def _refresh_shortcut_hints(self) -> None:
        """Keep tooltips and the sidebar legend showing the live bindings."""

        bindings = self.settings.shortcuts()
        key = self._shortcut_text

        if hasattr(self, "shortcut_legend"):
            lines = []
            for group in ("Sequences", "Review", "Navigation", "View"):
                entries = [
                    f"{key(action):<12} {label}"
                    for action, label, _default, action_group in SHORTCUT_ACTIONS
                    if action_group == group
                ]
                if entries:
                    lines.extend(entries)
            lines.append(f"{'Wheel':<12} slices · Shift ×5 · Ctrl zoom")
            self.shortcut_legend.setText("\n".join(lines))
        if hasattr(self, "verdict_segments"):
            hints = []
            for action, segment, label in (
                ("verdict_yes", "yes", "This is a microbleed"),
                ("verdict_no", "no", "This is not a microbleed"),
                ("verdict_unset", "unset", "No decision recorded"),
            ):
                button = self.verdict_segments.button(segment)
                if button is None:
                    continue
                if bindings.get(action):
                    hints.append(key(action))
                    button.setToolTip(f"{label}  ({key(action)})")
                else:
                    button.setToolTip(label)
            self.verdict_keys_label.setText(" · ".join(hints))
        for action_key, menu_action, name in getattr(self, "overlay_actions", []):
            # A tab in a menu item's text is where Qt prints the key.
            menu_action.setText(f"{name}\t{key(action_key)}")
        if hasattr(self, "lesion_zoom_btn"):
            self.lesion_zoom_btn.setToolTip(
                f"Frame a fixed field of view around the target in all views  ({key('lesion_zoom')})"
            )
            self.fit_btn.setToolTip(f"Fit the whole image in every view  ({key('fit_views')})")
        if hasattr(self, "prev_btn"):
            self.prev_btn.setToolTip(f"Previous case  ({key('prev_case')})")
            self.next_btn.setToolTip(f"Next case  ({key('next_case')})")
            self.prev_finding_btn.setToolTip(f"Previous finding  ({key('prev_finding')})")
            self.next_finding_btn.setToolTip(f"Next finding  ({key('next_finding')})")
        if hasattr(self, "save_next_btn"):
            self._update_save_buttons()
        # ``_apply_current_modality`` owns the sequence tooltips because it also
        # knows which sequences this case actually has.
        self._refresh_modality_tooltips()

    # ------------------------------------------------------------ case list --
    def _reload_case_list(self) -> None:
        self.all_cases = list_cases(self.db_path, self.reader_id, self.review_round)
        self._apply_case_filters()

    def _apply_case_filters(self) -> None:
        search = self.case_search.text().strip().lower() if hasattr(self, "case_search") else ""
        filtered: list[dict[str, Any]] = []
        for item in self.all_cases:
            if search and search not in str(item["case_id"]).lower():
                continue
            if getattr(self, "hide_missing_cb", None) and self.hide_missing_cb.isChecked() and item["file_status"] in {"all_missing", "missing_folder"}:
                continue
            if getattr(self, "complete_only_cb", None) and self.complete_only_cb.isChecked() and item["file_status"] != "complete":
                continue
            if getattr(self, "unverified_only_cb", None) and self.unverified_only_cb.isChecked() and item["reviewed_count"] >= item["finding_count"]:
                continue
            if getattr(self, "verified_only_cb", None) and self.verified_only_cb.isChecked() and item["reviewed_count"] < item["finding_count"]:
                continue
            if getattr(self, "source_unverified_cb", None) and self.source_unverified_cb.isChecked() and item["source_unverified_count"] <= 0:
                continue
            if getattr(self, "adjudication_cb", None) and self.adjudication_cb.isChecked() and item["adjudication_count"] <= 0:
                continue
            if getattr(self, "disagreement_cb", None) and self.disagreement_cb.isChecked() and item["disagreement_count"] <= 0:
                continue
            filtered.append(item)
        self.visible_cases = filtered
        if hasattr(self, "case_list"):
            previous_id = self.current_case_id
            self.case_list.blockSignals(True)
            self.case_list.clear()
            selected_row = -1
            for index, item in enumerate(filtered):
                reviewed = int(item["reviewed_count"])
                # Findings the reader has to decide on, including any they added.
                total = int(item["finding_count"])
                # File availability is intentionally not advertised in the
                # case list; it is shown inside the three-view workspace when
                # the case/modality is loaded. Filters still use inventory
                # state behind the scenes.
                progress = f"{reviewed}/{total} reviewed" if total else "no findings"
                text = f"{item['case_id']}      {_human_count(total, 'finding')}  ·  {progress}"
                list_item = QListWidgetItem(text)
                list_item.setData(Qt.ItemDataRole.UserRole, item["case_id"])
                if total and reviewed >= total:
                    list_item.setForeground(QColor(COLORS["success"]))
                elif reviewed:
                    list_item.setForeground(QColor(COLORS["warn"]))
                list_item.setToolTip(
                    f"{item['case_id']} · {item['reader_review_status']} by me\n"
                    f"My verifications: {item['reader_verified_count']} yes / {item['reader_not_verified_count']} no\n"
                    f"Source-unverified findings: {item['source_unverified_count']} · "
                    f"adjudication notes: {item['adjudication_count']} · "
                    f"reader disagreements: {item['disagreement_count']}"
                )
                self.case_list.addItem(list_item)
                if item["case_id"] == previous_id:
                    selected_row = index
            self.case_list.blockSignals(False)
            if selected_row >= 0:
                self.case_list.setCurrentRow(selected_row)
            self.visible_label.setText(f"{len(filtered)} / {len(self.all_cases)}")
            if hasattr(self, "queue_rail_caption"):
                self.queue_rail_caption.setText(
                    f"CASE QUEUE   {len(filtered)} / {len(self.all_cases)}"
                )
            self.visible_label.setToolTip(
                f"{len(filtered)} of {len(self.all_cases)} cases pass the current filters"
            )
            self._update_filter_badge()
            self._update_navigation_buttons()

    def _on_case_row_changed(self, row: int) -> None:
        if row < 0 or row >= self.case_list.count():
            return
        item = self.case_list.item(row)
        case_id = item.data(Qt.ItemDataRole.UserRole)
        if case_id and case_id != self.current_case_id:
            if not self.load_case(str(case_id)) and self.current_case_id:
                # The load was cancelled (unsaved review); put the highlight
                # back on the case that is actually open.
                self._select_case_in_list(self.current_case_id)

    def _update_navigation_buttons(self) -> None:
        if self.current_case_id is None:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return
        try:
            index = next(index for index, item in enumerate(self.visible_cases) if item["case_id"] == self.current_case_id)
        except StopIteration:
            index = -1
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(0 <= index < len(self.visible_cases) - 1)

    def previous_case(self) -> None:
        self._move_case(-1)

    def next_case(self) -> None:
        self._move_case(1)

    def _move_case(self, delta: int) -> None:
        if not self._confirm_dirty():
            return
        try:
            current = next(index for index, item in enumerate(self.visible_cases) if item["case_id"] == self.current_case_id)
        except StopIteration:
            current = 0
        new_index = current + int(delta)
        if 0 <= new_index < len(self.visible_cases):
            self.load_case(self.visible_cases[new_index]["case_id"])

    # -------------------------------------------------------------- loading --
    def load_case(self, case_id: str, *, force: bool = False) -> bool:
        if case_id == self.current_case_id and self._load_thread is None and not force:
            return True
        if not self._confirm_dirty():
            return False
        # Leaving a case behind is the moment a mask nobody decided on becomes
        # lost work, so it is the moment to say so.
        if not self._confirm_unconfirmed_segmentations():
            return False
        case = get_case(self.db_path, case_id)
        if case is None:
            self._set_status(f"Case {case_id} was not found in the shared datasheet.", COLORS["danger"])
            return False
        self._load_generation += 1
        generation = self._load_generation
        # Let a running prefetch stop after its current file: the case the
        # reader asked for should not compete with a speculative read for the
        # disk. Cancelling does not wait, so this never blocks the click.
        self._prefetch_timer.stop()
        if self._prefetch_worker is not None:
            self._prefetch_worker.cancel()
        if not self.settings.sticky_window:
            # Each case starts from its own automatic window unless the reader
            # asked to carry a manual contrast across cases.
            self._window_levels = {}
        self.current_case_id = case_id
        self.current_case = case
        self.targets = list_targets(self.db_path, case_id, self.reader_id, self.review_round)
        self.selected_target = None
        self.target_ras = None
        self.marker_ras = None
        self._review_dirty = False
        self.return_btn.setEnabled(False)
        self._populate_target_list()
        self._set_case_title(case)
        self._clear_review_form()
        paths = {modality: case.get(f"{modality}_path") for modality in MODALITY_ORDER}
        self._set_loading_placeholders(paths)
        self._select_case_in_list(case_id)
        self._update_navigation_buttons()
        self._mark_missing_modalities(paths)
        self._set_status(f"Loading {case_id}…", COLORS["accent"])
        self.case_status.setText("Reading the required AffineRestored files…")
        if self.targets:
            preferred_target_id = self.session.get("last_target_id") if case_id == self.session.get("last_case_id") else None
            preferred_index = next(
                (index for index, target in enumerate(self.targets) if target["target_id"] == preferred_target_id),
                0,
            )
            self._select_target_index(preferred_index, confirm=False)
        # NIfTI loading is deliberately deterministic on the GUI thread. The
        # old prototype used napari's own worker model, but a small custom
        # QThread wrapper caused native Windows crashes when a missing case
        # was opened or a window was closed during loading. The actual viewer
        # interaction remains native and continuous after this short load
        # step; correctness and process stability are more important here.
        self._load_case_sync(paths, generation)
        self._restore_slice_positions_if_resumed(case_id)
        self._schedule_session_save()
        self._schedule_prefetch()
        self._log_event(
            "case_loaded",
            session_id=self.session_id,
            reader_id=self.reader_id,
            review_round=self.review_round,
            case_id=case_id,
            details={"paths_present": {key: bool(value) for key, value in paths.items()}},
        )
        return True

    def _restore_slice_positions_if_resumed(self, case_id: str) -> None:
        if case_id != self.session.get("last_case_id"):
            return
        saved = {
            "axial": self.session.get("last_axial"),
            "coronal": self.session.get("last_coronal"),
            "sagittal": self.session.get("last_sagittal"),
        }
        if not any(value is not None for value in saved.values()):
            return
        for plane, value in saved.items():
            try:
                if value is not None:
                    self.view_panels[plane].canvas.set_slice(int(value))
            except (TypeError, ValueError):
                continue

    def _load_case_sync(self, paths: dict[str, str | None], generation: int) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._loading = True
        try:
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            volumes: dict[str, Volume | None] = {}
            errors: dict[str, str] = {}
            axcodes = self.settings.axcodes
            for modality in MODALITY_ORDER:
                path = paths.get(modality)
                if not path:
                    volumes[modality] = None
                    continue
                cached = self.volume_cache.get(path, axcodes)
                if cached is not None:
                    volumes[modality] = cached
                    continue
                try:
                    volume = load_volume(path, axcodes)
                except Exception as exc:
                    volumes[modality] = None
                    errors[modality] = f"{type(exc).__name__}: {exc}"
                else:
                    volumes[modality] = volume
                    self.volume_cache.put(path, axcodes, volume)
            if generation != self._load_generation or self._closing:
                return
            self.volumes = volumes
            self.load_errors = errors
            substituted = self._resolve_case_modality()
            self._load_label_volume()
            self._apply_current_modality()
            self._apply_zoom_preference()
            missing = [
                MODALITY_SHORT_LABELS[modality]
                for modality in MODALITY_BUTTON_ORDER
                if volumes.get(modality) is None
            ]
            # The segmented control already says which sequence is showing, so
            # this line only reports what is not there.
            availability = "all sequences available" if not missing else f"missing {', '.join(missing)}"
            active_label = MODALITY_LABELS[self.current_modality]
            loaded = volumes.get(self.current_modality) is not None
            self.case_status.setText(self._case_status_text(availability))
            if self._report_marker_placement():
                pass  # A finding that misses the image outranks every other note.
            elif substituted:
                self._set_status(
                    f"{MODALITY_LABELS[substituted]} is missing for {self.current_case_id}; "
                    f"opened {active_label} instead.",
                    COLORS["warn"],
                )
            else:
                self._set_status(
                    f"{self.current_case_id} · {active_label}" if loaded else f"{self.current_case_id} · no MRI available",
                    COLORS["success"] if loaded else COLORS["warn"],
                )
        finally:
            self._loading = False
            QApplication.restoreOverrideCursor()

    def _resolve_case_modality(self) -> str | None:
        """Pick the sequence this case opens on.

        Returns the preferred sequence when it had to be substituted, so the
        caller can say why the view is not showing what was asked for.
        """

        preference = self.settings.default_modality
        wanted = self.current_modality if preference == "last" else preference
        if wanted not in MODALITY_ORDER:
            wanted = "swi"
        if self.volumes.get(wanted) is not None:
            self.current_modality = wanted
            return None
        # Showing three "Not available" panels while a usable sequence sits
        # loaded in memory helps nobody.
        fallback = next(
            (key for key in MODALITY_BUTTON_ORDER if self.volumes.get(key) is not None),
            None,
        )
        if fallback is None:
            self.current_modality = wanted
            return None
        self.current_modality = fallback
        return wanted

    # ------------------------------------------------------- deferred writes --
    def _log_event(self, event_type: str, **kwargs: Any) -> None:
        """Queue an operation-log row; never make the reader wait for it."""

        db_path = self.db_path
        self._writer.submit(
            f"log {event_type}",
            lambda: log_event(db_path, event_type, **kwargs),
        )

    def _on_write_failed(self, message: str) -> None:
        # A dropped log line must not interrupt a review; say so once and move on.
        self._set_status(f"Could not write to the review database · {message}", COLORS["warn"])

    # ------------------------------------------------------------ prefetch --
    def _schedule_prefetch(self) -> None:
        """Queue a background read of the next case, shortly after this one."""

        if not self.settings.prefetch_enabled:
            return
        self._prefetch_timer.start()

    def _next_case_paths(self) -> list[str]:
        try:
            index = next(
                position
                for position, item in enumerate(self.visible_cases)
                if item["case_id"] == self.current_case_id
            )
        except StopIteration:
            return []
        if index + 1 >= len(self.visible_cases):
            return []
        case = get_case(self.db_path, str(self.visible_cases[index + 1]["case_id"]))
        if case is None:
            return []
        axcodes = self.settings.axcodes
        # Keep the case on screen at the head of the cache so prefetching the
        # next one cannot evict the sequences the reader is switching between.
        for modality in MODALITY_ORDER:
            volume = self.volumes.get(modality)
            if volume is not None:
                self.volume_cache.put(volume.path, axcodes, volume)
        paths = []
        # All three sequences of the next case are prefetched, starting with
        # the one it will open on, so 1/2/3 is instant there too.
        for modality in (self.current_modality, *MODALITY_BUTTON_ORDER):
            path = case.get(f"{modality}_path")
            if path and path not in paths and self.volume_cache.get(path, axcodes) is None:
                paths.append(str(path))
        return paths

    def _start_prefetch(self) -> None:
        # ``_loading`` matters because the synchronous load pumps the event
        # loop, so this timer can fire in the middle of a case change.
        if self._closing or self._loading or self._prefetch_thread is not None:
            return
        paths = self._next_case_paths()
        if not paths:
            return
        self._prefetch_generation += 1
        worker = PrefetchWorker(self._prefetch_generation, paths, self.settings.axcodes)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.loaded.connect(self._on_prefetched)
        # Qt aborts the process if a QThread is destroyed while it is still
        # running, so the only safe order is: worker finishes -> event loop
        # quits -> thread reports finished -> both objects are released.
        worker.finished.connect(thread.quit)
        thread.finished.connect(lambda t=thread, w=worker: self._release_prefetch(t, w))
        self._prefetch_worker = worker
        self._prefetch_thread = thread
        thread.start()

    def _on_prefetched(self, generation: int, path: str, volume: object) -> None:
        if generation != self._prefetch_generation or self._closing:
            return
        if isinstance(volume, Volume):
            self.volume_cache.put(path, self.settings.axcodes, volume)

    def _release_prefetch(self, thread: QThread, worker: PrefetchWorker) -> None:
        """Drop a finished prefetch, without disturbing a newer one."""

        if self._prefetch_thread is thread:
            self._prefetch_thread = None
            self._prefetch_worker = None
        try:
            worker.deleteLater()
            thread.deleteLater()
        except RuntimeError:
            # ``_stop_prefetch`` already released them and this queued signal
            # arrived afterwards; nothing left to do.
            pass

    def _stop_prefetch(self) -> None:
        """Stop a running prefetch and wait for the thread to actually exit."""

        worker, thread = self._prefetch_worker, self._prefetch_thread
        if worker is not None:
            worker.cancel()
        if thread is not None and thread.isRunning():
            thread.quit()
            # Cancellation is checked between files, so this waits for at most
            # one volume read.
            if not thread.wait(5000):  # pragma: no cover - would mean a hung read
                thread.terminate()
                thread.wait(1000)
        self._prefetch_thread = None
        self._prefetch_worker = None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()

    def _case_status_text(self, availability: str | None = None) -> str:
        """The one-line subtitle under the case name."""

        if availability is None:
            missing = [
                MODALITY_SHORT_LABELS[modality]
                for modality in MODALITY_BUTTON_ORDER
                if self.volumes.get(modality) is None
            ]
            availability = "all sequences available" if not missing else f"missing {', '.join(missing)}"
        return f"{availability}  ·  {orientation_summary(self.settings.orientation)}"

    def _set_loading_placeholders(self, paths: dict[str, str | None] | None = None) -> None:
        self.volumes = {modality: None for modality in MODALITY_ORDER}
        self.load_errors = {}
        # A sequence this case actually has is "loading", not "not available".
        expected = (paths or {}).get(self.current_modality) if paths is not None else None
        for panel in self.view_panels.values():
            if paths is None or expected:
                panel.canvas.set_loading(
                    f"Reading {MODALITY_LABELS[self.current_modality]} for {self.current_case_id}…"
                )
            else:
                panel.canvas.set_missing(
                    f"This case has no {MODALITY_LABELS[self.current_modality]} file.\n"
                    f"Expected: {MODALITY_SPECS[self.current_modality]['expected']}"
                )
            panel.canvas.update()

    def _set_case_title(self, case: dict[str, Any]) -> None:
        # ``case`` comes from ``case_inventory`` and carries no finding count,
        # so the loaded targets are the reliable source for this line.
        case_id = str(case["case_id"])
        self.case_title.setText(f"{case_id}   ·   {_human_count(len(self.targets), 'finding')}")
        self.case_status.setText("Loading the required AffineRestored files…")
        self._update_finding_buttons()

    def _select_case_in_list(self, case_id: str) -> None:
        for row in range(self.case_list.count()):
            if self.case_list.item(row).data(Qt.ItemDataRole.UserRole) == case_id:
                self.case_list.blockSignals(True)
                self.case_list.setCurrentRow(row)
                self.case_list.blockSignals(False)
                return

    def _populate_target_list(self) -> None:
        self.target_list.blockSignals(True)
        self.target_list.clear()
        halves: list[tuple[str, str]] = []
        for target in self.targets:
            source_verify = _verification_text(target.get("source_verify"))
            reader_status = _report_status(target.get("reader_verify"), target.get("reader_comment"))
            region = str(target.get("atlasregion") or "No region")
            moved = target.get("reader_moved_mm")
            # Two halves, joined by a space or a newline depending on how
            # much room the column has -- see _fit_finding_rows.  The second
            # half uses the same three words as the verdict buttons, without
            # the raw code the sheet stores: "Yes (1)" is spelled out in This
            # finding, immediately below the list.
            # Name the exception, not the default: nearly every finding comes
            # from the sheet, so "Source " on every row was 43px spent saying
            # nothing, while "Manual #..." is exactly what you want to spot.
            name = target["label"]
            if target.get("origin", "Source") == "Source":
                name = name.replace("Source ", "")
            head = f"{name}  ·  {region}"
            tail = f"me: {_short_status(target.get('reader_verify'), target.get('reader_comment'))}"
            tail += f"  ·  source: {_short_status(target.get('source_verify'))}"
            if moved:
                tail += f"  ·  moved {moved:.1f} mm"
            halves.append((head, tail))
            item = QListWidgetItem(f"{head}  ·  {tail}")
            item.setData(Qt.ItemDataRole.UserRole, target["target_id"])
            if target.get("reader_verify") is not None or str(target.get("reader_comment") or "").strip():
                item.setForeground(QColor(COLORS["success"]))
            item.setToolTip(
                f"{target['label']} · {target.get('origin', 'Source')}\n"
                f"RAS ({float(target['ras'][0]):.3f}, {float(target['ras'][1]):.3f}, {float(target['ras'][2]):.3f})\n"
                f"My review: {reader_status} · source verified: {source_verify}"
                + (f"\nMoved {moved:.1f} mm from the source position" if moved else "")
            )
            self.target_list.addItem(item)
        self._finding_row_halves = halves
        self.target_list.blockSignals(False)
        self._fit_finding_rows()
        self._update_finding_buttons()

    def _fit_finding_rows(self) -> None:
        """One line per finding while one line fits; two when it does not.

        A row wants 238px at the median across the 434 findings here and 270
        at the worst, so at the 330px the column opens at every one of them
        is on a single line.  The exceptions are a hand-added finding with a
        comment (393px, its id being a hex string) and a column dragged to
        its 300px floor.  Folding those beats eliding them, and which case
        applies depends on where the reader has put the handle -- so it is
        decided here rather than when the text is built.
        """

        list_widget = getattr(self, "target_list", None)
        if list_widget is None or not getattr(self, "_finding_row_halves", None):
            return
        metrics = list_widget.fontMetrics()
        # The padding, border and scrollbar around the text, measured from the
        # widget rather than assumed: the stylesheet owns those numbers.
        chrome = list_widget.viewport().width() - list_widget.contentsRect().width()
        chrome += 2 * 7 + 2 * 3 + 6  # item padding, list padding, breathing room
        available = list_widget.viewport().width() - chrome
        list_widget.blockSignals(True)
        for index, (head, tail) in enumerate(self._finding_row_halves):
            item = list_widget.item(index)
            if item is None:
                continue
            one_line = f"{head}  ·  {tail}"
            fits = metrics.horizontalAdvance(one_line) <= available
            wanted = one_line if fits else f"{head}\n{tail}"
            if item.text() != wanted:
                item.setText(wanted)
        list_widget.blockSignals(False)
        # Most cases hold a single finding: let the list shrink to fit so the
        # panel below it is not pushed off the bottom of a short window.
        rows = max(1, list_widget.count())
        row_height = list_widget.sizeHintForRow(0) if list_widget.count() else 30
        list_widget.setMaximumHeight(
            min(FINDING_LIST_MAX_HEIGHT, rows * max(row_height, 22) + 10)
        )

    def _on_target_row_changed(self, row: int) -> None:
        self._select_target_index(row, confirm=True)

    def _on_target_item_clicked(self, item: QListWidgetItem) -> None:
        """Clicking the finding already selected jumps back to it.

        ``currentRowChanged`` does not fire for the row that is already
        current, so after scrolling away there was no way back to a finding
        except selecting another one and returning to it.
        """

        target_id = item.data(Qt.ItemDataRole.UserRole)
        if self.selected_target is None or target_id != self.selected_target["target_id"]:
            return
        if self._views_are_on_the_marker():
            return
        self.return_to_finding()

    def _views_are_on_the_marker(self) -> bool:
        """True when every view already shows the selected finding."""

        if self.marker_ras is None or self.target_ras != self.marker_ras:
            return False
        return all(panel.canvas.marker_on_slice() for panel in self.view_panels.values())

    def _open_workspace_for(self, target: dict[str, Any] | None) -> None:
        """Start an unjudged finding where judging happens.

        Switching used to leave whatever tab and tool the last finding ended
        on, so a reader who had been painting arrived at a fresh case with a
        brush in hand and the Segment tab open -- offering the one thing that
        cannot be done there until a verdict exists.

        A finding that has already been judged is left alone: going back over
        your own segmentations is a real way to work, and moving the reader
        out of the Segment tab every time would break it.
        """

        if target is None or self.settings.keep_tool_on_switch:
            return
        judged = target.get("reader_verify") is not None or str(
            target.get("reader_comment") or ""
        ).strip()
        if judged:
            return
        self.show_panel_tab("review")
        self.set_tool("point")

    def _select_target_index(self, index: int, *, confirm: bool) -> None:
        if index < 0 or index >= len(self.targets):
            return
        if confirm and not self._confirm_dirty():
            return
        target = self.targets[index]
        self.selected_target = target
        # A new finding starts from its own recorded positions; an unsaved
        # move on the previous finding does not carry over.
        self.pending_ras = None
        self.selected_variant = self.reader_id if target.get("reader_ras") else "source"
        self._rebuild_position_variants(target)
        if self.target_list.currentRow() != index:
            # Keep the visible highlight in step with the finding that is
            # actually loaded, including the one selected automatically on load.
            self.target_list.blockSignals(True)
            self.target_list.setCurrentRow(index)
            self.target_list.blockSignals(False)
        # Use the position the selector is showing, which for a finding this
        # reader has already corrected is their own, not the source one.
        variant = self._current_variant()
        ras = tuple(float(value) for value in (variant["ras"] if variant else target["ras"]))
        self.target_ras = ras
        self.marker_ras = ras
        self._set_coordinate_spins(ras)
        self._set_manual_spins(ras)
        self.return_btn.setEnabled(True)
        self._apply_labels_to_views()
        self._load_review_form(target)
        self._open_workspace_for(target)
        self._apply_target_to_views(recenter=True)
        self._apply_zoom_preference()
        if not self._report_marker_placement():
            self._set_status(
                f"{target['label']} · RAS ({ras[0]:.3f}, {ras[1]:.3f}, {ras[2]:.3f})",
                COLORS["accent"],
            )
        self._schedule_session_save()
        self._log_event(
            "finding_selected",
            session_id=self.session_id,
            reader_id=self.reader_id,
            review_round=self.review_round,
            case_id=self.current_case_id,
            target_id=target["target_id"],
            details={"ras": ras, "origin": target.get("origin")},
        )

    # ------------------------------------------------------------ coordinates --
    def _set_coordinate_spins(self, ras: tuple[float, float, float] | None) -> None:
        for spin, value in zip((self.ras_l_spin, self.ras_p_spin, self.ras_s_spin), ras or (0.0, 0.0, 0.0)):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    def _set_manual_spins(self, ras: tuple[float, float, float] | None) -> None:
        if ras is None:
            return
        for spin, value in zip((self.manual_l_spin, self.manual_p_spin, self.manual_s_spin), ras):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    def jump_to_ras(self) -> None:
        values = tuple(float(spin.value()) for spin in (self.ras_l_spin, self.ras_p_spin, self.ras_s_spin))
        if not all(math.isfinite(value) for value in values):
            QMessageBox.warning(self, "Invalid RAS", "Enter three finite RAS coordinates.")
            return
        # Moving the crosshair is navigation, not a change of finding: the
        # selected finding, its list highlight and the unsaved review stay put
        # so ``Return to selected finding`` and ``Save review`` keep working.
        self.target_ras = values
        self._apply_target_to_views(recenter=True)
        self._set_manual_spins(values)
        self._set_status(f"Viewer centered on manual RAS ({values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f})", COLORS["accent"])
        if self.current_case_id:
            self._log_event(
                "manual_ras_jump",
                session_id=self.session_id,
                reader_id=self.reader_id,
                review_round=self.review_round,
                case_id=self.current_case_id,
                details={"ras": values},
            )
        self._schedule_session_save()

    def return_to_finding(self) -> None:
        if self.selected_target is None:
            return
        # Return to the position currently being shown, not necessarily the
        # source one.
        variant = self._current_variant()
        ras = variant["ras"] if variant else self.selected_target["ras"]
        self.target_ras = tuple(float(value) for value in ras)
        self.marker_ras = self.target_ras
        self._set_coordinate_spins(self.target_ras)
        self._apply_target_to_views(recenter=True)
        self._set_status(f"Returned to {self.selected_target['label']}", COLORS["accent"])
        self._log_event(
            "return_to_finding",
            session_id=self.session_id,
            reader_id=self.reader_id,
            review_round=self.review_round,
            case_id=self.current_case_id,
            target_id=self.selected_target["target_id"],
            details={"ras": self.target_ras},
        )

    def _apply_target_to_views(self, *, recenter: bool, planes: Iterable[str] | None = None) -> None:
        """Move the navigation cursor, and redraw the fixed markers."""

        volume = self.volumes.get(self.current_modality)
        neighbours = self._neighbour_voxels(volume)
        marker_voxel = None
        if volume is not None and self.marker_ras is not None:
            try:
                marker_voxel = ras_to_voxel(volume.affine, self.marker_ras)
            except Exception:
                marker_voxel = None
        names = set(planes) if planes is not None else set(self.view_panels)
        for name, panel in self.view_panels.items():
            panel.canvas.set_secondary_targets(neighbours)
            panel.canvas.set_marker_voxel(marker_voxel)
            if volume is None or self.target_ras is None:
                panel.canvas.set_target_voxel(None)
                continue
            try:
                voxel = ras_to_voxel(volume.affine, self.target_ras)
            except Exception:
                panel.canvas.set_target_voxel(None)
                continue
            panel.canvas.set_target_voxel(voxel, recenter=recenter and name in names)
        self._schedule_session_save()

    def marker_is_outside_the_volume(self) -> bool:
        """True when the selected finding's coordinate is not in this image."""

        volume = self.volumes.get(self.current_modality)
        if volume is None or self.marker_ras is None:
            return False
        try:
            voxel = ras_to_voxel(volume.affine, self.marker_ras)
        except Exception:
            return True
        return not voxel_in_bounds(voxel, volume.shape)

    def _report_marker_placement(self) -> bool:
        """Say so when the finding does not land inside the image.

        A coordinate that misses the volume means the workbook and the NIfTI
        disagree about the space they are in -- a wrong origin, or a file from
        another session.  Reviewing the clamped edge position would record a
        verdict about whatever tissue happens to be there.
        """

        if not self.marker_is_outside_the_volume():
            return False
        label = self.selected_target["label"] if self.selected_target else "This finding"
        sequence = MODALITY_LABELS[self.current_modality]
        self._set_status(
            f"{label} is outside the {sequence} volume — its coordinate and this image "
            "are not in the same space. Do not record a verdict from the edge crosshair.",
            COLORS["danger"],
        )
        return True

    def _neighbour_voxels(self, volume: Volume | None) -> list[np.ndarray]:
        """Display voxels of the other findings of this case."""

        if volume is None or not self.targets:
            return []
        selected_id = self.selected_target["target_id"] if self.selected_target else None
        voxels: list[np.ndarray] = []
        for target in self.targets:
            if target["target_id"] == selected_id:
                continue
            # Show each neighbour where this reader believes it is.
            ras = target.get("effective_ras") or target["ras"]
            try:
                voxels.append(ras_to_voxel(volume.affine, ras))
            except Exception:
                continue
        return voxels

    def _on_canvas_target_clicked(self, _plane: str, voxel: object) -> None:
        if not isinstance(voxel, np.ndarray) or self.volumes.get(self.current_modality) is None:
            return
        volume = self.volumes[self.current_modality]
        assert volume is not None
        values = tuple(float(value) for value in voxel_to_ras(volume.affine, voxel))
        # Clicking navigates only; the finding under review stays selected and
        # its marker stays where it is.
        self.target_ras = values
        self._set_coordinate_spins(values)
        self._set_manual_spins(values)
        linked = {name for name, panel in self.view_panels.items() if panel.sync_enabled}
        self._apply_target_to_views(recenter=True, planes=linked)
        self._set_status(f"Viewer centered on clicked voxel · RAS ({values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f})", COLORS["accent"])
        self._log_event(
            "viewer_coordinate_clicked",
            session_id=self.session_id,
            reader_id=self.reader_id,
            review_round=self.review_round,
            case_id=self.current_case_id,
            details={"ras": values, "voxel": tuple(float(value) for value in voxel)},
        )

    def _on_mouse_voxel_moved(self, plane: str, voxel: object) -> None:
        if plane in self.view_panels:
            # Remembered so F maximises the view the reader is working in.
            self._active_plane = plane
        if not isinstance(voxel, np.ndarray) or self.volumes.get(self.current_modality) is None:
            self._coord_label.setText(f"{PLANE_TITLES[plane]} · move over the image")
            return
        volume = self.volumes[self.current_modality]
        assert volume is not None
        try:
            ras = voxel_to_ras(volume.affine, voxel)
            self._coord_label.setText(
                f"{PLANE_TITLES[plane]} · voxel ({voxel[0]:.1f}, {voxel[1]:.1f}, {voxel[2]:.1f}) · RAS ({ras[0]:.3f}, {ras[1]:.3f}, {ras[2]:.3f})"
            )
        except Exception:
            self._coord_label.setText(f"{PLANE_TITLES[plane]} · mouse coordinate unavailable")

    def _on_slice_step(self, plane: str, steps: int) -> None:
        """Translate a relative wheel step into an absolute slice request."""

        if self.volumes.get(self.current_modality) is None or not steps:
            return
        current = self.view_panels[plane].canvas.slice_index
        self._on_slice_request(plane, current + int(steps))

    def _on_slice_request(self, plane: str, value: int) -> None:
        """Scrolling moves this plane only.

        The finding marker stays at its own coordinate, so scrolling off the
        lesion makes the crosshair disappear from that view instead of dragging
        it along -- that is how a reader checks whether a focus persists across
        slices, and it is the reason scrolling never edits the target.
        """

        panel = self.view_panels[plane]
        volume = self.volumes.get(self.current_modality)
        if volume is None:
            return
        panel.canvas.set_slice(value)
        if self.settings.scroll_moves_cursor and self.target_ras is not None:
            self._carry_cursor_to_slice(plane, volume, panel.canvas.slice_index)
        self._queue_slice_log(plane, panel.canvas.slice_index, sync=panel.sync_enabled)
        self._schedule_session_save()

    def _carry_cursor_to_slice(self, plane: str, volume: Volume, index: int) -> None:
        """Drag the shared cursor along with a scrolled view.

        Only the axis this plane scrolls moves; the other two keep the
        coordinate they had, so the cursor tracks the wheel without the other
        views jumping sideways.
        """

        try:
            voxel = ras_to_voxel(volume.affine, self.target_ras)
        except Exception:
            return
        voxel[PLANE_AXES[plane]] = float(index)
        self.target_ras = tuple(float(value) for value in voxel_to_ras(volume.affine, voxel))
        self._set_coordinate_spins(self.target_ras)
        for name, other in self.view_panels.items():
            if name == plane:
                continue
            try:
                other.canvas.set_target_voxel(
                    ras_to_voxel(volume.affine, self.target_ras),
                    recenter=other.sync_enabled,
                )
            except Exception:
                continue

    def _queue_slice_log(self, plane: str, value: int, *, sync: bool) -> None:
        """Record where scrolling stopped instead of every intermediate slice.

        Continuous wheel scrolling produced one SQLite connection, insert and
        commit per slice, which stutters badly on a synchronised folder.
        """

        self._pending_slice_log = {
            "case_id": self.current_case_id,
            "target_id": self.selected_target["target_id"] if self.selected_target else None,
            "details": {"plane": plane, "slice": int(value), "sync": bool(sync)},
        }
        self._slice_log_timer.start()

    def _flush_slice_log(self) -> None:
        pending = self._pending_slice_log
        self._pending_slice_log = None
        if not pending:
            return
        try:
            self._log_event(
                "slice_changed",
                session_id=self.session_id,
                reader_id=self.reader_id,
                review_round=self.review_round,
                case_id=pending["case_id"],
                target_id=pending["target_id"],
                details=pending["details"],
            )
        except Exception:
            # A log write must never interrupt review navigation.
            pass

    # ------------------------------------------------------------- modality --
    def set_modality(self, modality: str) -> None:
        """Switch sequence, keeping the reader's place in the volume."""

        modality = str(modality or "").lower()
        if modality not in MODALITY_ORDER:
            return
        if modality == self.current_modality:
            self.modality_segments.set_current_key(modality)
            return
        if self.current_case_id is not None and self.volumes.get(modality) is None:
            self.modality_segments.set_current_key(self.current_modality)
            self._set_status(
                f"{MODALITY_LABELS[modality]} is not available for {self.current_case_id}.",
                COLORS["warn"],
            )
            return
        self.current_modality = modality
        self.modality_segments.set_current_key(modality)
        # Comparing the same lesion across QSM/SWI/MIP must not throw the
        # reader back to Autofit, so the zoom and pan of each view are kept.
        self._apply_current_modality(reset_view=False)
        self._set_status(f"{MODALITY_LABELS[modality]}", COLORS["accent"])
        self._log_event(
            "modality_changed",
            session_id=self.session_id,
            reader_id=self.reader_id,
            review_round=self.review_round,
            case_id=self.current_case_id,
            target_id=self.selected_target["target_id"] if self.selected_target else None,
            details={"modality": modality},
        )
        self._schedule_session_save()

    def _mark_missing_modalities(self, paths: dict[str, str | None]) -> None:
        """Strike through the sequences this case does not have.

        Availability is known from the inventory before anything is read, so
        the reader can see it without clicking a button to find out.
        """

        for modality, button in (
            (key, self.modality_segments.button(key)) for key in MODALITY_BUTTON_ORDER
        ):
            if button is None:
                continue
            available = bool(paths.get(modality))
            font = button.font()
            font.setStrikeOut(not available)
            button.setFont(font)
            button.setEnabled(available)
            button.setToolTip(
                f"{MODALITY_LABELS[modality]}  "
                f"({self._shortcut_text('sequence_' + str(MODALITY_BUTTON_ORDER.index(modality) + 1))})"
                if available
                else f"{MODALITY_LABELS[modality]} is not available for this case"
            )

    def _refresh_modality_tooltips(self) -> None:
        paths = {
            modality: (self.current_case or {}).get(f"{modality}_path")
            for modality in MODALITY_ORDER
        }
        if self.current_case_id is None:
            paths = {modality: "-" for modality in MODALITY_ORDER}
        self._mark_missing_modalities(paths)

    def _apply_current_modality(self, *, reset_view: bool = True) -> None:
        modality = self.current_modality
        volume = self.volumes.get(modality)
        self._refresh_modality_tooltips()
        self.modality_segments.set_current_key(modality)
        for panel in self.view_panels.values():
            if volume is None:
                panel.canvas.set_missing()
            else:
                panel.canvas.set_volume(volume, reset_view=reset_view)
        if volume is None:
            if self.load_errors.get(modality):
                message = f"{MODALITY_LABELS[modality]} cannot be loaded.\n{self.load_errors[modality]}"
            else:
                expected = MODALITY_SPECS[modality]["expected"]
                if self.current_case is not None and not bool(self.current_case.get("folder_exists")):
                    message = f"Case folder is missing; required {MODALITY_LABELS[modality]} AffineRestored file is unavailable.\nExpected: {expected}"
                else:
                    message = f"Required {MODALITY_LABELS[modality]} AffineRestored file is missing.\nExpected: {expected}"
            for panel in self.view_panels.values():
                panel.canvas.set_missing(message)
            self.case_status.setText(message.replace("\n", "  "))
        else:
            self.case_status.setText(self._case_status_text())
        self._apply_window_preference()
        self._refresh_contrast_dialog()
        self._apply_target_to_views(recenter=True)
        self._toggle_crosshair()

    def _toggle_crosshair(self) -> None:
        for panel in self.view_panels.values():
            panel.canvas.set_show_target(self.target_crosshair_cb.isChecked())
            panel.canvas.set_show_mouse(self.mouse_crosshair_cb.isChecked())
            panel.canvas.set_show_directions(self.direction_cb.isChecked())

    def reset_views(self) -> None:
        for panel in self.view_panels.values():
            panel.canvas.reset_view()
        self.lesion_zoom_btn.setChecked(False)

    # ----------------------------------------------------------------- zoom --
    def set_lesion_focus(self, enabled: bool) -> None:
        """Frame the lesion field of view, or fall back to whole-image fit."""

        enabled = bool(enabled)
        for panel in self.view_panels.values():
            panel.canvas.set_lesion_fov(self.settings.lesion_fov_mm)
            panel.canvas.set_zoom_mode(LESION_ZOOM_LABEL if enabled else "Autofit")
        # A view without a target silently stays on fit; report what happened.
        applied = any(panel.canvas.lesion_focus for panel in self.view_panels.values())
        self.lesion_zoom_btn.setChecked(enabled and applied)
        if enabled and not applied:
            self._set_status("Select a finding or a coordinate before using lesion zoom.", COLORS["warn"])
        elif enabled:
            self._set_status(
                f"Lesion zoom · {self.settings.lesion_fov_mm:g} mm field of view", COLORS["accent"]
            )
        else:
            self._set_status("Whole-image fit", COLORS["dim"])

    def toggle_lesion_focus(self) -> None:
        self.set_lesion_focus(not self.lesion_zoom_btn.isChecked())

    def _apply_zoom_preference(self) -> None:
        """Apply the reader's auto-zoom preference to the current target."""

        if not self.settings.auto_zoom or self.target_ras is None:
            return
        if self.volumes.get(self.current_modality) is None:
            return
        for panel in self.view_panels.values():
            panel.canvas.set_lesion_fov(self.settings.lesion_fov_mm)
            panel.canvas.set_zoom_mode(LESION_ZOOM_LABEL)
        self.lesion_zoom_btn.setChecked(
            any(panel.canvas.lesion_focus for panel in self.view_panels.values())
        )

    # --------------------------------------------------------- segmentation --
    def _label_file(self) -> Path:
        return label_path(self.db_path, str(self.current_case_id), self.reader_id, self.review_round)

    def _label_reference(self) -> Volume | None:
        """The volume whose geometry the case's label file uses.

        One label file serves the whole case, so its grid has to be one of the
        sequences.  SWI first: it is the sequence a microbleed is read on and
        the one a brush stroke is almost always drawn on, so the mask is
        voxel-exact with respect to the image it was actually painted over.
        """

        for modality in LABEL_REFERENCE_ORDER:
            volume = self.volumes.get(modality)
            if volume is not None:
                return volume
        return None

    def _grid_mismatch(self) -> str | None:
        """Report sequences of this case that are not on one voxel grid.

        Every complete case in this dataset shares a grid across QSM, SWI and
        MIP, which is what lets one label file cover the case.  Nothing checked
        it, so a case that did not share one would have written a mask against
        the wrong image, or raised an index error mid-stroke.
        """

        reference = self._label_reference()
        if reference is None:
            return None
        reference_name = next(
            MODALITY_LABELS[modality]
            for modality in LABEL_REFERENCE_ORDER
            if self.volumes.get(modality) is reference
        )
        for modality in LABEL_REFERENCE_ORDER:
            volume = self.volumes.get(modality)
            if volume is None or volume is reference:
                continue
            same_shape = tuple(volume.shape) == tuple(reference.shape)
            same_affine = np.allclose(
                np.asarray(volume.affine), np.asarray(reference.affine), atol=1e-3
            )
            if not (same_shape and same_affine):
                return (
                    f"{MODALITY_LABELS[modality]} {tuple(volume.shape)} and "
                    f"{reference_name} {tuple(reference.shape)} are on different voxel "
                    "grids. One label file covers a whole case, so segmentation needs a "
                    "single grid and is unavailable here; verification and coordinates "
                    "are unaffected."
                )
        return None

    def _load_label_volume(self) -> None:
        """Read this reader's segmentation of the current case, if any."""

        self.label_volume = None
        self.label_values = {}
        self.label_sources = {}
        self.label_methods = {}
        self.label_settings = {}
        self.last_segmentation = None
        self._stored_roi_targets = set()
        self._roi_dirty = False
        self._roi_undo = []
        reference = self._label_reference()
        self._grid_problem = self._grid_mismatch()
        self._update_segment_availability()
        if reference is None or self.current_case_id is None:
            self._apply_labels_to_views()
            return
        if self._grid_problem:
            # Painting into an array that does not match the image on screen is
            # worse than not offering the tool at all.
            if self.active_tool in ("brush", "eraser"):
                self.set_tool(None)
            self._apply_labels_to_views()
            self._set_status(self._grid_problem, COLORS["warn"])
            return
        # A fresh label volume for this case invalidates every mask geometry
        # cached against the last one.
        self._others_cache = None
        self.label_volume = np.zeros(reference.shape, dtype=np.uint16)
        for target in self.targets:
            roi = target.get("roi")
            if roi:
                target_id = str(target["target_id"])
                self.label_values[target_id] = int(roi["label_value"])
                self.label_sources[target_id] = roi.get("generated_from")
                if roi.get("method"):
                    self.label_methods[target_id] = str(roi["method"])
                if roi.get("sensitivity") is not None:
                    self.label_settings[target_id] = {
                        "sensitivity": float(roi["sensitivity"]),
                        "radius_mm": float(roi.get("radius_mm") or 0.0),
                    }
                self._stored_roi_targets.add(target_id)
        path = self._label_file()
        if path.exists() and self.label_values:
            try:
                stored = load_volume(path, self.settings.axcodes)
            except Exception as exc:
                self._set_status(f"Could not read the saved segmentation: {exc}", COLORS["warn"])
            else:
                if stored.shape == reference.shape:
                    self.label_volume = np.asarray(stored.data, dtype=np.uint16)
                else:
                    self._set_status(
                        "The saved segmentation does not match this case's grid; it was not loaded.",
                        COLORS["warn"],
                    )
        self._apply_labels_to_views()
        self._update_roi_readout()

    def _label_value_for(self, target_id: str) -> int:
        value = self.label_values.get(str(target_id))
        if value is not None:
            return value
        used = set(self.label_values.values())
        value = next(candidate for candidate in range(1, 4096) if candidate not in used)
        self.label_values[str(target_id)] = value
        return value

    def _apply_labels_to_views(self) -> None:
        value = (
            self._label_value_for(self.selected_target["target_id"])
            if self.selected_target is not None
            else 1
        )
        for panel in self.view_panels.values():
            panel.canvas.set_label_volume(self.label_volume, value)

    def _selected_label_mask(self) -> np.ndarray | None:
        if self.label_volume is None or self.selected_target is None:
            return None
        return self.label_volume == self._label_value_for(self.selected_target["target_id"])

    def roi_volume_mm3(self) -> float:
        # The label volume's own grid, not the sequence on screen, so the
        # reading on the panel is the number that reaches the database.
        mask = self._selected_label_mask()
        reference = self._label_reference()
        if mask is None or reference is None:
            return 0.0
        return float(mask.sum()) * float(np.prod(reference.voxel_sizes))

    def _on_roi_stroke_started(self) -> None:
        """Open a new undo step; what changes gets recorded as it happens.

        Copying the whole label volume per stroke cost 23 MB on a real case
        here, and the twenty kept steps came to 461 MB on top of the cached
        image volumes.  A step now holds only the voxels it touched.
        """

        if self.label_volume is None:
            return
        self._roi_undo.append({"shape": tuple(self.label_volume.shape), "changes": []})
        del self._roi_undo[:-ROI_UNDO_STEPS]

    def _record_roi_change(self, indices: object, previous: object) -> None:
        """Add voxels, and the values they held, to the open undo step."""

        indices = np.asarray(indices, dtype=np.intp)
        if indices.size == 0 or self.label_volume is None:
            return
        if not self._roi_undo:
            self._on_roi_stroke_started()
            if not self._roi_undo:
                return
        step = self._roi_undo[-1]
        if step["shape"] != tuple(self.label_volume.shape):
            return
        step["changes"].append(
            (indices, np.asarray(previous, dtype=self.label_volume.dtype))
        )

    def _record_roi_mask_change(self, mask: np.ndarray) -> None:
        """Record an undo step for a whole-mask edit (clear, or generate)."""

        if self.label_volume is None:
            return
        indices = np.flatnonzero(np.asarray(mask).reshape(-1))
        self._on_roi_stroke_started()
        self._record_roi_change(indices, self.label_volume.reshape(-1)[indices])

    def _note_hand_edit(self) -> None:
        """Record that a human changed this mask after it was generated.

        A grown mask a reader corrected is neither purely automatic nor purely
        manual, and the results table should not claim either.
        """

        if self.selected_target is None:
            return
        target_id = str(self.selected_target["target_id"])
        current = self.label_methods.get(target_id)
        if current is None:
            self.label_methods[target_id] = "brush"
        elif current == "grow":
            self.label_methods[target_id] = "grow+brush"

    def _roi_undo_bytes(self) -> int:
        """How much the undo history costs; asserted on by the tests."""

        return sum(
            int(indices.nbytes + previous.nbytes)
            for step in self._roi_undo
            for indices, previous in step["changes"]
        )

    def _on_roi_painted(self, _plane: str) -> None:
        self._note_hand_edit()
        self._roi_dirty = True
        self._mark_review_dirty()
        for panel in self.view_panels.values():
            panel.canvas.update()
        self._queue_roi_readout()

    def _queue_roi_readout(self) -> None:
        """Update the mask readout soon, rather than on every mouse-move.

        This signal fires once per reported mouse position, and the readout
        costs 19 ms with the 3D window shut and 50 ms with the head in it --
        so the paint loop could keep up with 53 positions a second, or 20,
        against the 125 a mouse sends.  The brush lagged the cursor, and the
        dropped positions widened the gaps a stroke had to bridge.

        Nothing here is needed within a frame: it is a volume, a diameter and
        a slice count.  Collapsing a burst into one update at the end of it
        costs the reader an eighth of a second of staleness and gives the
        brush its event loop back.
        """

        if self._readout_timer.isActive():
            return
        self._readout_timer.start()

    def undo_roi(self) -> None:
        if not self._roi_undo or self.label_volume is None:
            self._set_status("Nothing to undo.", COLORS["dim"])
            return
        step = self._roi_undo.pop()
        if step["shape"] != tuple(self.label_volume.shape):
            return
        flat = self.label_volume.reshape(-1)
        # Reverse order, so a voxel painted over twice within one stroke ends up
        # holding the value it had before the stroke began.
        for indices, previous in reversed(step["changes"]):
            flat[indices] = previous
        self._roi_dirty = True
        self._apply_labels_to_views()
        self._update_roi_readout()
        self._set_status("Undid the last brush stroke.", COLORS["dim"])

    def clear_roi(self) -> None:
        mask = self._selected_label_mask()
        if mask is None or not mask.any():
            self._set_status("This finding has no segmentation.", COLORS["dim"])
            return
        self._record_roi_mask_change(mask)
        self.label_volume[mask] = 0
        self._roi_dirty = True
        self._mark_review_dirty()
        self._apply_labels_to_views()
        self._update_roi_readout()
        self._set_status("Cleared the segmentation of this finding.", COLORS["dim"])

    def grow_from_stroke(self) -> None:
        """Use what is already painted as the seed, and grow from there.

        Painting a rough scribble down the middle of a lesion and expanding it
        handles the shapes a single seed point does not: irregular lesions, and
        ones where the finding coordinate happens to sit on an unrepresentative
        voxel.
        """

        mask = self._selected_label_mask()
        if mask is None or not mask.any():
            self._set_status(
                "Paint a stroke inside the lesion first, then grow from it.", COLORS["warn"]
            )
            return
        self.auto_segment(seed_mask=mask)

    def auto_segment(self, seed_mask: np.ndarray | None = None) -> None:
        """Grow the mask from the finding, using the sequence on screen."""

        blocked = self.segmentation_block()
        if blocked:
            self._set_status(blocked, COLORS["warn"])
            return
        modality = self.current_modality
        if not can_segment(modality):
            QMessageBox.information(
                self,
                f"Not on the {MODALITY_LABELS.get(modality, modality)}",
                "A minimum-intensity projection smears a microbleed along the projection "
                "direction (about seven times, measured on this data), so a mask generated "
                "here would be a projection artefact.\n\nSwitch to SWI or QSM to generate, "
                "then come back to the MIP to check it.",
            )
            return
        volume = self.volumes.get(modality)
        if volume is None or self.label_volume is None or self.marker_ras is None:
            self._set_status("No volume to segment.", COLORS["warn"])
            return
        try:
            seed = seed_mask if seed_mask is not None else ras_to_voxel(volume.affine, self.marker_ras)
        except Exception:
            return
        sensitivity = float(self.sensitivity_spin.value())
        radius_mm = float(self.roi_radius_spin.value())
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            mask, details = segment_lesion(
                volume.data,
                seed,
                volume.voxel_sizes,
                # A microbleed is dark on SWI and bright on QSM.
                dark=modality != "qsm",
                sensitivity=sensitivity,
                radius_mm=radius_mm,
            )
        finally:
            QApplication.restoreOverrideCursor()
        target_id = str(self.selected_target["target_id"])
        value = self._label_value_for(target_id)
        # Everything the regrow touches: what it clears, plus what it adds.
        self._record_roi_mask_change((self.label_volume == value) | mask)
        self.label_volume[self.label_volume == value] = 0
        self.label_volume[mask] = value
        self.label_sources[target_id] = modality
        self.label_methods[target_id] = "grow"
        self.label_settings[target_id] = {"sensitivity": sensitivity, "radius_mm": radius_mm}
        self.last_segmentation = details
        self._roi_dirty = True
        self._mark_review_dirty()
        self._apply_labels_to_views()
        self._update_roi_readout()
        self._report_segmentation(details, seed_mask is not None, modality)

    def _report_segmentation(
        self, details: dict[str, Any], from_stroke: bool, modality: str
    ) -> None:
        """Say what came out, and when not to trust it.

        Region growing can stop for two reasons that are not "the lesion ends
        here": it ran into the safety cap, so its size is the cap's answer, or
        it never left the seed, so the reader pressed Generate and got nothing.
        Both looked like success before, which is how an unusable mask reaches
        the results table.
        """

        origin = "the painted stroke" if from_stroke else "the finding"
        where = MODALITY_LABELS[modality]
        if int(details.get("voxel_count", 0)) <= 1:
            self._set_status(
                f"Nothing grew from {origin} on {where}: this focus is too close to its "
                "background at the current sensitivity. Lower the sensitivity, or paint it "
                "by hand.",
                COLORS["warn"],
            )
            return
        summary = (
            f"{float(details['volume_mm3']):.1f} mm³ · ø {float(details['diameter_mm']):.1f} mm"
        )
        if details.get("reached_cap"):
            self._set_status(
                f"Grew from {origin} on {where} · {summary}, but it ran into the "
                f"{float(self.roi_radius_spin.value()):.0f} mm cap — the size is the cap's, "
                "not the lesion's. Check it, raise the cap, or raise the sensitivity.",
                COLORS["warn"],
            )
        elif details.get("suspect"):
            self._set_status(
                f"Grew from {origin} on {where} · {summary}, but it is "
                f"{float(details['longest_mm']):.0f} mm long — longer than a microbleed. "
                "It has probably followed a vessel; check it before saving.",
                COLORS["warn"],
            )
        else:
            self._set_status(
                f"Grew from {origin} on {where} · {summary}. Adjust with the brush, or "
                "change the sensitivity and grow again.",
                COLORS["success"],
            )

    def open_lesion_3d(self) -> None:
        """Open the mask inspector, or bring it back to the front."""

        if self._lesion_dialog is None:
            self._lesion_dialog = Lesion3DDialog(self)
            self._lesion_dialog.options_changed = self._refresh_lesion_3d
        # Shown first: the refresh below skips a window nobody is looking at,
        # which on the way in would mean this one.
        self._lesion_dialog.show()
        self._lesion_dialog.raise_()
        self._lesion_dialog.activateWindow()
        self._refresh_lesion_3d()

    def _refresh_lesion_3d(self) -> None:
        """Redraw the open inspector, if there is one.

        Called from the readout, which already runs on every change that can
        alter a mask -- a stroke, a Generate, an undo, a different finding.
        Left open beside the images it therefore follows the brush.
        """

        dialog = getattr(self, "_lesion_dialog", None)
        if dialog is None or not dialog.isVisible():
            return
        target = self.selected_target
        mask = self._selected_label_mask()
        reference = self._label_reference()
        if target is None or mask is None or reference is None or not mask.any():
            dialog.canvas.set_context(None, None, (1.0, 1.0), (0.0, 1.0), np.zeros(3))
            dialog.show_lesion(
                str(target["label"]) if target else "No finding",
                "",
                np.zeros((0, 4, 3)),
                np.zeros((0, 3)),
            )
            return

        spacing = np.asarray(reference.voxel_sizes[:3], dtype=np.float64)
        # Crop once and work on the crop.  Every full-volume pass over this
        # 256x256x176 label costs 16 ms, this method needs three of them, and
        # it runs from the readout on every brush stroke.  A microbleed is a
        # few hundred of those twelve million voxels.
        coords = np.argwhere(mask)
        low = coords.min(axis=0)
        crop = mask[
            low[0]:coords[:, 0].max() + 1,
            low[1]:coords[:, 1].max() + 1,
            low[2]:coords[:, 2].max() + 1,
        ]
        shape = lesion_shape(crop, reference.voxel_sizes)

        # Which sequence the head is drawn from is the reader's choice.  The
        # mask still belongs to the label reference's grid, so the two have to
        # be tied together in world space rather than by index -- see
        # _placement_centre.
        available = [key for key in MODALITY_BUTTON_ORDER if self.volumes.get(key) is not None]
        dialog.offer_brain_sources(
            available, MODALITY_LABELS, self._label_reference_modality() or ""
        )
        brain = self.volumes.get(dialog.brain_modality) or reference
        in_brain = dialog.wants_brain and self._brain_context(brain) is not None

        # In the brain the lesion is measured from the head's centre rather
        # than its own, or it would be drawn in the middle whatever its
        # coordinate says.  The crop moved the origin, so the centre it is
        # measured from moves with it.
        placement = self._placement_centre(reference, brain)
        centre = placement - low * spacing if in_brain else None
        quads, normals = lesion_surface(
            crop,
            reference.voxel_sizes,
            centre=centre,
            smooth=LESION_SMOOTH_PASSES if dialog.wants_smooth else 0,
        )
        corners = quads.reshape(-1, 3)
        extent = corners.max(axis=0) - corners.min(axis=0)
        radius = None
        tints = None
        own_faces = len(quads)
        where = ""
        if in_brain:
            context = self._brain_context(brain)
            fine, coarse, mm_fine, mm_coarse, window = context
            offset = coords.mean(axis=0) * spacing - placement
            dialog.canvas.set_context(
                fine, coarse, (mm_fine, mm_coarse), window, offset
            )
            brain_spacing = np.asarray(brain.voxel_sizes[:3], dtype=np.float64)
            radius = float(max(np.asarray(brain.shape[:3]) * brain_spacing) / 2.0)
            # Every mask in the case, not just this one: how the findings sit
            # relative to each other is the question the head is there for --
            # lobar against deep is what separates amyloid angiopathy from
            # hypertensive disease, and one lesion alone cannot show it.
            quads, normals, tints, others = self._case_surfaces(
                reference, placement, dialog.wants_smooth, quads, normals
            )
            where = (
                f"\n{abs(offset[0]):.0f} mm {'left' if offset[0] > 0 else 'right'}  ·  "
                f"{abs(offset[1]):.0f} mm {'posterior' if offset[1] > 0 else 'anterior'}  ·  "
                f"{abs(offset[2]):.0f} mm {'inferior' if offset[2] > 0 else 'superior'} "
                f"of the {MODALITY_SHORT_LABELS.get(dialog.brain_modality, 'volume')} centre"
            )
            if others:
                where += f"  ·  {_human_count(others, 'other mask')} shown in pink"
        else:
            dialog.canvas.set_context(None, None, (1.0, 1.0), (0.0, 1.0), np.zeros(3))
        dialog.show_lesion(
            f"{target['label']}  ·  {target.get('atlasregion') or 'no region'}",
            f"{shape['volume_mm3']:.1f} mm³  ·  ø {shape['diameter_mm']:.1f} mm  ·  "
            f"{int(shape['voxel_count'])} voxels  ·  {own_faces} faces\n"
            f"Bounding box {extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} mm "
            f"(L-R × P-A × I-S)" + where,
            quads,
            normals,
            radius_mm=radius,
            tints=tints,
        )

    def _case_surfaces(
        self,
        reference: Volume,
        volume_centre: np.ndarray,
        smooth: bool,
        own_quads: np.ndarray,
        own_normals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Every segmented finding in this case, the selected one in its own
        colour and the rest in the colour they already have on the slices.

        Built from the one label volume rather than a mask per finding: an
        equality test against a 256^3 array costs about 40 ms, and a case here
        holds up to twenty-five findings.  One pass over the non-zero voxels
        groups them all instead.
        """

        selected = QColor(COLORS["roi"])
        other = QColor(COLORS["neighbour"])
        own_tint = np.repeat(
            np.array([[selected.red(), selected.green(), selected.blue()]], dtype=np.float64),
            len(own_quads),
            axis=0,
        )
        volume = self.label_volume
        if volume is None or self.selected_target is None:
            return own_quads, own_normals, own_tint, 0
        mine = self._label_value_for(self.selected_target["target_id"])
        spacing = np.asarray(reference.voxel_sizes[:3], dtype=np.float64)

        # One pass over a 256^3 label volume costs about 130 ms, and this is
        # reached from the readout -- which fires on every brush stroke.  A
        # stroke only ever touches the selected mask, so the others are kept
        # until the selection, the case or the smoothing changes.
        key = (
            self.current_case_id,
            str(self.selected_target["target_id"]),
            bool(smooth),
            # Placing the head from another sequence shifts every lesion.
            tuple(np.round(volume_centre, 3)),
        )
        cached = self._others_cache
        if cached is not None and cached[0] == key:
            return self._with_others(own_quads, own_normals, own_tint, cached[1])

        filled = np.argwhere(volume)
        if not len(filled):
            self._others_cache = (key, (np.zeros((0, 4, 3)), np.zeros((0, 3)), np.zeros((0, 3)), 0))
            return own_quads, own_normals, own_tint, 0
        values = volume[filled[:, 0], filled[:, 1], filled[:, 2]]

        quads = [own_quads]
        normals = [own_normals]
        tints = [own_tint]
        count = 0
        for value in np.unique(values):
            if int(value) == int(mine):
                continue
            coords = filled[values == value]
            low = coords.min(axis=0)
            local = np.zeros(coords.max(axis=0) - low + 1, dtype=bool)
            local[tuple((coords - low).T)] = True
            # The crop moved the origin, so the centre it is measured from has
            # to move with it or the lesion lands in the wrong place.
            piece, piece_normals = lesion_surface(
                local,
                reference.voxel_sizes,
                centre=volume_centre - low * spacing,
                smooth=LESION_SMOOTH_PASSES if smooth else 0,
            )
            if not len(piece):
                continue
            quads.append(piece)
            normals.append(piece_normals)
            tints.append(
                np.repeat(
                    np.array([[other.red(), other.green(), other.blue()]], dtype=np.float64),
                    len(piece),
                    axis=0,
                )
            )
            count += 1
        others = (
            np.concatenate(quads[1:]) if count else np.zeros((0, 4, 3)),
            np.concatenate(normals[1:]) if count else np.zeros((0, 3)),
            np.concatenate(tints[1:]) if count else np.zeros((0, 3)),
            count,
        )
        self._others_cache = (key, others)
        return self._with_others(own_quads, own_normals, own_tint, others)

    @staticmethod
    def _with_others(own_quads, own_normals, own_tint, others):
        pieces, piece_normals, piece_tints, count = others
        if not count:
            return own_quads, own_normals, own_tint, 0
        return (
            np.concatenate([own_quads, pieces]),
            np.concatenate([own_normals, piece_normals]),
            np.concatenate([own_tint, piece_tints]),
            count,
        )

    def _label_reference_modality(self) -> str | None:
        """Which sequence's grid the case's label file uses."""

        for modality in LABEL_REFERENCE_ORDER:
            if self.volumes.get(modality) is not None:
                return modality
        return None

    def _placement_centre(self, reference: Volume, brain: Volume) -> np.ndarray:
        """Where, in the mask's own millimetre space, the head's centre lies.

        The mask is on the label reference's grid; the head may be drawn from
        a different sequence with its own shape and voxel size.  Subtracting
        the reference's own centre would then put the lesion wherever the two
        grids happen to disagree.  On this dataset the three sequences share a
        grid exactly, so the correction currently comes out as zero -- it is
        insurance against a case where they do not, not a fix for a visible
        fault, and the test for it builds a shifted grid on purpose.

        The two are tied together through the world coordinates their affines
        declare: take the reference's centre, ask where that point is in the
        head's grid, and shift by the difference.  Both volumes are reoriented
        to the same display axes, so this is a translation; a real rotation
        between the two sequences would need more, and these are affine
        restored to a common space precisely so there is none.
        """

        spacing = np.asarray(reference.voxel_sizes[:3], dtype=np.float64)
        centre_voxel = np.asarray(reference.shape[:3], dtype=np.float64) / 2.0
        own_centre = centre_voxel * spacing
        if brain is reference or brain.path == reference.path:
            return own_centre
        try:
            world = voxel_to_ras(reference.affine, centre_voxel)
            in_brain = np.asarray(ras_to_voxel(brain.affine, world), dtype=np.float64)
        except Exception:
            return own_centre
        brain_spacing = np.asarray(brain.voxel_sizes[:3], dtype=np.float64)
        drift = (in_brain - np.asarray(brain.shape[:3], dtype=np.float64) / 2.0) * brain_spacing
        return own_centre - drift

    def _brain_context(self, reference: Volume):
        """The resampled cubes for this case, built once and kept.

        Two of them: 96 for while the mouse is down (22.7 ms a frame for both
        depth halves) and 128 for when it stops (54.8 ms).  Built lazily --
        most readings never open this window, and paying 40 ms of resampling
        on every case load for a feature nobody asked for is the wrong trade.
        """

        key = (self.current_case_id, reference.path)
        cached = getattr(self, "_context_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            fine, mm_fine = isotropic_context(reference.data, reference.voxel_sizes, 128)
            coarse, mm_coarse = isotropic_context(reference.data, reference.voxel_sizes, 96)
        except Exception:
            self._context_cache = (key, None)
            return None
        # A signed parameter map has no brightness, only magnitude, and its
        # mean along a ray very nearly cancels: measured on this QSM, the
        # cube's median is 0.0000 against a 1st percentile of -0.0287, and the
        # projection came out a flat grey with no head in it.  Projecting the
        # distance from the map's own zero shows the tissue instead.  A
        # magnitude image like SWI has no negatives, so it is left alone.
        if float(np.percentile(fine, 1.0)) < 0.0:
            base = float(np.median(fine))
            fine = np.abs(fine - base)
            coarse = np.abs(coarse - base)
        # Window the projection, not the cube.  A ray's mean is far below any
        # single voxel because most of the ray is air, so cube percentiles put
        # the whole head in the bottom fifth of the ramp: measured on this
        # QSM, nothing at all reached mid-grey, and on the SWI 24% did against
        # 37% here.  One probe projection at build time settles it.
        probe = project_context(coarse, np.eye(3, dtype=np.float32))
        window = (
            (float(np.percentile(probe, 40.0)), float(np.percentile(probe, 99.5)))
            if probe.size
            else (0.0, 1.0)
        )
        if window[1] - window[0] < 1e-9:
            window = (window[0], window[0] + 1.0)
        context = (fine, coarse, mm_fine, mm_coarse, window)
        self._context_cache = (key, context)
        return context

    def _update_roi_readout(self) -> None:
        self._readout_timer.stop()
        self._refresh_lesion_3d()
        volume_mm3 = self.roi_volume_mm3()
        target_id = str(self.selected_target["target_id"]) if self.selected_target else ""
        source = self.label_sources.get(target_id)
        if volume_mm3 <= 0:
            self.roi_label.setText("no segmentation")
            self.roi_label.setToolTip("")
            return
        diameter = 2.0 * (3.0 * volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0)
        origin = f" · from {MODALITY_SHORT_LABELS.get(source, source)}" if source else ""
        # How many slices it spans, because a mask that looks complete on the
        # slice in front of you can still be one slice thick.
        mask = self._selected_label_mask()
        slices = ""
        if mask is not None and mask.any():
            axis = PLANE_AXES["axial"]
            present = np.unique(np.nonzero(mask)[axis])
            slices = f" · {len(present)} slice{'' if len(present) == 1 else 's'}"
        self.roi_label.setText(f"{volume_mm3:.1f} mm³ · ø {diameter:.1f} mm{slices}{origin}")
        method = self.label_methods.get(target_id)
        settings = self.label_settings.get(target_id) or {}
        detail = [f"Method: {method}"] if method else []
        if settings:
            detail.append(
                f"Grown at sensitivity {settings['sensitivity']:g}, "
                f"cap {settings['radius_mm']:g} mm"
            )
        self.roi_label.setToolTip("\n".join(detail))

    def _write_label_volume(self) -> dict[str, Any] | None:
        """Persist the label volume and its rows; called from Save review."""

        if self.label_volume is None or self.current_case_id is None:
            return None
        reference = self._label_reference()
        if reference is None:
            return None
        path = self._label_file()
        used = np.any(self.label_volume)
        try:
            if used:
                path.parent.mkdir(parents=True, exist_ok=True)
                save_label_volume(path, self.label_volume, reference)
            elif path.exists():
                path.unlink()
        except Exception as exc:
            QMessageBox.critical(self, "Could not save the segmentation", f"{type(exc).__name__}: {exc}")
            return None
        voxel_mm3 = float(np.prod(reference.voxel_sizes))
        written = {}
        for target in self.targets:
            target_id = str(target["target_id"])
            value = self.label_values.get(target_id)
            if value is None:
                continue
            mask = self.label_volume == value
            count = int(np.count_nonzero(mask))
            if count == 0 and target_id not in self._stored_roi_targets:
                # Merely selecting a finding reserves it a label value.  That is
                # not a segmentation, and writing a delete plus a log entry for
                # it costs two SQLite connections on the GUI thread per finding.
                continue
            # The centre of the drawn voxels, so the results table says where
            # the segmentation is and not only how big it is.
            centroid = None
            if count:
                centre = np.asarray(np.nonzero(mask), dtype=float).mean(axis=1)
                centroid = tuple(float(v) for v in voxel_to_ras(reference.affine, centre))
            save_roi(
                self.db_path,
                target_id=target_id,
                case_id=self.current_case_id,
                reader_id=self.reader_id,
                review_round=self.review_round,
                label_value=value,
                path=path,
                voxel_count=count,
                volume_mm3=count * voxel_mm3,
                generated_from=self.label_sources.get(target_id),
                centroid_ras=centroid,
                session_id=self.session_id,
                method=self.label_methods.get(target_id),
                sensitivity=(self.label_settings.get(target_id) or {}).get("sensitivity"),
                radius_mm=(self.label_settings.get(target_id) or {}).get("radius_mm"),
            )
            written[target_id] = count
        self._roi_dirty = False
        return written

    # ---------------------------------------------------------------- tools --
    def segmentation_block(self) -> str | None:
        """Why this finding cannot be segmented right now, or ``None``.

        Two reasons.  The label grid may not match the sequence on screen, in
        which case a stroke would land in the wrong place.  Or this reader has
        judged the finding **not** a microbleed -- there is then nothing to
        outline, and a mask saved against a "no" is a contradiction that would
        travel all the way into the export and the agreement table.
        """

        if self._grid_problem:
            return self._grid_problem
        if self.selected_target is None:
            return "Select a finding before segmenting."
        if getattr(self, "_verdict", None) == 0:
            return (
                "Marked \u201cnot a microbleed\u201d, so there is nothing to outline. "
                "Set the verdict back to Yes or Not set to segment it."
            )
        return None

    def _update_segment_availability(self) -> None:
        # Reachable from _show_verdict, which the form clears during start-up
        # before the segment tab exists.
        if not hasattr(self, "segment_block_label"):
            return
        reason = self.segmentation_block()
        blocked = reason is not None
        for widget in (
            self.auto_roi_btn,
            self.grow_stroke_btn,
            self.sensitivity_spin,
            self.roi_radius_spin,
            self.brush_spin,
            self.brush_3d_cb,
            self.undo_roi_btn,
            self.save_segment_btn,
        ):
            widget.setEnabled(not blocked)
        for name in ("brush", "eraser"):
            button = self.tool_buttons.get(name)
            if button is not None:
                button.setEnabled(not blocked)
        # Clear is deliberately left alone.  A mask drawn before the verdict
        # changed is still the reader's to remove, and locking them out of
        # removing it is how a contradiction becomes permanent.
        self.segment_block_label.setText(reason or "")
        self.segment_block_label.setVisible(blocked)
        if blocked and self.active_tool in ("brush", "eraser"):
            self.set_tool(None)

    def set_tool(self, tool: str | None) -> None:
        """Activate a tool, or ``None`` for plain navigation."""

        if tool is not None and tool not in self.tool_buttons:
            return
        blocked = self.segmentation_block() if tool in ("brush", "eraser") else None
        if blocked:
            # Arming a brush that cannot paint anywhere is worse than saying no.
            for name, button in self.tool_buttons.items():
                button.setChecked(False)
            self.active_tool = None
            self._set_status(blocked, COLORS["warn"])
            return
        self.active_tool = tool
        for name, button in self.tool_buttons.items():
            if button.isChecked() != (name == tool):
                button.setChecked(name == tool)
        for panel in self.view_panels.values():
            panel.canvas.set_pick_enabled(tool == "point")
            panel.canvas.set_paint_mode(
                {"brush": "paint", "eraser": "erase"}.get(tool)
            )
        if tool in ("brush", "eraser"):
            # The controls for the tool just picked are one tab away.
            self.show_panel_tab("segment")
            if not can_segment(self.current_modality):
                usable = [
                    MODALITY_LABELS[key]
                    for key in MODALITY_BUTTON_ORDER
                    if can_segment(key) and self.volumes.get(key) is not None
                ]
                self._set_status(
                    f"{MODALITY_LABELS.get(self.current_modality, 'This sequence')} "
                    "cannot be segmented"
                    + (f"; switch to {' or '.join(usable)}." if usable else "."),
                    COLORS["warn"],
                )
            elif tool == "brush":
                self._set_status(
                    "Brush: drag with the left button to paint. Right-drag is still contrast.",
                    COLORS["accent"],
                )
            else:
                self._set_status(
                    "Eraser: drag with the left button to rub out. Right-drag is still contrast.",
                    COLORS["accent"],
                )
        elif tool == "point":
            self._set_status(
                "Point tool: click in a view to place the crosshair, then use "
                "Move here or Add microbleed.",
                COLORS["accent"],
            )
        else:
            self._set_status("Navigation only; clicks no longer move the crosshair.", COLORS["dim"])

    # ---------------------------------------------------------- review tabs --
    def current_panel_tab(self) -> str:
        index = self.panel_tabs.currentIndex()
        if 0 <= index < len(self._panel_tab_keys):
            return self._panel_tab_keys[index]
        return ""

    def show_panel_tab(self, key: str) -> None:
        if key in self._panel_tab_keys:
            self.panel_tabs.setCurrentIndex(self._panel_tab_keys.index(key))

    def save_and_segment(self) -> bool:
        """Save the verdict, then pick up the brush on the same finding.

        The segmentation panel is always on screen now, so this no longer has
        to reveal anything -- what it saves the reader is the reach for the
        toolbar between deciding and drawing.
        """

        blocked = self.segmentation_block()
        if blocked:
            self._set_status(blocked, COLORS["warn"])
            return False
        if not self.save_current_review(advance=False):
            return False
        self.show_panel_tab("segment")
        self.set_tool("brush")
        return True

    def unconfirmed_segmentations(self) -> list[dict[str, Any]]:
        """Findings of this case with a mask but no verdict from this reader.

        Drawing without deciding is work that will not reach the results: the
        results table is keyed on the verdict, so a mask beside "not set" is
        invisible to the analysis.  Doing nothing at all is not worth
        mentioning -- moving through cases to look at them is normal.
        """

        pending: list[dict[str, Any]] = []
        for target in self.targets:
            target_id = str(target["target_id"])
            decided = target.get("reader_verify") is not None or str(
                target.get("reader_comment") or ""
            ).strip()
            if decided:
                continue
            drawn = bool(target.get("roi"))
            if not drawn and self.label_volume is not None:
                value = self.label_values.get(target_id)
                drawn = bool(value and np.any(self.label_volume == value))
            if drawn:
                pending.append(target)
        return pending

    def _confirm_unconfirmed_segmentations(self) -> bool:
        """Ask before leaving a case whose masks carry no verdict."""

        pending = self.unconfirmed_segmentations()
        if not pending:
            return True
        names = ", ".join(str(target["label"]) for target in pending[:4])
        if len(pending) > 4:
            names += f" and {len(pending) - 4} more"
        prompt = QMessageBox(self)
        prompt.setWindowTitle("Segmented but not decided")
        prompt.setText(
            f"{_human_count(len(pending), 'finding')} in {self.current_case_id} "
            "has a segmentation but no verdict."
        )
        prompt.setInformativeText(
            f"{names}\n\nThe results are keyed on the verdict, so a mask without one "
            "does not reach them. Go back and record yes or no?"
        )
        back = prompt.addButton("Go back", QMessageBox.ButtonRole.AcceptRole)
        prompt.addButton("Leave it", QMessageBox.ButtonRole.DestructiveRole)
        prompt.exec()
        if prompt.clickedButton() is back:
            index = next(
                (
                    position
                    for position, target in enumerate(self.targets)
                    if target["target_id"] == pending[0]["target_id"]
                ),
                -1,
            )
            if index >= 0:
                self._select_target_index(index, confirm=False)
            self.show_panel_tab("review")
            return False
        return True

    def _toggle_roi_overlay(self, visible: bool) -> None:
        for panel in self.view_panels.values():
            panel.canvas.set_show_labels(bool(visible))

    def _set_brush_radius(self, millimetres: float) -> None:
        for panel in self.view_panels.values():
            panel.canvas.set_brush_radius(float(millimetres))

    def _set_brush_3d(self, spherical: bool) -> None:
        for panel in self.view_panels.values():
            panel.canvas.set_brush_3d(bool(spherical))

    def _set_roi_outline(self, outline: bool) -> None:
        self.settings.set_roi_outline(bool(outline))
        for panel in self.view_panels.values():
            panel.canvas.set_label_outline(bool(outline))

    def step_brush_radius(self, delta: float) -> None:
        """Resize the brush from the keyboard, without leaving the image.

        Reaching for the spin box means looking away from the lesion, which is
        the one thing a reader correcting a mask voxel by voxel cannot do.
        """

        if self.active_tool not in ("brush", "eraser"):
            self._set_status("Pick the Brush or the Eraser first.", COLORS["dim"])
            return
        self.brush_spin.setValue(self.brush_spin.value() + float(delta))
        self._set_status(f"Brush radius {self.brush_spin.value():.1f} mm", COLORS["accent"])

    def toggle_point_tool(self) -> None:
        self.set_tool(None if self.active_tool == "point" else "point")

    def clear_picked_position(self) -> None:
        """Drop a picked position and put the cursor back on the finding.

        Right-clicking (without dragging) or pressing Escape undoes a pick that
        landed in the wrong place, without having to pick again or save.
        """

        if self.marker_ras is None:
            return
        if self.target_ras == self.marker_ras:
            self._set_status("Nothing to cancel; the crosshair is on the finding.", COLORS["dim"])
            return
        self.target_ras = self.marker_ras
        self._set_coordinate_spins(self.target_ras)
        self._set_manual_spins(self.target_ras)
        self._apply_target_to_views(recenter=True)
        self._apply_zoom_preference()
        self._set_status("Cancelled the picked position.", COLORS["dim"])

    # ------------------------------------------------------------- position --
    def _variant_label(self, variant: dict[str, Any], *, compact: bool = False) -> str:
        if variant["key"] == "source":
            return "Source"
        moved = variant.get("moved_mm")
        suffix = "" if compact else " from source"
        distance = f" · {moved:.1f} mm{suffix}" if moved else ""
        if variant["key"] == self.reader_id:
            return f"Mine{distance}"
        return f"{variant['reader_id']}{distance}"

    def _rebuild_position_variants(self, target: dict[str, Any] | None) -> None:
        """Refresh the position choices for the finding under review."""

        self.position_variants = list(target.get("position_variants") or []) if target else []
        if target and self.pending_ras is not None:
            # An unsaved "Move here" replaces my saved position in the list.
            mine = next(
                (item for item in self.position_variants if item["key"] == self.reader_id), None
            )
            source_ras = tuple(target.get("source_ras") or target["ras"])
            entry = {
                "key": self.reader_id,
                "reader_id": self.reader_id,
                "review_round": self.review_round,
                "ras": self.pending_ras,
                "moved_mm": _distance_mm(source_ras, self.pending_ras),
                "updated_at": None,
                "pending": True,
            }
            if mine is None:
                self.position_variants.append(entry)
            else:
                self.position_variants[self.position_variants.index(mine)] = entry
        self._refresh_position_combo()

    def _refresh_position_combo(self) -> None:
        self.position_combo.blockSignals(True)
        self.position_combo.clear()
        for variant in self.position_variants:
            label = self._variant_label(variant)
            if variant.get("pending"):
                label += "  (unsaved)"
            self.position_combo.addItem(label, variant["key"])
        index = self.position_combo.findData(self.selected_variant)
        if index < 0:
            index = 0
            self.selected_variant = (
                self.position_variants[0]["key"] if self.position_variants else "source"
            )
        self.position_combo.setCurrentIndex(index)
        self.position_combo.blockSignals(False)
        self.position_combo.setEnabled(len(self.position_variants) > 1)
        self.move_here_btn.setEnabled(self.selected_target is not None)
        self._update_position_hint()

    def _update_position_hint(self) -> None:
        """Say what saving would do, without treating browsing as an edit."""

        variant = self._current_variant()
        if variant is None or self.selected_target is None:
            self.position_hint.setText("")
            return
        saved = self.selected_target.get("reader_ras")
        shown = tuple(float(value) for value in variant["ras"])
        if variant.get("pending"):
            text = "unsaved move"
        elif variant["key"] == "source":
            text = "saving clears yours" if saved is not None else ""
        elif saved is None or (_distance_mm(saved, shown) or 0.0) > 1e-6:
            text = "saving adopts this"
        else:
            text = ""
        self.position_hint.setText(text)

    def _current_variant(self) -> dict[str, Any] | None:
        return next(
            (item for item in self.position_variants if item["key"] == self.selected_variant),
            None,
        )

    def _on_variant_selected(self, index: int) -> None:
        if index < 0 or index >= len(self.position_variants):
            return
        variant = self.position_variants[index]
        # No early return on an unchanged key: the shown position and the key
        # can disagree right after a case loads, and the click has to fix that.
        self.selected_variant = str(variant["key"])
        self._show_selected_variant()
        # Looking at another reader's position is not an edit, so it must not
        # mark the review dirty; the hint next to the selector says what saving
        # would do instead of nagging on the way out.
        self._update_position_hint()
        if variant["key"] == "source":
            message = "Showing the source position."
        elif variant["key"] == self.reader_id:
            message = "Showing your position for this finding."
        else:
            message = (
                f"Showing {variant['reader_id']}'s position. "
                "Saving would record it as yours."
            )
        self._set_status(message, COLORS["accent"])

    def _show_selected_variant(self, *, recenter: bool = True) -> None:
        variant = self._current_variant()
        if variant is None:
            return
        ras = tuple(float(value) for value in variant["ras"])
        self.marker_ras = ras
        if recenter:
            self.target_ras = ras
            self._set_coordinate_spins(ras)
        self._apply_target_to_views(recenter=recenter)
        self._apply_zoom_preference()
        self._update_ghost_markers()

    def _update_ghost_markers(self) -> None:
        """Draw the source position faintly whenever a correction is shown."""

        volume = self.volumes.get(self.current_modality)
        source = next((item for item in self.position_variants if item["key"] == "source"), None)
        ghost_voxel = None
        if (
            volume is not None
            and source is not None
            and self.selected_variant != "source"
        ):
            try:
                ghost_voxel = ras_to_voxel(volume.affine, source["ras"])
            except Exception:
                ghost_voxel = None
        for panel in self.view_panels.values():
            panel.canvas.set_ghost_voxel(ghost_voxel)

    def move_finding_here(self) -> None:
        """Record the crosshair position as this reader's correction."""

        if self.selected_target is None:
            self._set_status("Select a finding before moving it.", COLORS["warn"])
            return
        if self.target_ras is None:
            self._set_status("Click the true position in a view first.", COLORS["warn"])
            return
        chosen = tuple(float(value) for value in self.target_ras)
        snapped_by = 0.0
        if self.settings.snap_to_lesion:
            refined = self._snapped_ras(chosen)
            if refined is not None:
                snapped_by = _distance_mm(chosen, refined) or 0.0
                chosen = refined
                self.target_ras = chosen
                self._set_coordinate_spins(chosen)
                self._apply_target_to_views(recenter=False)
        source_ras = tuple(
            self.selected_target.get("source_ras") or self.selected_target["ras"]
        )
        distance = _distance_mm(source_ras, chosen) or 0.0
        if distance < 1e-6:
            self.selected_variant = "source"
            self.pending_ras = None
        else:
            self.pending_ras = chosen
            self.selected_variant = self.reader_id
        self._rebuild_position_variants(self.selected_target)
        self._show_selected_variant(recenter=False)
        self._mark_review_dirty()
        refined = f" (snapped {snapped_by:.1f} mm onto the focus)" if snapped_by >= 0.05 else ""
        self._set_status(
            f"Moved {self.selected_target['label']} {distance:.1f} mm from the source "
            f"position{refined}. Save to record it.",
            COLORS["accent"],
        )

    def _snapped_ras(self, ras: tuple[float, float, float]) -> tuple[float, float, float] | None:
        """The nearest local intensity extremum to a coordinate, in RAS.

        A microbleed is dark on SWI and bright on QSM; the MIP smears it along
        the projection direction, so snapping there would move the point onto
        an artefact and it is left alone.
        """

        modality = self.current_modality
        volume = self.volumes.get(modality)
        if volume is None or not can_segment(modality):
            # Snapping on a projection would move the point onto an artefact.
            return None
        try:
            voxel = ras_to_voxel(volume.affine, ras)
        except (ValueError, TypeError, np.linalg.LinAlgError):
            # A coordinate this affine cannot express; nothing to refine.
            return None
        snapped = snap_to_extremum(
            volume.data, voxel, volume.voxel_sizes,
            dark=modality != "qsm", radius_mm=self.settings.snap_radius_mm,
        )
        return tuple(float(value) for value in voxel_to_ras(volume.affine, snapped))

    def _distance_to_focus(self, ras: tuple[float, float, float] | None) -> float | None:
        """How far a coordinate sits from anything focal, in millimetres.

        Measured here because the volume is already in memory; recomputing it
        at export time would mean re-reading every NIfTI in the study.
        """

        if ras is None:
            return None
        snapped = self._snapped_ras(ras)
        if snapped is None:
            return None
        return _distance_mm(ras, snapped)

    def _variant_markers_for_peek(self) -> list[tuple[np.ndarray, str, str]]:
        volume = self.volumes.get(self.current_modality)
        if volume is None:
            return []
        markers: list[tuple[np.ndarray, str, str]] = []
        for position, variant in enumerate(self.position_variants):
            try:
                voxel = ras_to_voxel(volume.affine, variant["ras"])
            except Exception:
                continue
            colour = (
                COLORS["variant_source"]
                if variant["key"] == "source"
                else VARIANT_COLORS[position % len(VARIANT_COLORS)]
            )
            markers.append((voxel, colour, self._variant_label(variant, compact=True)))
        return markers

    def set_variant_peek(self, visible: bool) -> None:
        """Hold-to-compare: show every recorded position at once."""

        visible = bool(visible) and len(self.position_variants) > 1
        markers = self._variant_markers_for_peek() if visible else []
        for panel in self.view_panels.values():
            panel.canvas.set_variant_markers(markers)
            panel.canvas.set_show_variants(visible)
        if visible:
            self._set_status(
                f"Showing all {len(self.position_variants)} recorded positions.", COLORS["accent"]
            )

    # ------------------------------------------------------ window / level --
    def _on_window_changed(self, plane: str, low: float, high: float) -> None:
        """Mirror one view's window to the other two and remember it."""

        if self._applying_window:
            return
        self._applying_window = True
        try:
            for name, panel in self.view_panels.items():
                if name != plane:
                    panel.canvas.set_window_limits(low, high, notify=False)
            canvas = self.view_panels[plane].canvas
            level, window = canvas.window_level
            if canvas.window_limits == canvas._auto_window:
                self._window_levels.pop(self.current_modality, None)
            else:
                self._window_levels[self.current_modality] = (level, window)
            if self.contrast_dialog is not None and self.contrast_dialog.isVisible():
                self.contrast_dialog.show_values(level, window)
        finally:
            self._applying_window = False

    def _apply_window_preference(self) -> None:
        """Restore a remembered window for the sequence now on screen."""

        remembered = self._window_levels.get(self.current_modality)
        if remembered is None:
            return
        self._applying_window = True
        try:
            for panel in self.view_panels.values():
                panel.canvas.set_window_level(*remembered, notify=False)
        finally:
            self._applying_window = False

    def set_window_level(self, level: float, window: float) -> None:
        self._applying_window = True
        try:
            for panel in self.view_panels.values():
                panel.canvas.set_window_level(level, window, notify=False)
        finally:
            self._applying_window = False
        self._window_levels[self.current_modality] = (float(level), float(window))
        if self.contrast_dialog is not None and self.contrast_dialog.isVisible():
            self.contrast_dialog.show_values(level, window)

    def reset_window_level(self) -> None:
        self._applying_window = True
        try:
            for panel in self.view_panels.values():
                panel.canvas.reset_window(notify=False)
        finally:
            self._applying_window = False
        self._window_levels.pop(self.current_modality, None)
        self._refresh_contrast_dialog()
        self._set_status(
            f"{MODALITY_LABELS[self.current_modality]} contrast back to automatic.", COLORS["dim"]
        )

    def open_contrast_dialog(self) -> None:
        if self.contrast_dialog is None:
            dialog = ContrastDialog(self)
            dialog.valuesChanged.connect(self.set_window_level)
            dialog.resetRequested.connect(self.reset_window_level)
            dialog.stickyChanged.connect(self._set_sticky_window)
            self.contrast_dialog = dialog
        self._refresh_contrast_dialog()
        self.contrast_dialog.show()
        self.contrast_dialog.raise_()
        self.contrast_dialog.activateWindow()

    def _refresh_contrast_dialog(self) -> None:
        if self.contrast_dialog is None:
            return
        canvas = self.view_panels["axial"].canvas
        level, window = canvas.window_level
        _auto_level, auto_window = canvas.auto_window_level
        self.contrast_dialog.configure(
            MODALITY_LABELS[self.current_modality],
            level,
            window,
            auto_window,
            self.settings.sticky_window,
        )

    def _set_sticky_window(self, sticky: bool) -> None:
        self.settings.set_sticky_window(bool(sticky))
        if not sticky:
            self._window_levels = {}
        self._set_status(
            "Manual contrast will be kept when changing case."
            if sticky
            else "Contrast returns to automatic on the next case.",
            COLORS["dim"],
        )

    def toggle_maximized_view(self, plane: str | None) -> None:
        """Show a single plane full size, or restore the four-cell workspace."""

        if plane is None:
            plane = self._hovered_plane() or self._active_plane
        if plane not in self.view_panels:
            return
        target = None if self._maximized_plane == plane else plane
        self._maximized_plane = target
        for name, panel in self.view_panels.items():
            panel.setVisible(target is None or name == target)
            panel.maximize_btn.setChecked(target == name)
            panel.maximize_btn.setToolTip(
                "Restore the three-view workspace (F)" if target == name else "Show only this view (F)"
            )
        self.location_panel.setVisible(target is None)
        self._set_grid_stretch(target)
        if target is not None:
            self._active_plane = target

    def _hovered_plane(self) -> str | None:
        for name, panel in self.view_panels.items():
            if panel.isVisible() and panel.canvas.underMouse():
                return name
        return None

    def open_settings(self, *, tab: int = 0) -> None:
        before_orientation = self.settings.orientation
        before_shortcuts = self.settings.shortcuts()
        dialog = SettingsDialog(self.settings, self)
        tabs = dialog.findChild(QTabWidget)
        if tabs is not None:
            tabs.setCurrentIndex(int(tab))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        for panel in self.view_panels.values():
            panel.canvas.set_lesion_fov(self.settings.lesion_fov_mm)
            panel.canvas.set_smooth_zoom(self.settings.smooth_zoom)
        self._update_save_buttons()
        if self.settings.shortcuts() != before_shortcuts:
            self._bind_shortcuts()
        else:
            self._refresh_shortcut_hints()
        if self.settings.orientation != before_orientation:
            # Reorientation happens while a volume is read, so the case has to
            # be loaded again for the new display preset to take effect.
            self._set_status(
                f"Display preset: {orientation_summary(self.settings.orientation)} · reloading the case…",
                COLORS["accent"],
            )
            if self.current_case_id:
                self.load_case(self.current_case_id, force=True)
            else:
                self.case_status.setText(self._case_status_text())
            return
        if self.settings.auto_zoom:
            self._apply_zoom_preference()
        self._set_status("Preferences updated.", COLORS["success"])

    # --------------------------------------------------------------- export --
    def export_reviews(self, out_path: Path | None = None) -> Path | None:
        """Write the analysis table for this dataset."""

        if out_path is None:
            suggested = Path(self.dataset.review_db).with_name(
                f"{Path(self.dataset.workbook).stem}_reviews.xlsx"
            )
            chosen, _filter = QFileDialog.getSaveFileName(
                self,
                "Export review results",
                str(suggested),
                "Excel workbook (*.xlsx);;CSV file (*.csv)",
            )
            if not chosen:
                return None
            out_path = Path(chosen)
        try:
            result = export_reviews(self.db_path, out_path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not export the reviews", f"{type(exc).__name__}: {exc}")
            return None
        self._set_status(
            f"Exported {result['findings']} findings from {result['readers']} "
            f"{'reader' if result['readers'] == 1 else 'readers'} to {Path(result['path']).name}"
            + (f" · {result['disagreements']} disagreements" if result["disagreements"] else ""),
            COLORS["success"],
        )
        self._log_event(
            "reviews_exported",
            session_id=self.session_id,
            reader_id=self.reader_id,
            review_round=self.review_round,
            details=result,
        )
        return Path(result["path"])

    # -------------------------------------------------------------- dataset --
    # ---------------------------------------------------------- queue column --
    def _fit_queue_to_screen(self) -> None:
        """Fold the case queue when the screen cannot seat it.

        Four columns want 1680px with the queue open.  On a 1366 or 1600 wide
        display that is more than there is, and Qt would answer by making the
        window wider than the screen -- so the queue folds instead, which is
        exactly the escape it exists for.  It comes back on its own if the
        reader plugs in a wider screen, unless they folded it themselves.
        """

        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry().width()
        if self.queue_pinned and available < QUEUE_PINNED_MIN_WIDTH:
            self.set_queue_pinned(False, remember=False)
            self._queue_auto_folded = True
            self._set_status(
                f"This screen is {available}px wide; the case queue needs "
                f"{QUEUE_PINNED_MIN_WIDTH} to sit beside the images, so it is folded. "
                "Hover the strip to look at it.",
                COLORS["dim"],
            )
        elif self._queue_auto_folded and available >= QUEUE_PINNED_MIN_WIDTH:
            self._queue_auto_folded = False
            if self.settings.queue_pinned:
                self.set_queue_pinned(True, remember=False)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        # Once, when the window first knows which screen it is on.
        if not self._checked_screen_width:
            self._checked_screen_width = True
            QTimer.singleShot(0, self._fit_queue_to_screen)

    def moveEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().moveEvent(event)
        # Dragged to another monitor, which may be a different width.
        if self._checked_screen_width:
            self._fit_queue_to_screen()

    def _build_queue_rail(self) -> QWidget:
        """The strip the case queue folds down to.

        Hovering it brings the queue back as an overlay rather than by widening
        the column, because pushing the images aside on every pass of the mouse
        would re-fit and re-centre them each time -- the queue is worth a look,
        not a reflow.
        """

        rail = QFrame()
        rail.setObjectName("Card")
        rail.setFixedWidth(26)
        rail.setCursor(Qt.CursorShape.PointingHandCursor)
        rail.setToolTip("Case queue — hover to look, click to keep it open")
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(2, 6, 2, 6)
        layout.setSpacing(6)
        pin = QPushButton()
        pin.setObjectName("IconButton")
        pin.setIcon(_stroke_icon("expand-right", 16))
        pin.setIconSize(QSize(16, 16))
        pin.setToolTip("Keep the case queue open")
        pin.clicked.connect(lambda _checked=False: self.set_queue_pinned(True))
        layout.addWidget(pin)
        self.queue_rail_caption = VerticalLabel("CASE QUEUE")
        font = self.queue_rail_caption.font()
        font.setPointSize(8)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.0)
        self.queue_rail_caption.setFont(font)
        layout.addWidget(self.queue_rail_caption, 1, Qt.AlignmentFlag.AlignHCenter)
        rail.installEventFilter(self)
        self.queue_rail_pin = pin
        return rail

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        """Show the folded queue while the mouse is on it, hide it after."""

        if watched is getattr(self, "target_list", None):
            if event.type() == QEvent.Type.Resize:
                self._fit_finding_rows()
        elif watched is getattr(self, "queue_rail", None):
            if event.type() == QEvent.Type.Enter and not self.queue_pinned:
                self._show_queue_overlay(True)
        elif watched is getattr(self, "queue_panel", None):
            if event.type() == QEvent.Type.Leave and not self.queue_pinned:
                # Leaving into a menu the panel itself opened is not leaving.
                if QApplication.activePopupWidget() is None:
                    self._show_queue_overlay(False)
        return super().eventFilter(watched, event)

    def set_right_column_width(self, width: int) -> None:
        """Put the splitter handle where the reader last left it."""

        width = max(RIGHT_COLUMN_MIN_WIDTH, min(RIGHT_COLUMN_MAX_WIDTH, int(width)))
        total = max(self.body_split.width(), width + 200)
        self.body_split.setSizes([total - width, width])
        self._fit_finding_rows()

    def set_queue_pinned(self, pinned: bool, *, remember: bool = True) -> None:
        """Keep the case queue open, or fold it down to its strip.

        The window's own minimum width follows: open, the four columns need
        1680px and Qt will not let the window be smaller; folded, 1340 is
        enough.  Declaring one number for both would either forbid a size that
        works perfectly well or promise one the layout cannot honour -- the
        old fixed 1340 was the second kind, and Qt quietly refused to make the
        window that small.

        ``remember`` is false when the window folded the queue itself because
        the screen is too narrow: that is not the reader changing their mind.
        """

        self.queue_pinned = bool(pinned)
        if remember:
            self.settings.set_queue_pinned(self.queue_pinned)
            self._queue_auto_folded = False
        self.setMinimumWidth(
            QUEUE_PINNED_MIN_WIDTH if self.queue_pinned else MINIMUM_WINDOW_WIDTH
        )
        self.queue_rail.setVisible(not self.queue_pinned)
        layout = self.centralWidget().layout()
        if self.queue_pinned:
            if layout.indexOf(self.queue_panel) < 0:
                layout.insertWidget(1, self.queue_panel)
            self.queue_panel.setMinimumWidth(QUEUE_COLUMN_WIDTH)
            self.queue_panel.setMaximumWidth(QUEUE_COLUMN_WIDTH + 60)
            self.queue_panel.show()
        else:
            # Taking it out of the layout is what gives the width back; hiding
            # alone leaves an item that reclaims the space the moment the
            # overlay shows it again.
            layout.removeWidget(self.queue_panel)
            self.queue_panel.setParent(self.centralWidget())
            self.queue_panel.hide()

    def _show_queue_overlay(self, visible: bool) -> None:
        """Float the folded queue over the images, anchored to the rail."""

        if self.queue_pinned:
            return
        if not visible:
            self.queue_panel.hide()
            return
        central = self.centralWidget()
        central.layout().removeWidget(self.queue_panel)
        self.queue_panel.setParent(central)
        rail = self.queue_rail.geometry()
        self.queue_panel.setMinimumWidth(QUEUE_COLUMN_WIDTH)
        self.queue_panel.setMaximumWidth(QUEUE_COLUMN_WIDTH)
        self.queue_panel.setGeometry(
            rail.right() + 1, rail.top(), QUEUE_COLUMN_WIDTH, rail.height()
        )
        self.queue_panel.show()
        self.queue_panel.raise_()

    def switch_session(self) -> bool:
        """Change reader or review round without leaving the application.

        Reopening the dataset the window is already on runs the same reader and
        round chooser as startup, so handing the workstation to the other
        reader, or beginning a second round, costs a dialog instead of a
        restart.  Nothing is torn down until the new session exists.
        """

        if not self._confirm_dirty():
            return False
        if not self._confirm_unconfirmed_segmentations():
            return False
        return self.switch_dataset(self.dataset)

    def change_dataset(self) -> bool:
        """Open another study, ending the current review session cleanly."""

        if not self._confirm_dirty():
            return False
        dialog = DatasetDialog(self.settings, self.dataset, self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.dataset is None:
            return False
        dataset = dialog.dataset
        if dataset == self.dataset:
            self._set_status("That dataset is already open.", COLORS["dim"])
            return False
        return self.switch_dataset(dataset)

    def switch_dataset(self, dataset: Dataset, session: dict[str, Any] | None = None) -> bool:
        """Point the window at another dataset.

        Nothing is torn down until the new store opens and a session is
        established there, so a failure leaves the current review untouched.
        ``session`` skips the reader prompt when the caller already has one.
        """

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            report = initialize_store(dataset.workbook, dataset.data_root, dataset.review_db)
        except SourceReadError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Could not open that dataset", str(exc))
            return False
        except Exception as exc:  # pragma: no cover - filesystem-specific
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Could not open that dataset", f"{type(exc).__name__}: {exc}")
            return False
        QApplication.restoreOverrideCursor()

        if session is None:
            reader_dialog = ReaderDialog(
                dataset.review_db,
                self,
                initial_reader=self.reader_id,
                dataset=dataset,
                settings=self.settings,
            )
            if reader_dialog.exec() != QDialog.DialogCode.Accepted or reader_dialog.session is None:
                return False
            session = reader_dialog.session

        # The old session is only closed once the new one exists.
        self._slice_log_timer.stop()
        self._flush_slice_log()
        self._save_session_state()
        self._log_event(
            "session_closed",
            session_id=self.session_id,
            reader_id=self.reader_id,
            review_round=self.review_round,
            case_id=self.current_case_id,
            details={"reason": "dataset_switched"},
        )
        try:
            close_session(self.db_path, self.session_id)
        except Exception:
            pass
        # Queued writes belong to the database we are leaving, so they have to
        # land before ``db_path`` changes underneath them.
        self._writer.stop()
        self._writer = DatabaseWriter(self)
        self._writer.failed.connect(self._on_write_failed)
        self._writer.start()

        self.dataset = dataset
        self.db_path = Path(dataset.review_db)
        self.data_root = Path(dataset.data_root)
        self.session = dict(session)
        self.reader_id = str(session["reader_id"])
        self.review_round = int(session["review_round"])
        self.session_id = str(session["session_id"])
        self.settings.remember_dataset(dataset)

        self._prefetch_timer.stop()
        self._stop_prefetch()
        self.volume_cache.clear()
        self._window_levels = {}
        self.current_case = None
        self.current_case_id = None
        self.targets = []
        self.selected_target = None
        self.target_ras = None
        self.marker_ras = None
        self.volumes = {modality: None for modality in MODALITY_ORDER}
        self.load_errors = {}
        self._review_dirty = False
        self._pending_slice_log = None
        self._clear_review_form()
        self._populate_target_list()
        self._set_loading_placeholders()
        self.setWindowTitle(f"{APP_TITLE} · {self.reader_id} · round {self.review_round}")
        self.session_label.setText(f"{self.reader_id} · round {self.review_round}")
        self.session_label.setToolTip(
            f"{self.reader_id} · round {self.review_round}\n"
            "Reader reports are shared with every reader of this datasheet"
        )
        self.more_btn.setToolTip(self._dataset_tooltip())
        self.case_title.setText("No case loaded")
        self.case_status.setText(f"{dataset.name} · select a case from the queue")
        self._reload_case_list()
        self._log_event(
            "dataset_opened",
            session_id=self.session_id,
            reader_id=self.reader_id,
            review_round=self.review_round,
            details={**dataset.as_dict(), "cases": report.get("case_count"), "findings": report.get("source_count")},
        )
        self._set_status(
            f"Opened {dataset.name} · {report.get('case_count', 0)} cases · "
            f"{report.get('source_count', 0)} findings",
            COLORS["success"],
        )
        if self.visible_cases:
            self.load_case(self.visible_cases[0]["case_id"])
        return True

    def _dataset_tooltip(self) -> str:
        return (
            f"Current dataset: {self.dataset.name}\n"
            f"Workbook: {self.dataset.workbook}\n"
            f"MRI folder: {self.dataset.data_root}\n"
            f"Reviews: {self.dataset.review_db}\n\n"
            "Click to open a different dataset."
        )

    def _update_save_buttons(self) -> None:
        """Keep the tooltips honest about where the keyboard lands.

        Each button is its own destination now -- stay, draw, move on -- so
        relabelling one of them when the preference changes would leave two
        buttons claiming the same thing.  What the preference still decides is
        where Ctrl+S goes, so the shortcut is shown on whichever button does
        the same thing.
        """

        advances = self.settings.save_advances
        key = self._shortcut_text("save_review")
        for button in (self.save_next_btn, self.segment_next_btn):
            button.setToolTip(
                "Save and move to the next finding, or the next case"
                + (f"  ({key})" if advances else "")
            )
        for button in (self.save_review_btn, self.segment_save_btn):
            button.setToolTip(
                "Save and stay on this finding" + (f"  ({key})" if not advances else "")
            )


    # -------------------------------------------------------- finding steps --
    def step_finding(self, delta: int) -> None:
        """Move to the neighbouring finding of the current case."""

        if not self.targets:
            return
        if self.selected_target is None:
            self._select_target_index(0, confirm=True)
            return
        try:
            current = next(
                index
                for index, target in enumerate(self.targets)
                if target["target_id"] == self.selected_target["target_id"]
            )
        except StopIteration:
            current = 0
        new_index = current + int(delta)
        if new_index < 0 or new_index >= len(self.targets):
            self._set_status(
                "First finding of this case." if new_index < 0 else "Last finding of this case.",
                COLORS["dim"],
            )
            return
        self._select_target_index(new_index, confirm=True)

    def _update_finding_buttons(self) -> None:
        count = len(self.targets)
        index = -1
        if self.selected_target is not None:
            index = next(
                (
                    position
                    for position, target in enumerate(self.targets)
                    if target["target_id"] == self.selected_target["target_id"]
                ),
                -1,
            )
        self.prev_finding_btn.setEnabled(index > 0)
        self.next_finding_btn.setEnabled(0 <= index < count - 1)
        if count == 0:
            self.target_count_label.setText("no findings")
        elif index >= 0:
            self.target_count_label.setText(f"{index + 1} of {count}")
        else:
            self.target_count_label.setText(_human_count(count, "finding"))

    # ----------------------------------------------------------- review form --
    VERDICT_KEYS = {1: "yes", 0: "no", None: "unset"}

    def _show_verdict(self, verify: Any) -> None:
        value = 1 if verify in (1, "1") else 0 if verify in (0, "0") else None
        self._verdict = value
        self.verdict_segments.set_current_key(self.VERDICT_KEYS[value])
        colour = {1: COLORS["success"], 0: COLORS["danger"]}.get(value, COLORS["dim"])
        self.verdict_summary.setText(_verification_text(value))
        self.verdict_summary.setStyleSheet(f"color: {colour};")
        self._update_segment_availability()

    def _on_verdict_selected(self, key: str) -> None:
        value = 1 if key == "yes" else 0 if key == "no" else None
        if value == self._verdict:
            return
        self._verdict = value
        self._update_segment_availability()
        self._mark_review_dirty()

    def set_verdict(self, value: int | None) -> None:
        """Record a verdict from the keyboard."""

        if self.selected_target is None:
            self._set_status("Select a finding before recording a verdict.", COLORS["warn"])
            return
        self._show_verdict(value)
        self._mark_review_dirty()
        self._set_status(
            f"Verdict: {_verification_text(value)} · Ctrl+S to save"
            + (" and continue" if self.settings.save_advances else ""),
            COLORS["accent"],
        )

    def _set_combo_value(self, combo, value: Any) -> None:
        index = combo.findData(str(value or ""))
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _clear_review_form(self) -> None:
        self._updating_form = True
        self._show_verdict(None)
        self._set_combo_value(self.certainty_combo, "")
        self._set_combo_value(self.mimic_combo, "")
        self.comment_edit.clear()
        self.source_report.clear()
        self.reports_browser.clear()
        self.source_summary.setText("—")
        self.position_hint.setText("")
        self._updating_form = False
        self._review_dirty = False
        self._update_dirty_indicator()
        self._update_finding_buttons()

    def _load_review_form(self, target: dict[str, Any]) -> None:
        self._updating_form = True
        self._show_verdict(target.get("reader_verify"))
        self._set_combo_value(self.certainty_combo, target.get("reader_certainty"))
        self._set_combo_value(self.mimic_combo, target.get("reader_mimic"))
        self.comment_edit.setPlainText(str(target.get("reader_comment") or ""))
        self._updating_form = False
        self._review_dirty = False
        self._update_dirty_indicator()
        self._update_finding_buttons()
        self._update_roi_readout()
        # One line of source context, so the reader does not have to open the
        # Reports section for the facts that matter while deciding.
        summary = [str(target.get("atlasregion") or "no region recorded")]
        if str(target.get("origin", "Source")) == "Manual":
            summary.append(f"added manually by {target.get('created_by') or 'a reader'}")
        else:
            summary.append(f"source: {_verification_text(target.get('source_verify')).lower()}")
        if target.get("source_readers"):
            summary.append(str(target["source_readers"]))
        if str(target.get("source_need_adjudicate") or "").strip():
            summary.append("needs adjudication")
        if str(target.get("source_comments") or "").strip():
            summary.append("has source comment")
        reports = target.get("reader_reports") or []
        others = [report for report in reports if report.get("reader_id") != self.reader_id]
        if others:
            summary.append(_human_count(len(others), "other reader report"))
        self.source_summary.setText("  ·  ".join(summary))
        source_lines = [
            f"Origin: {target.get('origin', 'Source')}",
            f"RAS: ({float(target['ras'][0]):.5f}, {float(target['ras'][1]):.5f}, {float(target['ras'][2]):.5f})",
            f"Source verified: {_verification_text(target.get('source_verify'))}",
            f"Source readers: {target.get('source_readers') or 'Not recorded'}",
            f"Need adjudicate: {target.get('source_need_adjudicate') or 'Not recorded'}",
            f"Source comments: {target.get('source_comments') or 'None'}",
        ]
        self.source_report.setPlainText("\n".join(source_lines))
        reports = target.get("reader_reports") or []
        if not reports:
            self.reports_browser.setPlainText("No reader reports yet.")
        else:
            source_ras = tuple(target.get("source_ras") or target["ras"])
            lines: list[str] = []
            for report in reports:
                moved = ""
                if report.get("ras_l") is not None:
                    corrected = (report["ras_l"], report["ras_p"], report["ras_s"])
                    distance = _distance_mm(source_ras, corrected) or 0.0
                    moved = (
                        f"\n  Moved {distance:.1f} mm to "
                        f"({corrected[0]:.3f}, {corrected[1]:.3f}, {corrected[2]:.3f})"
                    )
                lines.append(
                    f"{report.get('reader_id', 'Unknown')} · round {report.get('review_round', '?')} · "
                    f"{_report_status(report.get('verify'), report.get('comment'))}\n"
                    f"  {str(report.get('comment') or 'No comment').strip()}{moved}\n"
                    f"  Updated: {report.get('updated_at') or '—'}"
                )
            self.reports_browser.setPlainText("\n\n".join(lines))

    def _mark_review_dirty(self) -> None:
        if not self._updating_form:
            self._review_dirty = True
            self._update_dirty_indicator()

    def _update_dirty_indicator(self) -> None:
        if not self._review_dirty:
            self.dirty_label.setText("")
        else:
            self.dirty_label.setText("unsaved · ROI" if self._roi_dirty else "unsaved")

    def _confirm_dirty(self) -> bool:
        if not self._review_dirty or self.selected_target is None:
            return True
        prompt = QMessageBox(self)
        prompt.setWindowTitle("Unsaved review")
        prompt.setText("The current reader report has unsaved changes.")
        prompt.setInformativeText("Save it before moving to another case, finding or coordinate?")
        save_button = prompt.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        discard_button = prompt.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        prompt.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        prompt.exec()
        clicked = prompt.clickedButton()
        if clicked is save_button:
            # Saving from this prompt must not also jump somewhere else; the
            # caller is already on its way to another finding or case.
            return self.save_current_review(advance=False)
        if clicked is discard_button:
            self._review_dirty = False
            self._update_dirty_indicator()
            return True
        return False

    def save_current_review(self, *, advance: bool | None = None) -> bool:
        if self.selected_target is None or self.current_case_id is None:
            self._set_status("Select a source or manual finding before saving a review.", COLORS["warn"])
            return False
        verify = self._verdict
        comment = self.comment_edit.toPlainText().strip() or None
        # You save the position you are looking at: selecting another reader's
        # correction and saving adopts it, selecting Source releases yours.
        variant = self._current_variant()
        corrected_ras = None
        if variant is not None and variant["key"] != "source":
            corrected_ras = tuple(float(value) for value in variant["ras"])
        # Measured now, while the image is in memory: how far the coordinate
        # this reader stands behind sits from anything focal.
        standing_behind = corrected_ras or tuple(
            float(value) for value in self.selected_target["ras"]
        )
        try:
            save_review(
                self.db_path,
                target_id=str(self.selected_target["target_id"]),
                case_id=self.current_case_id,
                reader_id=self.reader_id,
                review_round=self.review_round,
                verify=verify,
                comment=comment,
                corrected_ras=corrected_ras,
                session_id=self.session_id,
                snap_mm=self._distance_to_focus(standing_behind),
                certainty=str(self.certainty_combo.currentData() or "") or None,
                mimic=str(self.mimic_combo.currentData() or "") or None,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Could not save review", str(exc))
            return False
        if self._roi_dirty:
            self._write_label_volume()
        self._review_dirty = False
        self._update_dirty_indicator()
        current_target_id = self.selected_target["target_id"]
        self.targets = list_targets(self.db_path, self.current_case_id, self.reader_id, self.review_round)
        self._populate_target_list()
        row = next((index for index, target in enumerate(self.targets) if target["target_id"] == current_target_id), -1)
        if row >= 0:
            self.target_list.blockSignals(True)
            self.target_list.setCurrentRow(row)
            self.target_list.blockSignals(False)
            self.selected_target = self.targets[row]
            self._load_review_form(self.selected_target)
            # The position was written along with the verdict, so nothing is
            # pending any more.  Without this the combo kept offering the
            # "(unsaved)" entry and the hint beside Move still read "unsaved
            # move" for a position that had just been stored -- the reader is
            # then looking at a warning about work that is already safe.
            self.pending_ras = None
            self.selected_variant = (
                self.reader_id if self.selected_target.get("reader_ras") else "source"
            )
            self._rebuild_position_variants(self.selected_target)
        self._reload_case_list()
        saved_label = self.selected_target["label"] if self.selected_target else "review"
        self._set_status(f"Saved {saved_label} · {_verification_text(verify)}", COLORS["success"])
        if advance is None:
            advance = self.settings.save_advances
        if advance:
            self._advance_after_save(saved_label)
        return True

    def _advance_after_save(self, saved_label: str) -> None:
        """Move to the next microbleed, and only then to the next case.

        "Next" means the next finding in the list, decided or not.  It used to
        skip anything already reviewed and jump straight out of the case, so
        where the button landed depended on what the reader had done earlier,
        and a finding they wanted to look at again was unreachable from it.
        Walking the list in order is the thing a reader can predict.
        """

        current = next(
            (
                index
                for index, target in enumerate(self.targets)
                if self.selected_target is not None
                and target["target_id"] == self.selected_target["target_id"]
            ),
            -1,
        )
        if 0 <= current < len(self.targets) - 1:
            self._select_target_index(current + 1, confirm=False)
            self._set_status(
                f"Saved {saved_label} · finding {current + 2} of {len(self.targets)}",
                COLORS["success"],
            )
            return
        try:
            case_index = next(
                index
                for index, item in enumerate(self.visible_cases)
                if item["case_id"] == self.current_case_id
            )
        except StopIteration:
            case_index = -1
        if 0 <= case_index < len(self.visible_cases) - 1:
            next_case = self.visible_cases[case_index + 1]["case_id"]
            self.load_case(next_case)
            self._set_status(f"Saved {saved_label} · opened {next_case}", COLORS["success"])
            return
        self._set_status(
            f"Saved {saved_label} · that was the last finding of the last visible case.",
            COLORS["success"],
        )

    # ---------------------------------------------------------- manual point --
    def _removal_entry(self, target_id: str) -> dict[str, Any]:
        """Whether this finding can be removed, and how to say so.

        Kept apart from the menu so the answer is testable without opening
        one, and so the button and the context menu cannot disagree.
        """

        blockers = manual_deletion_blockers(self.db_path, str(target_id), self.reader_id)
        return {
            "allowed": not blockers,
            "struck_through": bool(blockers),
            "reason": "\n".join(blockers)
            if blockers
            else "Delete this finding and your work on it",
        }

    def _show_finding_menu(self, position: QPoint) -> None:
        """Right-click a finding: act on the one under the cursor.

        Removing lives here as well as beside Add, because the list is where a
        reader notices they added something twice.  The entry is disabled with
        the reason in its tooltip rather than hidden -- "why can I not delete
        this" is exactly the thing worth saying, and a missing menu item says
        nothing.
        """

        item = self.target_list.itemAt(position)
        if item is None:
            return
        target_id = str(item.data(Qt.ItemDataRole.UserRole))
        target = next(
            (entry for entry in self.targets if str(entry["target_id"]) == target_id), None
        )
        if target is None:
            return
        menu = QMenu(self.target_list)
        show = menu.addAction(f"Go to {target['label']}")
        menu.addSeparator()
        entry = self._removal_entry(target_id)
        remove = menu.addAction("Remove finding…")
        remove.setEnabled(entry["allowed"])
        remove.setToolTip(entry["reason"])
        if entry["struck_through"]:
            # Greying alone was not enough to read as disabled, and the reason
            # is worth keeping, so the entry stays with a line through it.
            font = remove.font()
            font.setStrikeOut(True)
            remove.setFont(font)
        menu.setToolTipsVisible(True)
        chosen = menu.exec(self.target_list.viewport().mapToGlobal(position))
        if chosen is show:
            self._select_target_id(target_id)
        elif chosen is remove:
            self._select_target_id(target_id)
            self.remove_manual_microbleed()

    def _select_target_id(self, target_id: str) -> None:
        index = next(
            (
                position
                for position, target in enumerate(self.targets)
                if str(target["target_id"]) == str(target_id)
            ),
            -1,
        )
        if index >= 0:
            self._select_target_index(index, confirm=False)

    def _confirm_no_finding_there(self, ras: tuple[float, float, float]) -> bool:
        """Stop a second copy of a finding that is already in the list.

        The mistake is ordinary: the reader spots a lesion, does not notice it
        is already row 17, and adds it again -- and a duplicate is worse than a
        miss, because it inflates the count and half the readers review each
        copy.  Measured over every within-case pair in this dataset, two real
        findings never come within 2.03 mm, so a point inside a millimetre is
        the same lesion and one inside three is worth a question.
        """

        if self.current_case_id is None:
            return True
        try:
            nearby = findings_near(self.db_path, self.current_case_id, ras)
        except Exception as exc:
            # Let the reader add it -- a duplicate can be removed, a blocked
            # workflow cannot -- but do not let the failure pass in silence.
            self._set_status(
                f"Could not check for an existing finding here ({type(exc).__name__}); "
                "adding anyway.",
                COLORS["warn"],
            )
            return True
        if not nearby:
            return True
        closest = nearby[0]
        distance = float(closest["distance_mm"])
        prompt = QMessageBox(self)
        prompt.setWindowTitle("There is already a finding here")
        if distance <= SAME_FINDING_MM:
            prompt.setIcon(QMessageBox.Icon.Warning)
            prompt.setText(
                f"{closest['label']} is {distance:.2f} mm from this point — that is the "
                "same lesion."
            )
            prompt.setInformativeText(
                "Two findings in this dataset are never closer than 2 mm, so this would "
                "be a duplicate. To move that finding instead, select it and use Position."
            )
            go = prompt.addButton("Go to it", QMessageBox.ButtonRole.AcceptRole)
            prompt.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            prompt.exec()
            if prompt.clickedButton() is go:
                self._select_target_id(str(closest["target_id"]))
            return False
        prompt.setIcon(QMessageBox.Icon.Question)
        prompt.setText(f"{closest['label']} is {distance:.2f} mm from this point.")
        prompt.setInformativeText(
            "Findings this close are usually the same one seen twice. Add it anyway "
            "only if you can see two separate foci."
        )
        go = prompt.addButton("Go to it", QMessageBox.ButtonRole.AcceptRole)
        anyway = prompt.addButton("Add anyway", QMessageBox.ButtonRole.DestructiveRole)
        prompt.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        prompt.exec()
        if prompt.clickedButton() is go:
            self._select_target_id(str(closest["target_id"]))
            return False
        return prompt.clickedButton() is anyway

    def remove_manual_microbleed(self) -> None:
        """Delete a finding this reader added, and their own work on it."""

        if self.selected_target is None:
            self._set_status("Select the finding you want to remove.", COLORS["warn"])
            return
        target_id = str(self.selected_target["target_id"])
        label = str(self.selected_target["label"])
        blockers = manual_deletion_blockers(self.db_path, target_id, self.reader_id)
        if blockers:
            QMessageBox.information(self, f"Cannot remove {label}", "\n\n".join(blockers))
            return
        prompt = QMessageBox(self)
        prompt.setIcon(QMessageBox.Icon.Warning)
        prompt.setWindowTitle("Remove this finding")
        prompt.setText(f"Remove {label} from {self.current_case_id}?")
        prompt.setInformativeText(
            "Your verdict and segmentation for it go too — they describe a finding that "
            "will not exist. This cannot be undone."
        )
        remove = prompt.addButton("Remove", QMessageBox.ButtonRole.DestructiveRole)
        prompt.addButton("Keep it", QMessageBox.ButtonRole.RejectRole)
        prompt.exec()
        if prompt.clickedButton() is not remove:
            return

        # Clear its voxels before the row goes: the label file holds the whole
        # case, and an integer nothing decodes any more is worse than no mask.
        value = self.label_values.pop(target_id, None)
        if value is not None and self.label_volume is not None:
            mask = self.label_volume == value
            if np.any(mask):
                self.label_volume[mask] = 0
                self._roi_dirty = True
            # The 3D view keeps the other findings' geometry until the
            # selection changes, because a brush stroke cannot alter it -- a
            # deletion can.  Every route into this method selects the finding
            # first, so the key changes anyway and this line is belt and
            # braces rather than a fix for anything reachable today; it is
            # here so a future caller that deletes without reselecting does
            # not leave a removed mask in the head.
            self._others_cache = None
        self.label_sources.pop(target_id, None)
        self.label_methods.pop(target_id, None)
        self.label_settings.pop(target_id, None)
        self._stored_roi_targets.discard(target_id)
        try:
            removed = delete_manual_annotation(
                self.db_path,
                target_id=target_id,
                reader_id=self.reader_id,
                session_id=self.session_id,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Could not remove the finding", str(exc))
            return

        self.selected_target = None
        self._review_dirty = False
        self.targets = list_targets(
            self.db_path, self.current_case_id, self.reader_id, self.review_round
        )
        if self._roi_dirty:
            self._write_label_volume()
        self._populate_target_list()
        self._roi_undo = []
        if self.targets:
            self._select_target_index(0, confirm=False)
        else:
            self._clear_review_form()
            self._apply_labels_to_views()
        self._reload_case_list()
        self._update_finding_buttons()
        carried = []
        if removed["reviews"]:
            carried.append(_human_count(removed["reviews"], "review"))
        if removed["segmentations"]:
            carried.append(_human_count(removed["segmentations"], "segmentation"))
        detail = f" and {' and '.join(carried)}" if carried else ""
        self._set_status(f"Removed {label}{detail}.", COLORS["success"])

    def add_manual_microbleed(self) -> None:
        if self.current_case_id is None:
            QMessageBox.information(self, "No case selected", "Select a case before adding a manual microbleed.")
            return
        if not self._confirm_dirty():
            return
        ras = tuple(float(spin.value()) for spin in (self.manual_l_spin, self.manual_p_spin, self.manual_s_spin))
        if not self._confirm_no_finding_there(ras):
            return
        try:
            target_id = add_manual_annotation(
                self.db_path,
                case_id=self.current_case_id,
                ras=ras,
                reader_id=self.reader_id,
                review_round=self.review_round,
                atlasregion=self.manual_region_edit.text().strip() or None,
                initial_note=self.manual_note_edit.toPlainText().strip() or None,
                session_id=self.session_id,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Could not add manual microbleed", str(exc))
            return
        self.targets = list_targets(self.db_path, self.current_case_id, self.reader_id, self.review_round)
        self._populate_target_list()
        row = next((index for index, target in enumerate(self.targets) if target["target_id"] == target_id), -1)
        if row >= 0:
            self.target_list.setCurrentRow(row)
        self.manual_region_edit.clear()
        self.manual_note_edit.clear()
        self._set_status("Added a manual microbleed annotation to the shared datasheet.", COLORS["success"])

    # --------------------------------------------------------------- refresh --
    def refresh_inventory(self) -> None:
        try:
            counts = refresh_inventory_store(self.db_path, self.data_root)
        except Exception as exc:
            QMessageBox.critical(self, "Could not refresh file inventory", str(exc))
            return
        # A rescan can point a case at a different file, so cached volumes
        # cannot be trusted afterwards.
        self._prefetch_timer.stop()
        self._stop_prefetch()
        self.volume_cache.clear()
        self._reload_case_list()
        if self.current_case_id:
            self.current_case = get_case(self.db_path, self.current_case_id)
            self.targets = list_targets(self.db_path, self.current_case_id, self.reader_id, self.review_round)
            self._populate_target_list()
            self._set_case_title(self.current_case or {"case_id": self.current_case_id, "source_count": len(self.targets), "file_status": ""})
            self.load_case(self.current_case_id, force=True)
        self._set_status(
            f"Inventory refreshed · complete {counts.get('complete', 0)} · missing/partial cases are shown by filters.",
            COLORS["success"],
        )
        self._log_event(
            "inventory_refreshed",
            session_id=self.session_id,
            reader_id=self.reader_id,
            review_round=self.review_round,
            details=counts,
        )

    # ----------------------------------------------------------- session/log --
    def _schedule_session_save(self) -> None:
        self._session_timer.start()

    def _save_session_state(self) -> None:
        if self._closing and not self._writer.isRunning():
            return
        state = {
            "case_id": self.current_case_id,
            "target_id": self.selected_target["target_id"] if self.selected_target else None,
            "modality": self.current_modality,
            "axial": self.view_panels["axial"].canvas.slice_index,
            "coronal": self.view_panels["coronal"].canvas.slice_index,
            "sagittal": self.view_panels["sagittal"].canvas.slice_index,
            "filters": {
                "search": self.case_search.text(),
                "hide_missing": self.hide_missing_cb.isChecked(),
                "complete_only": self.complete_only_cb.isChecked(),
                "unverified_only": self.unverified_only_cb.isChecked(),
                "verified_only": self.verified_only_cb.isChecked(),
                "source_unverified": self.source_unverified_cb.isChecked(),
                "adjudication": self.adjudication_cb.isChecked(),
                "disagreement": self.disagreement_cb.isChecked(),
            },
        }
        db_path, session_id = self.db_path, self.session_id
        self._writer.submit(
            "session state", lambda: save_session_state(db_path, session_id, state)
        )

    def _set_status(self, message: str, color: str = COLORS["dim"]) -> None:
        # ``showMessage`` hides every non-permanent status widget for as long as
        # the message is displayed, which used to blank this label completely.
        self._status_bar.clearMessage()
        self._status_label.setText(message)
        self._status_label.setStyleSheet(f"color:{color}; font-size:9pt;")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._closing:
            event.accept()
            return
        if not self._confirm_dirty():
            event.ignore()
            return
        if not self._confirm_unconfirmed_segmentations():
            event.ignore()
            return
        self._closing = True
        self._prefetch_timer.stop()
        self._stop_prefetch()
        self._slice_log_timer.stop()
        self._flush_slice_log()
        self._save_session_state()
        self._log_event(
            "session_closed",
            session_id=self.session_id,
            reader_id=self.reader_id,
            review_round=self.review_round,
            case_id=self.current_case_id,
            target_id=self.selected_target["target_id"] if self.selected_target else None,
            details={},
        )
        db_path, session_id = self.db_path, self.session_id
        self._writer.submit("close session", lambda: close_session(db_path, session_id))
        # Everything queued above is written before the thread ends; the wait is
        # bounded so a stuck lock cannot keep the window open.
        self._writer.stop()
        event.accept()


def _initialize(dataset: Dataset | None = None) -> dict[str, Any]:
    dataset = dataset or default_dataset()
    return initialize_store(dataset.workbook, dataset.data_root, dataset.review_db)


LOG_PATH = BASE_DIR / "microbleed_viewer.log"
_LOG_HANDLE = None


# Windows groups taskbar buttons by this string; it only has to be unique and
# stable, so it names the tool rather than any institution.
APP_MODEL_ID = "MicrobleedReviewViewer.Desktop"


def apply_application_identity(app) -> "QIcon | None":
    """Give the application its icon everywhere Windows and Qt will show one.

    ``setWindowIcon`` on the application covers the main window, every dialog
    and every message box, because Qt hands top-level windows the application
    icon unless they set their own.

    The taskbar needs one more step. Windows groups a window under the
    executable that hosts it, which here is ``python.exe`` -- so without an
    explicit AppUserModelID the taskbar button shows the Python logo no matter
    what Qt was told. Setting the id makes the shell treat this as its own
    application and use the icon below.
    """

    icon_path = icon_file()
    icon = QIcon(str(icon_path)) if icon_path is not None else None
    if icon is not None and not icon.isNull():
        app.setWindowIcon(icon)
    else:
        icon = None

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_MODEL_ID)
        except Exception:  # pragma: no cover - a wrong taskbar icon is not fatal
            pass
    return icon


def install_diagnostics(log_path: Path | None = None) -> Path:
    """Send Qt messages and any fatal crash to a log file.

    A crash inside Qt prints to stderr and aborts, which is invisible when the
    viewer is started from a ``.bat`` that closes with it -- the window simply
    disappears. Writing both streams to a file turns the next one into
    something diagnosable.
    """

    global _LOG_HANDLE
    path = Path(log_path or LOG_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover - read-only location
        return path
    _LOG_HANDLE = handle
    handle.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} viewer started ===\n")
    faulthandler.enable(handle)

    levels = {
        QtMsgType.QtDebugMsg: "debug",
        QtMsgType.QtInfoMsg: "info",
        QtMsgType.QtWarningMsg: "warning",
        QtMsgType.QtCriticalMsg: "critical",
        QtMsgType.QtFatalMsg: "fatal",
    }

    def handler(mode, context, message) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        handle.write(f"{stamp} {levels.get(mode, 'message')}: {message}\n")

    qInstallMessageHandler(handler)
    return path


SYNC_FOLDER_MARKERS = ("onedrive", "dropbox", "google drive", "googledrive", "box sync", "icloud")


def synced_folder_name(path: Path) -> str | None:
    """Name of the sync client whose folder holds ``path``, if any."""

    text = str(Path(path).resolve()).lower()
    for marker in SYNC_FOLDER_MARKERS:
        if marker in text:
            return marker
    for variable in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        root = os.environ.get(variable)
        if root and text.startswith(str(Path(root).resolve()).lower()):
            return "onedrive"
    return None


def _warn_about_synced_database(settings: ViewerSettings, dataset: Dataset) -> None:
    """A synced review database is the usual cause of a frozen window.

    The sync client takes the file to upload it, SQLite waits for the lock, and
    the viewer stops repainting until the lock clears.
    """

    marker = synced_folder_name(dataset.review_db)
    if marker is None:
        return
    key = f"warned/synced_db/{dataset.review_db}"
    if settings.store.value(key, False) in (True, "true", "True", 1, "1"):
        return
    prompt = QMessageBox()
    prompt.setIcon(QMessageBox.Icon.Warning)
    prompt.setWindowTitle("Review database is in a synchronised folder")
    prompt.setText(
        f"The review database sits inside a {marker.title()} folder, which can make the "
        "viewer freeze."
    )
    prompt.setInformativeText(
        "While the sync client holds the file, the viewer has to wait for it, and the "
        "window stops responding until the lock clears.\n\n"
        "Keeping the database on a local disk avoids this. Use Dataset to point at a "
        "local path -- the MRI folder and the workbook can stay where they are.\n\n"
        f"Database: {dataset.review_db}"
    )
    prompt.addButton("Continue anyway", QMessageBox.ButtonRole.AcceptRole)
    silence = prompt.addButton("Do not warn me for this database", QMessageBox.ButtonRole.RejectRole)
    prompt.exec()
    if prompt.clickedButton() is silence:
        settings.store.setValue(key, True)
        settings.store.sync()


def _report_source_state(report: dict[str, Any], dataset: Dataset) -> None:
    # The reader has to know when the workbook and the store have diverged.
    if report.get("source_changed"):
        prompt = QMessageBox()
        prompt.setIcon(QMessageBox.Icon.Warning)
        prompt.setWindowTitle("Source workbook changed")
        prompt.setText("The findings workbook has changed since this review store was created.")
        prompt.setInformativeText(
            "Importing brings in rows that were added and refreshes the source columns of "
            "rows that were edited. Reader verifications, comments and manual findings are "
            "never touched.\n\n"
            f"Workbook: {dataset.workbook}\nReview store: {report.get('db_path')}"
        )
        import_button = prompt.addButton("Import changes", QMessageBox.ButtonRole.AcceptRole)
        prompt.addButton("Keep as it is", QMessageBox.ButtonRole.RejectRole)
        prompt.exec()
        if prompt.clickedButton() is not import_button:
            return
        try:
            result = reimport_source(dataset.review_db, dataset.workbook)
        except SourceReadError as exc:
            QMessageBox.critical(None, "Could not import the workbook changes", str(exc))
            return
        removed = result["removed_from_workbook"]
        QMessageBox.information(
            None,
            "Workbook changes imported",
            f"Added {result['added']} findings.\n"
            f"Refreshed {result['updated']} existing findings.\n"
            f"{result['unchanged']} were already up to date."
            + (
                f"\n\n{removed} finding(s) are in the review store but no longer in the workbook. "
                "They were kept, because they may already have been reviewed."
                if removed
                else ""
            ),
        )
    elif report.get("source_unavailable"):
        QMessageBox.warning(
            None,
            "Source workbook unavailable",
            "The original source workbook could not be read right now. The findings already "
            "imported into the review store are used instead and the review can continue.\n\n"
            f"Workbook: {dataset.workbook}",
        )


def _remember_config(config: dict, dataset: Dataset) -> dict:
    """Apply what the dialog chose, and write it down.

    Written rather than kept in memory: the point of the dialog is that the
    next launch does not ask again.  A failure to write is worth saying but
    not worth stopping for -- the session in front of the reader is configured
    either way.
    """

    config = dict(config)
    config["paths"] = {
        "workbook": str(dataset.workbook),
        "data_root": str(dataset.data_root),
        "review_database": str(dataset.review_db),
    }
    applied = apply_dataset_config(config)
    try:
        dataset_config.save(applied)
    except OSError as exc:
        QMessageBox.warning(
            None,
            "Could not save config.json",
            f"The dataset is open, but this choice will not be remembered.\n\n{exc}",
        )
    return applied


def main() -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    install_diagnostics()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    # Before the first dialog: the dataset and reader dialogs open ahead of the
    # main window, and they should already carry the mark.
    apply_application_identity(app)
    app.setStyle("Fusion")
    app.setStyleSheet(GLOBAL_STYLE)
    settings = ViewerSettings()
    # How this dataset is shaped: which sheet, which columns, what the files
    # are called.  Missing file means the defaults, which is what the study
    # this was written for uses.
    try:
        config = apply_dataset_config(dataset_config.load())
    except dataset_config.ConfigError as exc:
        QMessageBox.critical(
            None,
            "config.json cannot be used",
            f"{exc}\n\nStarting with the built-in defaults; "
            "fix the file or set the format in the dataset dialog.",
        )
        config = apply_dataset_config(None)
    # The launcher's environment decides the starting dataset; another one can
    # be opened from the reader dialog before any review begins.
    dataset = default_dataset(config)
    while True:
        problems = dataset.problems()
        if problems:
            QMessageBox.warning(None, "Dataset unavailable", "\n".join(problems))
            report = None
        else:
            try:
                report = _initialize(dataset)
            except SourceReadError as exc:
                QMessageBox.critical(None, "Microbleed Review could not start", str(exc))
                report = None
        if report is None:
            chooser = DatasetDialog(settings, dataset, config=config)
            if chooser.exec() != QDialog.DialogCode.Accepted or chooser.dataset is None:
                return 2
            dataset = chooser.dataset
            config = _remember_config(chooser.config, dataset)
            continue
        _report_source_state(report, dataset)
        _warn_about_synced_database(settings, dataset)
        reader_dialog = ReaderDialog(
            dataset.review_db,
            initial_reader=str(settings.store.value("session/last_reader", "") or ""),
            dataset=dataset,
            allow_dataset_change=True,
            settings=settings,
        )
        if reader_dialog.exec() == QDialog.DialogCode.Accepted and reader_dialog.session is not None:
            break
        if not reader_dialog.change_dataset_requested:
            return 0
        chooser = DatasetDialog(settings, dataset, config=config)
        if chooser.exec() == QDialog.DialogCode.Accepted and chooser.dataset is not None:
            dataset = chooser.dataset
            config = _remember_config(chooser.config, dataset)

    settings.remember_dataset(dataset)
    settings.store.setValue("session/last_reader", reader_dialog.reader_id)
    window = MicrobleedViewer(
        dataset.review_db,
        dataset.data_root,
        reader_dialog.session,
        settings=settings,
        dataset=dataset,
    )
    window.show()
    window.raise_()
    window.activateWindow()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - normal desktop launch
    raise SystemExit(main())
