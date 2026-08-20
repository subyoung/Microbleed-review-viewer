from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

VIEWER_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = VIEWER_DIR.parent
sys.path.insert(0, str(VIEWER_DIR))

try:
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover - allows core-only environments to run
    QApplication = None


def _sample_case(db_path) -> str:
    """The case the data-backed tests work on.

    Discovered rather than named: the source should not carry a study's
    identifiers, and a test that hard-codes one only runs on one dataset.
    Ordering follows the case queue, so this is the case the viewer opens.
    """

    import review_store

    listing = review_store.list_cases(db_path, "sample", 1)
    complete = [item for item in listing if item["file_status"] == "complete"]
    if not complete:
        # These tests read images.  A dataset with none -- the example
        # workbook in examples/, for instance, which ships no MRI -- exercises
        # the store and nothing else, and saying so beats failing.
        raise unittest.SkipTest(
            "The configured dataset has no case with all three sequences."
        )
    return str(complete[0]["case_id"])


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class DesktopViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = os.environ.get("TEST_SOURCE_XLSX")
        if not source:
            raise unittest.SkipTest("Set TEST_SOURCE_XLSX to a readable copy of the source workbook.")
        cls.app = QApplication.instance() or QApplication([])
        cls.source = Path(source)
        cls.data_root = Path(os.environ.get("TEST_DATA_ROOT", PROJECT_DIR / "Data"))

    def setUp(self) -> None:
        from PySide6.QtCore import QSettings

        import desktop_app
        import review_store

        self.desktop_app = desktop_app
        self.review_store = review_store
        self.temp_dir = Path(tempfile.mkdtemp(prefix="microbleed_desktop_test_"))
        self.db_path = self.temp_dir / "review.sqlite"
        review_store.initialize_store(self.source, self.data_root, self.db_path)
        self.case_id = _sample_case(self.db_path)
        self.session = review_store.start_new_session(self.db_path, "Desktop Test Reader")
        # Preferences go to a throwaway ini so a test run never depends on, or
        # disturbs, the reader preferences of the logged-in user.
        self.settings = desktop_app.ViewerSettings(
            QSettings(str(self.temp_dir / "prefs.ini"), QSettings.Format.IniFormat)
        )
        self.window = desktop_app.MicrobleedViewer(
            self.db_path, self.data_root, self.session, settings=self.settings
        )
        self.window.show()
        self.assertTrue(
            self._wait_for(
                lambda: self.window.current_case_id == self.case_id
                and self.window._load_thread is None
                and all(self.window.volumes[modality] is not None for modality in ("qsm", "swi", "mip"))
            )
        )

    def tearDown(self) -> None:
        # Closing with unsaved changes, or with a mask nobody decided on, opens
        # a modal prompt on purpose -- which would block a headless run for
        # ever.  Both prompts read state, so clear the state rather than the
        # prompt: emptying the finding list leaves nothing to ask about.
        self.window._review_dirty = False
        self.window.targets = []
        self.window.close()
        self.app.processEvents()
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _wait_for(self, predicate, timeout_ms: int = 6000) -> bool:
        from PySide6.QtCore import QElapsedTimer

        timer = QElapsedTimer()
        timer.start()
        while not predicate() and timer.elapsed() < timeout_ms:
            self.app.processEvents()
        return bool(predicate())

    # ----------------------------------------------- external segmentations --
    def _external_folder(self, *, probability: bool = False) -> Path:
        """A results folder for the open case, on the open case's own grid.

        Written through the same path a saved mask takes, so the file that
        comes back is voxel-for-voxel the one the viewer would compare
        against -- a test that writes its own affine would pass while the
        viewer refused every real file.
        """

        import nibabel as nib

        from imaging import save_label_volume, to_source_orientation

        reference = self.window._label_reference()
        assert reference is not None
        folder = self.temp_dir / "results" / self.case_id / "models"
        folder.mkdir(parents=True, exist_ok=True)

        mask = np.zeros(reference.shape, dtype=np.uint16)
        centre = tuple(int(size // 2) for size in reference.shape)
        mask[centre[0] - 1:centre[0] + 2, centre[1] - 1:centre[1] + 2, centre[2]] = 1
        # A second detection, far enough away to stay a second detection.
        mask[centre[0] + 20, centre[1] + 20, centre[2]] = 1
        save_label_volume(folder / "model_prediction.nii.gz", mask, reference)

        if probability:
            soft = np.zeros(reference.shape, dtype=np.float32)
            soft[centre[0] - 1:centre[0] + 2, centre[1] - 1:centre[1] + 2, centre[2]] = 0.9
            soft[centre[0] + 3, centre[1], centre[2]] = 0.4
            # Enough distinct values that it reads as a probability map.
            soft[0, 0, :16] = np.linspace(0.01, 0.3, 16)
            restored = to_source_orientation(soft, reference)
            nib.save(
                nib.Nifti1Image(
                    np.asarray(restored.dataobj, dtype=np.float32),
                    np.asarray(restored.affine, dtype=np.float64),
                ),
                str(folder / "model_probability.nii.gz"),
            )
        return self.temp_dir / "results"

    def _turn_on_developer_mode(self, root: Path, pattern: str = "./models") -> None:
        self.settings.set_external_results(str(root), pattern)
        self.window._set_developer_mode(True)
        self.window._refresh_external_panel()

    def test_the_external_tab_is_hidden_until_developer_mode_is_turned_on(self) -> None:
        index = self.window._panel_tab_keys.index("external")
        self.assertFalse(
            self.window.panel_tabs.isTabVisible(index),
            "a reader is shown a detector's guesses without asking for them",
        )
        self._turn_on_developer_mode(self._external_folder())
        self.assertTrue(self.window.panel_tabs.isTabVisible(index))

        # And it is not there for the cases that have nothing behind it.
        self._turn_on_developer_mode(self.temp_dir / "empty_results")
        self.assertFalse(
            self.window.panel_tabs.isTabVisible(index),
            "an empty tab on every case is a thing to click and find nothing in",
        )
        self._turn_on_developer_mode(self._external_folder())

        # And it puts everything away again, including what is on the images.
        self.window.external_combo.setCurrentIndex(1)
        self.assertIsNotNone(self.window.external_mask)
        self.window._set_developer_mode(False)
        self.assertFalse(self.window.panel_tabs.isTabVisible(index))
        self.assertIsNone(self.window.external_mask)
        self.assertIsNone(self.window.view_panels["axial"].canvas._external_volume)

    def test_cases_with_a_result_are_marked_in_the_queue(self) -> None:
        rows = lambda: [
            self.window.case_list.item(index).text()
            for index in range(self.window.case_list.count())
        ]
        self.assertFalse([row for row in rows() if "*" in row])

        self._turn_on_developer_mode(self._external_folder())
        marked = [row for row in rows() if "*" in row]
        self.assertEqual(len(marked), 1, "the one case with a result is not marked")
        self.assertTrue(
            marked[0].startswith(f"{self.case_id}*"),
            f"the mark is not on the identifier: {marked[0]!r}",
        )

        # And the filter for them, which only exists in developer mode.
        self.window.external_only_cb.setChecked(True)
        self.assertEqual(len(rows()), 1, "the filter did not narrow to the marked case")
        self.window.external_only_cb.setChecked(False)
        self.assertGreater(len(rows()), 1)

    def test_both_folder_layouts_are_readable(self) -> None:
        root = self._external_folder()
        self._turn_on_developer_mode(root, "./models")
        self.assertEqual(self.window.external_combo.count(), 2)
        # The same run, described the other way round: nothing sits directly
        # in the case folder, so nothing is found.
        self._turn_on_developer_mode(root, ".")
        self.assertEqual(self.window.external_combo.count(), 1)
        self.assertIn("Nothing for", self.window.external_note.text())

    def test_a_binary_result_is_shown_counted_and_offered_no_threshold(self) -> None:
        self._turn_on_developer_mode(self._external_folder())
        self.window.external_combo.setCurrentIndex(1)

        self.assertEqual(self.window.external_kind, "mask")
        self.assertTrue(
            self.window.external_threshold_row.isHidden(),
            "a threshold control over a hard segmentation is a knob that does nothing",
        )
        self.assertEqual(len(self.window.external_blobs), 2)
        self.assertEqual(self.window.external_blob_list.count(), 2)
        self.assertEqual(int(np.count_nonzero(self.window.external_mask)), 10)
        for panel in self.window.view_panels.values():
            self.assertIsNotNone(panel.canvas._external_volume)

    def test_a_probability_map_is_recognised_and_follows_its_threshold(self) -> None:
        self._turn_on_developer_mode(self._external_folder(probability=True))
        names = [
            self.window.external_combo.itemText(index)
            for index in range(self.window.external_combo.count())
        ]
        self.window.external_combo.setCurrentIndex(names.index("model_probability"))

        self.assertEqual(self.window.external_kind, "probability")
        self.assertFalse(self.window.external_threshold_row.isHidden())
        self.assertAlmostEqual(self.window.external_threshold_spin.value(), 0.5)
        self.assertEqual(int(np.count_nonzero(self.window.external_mask)), 9)
        self.assertIn("0.50", self.window.external_readout.text())

        # Moving the number changes nothing until Apply: re-thresholding
        # twelve million voxels on every step of a drag is the lurch this
        # replaced.
        self.window.external_threshold_spin.setValue(0.35)
        self.assertEqual(int(np.count_nonzero(self.window.external_mask)), 9)
        self.assertTrue(self.window.external_apply_btn.isEnabled())
        self.assertIn("press Apply", self.window.external_readout.text())

        self.window._apply_external_threshold()
        self.assertEqual(int(np.count_nonzero(self.window.external_mask)), 10)
        self.assertFalse(self.window.external_apply_btn.isEnabled())
        self.assertNotIn("press Apply", self.window.external_readout.text())

        self.window.external_threshold_spin.setValue(0.95)
        self.window._apply_external_threshold()
        self.assertEqual(int(np.count_nonzero(self.window.external_mask)), 0)
        self.assertIn("nothing above this threshold", self.window.external_readout.text())

        # The slider and the box are two handles on one number.
        self.window.external_threshold_slider.setValue(35)
        self.assertAlmostEqual(self.window.external_threshold_spin.value(), 0.35)
        self.window._apply_external_threshold()
        self.assertEqual(int(np.count_nonzero(self.window.external_mask)), 10)

    def test_clicking_a_detection_moves_the_views_to_it(self) -> None:
        self._turn_on_developer_mode(self._external_folder())
        self.window.external_combo.setCurrentIndex(1)
        before = tuple(self.window.target_ras)
        marker = tuple(self.window.marker_ras)

        self.window.external_blob_list.setCurrentRow(0)
        self.assertNotEqual(tuple(self.window.target_ras), before)
        canvas = self.window.view_panels["axial"].canvas
        self.assertTrue(
            np.any(canvas._external_slice()),
            "the views moved somewhere the detection is not",
        )
        # Navigation only. The finding under review keeps its own coordinate,
        # or looking at a model's output would quietly rewrite the data.
        self.assertEqual(tuple(self.window.marker_ras), marker)

    def test_comparison_scores_the_result_against_every_mask_of_the_case(self) -> None:
        self._turn_on_developer_mode(self._external_folder())
        self.window.external_combo.setCurrentIndex(1)
        self.window.external_compare_cb.setChecked(True)
        self.assertIn("not segmented", self.window.external_readout.text())

        # Paint the reader's own mask over exactly half of the model's, and
        # the sentence has to say so.
        mask = np.asarray(self.window.external_mask)
        coords = np.argwhere(mask)[:5]
        value = self.window._label_value_for(self.window.selected_target["target_id"])
        self.window.label_volume[tuple(coords.T)] = value
        self.window._update_external_readout()

        text = self.window.external_readout.text()
        self.assertIn("Dice 0.667", text)
        # Counted off the volume that was scored, not off what has been saved:
        # this mask has not reached the database and it is still in the dice.
        self.assertIn("your 1 mask", text)
        self.assertIn("both 5", text)
        self.assertIn("theirs only 5", text)

        # In compare mode the canvas draws the two together rather than the
        # reader's mask twice.
        canvas = self.window.view_panels["axial"].canvas
        self.assertTrue(canvas._external_compare)

        # Comparing against an overlay that is switched off would show the
        # model's mask and none of the reader's, which reads as a model that
        # found what the reader missed.
        self.window.show_roi_cb.setChecked(False)
        self.window.external_show_cb.setChecked(False)
        self.window.external_compare_cb.setChecked(False)
        self.window.external_compare_cb.setChecked(True)
        self.assertTrue(self.window.show_roi_cb.isChecked())
        self.assertTrue(self.window.external_show_cb.isChecked())
        self.assertTrue(self.window.view_panels["axial"].canvas._external_compare)

        # And the empty entry is the way back out.
        self.window.external_combo.setCurrentIndex(0)
        self.assertIsNone(self.window.external_mask)
        self.assertIsNone(canvas._external_volume)

    def test_a_result_on_another_grid_is_refused_rather_than_resampled(self) -> None:
        import nibabel as nib

        import external_results

        root = self._external_folder()
        self._turn_on_developer_mode(root)
        self.window.external_combo.setCurrentIndex(1)

        wrong = self.temp_dir / "wrong_grid.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.uint8), np.eye(4)), str(wrong))
        self.window.load_external_result(
            external_results.ExternalResult(self.case_id, wrong.name, wrong)
        )
        self.assertIsNone(self.window.external_mask)
        self.assertIn("not on this case's voxel grid", self.window._status_label.text())

    def test_the_result_is_dropped_when_the_case_changes(self) -> None:
        self._turn_on_developer_mode(self._external_folder())
        self.window.external_combo.setCurrentIndex(1)
        self.assertIsNotNone(self.window.external_mask)

        other = next(
            item["case_id"]
            for item in self.window.all_cases
            if item["case_id"] != self.case_id
        )
        self.window.load_case(other)
        self.assertIsNone(
            self.window.external_mask,
            "one case's model output stayed on screen over another case's images",
        )
        self.assertIsNone(self.window.view_panels["axial"].canvas._external_volume)

    def test_the_two_overlays_have_a_key_each(self) -> None:
        """S is this reader's mask; Shift+S is the other one.

        One key for both would mean a reader comparing two masks has to reach
        for the mouse to hide either of them.
        """

        self._turn_on_developer_mode(self._external_folder())
        self.window.external_combo.setCurrentIndex(1)
        canvas = self.window.view_panels["axial"].canvas

        self.assertEqual(self.settings.shortcut("toggle_external_overlay"), "Shift+S")
        self.assertNotEqual(
            self.settings.shortcut("toggle_roi_overlay"),
            self.settings.shortcut("toggle_external_overlay"),
        )

        self.window._toggle_external_overlay_key()
        self.assertIsNone(canvas._external_volume, "Shift+S did not take it off")
        self.assertTrue(canvas._show_labels, "Shift+S took the reader's own mask off too")
        self.window._toggle_external_overlay_key()
        self.assertIsNotNone(canvas._external_volume)

        self.window.show_roi_cb.toggle()
        self.assertFalse(canvas._show_labels)
        self.assertIsNotNone(canvas._external_volume, "S took the external mask off too")

    def test_a_case_load_never_leaves_the_window_open_to_clicks(self) -> None:
        """The progress dialog is modal, or it is decoration.

        Loading holds the GUI thread, and the events that arrive during it are
        replayed afterwards unless something else is there to take them. The
        dialog is that something else.
        """

        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QPushButton

        progress = self.desktop_app.LoadProgress(self.window, "Opening…", steps=2)
        try:
            self.assertEqual(
                progress._dialog.windowModality(), Qt.WindowModality.ApplicationModal
            )
            self.assertIsNone(
                progress._dialog.findChild(QPushButton), "a cancel that cannot cancel"
            )
            # Not shown yet: a 470 ms local load should not flash a dialog.
            self.assertFalse(progress._dialog.isVisible())
            progress.show_now()
            self.assertTrue(progress._dialog.isVisible())
            progress.busy("Downloading…")
            self.assertEqual((progress._dialog.minimum(), progress._dialog.maximum()), (0, 0))
            progress.measured(2, 1)
            self.assertEqual(progress._dialog.value(), 1)
        finally:
            progress.close()

    def test_finding_and_manual_ras_use_the_same_physical_target(self) -> None:
        target = self.review_store.list_targets(self.db_path, self.case_id, "Desktop Test Reader", 1)[0]
        expected = (float(target["ras"][0]), float(target["ras"][1]), float(target["ras"][2]))
        self.assertEqual(tuple(round(value, 5) for value in self.window.target_ras), tuple(round(value, 5) for value in expected))
        self.assertEqual(self.window.view_panels["axial"].canvas.slice_index, 55)
        self.assertEqual(self.window.view_panels["coronal"].canvas.slice_index, 64)
        self.assertEqual(self.window.view_panels["sagittal"].canvas.slice_index, 152)

        for spin, value in zip((self.window.ras_l_spin, self.window.ras_p_spin, self.window.ras_s_spin), expected):
            spin.setValue(value)
        self.window.jump_to_ras()
        self.assertEqual(tuple(round(value, 5) for value in self.window.target_ras), tuple(round(value, 5) for value in expected))
        self.assertEqual(self.window.view_panels["axial"].canvas.slice_index, 55)
        # A coordinate jump is navigation only. The finding stays selected so
        # the reader can still save a review and return to it.
        self.assertIsNotNone(self.window.selected_target)
        self.assertEqual(self.window.selected_target["target_id"], target["target_id"])
        self.assertEqual(self.window.target_list.currentRow(), 0)

    def test_navigation_keeps_the_finding_selected_and_return_works(self) -> None:
        import numpy as np

        finding_ras = tuple(self.window.target_ras)
        label = self.window.selected_target["label"]
        self.window._on_canvas_target_clicked("axial", np.asarray([12.0, 14.0, 16.0]))
        self.assertIsNotNone(self.window.selected_target)
        self.assertEqual(self.window.selected_target["label"], label)
        self.assertNotEqual(tuple(self.window.target_ras), finding_ras)
        self.assertTrue(self.window.return_btn.isEnabled())
        self.window.return_to_finding()
        self.assertEqual(
            tuple(round(value, 6) for value in self.window.target_ras),
            tuple(round(value, 6) for value in finding_ras),
        )
        self.window.comment_edit.setPlainText("still saveable after navigating")
        self.assertTrue(self.window.save_current_review())

    def _send_wheel(self, canvas, angle: int = 120, modifiers=None):
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QWheelEvent

        if modifiers is None:
            modifiers = Qt.KeyboardModifier.NoModifier
        position = QPointF(canvas.width() / 2, canvas.height() / 2)
        event = QWheelEvent(
            position,
            canvas.mapToGlobal(position),
            QPoint(0, 0),
            QPoint(0, angle),
            Qt.MouseButton.NoButton,
            modifiers,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        self.app.sendEvent(canvas, event)
        self.app.processEvents()

    def test_wheel_notch_moves_exactly_one_slice(self) -> None:
        from PySide6.QtCore import Qt

        axial = self.window.view_panels["axial"].canvas
        coronal = self.window.view_panels["coronal"].canvas
        start = axial.slice_index
        coronal_start = coronal.slice_index
        self._send_wheel(axial, 120)
        self.assertEqual(axial.slice_index, start + 1)
        self.assertEqual(coronal.slice_index, coronal_start)
        self._send_wheel(axial, -120)
        self.assertEqual(axial.slice_index, start)
        self._send_wheel(axial, 120, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(axial.slice_index, start + 5)

    def test_zoom_combo_follows_the_canvas(self) -> None:
        from PySide6.QtCore import Qt

        panel = self.window.view_panels["axial"]
        self._send_wheel(panel.canvas, 120, Qt.KeyboardModifier.ControlModifier)
        self.assertNotEqual(panel.canvas.zoom_text, "Autofit")
        self.assertEqual(panel.zoom_combo.currentText(), panel.canvas.zoom_text)
        self.window.reset_views()
        self.assertEqual(panel.zoom_combo.currentText(), "Autofit")

    def test_modality_switch_keeps_zoom_and_target(self) -> None:
        axial = self.window.view_panels["axial"].canvas
        axial.set_zoom_mode("200%")
        target = tuple(self.window.target_ras)
        self.window.set_modality("qsm")
        self.assertEqual(self.window.current_modality, "qsm")
        self.assertEqual(axial.zoom_text, "200%")
        self.assertEqual(tuple(self.window.target_ras), target)

    def test_status_message_and_live_coordinate_are_both_visible(self) -> None:
        import numpy as np

        self.window._set_status("status message")
        self.window._on_mouse_voxel_moved("axial", np.asarray([10.0, 20.0, 30.0]))
        self.assertTrue(self.window._status_label.isVisible())
        self.assertTrue(self.window._coord_label.isVisible())
        self.assertEqual(self.window._status_label.text(), "status message")
        self.assertIn("RAS", self.window._coord_label.text())

    def test_slice_logging_is_debounced(self) -> None:
        from PySide6.QtCore import QElapsedTimer

        case_id = self.window.current_case_id

        def slice_rows() -> int:
            entries = self.review_store.recent_case_log(self.db_path, case_id, limit=5000)
            return sum(1 for item in entries if item["event_type"] == "slice_changed")

        # Writes are queued now, so settle the queue before counting.
        self.assertTrue(self._wait_for(lambda: self.window._writer.pending() == 0))
        before = slice_rows()
        axial = self.window.view_panels["axial"].canvas
        for _ in range(10):
            self._send_wheel(axial, 120)
        self.assertEqual(slice_rows() - before, 0)
        timer = QElapsedTimer()
        timer.start()
        while timer.elapsed() < 1400:
            self.app.processEvents()
        self.assertTrue(self._wait_for(lambda: self.window._writer.pending() == 0))
        # Ten wheel steps leave one row: where the scrolling stopped.
        self.assertEqual(slice_rows() - before, 1)

    def test_sync_scroll_zoom_and_modality_switch(self) -> None:
        self.window._on_slice_request("axial", 56)
        self.assertEqual(self.window.view_panels["axial"].canvas.slice_index, 56)
        self.assertEqual(self.window.view_panels["coronal"].canvas.slice_index, 64)
        self.assertEqual(self.window.view_panels["sagittal"].canvas.slice_index, 152)

        axial = self.window.view_panels["axial"].canvas
        axial.set_zoom_mode("200%")
        self.assertAlmostEqual(axial._zoom_multiplier, 2.0)
        axial.reset_view()
        self.assertEqual(axial.zoom_mode, "fit")

        self.window.set_modality("qsm")
        self.assertEqual(self.window.current_modality, "qsm")
        self.assertIsNotNone(self.window.volumes["qsm"])
        self.assertEqual(self.window.modality_segments.current_key(), "qsm")
        self.window.set_modality("mip")
        self.assertEqual(self.window.current_modality, "mip")
        self.assertIsNotNone(self.window.volumes["mip"])

    def test_coronal_and_sagittal_display_superior_to_inferior_without_flip(self) -> None:
        from imaging import Volume

        data = np.zeros((4, 5, 6), dtype=np.float32)
        for z in range(data.shape[2]):
            data[:, :, z] = float(z)
        volume = Volume(
            path="synthetic",
            data=data,
            affine=np.eye(4, dtype=np.float64),
            shape=data.shape,
            voxel_sizes=(1.0, 1.0, 1.0),
        )

        for plane in ("coronal", "sagittal"):
            canvas = self.desktop_app.SliceCanvas(plane)
            canvas.set_volume(volume)
            canvas.set_slice(1)
            displayed = canvas._display_array()
            self.assertIsNotNone(displayed)
            np.testing.assert_array_equal(displayed[0], 0.0)
            np.testing.assert_array_equal(displayed[-1], 5.0)

        sagittal = self.desktop_app.SliceCanvas("sagittal")
        sagittal.set_volume(volume)
        sagittal.set_slice(1)
        # Pixmap coordinates, so the middle of pixel (3, 4) is (3.5, 4.5); see
        # ``CanvasGeometryTests`` for why that half pixel matters.
        np.testing.assert_array_equal(sagittal._voxel_from_image_point(3.5, 4.5), [1.0, 3.0, 4.0])

    def test_wheel_pixel_delta_requires_accumulation(self) -> None:
        remainder = 0.0
        steps = []
        for _ in range(3):
            step, remainder = self.desktop_app.quantize_wheel_delta(
                20, pixel=True, remainder=remainder
            )
            steps.append(step)
        self.assertEqual(steps, [0, 0, 0])
        step, remainder = self.desktop_app.quantize_wheel_delta(
            20, pixel=True, remainder=remainder
        )
        self.assertEqual(step, 1)
        self.assertAlmostEqual(remainder, 0.0)
        step, remainder = self.desktop_app.quantize_wheel_delta(
            120, pixel=False, remainder=0.0
        )
        self.assertEqual(step, 1)
        self.assertAlmostEqual(remainder, 0.0)

    def test_wheel_zoom_uses_fine_steps(self) -> None:
        canvas = self.desktop_app.SliceCanvas("axial")
        canvas.step_zoom(1)
        self.assertAlmostEqual(canvas._zoom_multiplier, 1.10, places=6)
        canvas.step_zoom(-1)
        self.assertAlmostEqual(canvas._zoom_multiplier, 1.00, places=6)

    def test_sagittal_geometry_uses_voxel_spacing(self) -> None:
        from imaging import Volume

        data = np.zeros((4, 5, 6), dtype=np.float32)
        volume = Volume(
            path="synthetic",
            data=data,
            affine=np.eye(4, dtype=np.float64),
            shape=data.shape,
            voxel_sizes=(2.0, 3.0, 4.0),
        )
        canvas = self.desktop_app.SliceCanvas("sagittal")
        canvas.resize(600, 600)
        canvas.set_volume(volume)
        image = canvas._display_array()
        self.assertIsNotNone(image)
        rect, _scales = canvas._draw_geometry(int(image.shape[1]), int(image.shape[0]))
        expected_ratio = (data.shape[1] * 3.0) / (data.shape[2] * 4.0)
        self.assertAlmostEqual(rect.width() / rect.height(), expected_ratio, places=6)

    def test_crosshair_toggles_and_missing_case_are_explicit(self) -> None:
        self.window.target_crosshair_cb.setChecked(False)
        self.window.mouse_crosshair_cb.setChecked(False)
        self.assertTrue(all(not panel.canvas._show_target for panel in self.window.view_panels.values()))
        self.assertTrue(all(not panel.canvas._show_mouse for panel in self.window.view_panels.values()))

        self.window.hide_missing_cb.setChecked(False)
        unusable = next(
            (
                item["case_id"]
                for item in self.window.all_cases
                if item["file_status"] == "all_missing"
            ),
            None,
        )
        if unusable is None:
            self.skipTest("The configured dataset has no case with unusable files.")
        self.window.load_case(unusable)
        self.assertTrue(
            self._wait_for(
                lambda: self.window.current_case_id == unusable
                and self.window._load_thread is None
            )
        )
        self.assertTrue(all(volume is None for volume in self.window.volumes.values()))
        # Cases open on SWI by default, so that is the sequence reported here.
        message = self.window.view_panels["axial"].canvas._missing_message
        self.assertIn("Required SWI AffineRestored file is missing", message)
        self.assertIn("_GRE4_SWI_AffineRestored.nii.gz", message)

    def test_reader_review_is_persisted_and_visible_to_other_reader(self) -> None:
        self.window.set_verdict(1)
        self.window.comment_edit.setPlainText("Desktop GUI QA")
        self.assertTrue(self.window.save_current_review(advance=False))
        saved = self.review_store.list_targets(self.db_path, self.case_id, "Other Reader", 1)[0]
        self.assertTrue(any(report["comment"] == "Desktop GUI QA" for report in saved["reader_reports"]))

    def test_resume_restores_last_case_modality_and_filter_state(self) -> None:
        # "Last used" is the preference under which the session's sequence is
        # meant to be restored.
        self.settings.update(
            auto_zoom=True, lesion_fov_mm=60.0, save_advances=True, default_modality="last"
        )
        self.window.set_modality("qsm")
        self.window.case_search.setText(self.case_id)
        self.window._save_session_state()
        session_id = self.session["session_id"]
        self.window.close()
        resumed = self.review_store.resume_session(self.db_path, session_id)
        self.window = self.desktop_app.MicrobleedViewer(
            self.db_path, self.data_root, resumed, settings=self.settings
        )
        self.window.show()
        self.assertTrue(
            self._wait_for(
                lambda: self.window.current_case_id == self.case_id
                and self.window.current_modality == "qsm"
                and self.window._load_thread is None
                and self.window.volumes["qsm"] is not None
            )
        )
        self.assertEqual(self.window.case_search.text(), self.case_id)

    # ------------------------------------------------- reading-workflow UI --
    def test_cases_open_on_swi_with_the_lesion_framed(self) -> None:
        self.assertEqual(self.settings.default_modality, "swi")
        self.assertEqual(self.window.current_modality, "swi")
        self.assertEqual(self.window.modality_segments.current_key(), "swi")
        canvas = self.window.view_panels["axial"].canvas
        self.assertTrue(canvas.lesion_focus)
        self.assertTrue(self.window.lesion_zoom_btn.isChecked())

        # The framed field of view must match the preference, and the target
        # must sit in the middle of the viewport.
        image = canvas._display_array()
        rect, (scale_x, _scale_y) = canvas._draw_geometry(int(image.shape[1]), int(image.shape[0]))
        point = canvas._target_image_point()
        self.assertAlmostEqual(rect.left() + point[0] * scale_x, canvas.width() / 2, delta=1.5)
        millimetres_per_pixel = float(canvas.volume.voxel_sizes[0]) / scale_x
        usable = min(canvas.width(), canvas.height()) - 56.0
        self.assertAlmostEqual(usable * millimetres_per_pixel, self.settings.lesion_fov_mm, delta=1.0)

    def test_lesion_zoom_toggles_and_survives_a_sequence_switch(self) -> None:
        canvas = self.window.view_panels["axial"].canvas
        self.window.toggle_lesion_focus()
        self.assertFalse(canvas.lesion_focus)
        self.assertEqual(self.window.view_panels["axial"].zoom_combo.currentText(), "Autofit")
        self.window.toggle_lesion_focus()
        self.assertTrue(canvas.lesion_focus)
        self.assertEqual(
            self.window.view_panels["axial"].zoom_combo.currentText(),
            self.desktop_app.LESION_ZOOM_LABEL,
        )
        self.window.set_modality("qsm")
        self.assertTrue(canvas.lesion_focus)

    def test_verdict_keys_drive_the_segmented_control(self) -> None:
        self.window.set_verdict(1)
        self.assertEqual(self.window._verdict, 1)
        self.assertEqual(self.window.verdict_segments.current_key(), "yes")
        self.assertTrue(self.window._review_dirty)
        self.window.set_verdict(0)
        self.assertEqual(self.window.verdict_segments.current_key(), "no")
        self.window.set_verdict(None)
        self.assertEqual(self.window.verdict_segments.current_key(), "unset")

    def test_saving_advances_to_the_next_case_when_the_case_is_done(self) -> None:
        first_case = self.window.current_case_id
        self.window.set_verdict(0)
        self.assertTrue(self.window.save_current_review())
        self.assertNotEqual(self.window.current_case_id, first_case)
        saved = self.review_store.list_targets(
            self.db_path, first_case, "Desktop Test Reader", 1
        )
        self.assertEqual(saved[0]["reader_verify"], 0)

    def test_saving_can_be_kept_on_the_same_finding(self) -> None:
        self.settings.update(
            auto_zoom=True, lesion_fov_mm=60.0, save_advances=False, default_modality="swi"
        )
        self.window._update_save_buttons()
        first_case = self.window.current_case_id
        self.window.set_verdict(1)
        self.assertTrue(self.window.save_current_review())
        self.assertEqual(self.window.current_case_id, first_case)

    def test_missing_preferred_sequence_falls_back_to_an_available_one(self) -> None:
        self.window.volumes["swi"] = None
        substituted = self.window._resolve_case_modality()
        self.assertEqual(substituted, "swi")
        self.assertEqual(self.window.current_modality, "qsm")
        self.window.volumes["qsm"] = None
        self.window.volumes["mip"] = None
        self.assertIsNone(self.window._resolve_case_modality())

    def test_maximizing_a_view_hides_the_others_and_enlarges_it(self) -> None:
        panel = self.window.view_panels["coronal"]
        before = (panel.canvas.width(), panel.canvas.height())
        self.window.toggle_maximized_view("coronal")
        self.app.processEvents()
        visible = [name for name, item in self.window.view_panels.items() if item.isVisible()]
        self.assertEqual(visible, ["coronal"])
        # Hiding the other panels is not enough: their rows and columns keep
        # their stretch unless it is moved onto the visible cell.
        after = (panel.canvas.width(), panel.canvas.height())
        self.assertGreater(after[0], before[0])
        self.assertGreater(after[1], before[1])

        self.window.toggle_maximized_view("coronal")
        self.app.processEvents()
        self.assertTrue(all(item.isVisible() for item in self.window.view_panels.values()))
        # Back to its own cell, the same size as the other small one.
        restored = (panel.canvas.width(), panel.canvas.height())
        self.assertLess(restored[0], after[0])
        self.assertLess(restored[1], after[1])
        # Sagittal shares the right column with coronal, so they are the same
        # width; axial shares the left one with the location panel, which has a
        # wider minimum of its own.
        sibling = self.window.view_panels["sagittal"].canvas
        self.assertAlmostEqual(restored[0], sibling.width(), delta=2)
        self.assertAlmostEqual(restored[1], sibling.height(), delta=2)

    # ------------------------------------------------- fixed finding marker --
    def test_the_finding_marker_stays_put_while_scrolling(self) -> None:
        canvas = self.window.view_panels["axial"].canvas
        marker_ras = tuple(self.window.marker_ras)
        marker_voxel = canvas._marker_voxel.copy()
        start = canvas.slice_index
        self.assertTrue(canvas.marker_on_slice())

        self.window._on_slice_step("axial", 4)
        self.assertEqual(canvas.slice_index, start + 4)
        # The marker did not move in physical space, in voxels, or on screen.
        self.assertEqual(tuple(self.window.marker_ras), marker_ras)
        np.testing.assert_array_equal(canvas._marker_voxel, marker_voxel)
        # And it is no longer drawn, because this slice does not contain it.
        self.assertFalse(canvas.marker_on_slice())

        self.window._on_slice_step("axial", -4)
        self.assertEqual(canvas.slice_index, start)
        self.assertTrue(canvas.marker_on_slice())

    def test_scrolling_one_view_does_not_move_the_others(self) -> None:
        axial = self.window.view_panels["axial"].canvas
        coronal = self.window.view_panels["coronal"].canvas
        sagittal = self.window.view_panels["sagittal"].canvas
        before = (coronal.slice_index, sagittal.slice_index)
        coronal_marker = coronal._marker_voxel.copy()
        self.window._on_slice_step("axial", 6)
        self.assertEqual((coronal.slice_index, sagittal.slice_index), before)
        # The marker keeps its place inside the other planes as well, so it no
        # longer slides across the coronal image while the axial scrolls.
        np.testing.assert_array_equal(coronal._marker_voxel, coronal_marker)
        self.assertNotEqual(axial.slice_index, coronal.slice_index)

    def test_selecting_a_finding_recentres_and_repins_the_marker(self) -> None:
        canvas = self.window.view_panels["axial"].canvas
        self.window._on_slice_step("axial", 7)
        self.assertFalse(canvas.marker_on_slice())
        self.window.return_to_finding()
        self.assertTrue(canvas.marker_on_slice())

    def test_other_findings_of_the_case_are_marked(self) -> None:
        multi = next(
            (
                case["case_id"]
                for case in self.window.all_cases
                if case["source_count"] >= 3 and case["file_status"] == "complete"
            ),
            None,
        )
        if multi is None:
            self.skipTest("No complete multi-finding case in this data root.")
        self.window.load_case(multi)
        self.assertTrue(
            self._wait_for(
                lambda: self.window.current_case_id == multi
                and self.window.volumes.get(self.window.current_modality) is not None
            )
        )
        canvas = self.window.view_panels["axial"].canvas
        self.assertEqual(len(canvas._secondary_voxels), len(self.window.targets) - 1)
        self.assertEqual(self.window.target_count_label.text(), f"1 of {len(self.window.targets)}")
        self.window.step_finding(1)
        self.assertEqual(self.window.target_count_label.text(), f"2 of {len(self.window.targets)}")

    # -------------------------------------------------- orientation presets --
    def test_display_preset_mirrors_the_view_without_moving_the_finding(self) -> None:
        import numpy as np

        from imaging import clamp_voxel, preset_axcodes, ras_to_voxel

        canvas = self.window.view_panels["axial"].canvas
        self.assertEqual(tuple(canvas.volume.orientation), ("L", "P", "I"))
        self.assertEqual(
            self.desktop_app.plane_directions("axial", tuple(canvas.volume.orientation))[0], "R"
        )
        finding_ras = tuple(self.window.target_ras)
        before = self.window.volumes["swi"]
        voxel_before = ras_to_voxel(before.affine, finding_ras)
        value_before = float(before.data[tuple(int(x) for x in clamp_voxel(voxel_before, before.shape))])

        self.settings.store.setValue("display/orientation", "neurological")
        self.assertEqual(self.settings.axcodes, preset_axcodes("neurological"))
        self.window.load_case(self.window.current_case_id, force=True)
        self.assertTrue(
            self._wait_for(lambda: self.window.volumes.get("swi") is not None)
        )

        canvas = self.window.view_panels["axial"].canvas
        self.assertEqual(tuple(canvas.volume.orientation), ("R", "P", "I"))
        # The label follows the array, so the left of the image is now L.
        self.assertEqual(
            self.desktop_app.plane_directions("axial", tuple(canvas.volume.orientation))[0], "L"
        )
        # The finding did not move in physical space, and still lands on the
        # same tissue; only its column index is mirrored.
        self.assertEqual(tuple(self.window.target_ras), finding_ras)
        after = self.window.volumes["swi"]
        voxel_after = ras_to_voxel(after.affine, finding_ras)
        value_after = float(after.data[tuple(int(x) for x in clamp_voxel(voxel_after, after.shape))])
        self.assertEqual(value_before, value_after)
        self.assertAlmostEqual(
            float(voxel_before[0]) + float(voxel_after[0]), before.shape[0] - 1, places=4
        )
        np.testing.assert_allclose(voxel_before[1:], voxel_after[1:], atol=1e-6)

    def test_clicking_the_mirrored_voxel_returns_the_same_ras(self) -> None:
        import numpy as np

        canvas = self.window.view_panels["axial"].canvas
        voxel = np.asarray([40.0, 55.0, 60.0])
        self.window._on_canvas_target_clicked("axial", voxel)
        ras_radiological = tuple(self.window.target_ras)

        self.settings.store.setValue("display/orientation", "neurological")
        self.window.load_case(self.window.current_case_id, force=True)
        self.assertTrue(self._wait_for(lambda: self.window.volumes.get("swi") is not None))
        width = self.window.volumes["swi"].shape[0] - 1
        mirrored = np.asarray([width - voxel[0], voxel[1], voxel[2]])
        self.window._on_canvas_target_clicked("axial", mirrored)
        for left, right in zip(ras_radiological, self.window.target_ras):
            self.assertAlmostEqual(left, right, places=6)

    # ---------------------------------------------------------- view layout --
    def test_the_window_is_three_columns(self) -> None:
        """Queue on the left at full height, images and the decision in the
        middle, reference and drawing on the right.

        The toolbar sits over the middle and right only: a heading above a
        list of cases would be a heading about something else.
        """

        central = self.window.centralWidget()

        def box(widget):
            top_left = widget.mapTo(central, widget.rect().topLeft())
            return top_left.x(), top_left.y(), widget.width(), widget.height()

        queue = box(self.window.queue_panel)
        toolbar = box(self.window.case_title.parentWidget().parentWidget())
        axial = box(self.window.view_panels["axial"])
        right = box(self.window.right_column)

        self.assertLess(queue[0], toolbar[0], "the toolbar reaches over the case queue")
        self.assertLess(queue[1], toolbar[1] + 4, "the queue does not start at the top")
        self.assertGreater(queue[3], axial[3], "the queue is not full height")
        self.assertLess(axial[0], right[0])
        self.assertGreaterEqual(right[1], toolbar[1] + toolbar[3], "the right column is under the toolbar")

        findings = box(self.window.findings_panel)
        details = box(self.window.details_panel)
        self.assertGreaterEqual(findings[0], right[0], "the findings list left the right column")
        self.assertLess(
            findings[1], details[1], "the findings list is not above This finding"
        )

        grid = self.window.view_grid
        placed = {}
        for plane, panel in self.window.view_panels.items():
            row, column, _rs, _cs = grid.getItemPosition(grid.indexOf(panel))
            placed[plane] = (row, column)
        row, column, _rs, _cs = grid.getItemPosition(grid.indexOf(self.window.location_panel))
        self.assertEqual(placed["axial"], (0, 0))
        self.assertEqual(placed["sagittal"], (0, 1))
        self.assertEqual((row, column), (1, 0), "deciding where the finding is left the grid")
        self.assertEqual(placed["coronal"], (1, 1))

    def test_the_declared_minimum_width_is_one_the_layout_can_honour(self) -> None:
        """A minimum the layout cannot meet is not a minimum.

        The window said 1340 while the four columns needed 1680 with the case
        queue open, so Qt quietly refused to make it that small and the number
        described nothing.  There are two now, and the window swaps between
        them as the queue folds.
        """

        for pinned, declared in (
            (True, self.desktop_app.QUEUE_PINNED_MIN_WIDTH),
            (False, self.desktop_app.MINIMUM_WINDOW_WIDTH),
        ):
            self.window.set_queue_pinned(pinned)
            self.window.resize(declared, 800)
            self._wait_for(lambda: False, timeout_ms=250)
            with self.subTest(queue="pinned" if pinned else "folded"):
                self.assertEqual(self.window.minimumWidth(), declared)
                self.assertEqual(
                    self.window.width(),
                    declared,
                    "the window would not shrink to the width it claims",
                )
                for key, scroll in self.window.panel_pages.items():
                    self.assertLessEqual(
                        scroll.widget().minimumSizeHint().width(),
                        scroll.viewport().width(),
                        f"the {key} panel is clipped at the declared minimum",
                    )
        self.window.set_queue_pinned(True)

    def test_a_narrow_screen_folds_the_queue_without_forgetting_the_choice(self) -> None:
        """Otherwise Qt answers by making the window wider than the display.

        Folding is the escape this column was built for, so the window takes
        it -- but it is the window's decision, not the reader's, so the
        preference is left alone and the queue comes back on a wider screen.
        """

        self.window.set_queue_pinned(True)
        self.assertTrue(self.window.settings.queue_pinned)
        available = (
            self.window.screen() or self.app.primaryScreen()
        ).availableGeometry().width()
        self.assertLess(
            available,
            self.desktop_app.QUEUE_PINNED_MIN_WIDTH,
            "this check needs a screen narrower than the pinned layout",
        )

        self.window._queue_auto_folded = False
        self.window._fit_queue_to_screen()
        self._wait_for(lambda: False, timeout_ms=150)
        self.assertFalse(self.window.queue_pinned)
        self.assertTrue(self.window._queue_auto_folded)
        self.assertTrue(
            self.window.settings.queue_pinned,
            "the window overwrote the reader's own choice",
        )
        self.assertIn("folded", self.window._status_label.text().lower())

        # Folding it by hand is the reader's decision and is remembered.
        self.window.set_queue_pinned(False)
        self.assertFalse(self.window.settings.queue_pinned)
        self.assertFalse(self.window._queue_auto_folded)

    def test_the_folded_rail_reads_as_a_word(self) -> None:
        """One capital per line is what a 26px strip does to a plain label."""

        caption = self.window.queue_rail_caption
        self.assertIsInstance(caption, self.desktop_app.VerticalLabel)
        self.assertIn("CASE QUEUE", caption.text())
        # Rotated: it asks for a tall, narrow box rather than a wide one.
        hint = caption.sizeHint()
        self.assertGreater(hint.height(), hint.width() * 3)
        self.assertLessEqual(hint.width(), self.window.queue_rail.width())

    def test_double_click_picks_a_position_instead_of_resetting_the_zoom(self) -> None:
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        canvas = self.window.view_panels["axial"].canvas
        canvas.set_zoom_mode("200%")
        before_zoom = canvas.zoom_text
        position = QPointF(canvas.width() / 2 + 12, canvas.height() / 2 + 8)
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonDblClick,
            position,
            canvas.mapToGlobal(position),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas.mouseDoubleClickEvent(event)
        self.app.processEvents()
        # The zoom is untouched: an accidental double-click while picking a
        # position no longer throws the view back to Autofit.
        self.assertEqual(canvas.zoom_text, before_zoom)

    def test_clicking_the_current_finding_jumps_back_to_it(self) -> None:
        canvas = self.window.view_panels["axial"].canvas
        item = self.window.target_list.item(self.window.target_list.currentRow())
        # Already on the finding: clicking it changes nothing.
        self.window._on_target_item_clicked(item)
        self.assertTrue(canvas.marker_on_slice())

        self.window._on_slice_step("axial", 5)
        self.assertFalse(canvas.marker_on_slice())
        # Clicking the row that is already selected brings the views back.
        self.window._on_target_item_clicked(item)
        self.assertTrue(canvas.marker_on_slice())
        self.assertEqual(self.window.target_ras, self.window.marker_ras)

    def test_the_clicked_position_is_marked_in_every_view(self) -> None:
        import numpy as np

        from imaging import ras_to_voxel

        volume = self.window.volumes[self.window.current_modality]
        voxel = ras_to_voxel(volume.affine, self.window.marker_ras)
        clicked = np.asarray([voxel[0] + 9.0, voxel[1] + 7.0, voxel[2]])
        self.window._on_canvas_target_clicked("axial", clicked)
        for plane, panel in self.window.view_panels.items():
            canvas = panel.canvas
            self.assertIsNotNone(canvas._target_voxel, plane)
            # The cursor is somewhere else than the finding, so each view has
            # something to draw as the clicked guide.
            self.assertFalse(
                np.allclose(canvas._target_voxel, canvas._marker_voxel, atol=0.5), plane
            )
        self.assertTrue(self.window.view_panels["axial"].canvas._show_mouse)

    # ----------------------------------------------------------- panel layout --
    def test_the_review_panel_fits_without_clipping(self) -> None:
        """Nothing in the reading panels may be cut off at a supported size.

        A QLabel that cannot wrap reports its whole text as its minimum width,
        so a single long help sentence used to force a panel to 1738 px and
        everything past the viewport was simply unreachable.
        """

        from PySide6.QtWidgets import QWidget

        minimum = self.window.minimumSize()
        for width, height in ((minimum.width(), minimum.height()), (1540, 960), (1920, 1080)):
            self.window.resize(width, height)
            self._wait_for(lambda: False, timeout_ms=200)
            areas = dict(self.window.panel_pages)
            areas["details"] = self.window.details_panel
            for key, scroll in areas.items():
                if key in self.window.panel_pages:
                    self.window.show_panel_tab(key)
                    self._wait_for(lambda: False, timeout_ms=100)
                panel = scroll.widget()
                viewport = scroll.viewport().width()
                clipped = [
                    type(child).__name__
                    for child in panel.findChildren(QWidget)
                    if child.isVisible()
                    and child.width() > 12
                    and child.mapTo(panel, child.rect().topLeft()).x() + child.width() > viewport + 1
                ]
                with self.subTest(size=(width, height), panel=key):
                    self.assertLessEqual(
                        panel.minimumSizeHint().width(),
                        viewport,
                        f"the {key} panel demands {panel.minimumSizeHint().width()}px "
                        f"but has {viewport}px",
                    )
                    self.assertEqual(clipped, [], f"{len(clipped)} widgets clipped")

    def test_the_reading_panels_need_no_scrolling(self) -> None:
        """The panels a reader touches on every finding have to fit.

        Measured before the columns: stacked in one place the contents wanted
        471px of a 325px viewport with everything collapsed, and 1219px with
        the segmentation open -- two and a half screenfuls at 1920x1080.
        """

        minimum = self.window.minimumSize()
        self.window.resize(minimum.width(), minimum.height())
        self._wait_for(lambda: False, timeout_ms=250)
        for key, scroll in self.window.panel_pages.items():
            self.window.show_panel_tab(key)
            self._wait_for(lambda: False, timeout_ms=100)
            wanted = scroll.widget().sizeHint().height()
            available = scroll.viewport().height()
            with self.subTest(panel=key):
                self.assertLessEqual(
                    wanted,
                    available,
                    f"the {key} panel wants {wanted}px of a {available}px viewport",
                )

    def test_the_finding_stays_visible_whichever_tab_is_open(self) -> None:
        """Nothing per-finding may hide behind a tab.

        One case here holds twenty-five findings, and painting without being
        able to see which one is selected is painting blind.  The list is in
        the right column now and the verdict and unsaved flag are above the
        tab bar; what matters for both is that neither tab can cover them.
        """

        self.window.resize(1920, 1080)
        self._wait_for(lambda: False, timeout_ms=200)
        for widget, name in (
            (self.window.target_list, "finding list"),
            (self.window.dirty_label, "unsaved flag"),
            (self.window.verdict_summary, "verdict"),
        ):
            for key, scroll in self.window.panel_pages.items():
                self.assertFalse(
                    scroll.widget().isAncestorOf(widget),
                    f"the {name} is inside the {key} tab and vanishes when you leave it",
                )

        self.window.set_verdict(1)
        self.window.show_panel_tab("segment")
        self._wait_for(lambda: False, timeout_ms=150)
        self.assertTrue(self.window.target_list.isVisible())
        self.assertTrue(self.window.right_column.isAncestorOf(self.window.target_list))
        self.assertIn("yes", self.window.verdict_summary.text().lower())
        self.assertIn("unsaved", self.window.dirty_label.text().lower())
        # And the tools that tab exists for are the ones now on screen.
        self.assertTrue(self.window.auto_roi_btn.isVisible())
        self.assertTrue(self.window.brush_spin.isVisible())

    def test_a_findings_row_fits_however_wide_the_column_is_dragged(self) -> None:
        """A row is never cut off, at any width the handle can reach.

        Two ways to be wrong here, and the panel has to avoid both: eliding
        the row, and folding it in half when it would have fitted.  Which of
        those applies depends on the font, so the check is against what the
        widget itself measures rather than against a pixel count -- the
        numbers in the docs were once taken under the offscreen platform's
        fallback font, which is about 40% wider than the real one, and every
        conclusion drawn from them was wrong.
        """

        from desktop_app import RIGHT_COLUMN_MAX_WIDTH, RIGHT_COLUMN_MIN_WIDTH

        self.window.resize(1920, 1080)
        self._wait_for(lambda: False, timeout_ms=250)
        listing = self.window.target_list
        for width in (RIGHT_COLUMN_MIN_WIDTH, 330, 404, RIGHT_COLUMN_MAX_WIDTH):
            self.window.set_right_column_width(width)
            self._wait_for(lambda: False, timeout_ms=120)
            with self.subTest(column=self.window.right_column.width()):
                metrics = listing.fontMetrics()
                room = listing.viewport().width()
                for index, (head, tail) in enumerate(self.window._finding_row_halves):
                    one_line = f"{head}  ·  {tail}"
                    text = listing.item(index).text()
                    if metrics.horizontalAdvance(one_line) + 40 <= room:
                        self.assertNotIn(
                            "\n", text, "folded a row that had room for one line"
                        )
                    else:
                        self.assertIn("\n", text, "left a row on a line it does not fit")
                    # Nothing is dropped either way.
                    self.assertIn(head, text.replace("\n", "  ·  "))
                    self.assertIn(tail, text.replace("\n", "  ·  "))
                # The 300px floor is set by the controls under the list, not by
                # the list, so a row is only guaranteed whole from the width
                # the column opens at upwards.
                if self.window.right_column.width() >= self.desktop_app.RIGHT_COLUMN_DEFAULT_WIDTH:
                    self.assertLessEqual(
                        listing.sizeHintForColumn(0),
                        room,
                        "a findings row is cut off at this column width",
                    )

    def test_the_reference_column_can_be_dragged_between_its_limits(self) -> None:
        """It was draggable before the three-column rebuild took the handle
        away, and how much room reference material deserves is the reader's
        call, not the window's.  What the window owes is limits and memory.
        """

        from desktop_app import RIGHT_COLUMN_MAX_WIDTH, RIGHT_COLUMN_MIN_WIDTH

        self.window.resize(1920, 1080)
        self._wait_for(lambda: False, timeout_ms=200)
        self.assertIs(self.window.body_split.widget(1), self.window.right_column)
        self.assertFalse(self.window.body_split.childrenCollapsible())

        self.window.body_split.setSizes([100, 4000])
        self._wait_for(lambda: False, timeout_ms=120)
        self.assertEqual(self.window.right_column.width(), RIGHT_COLUMN_MAX_WIDTH)
        self.window.body_split.setSizes([4000, 10])
        self._wait_for(lambda: False, timeout_ms=120)
        self.assertEqual(self.window.right_column.width(), RIGHT_COLUMN_MIN_WIDTH)

        # Remembered, and restored on the next window.
        self.window.set_right_column_width(372)
        self.window.settings.set_right_column_width(self.window.right_column.width())
        self.assertEqual(self.window.settings.right_column_width, 372)

        # Widening the window afterwards gives the extra room to the images.
        self.window.set_right_column_width(340)
        self._wait_for(lambda: False, timeout_ms=120)
        before = self.window.right_column.width()
        self.window.resize(1920 + 200, 1080)
        self._wait_for(lambda: False, timeout_ms=200)
        self.assertEqual(self.window.right_column.width(), before)

    def test_the_row_says_which_findings_are_not_from_the_sheet(self) -> None:
        """Name the exception, not the default.

        Every row used to start with "Source ", 43px spent saying what all
        but a handful of findings are.  A finding this reader added is the
        one worth spotting -- it is the only kind that can be deleted.
        """

        # Placed by hand well away from the source findings, so no duplicate
        # guard gets in the way of what this is checking.
        far = tuple(v + 40.0 for v in self.window.targets[0]["ras"])
        self.review_store.add_manual_annotation(
            self.db_path,
            case_id=self.window.current_case_id,
            ras=far,
            reader_id=self.window.reader_id,
            review_round=self.window.review_round,
            atlasregion="pons",
            session_id=self.window.session_id,
        )
        self.window.targets = self.review_store.list_targets(
            self.db_path,
            self.window.current_case_id,
            self.window.reader_id,
            self.window.review_round,
        )
        self.window._populate_target_list()
        rows = [
            self.window.target_list.item(i).text().split("\n", 1)[0]
            for i in range(self.window.target_list.count())
        ]
        manual = [row for row in rows if "Manual" in row]
        self.assertEqual(len(manual), 1, rows)
        self.assertTrue(
            all(not row.startswith("Source") for row in rows),
            f"the default origin is still spelled out: {rows}",
        )
        self.assertTrue(
            any(row.startswith("#") for row in rows), f"no sheet finding left: {rows}"
        )

    def test_every_overlay_can_be_toggled_from_the_keyboard(self) -> None:
        """These are read-time decisions, not settings.

        Turning the mouse crosshair off to look at what is under it, or the
        direction labels off to see a corner, is something done mid-finding --
        and it was three clicks into a menu.  The keys come from the same
        table as every other binding, so the menu prints whatever they are
        bound to now.
        """

        callbacks = self.window._shortcut_callbacks()
        canvas = self.window.view_panels["axial"].canvas
        for action, checkbox, shown in (
            ("overlay_target", self.window.target_crosshair_cb, lambda: canvas._show_target),
            ("overlay_mouse", self.window.mouse_crosshair_cb, lambda: canvas._show_mouse),
            ("overlay_labels", self.window.direction_cb, lambda: canvas._show_directions),
        ):
            with self.subTest(action=action):
                self.assertIn(action, self.window._shortcuts, "not bound to a key")
                before = checkbox.isChecked()
                callbacks[action]()
                self._wait_for(lambda: False, timeout_ms=80)
                self.assertNotEqual(checkbox.isChecked(), before)
                self.assertEqual(shown(), checkbox.isChecked(), "the view did not follow")
                callbacks[action]()
                self._wait_for(lambda: False, timeout_ms=80)
                self.assertEqual(checkbox.isChecked(), before)

        # The menu says which key, and keeps saying it after a rebinding.
        texts = {name: action.text() for _key, action, name in self.window.overlay_actions}
        self.assertIn("Finding crosshair\tX", texts["Finding crosshair"])
        self.window.settings.set_shortcuts({"overlay_mouse": "Ctrl+M"})
        self.window._bind_shortcuts()
        texts = {name: action.text() for _key, action, name in self.window.overlay_actions}
        self.assertIn("Ctrl+M", texts["Mouse crosshair"])

    def test_no_two_actions_share_a_default_key(self) -> None:
        """A collision makes one of the two silently dead."""

        from collections import Counter

        counts = Counter(
            default
            for _action, _label, default, _group in self.desktop_app.SHORTCUT_ACTIONS
            if default
        )
        self.assertEqual(
            [key for key, count in counts.items() if count > 1],
            [],
            "two actions default to the same key",
        )

    def test_the_three_save_buttons_share_the_row(self) -> None:
        """"+ Next" took every spare pixel, which left "+ Segment" without
        room for its own name and read as though the other two were
        afterthoughts."""

        self.window.resize(1920, 1080)
        self._wait_for(lambda: False, timeout_ms=250)
        widths = [
            self.window.save_review_btn.width(),
            self.window.save_segment_btn.width(),
            self.window.save_next_btn.width(),
        ]
        self.assertLessEqual(max(widths) - min(widths), 4, f"uneven save row: {widths}")
        self.assertEqual(self.window.save_segment_btn.text(), "+ Segment")
        self.assertGreaterEqual(
            self.window.save_segment_btn.width(),
            self.window.save_segment_btn.minimumSizeHint().width(),
            "the segment button cannot show its own name",
        )

    def test_a_no_verdict_locks_the_segmentation_tools(self) -> None:
        """A mask saved against "not a microbleed" is a contradiction.

        It would travel into the export and into the agreement table, where
        nothing downstream knows which of the two to believe.  Clear stays
        enabled on purpose: a mask drawn before the verdict changed is still
        the reader's to remove, and locking them out of removing it is how the
        contradiction becomes permanent.
        """

        self.window.set_verdict(1)
        self._wait_for(lambda: False, timeout_ms=80)
        self.assertIsNone(self.window.segmentation_block())
        self.window.set_tool("brush")
        self.assertEqual(self.window.active_tool, "brush")

        self.window.set_verdict(0)
        self._wait_for(lambda: False, timeout_ms=80)
        reason = self.window.segmentation_block()
        self.assertIsNotNone(reason)
        self.assertIn("not a microbleed", reason)
        # The brush already in hand is put down, not left armed over a mask
        # that may no longer be saved.
        self.assertIsNone(self.window.active_tool)
        for name, widget in (
            ("Generate", self.window.auto_roi_btn),
            ("Grow stroke", self.window.grow_stroke_btn),
            ("brush size", self.window.brush_spin),
            ("+ Segment", self.window.save_segment_btn),
            ("brush tool", self.window.tool_buttons["brush"]),
            ("eraser tool", self.window.tool_buttons["eraser"]),
        ):
            self.assertFalse(widget.isEnabled(), f"{name} is still usable")
        self.assertTrue(self.window.clear_roi_btn.isEnabled(), "Clear was taken away too")
        self.assertTrue(self.window.segment_block_label.isVisible())
        self.assertIn("not a microbleed", self.window.segment_block_label.text())

        # And the paths that do not go through a button say no as well.
        self.window.set_tool("brush")
        self.assertIsNone(self.window.active_tool)
        self.assertFalse(self.window.save_and_segment())
        self.window.auto_segment()
        self.assertIn("not a microbleed", self.window._status_label.text())

        self.window.set_verdict(None)
        self._wait_for(lambda: False, timeout_ms=80)
        self.assertIsNone(self.window.segmentation_block())
        self.assertTrue(self.window.auto_roi_btn.isEnabled())
        self.assertFalse(self.window.segment_block_label.isVisible())

    def test_an_unjudged_finding_opens_ready_to_judge(self) -> None:
        """Arriving with a brush in hand offers the one thing you cannot do.

        Segmentation needs a verdict first, so a finding nobody has judged
        opens on Review with the Point tool -- deciding where it is, is what
        happens next.  A finding already judged is left alone: going back over
        your own segmentations is a real way to work.
        """

        multi = next(
            (
                item["case_id"]
                for item in self.window.all_cases
                if item["finding_count"] >= 2 and item["file_status"] == "complete"
            ),
            None,
        )
        if multi is None:
            self.skipTest("No complete multi-finding case in this data root.")
        self.window.load_case(multi)
        self.assertTrue(
            self._wait_for(
                lambda: self.window.current_case_id == multi
                and self.window._load_thread is None
                and len(self.window.targets) >= 2
            )
        )
        self.assertEqual(self.window.current_panel_tab(), "review")
        self.assertEqual(self.window.active_tool, "point")

        # Judge this one and start painting it.
        self.window.set_verdict(1)
        self.assertTrue(self.window.save_current_review(advance=False))
        self.window.set_tool("brush")
        self.assertEqual(self.window.current_panel_tab(), "segment")

        self.window.step_finding(1)
        self._wait_for(lambda: False, timeout_ms=150)
        self.assertEqual(self.window.current_panel_tab(), "review")
        self.assertEqual(self.window.active_tool, "point")

        # Back to the one already judged: nothing moves.
        self.window.set_tool("brush")
        self.window.step_finding(-1)
        self._wait_for(lambda: False, timeout_ms=150)
        self.assertEqual(self.window.current_panel_tab(), "segment")
        self.assertEqual(self.window.active_tool, "brush")

    def test_the_preference_can_leave_the_tab_and_tool_alone(self) -> None:
        """Going back over your own segmentations is a different job."""

        self.settings.update(
            auto_zoom=self.settings.auto_zoom,
            lesion_fov_mm=self.settings.lesion_fov_mm,
            save_advances=self.settings.save_advances,
            default_modality=self.settings.default_modality,
            keep_tool_on_switch=True,
        )
        self.assertTrue(self.settings.keep_tool_on_switch)
        multi = next(
            (
                item["case_id"]
                for item in self.window.all_cases
                if item["finding_count"] >= 2 and item["file_status"] == "complete"
            ),
            None,
        )
        if multi is None:
            self.skipTest("No complete multi-finding case in this data root.")
        self.window.load_case(multi)
        self.assertTrue(
            self._wait_for(
                lambda: self.window.current_case_id == multi
                and self.window._load_thread is None
                and len(self.window.targets) >= 2
            )
        )
        # Saved, so stepping away does not stop to ask about unsaved work.
        self.window.set_verdict(1)
        self.assertTrue(self.window.save_current_review(advance=False))
        self.window.set_tool("brush")
        self.window.step_finding(1)
        self._wait_for(lambda: False, timeout_ms=150)
        self.assertEqual(self.window.current_panel_tab(), "segment")
        self.assertEqual(self.window.active_tool, "brush")

    def test_the_3d_window_shows_the_mask_and_follows_it(self) -> None:
        """A check you run on a mask you just drew, left open while drawing.

        Whether the grower slipped down a vessel, or a stroke is one slice
        thick, is one drag here against a scroll through the stack.  It is
        worth nothing if it shows a mask that is no longer there, so it
        redraws from the same hook as the readout.
        """

        self.window.set_verdict(1)
        self.window.auto_segment()
        self.assertTrue(self._wait_for(lambda: self.window.roi_volume_mm3() > 0))

        self.window.open_lesion_3d()
        self._wait_for(lambda: False, timeout_ms=150)
        dialog = self.window._lesion_dialog
        self.assertIsNotNone(dialog)
        self.assertTrue(dialog.isVisible())
        self.assertTrue(dialog.canvas.has_surface())
        self.assertIn(str(self.window.selected_target["label"]), dialog.heading.text())
        self.assertIn("mm³", dialog.measurements.text())
        self.assertIn("faces", dialog.measurements.text())

        # It is not modal: the reader keeps working while it stands open.
        self.assertFalse(dialog.isModal())

        # Clearing the mask empties it rather than leaving yesterday's lesion.
        self.window.clear_roi()
        self._wait_for(lambda: False, timeout_ms=150)
        self.assertFalse(dialog.canvas.has_surface())

    def test_turning_the_lesion_cannot_roll_it_over_or_lose_it(self) -> None:
        """Past the pole the lesion flips mid-drag with nothing to right it,
        and an unbounded zoom leaves an empty window with no way back."""

        canvas = self.desktop_app.LesionCanvas()
        canvas.resize(300, 300)
        for _ in range(60):
            canvas.turn(0.3, 0.3)
        _yaw, pitch, _zoom = canvas.angles
        self.assertLessEqual(abs(pitch), 1.5)
        for _ in range(80):
            canvas.step_zoom(1)
        self.assertLessEqual(canvas.angles[2], 6.0)
        for _ in range(200):
            canvas.step_zoom(-1)
        self.assertGreaterEqual(canvas.angles[2], 0.4)
        # Double-click puts it all back.
        canvas.reset_view()
        self.assertEqual(
            canvas.angles,
            (self.desktop_app._DEFAULT_YAW, self.desktop_app._DEFAULT_PITCH, 1.0),
        )

    def test_saving_clears_the_unsaved_move_beside_the_position(self) -> None:
        """It said "unsaved move" about a position already written.

        The verdict and the position are saved together, but only the verdict
        half of the panel was rebuilt afterwards, so the combo went on
        offering an "(unsaved)" entry and the warning beside Move stayed up --
        a reader looking at a warning about work that is already safe either
        saves twice or stops trusting the indicator.
        """

        # The same two fields "Move" sets: an unsaved position of my own.
        moved = tuple(value + 2.0 for value in self.window.target_ras)
        self.window.pending_ras = moved
        self.window.selected_variant = self.window.reader_id
        self.window._rebuild_position_variants(self.window.selected_target)
        self._wait_for(lambda: False, timeout_ms=100)
        self.assertEqual(self.window.position_hint.text(), "unsaved move")

        self.window.set_verdict(1)
        self.assertTrue(self.window.save_current_review(advance=False))
        self._wait_for(lambda: False, timeout_ms=150)
        self.assertIsNone(self.window.pending_ras)
        self.assertEqual(self.window.position_hint.text(), "")
        self.assertNotIn(
            "unsaved",
            self.window.position_combo.currentText().lower(),
            "the position list still offers an unsaved entry",
        )

    def test_the_3d_window_can_put_the_lesion_back_in_the_head(self) -> None:
        """Where a microbleed sits is half of what a reader is judging.

        Lobar against deep is the distinction between amyloid angiopathy and
        hypertensive microangiopathy, and it is hard to hold in mind from
        three slice views.  The head is a mean projection of the same volume
        the mask was painted on, so no brain extraction is involved and
        nothing here is a boundary to measure against.
        """

        self.window.set_verdict(1)
        self.window.auto_segment()
        self.assertTrue(self._wait_for(lambda: self.window.roi_volume_mm3() > 0))
        self.window.open_lesion_3d()
        self._wait_for(lambda: False, timeout_ms=150)
        dialog = self.window._lesion_dialog
        canvas = dialog.canvas

        self.assertFalse(canvas.has_context(), "the brain is on without being asked for")
        lesion_scale = canvas._radius_mm

        dialog.brain_cb.setChecked(True)
        self._wait_for(lambda: False, timeout_ms=400)
        self.assertTrue(canvas.has_context())
        self.assertGreater(canvas._radius_mm, lesion_scale * 5, "the view did not zoom out to the head")
        self.assertTrue(dialog.brain_alpha.isEnabled())
        # The lesion is no longer centred: it sits where its coordinate says.
        self.assertGreater(float(np.abs(canvas._lesion_centre_mm).max()), 1.0)
        # The readout names the sequence the head is drawn from.
        self.assertRegex(dialog.measurements.text(), r"of the \w+ centre")

        dialog.brain_cb.setChecked(False)
        self._wait_for(lambda: False, timeout_ms=200)
        self.assertFalse(canvas.has_context())
        self.assertAlmostEqual(canvas._radius_mm, lesion_scale, places=3)

    def test_smoothing_changes_the_picture_and_not_the_numbers(self) -> None:
        """A smoothed mesh is a rendering choice, not a measurement.

        Volume, diameter and voxel count are counted from the voxels, so they
        must read the same either way -- otherwise a display option would be
        quietly editing the result that reaches the export.
        """

        self.window.set_verdict(1)
        self.window.auto_segment()
        self.assertTrue(self._wait_for(lambda: self.window.roi_volume_mm3() > 0))
        self.window.open_lesion_3d()
        self._wait_for(lambda: False, timeout_ms=150)
        dialog = self.window._lesion_dialog

        blocky = np.array(dialog.canvas._quads, copy=True)
        readout = dialog.measurements.text()

        dialog.smooth_cb.setChecked(True)
        self._wait_for(lambda: False, timeout_ms=200)
        smooth = dialog.canvas._quads
        self.assertEqual(len(smooth), len(blocky), "smoothing changed the face count")
        self.assertFalse(np.allclose(smooth, blocky), "Smooth did nothing")
        for measurement in readout.split("\n")[0].split("·"):
            self.assertIn(
                measurement.strip(),
                dialog.measurements.text(),
                "a display option changed a measurement",
            )

    def test_the_3d_view_opens_with_the_head_upright(self) -> None:
        """The display array is (L, P, I), so an unrotated view of it looks
        straight down the I axis -- a head lying on its back, which is what
        this window used to open on.  Superior belongs at the top."""

        canvas = self.desktop_app.LesionCanvas()
        canvas.resize(300, 300)
        rotation = canvas._rotation()

        def on_screen(axis: int):
            direction = np.zeros(3)
            direction[axis] = 1.0
            return direction @ rotation.T

        # Axis 2 is inferior, and inferior points down the screen (y grows
        # downwards), so superior is up.
        inferior = on_screen(2)
        self.assertGreater(inferior[1], 0.9, f"inferior is not downwards: {inferior}")
        self.assertLess(abs(inferior[0]), 0.2, "the head is tilted sideways")
        # Left-right runs mostly across the screen rather than into it, so the
        # opening view is a three-quarter and not a flat profile.
        self.assertGreater(abs(on_screen(0)[0]), 0.5)

    def test_dragging_sideways_spins_the_head_the_way_it_is_pushed(self) -> None:
        """Adding the delta sent the far side that way instead, which reads as
        the model turning backwards under the finger."""

        canvas = self.desktop_app.LesionCanvas()
        canvas.resize(300, 300)
        # Axis 1 is posterior, so the anterior pole -- the part of the head
        # facing the reader when the view opens -- is its negative.
        anterior = np.array([0.0, -1.0, 0.0])
        superior = np.array([0.0, 0.0, -1.0])

        def on_screen(point):
            return point @ canvas._rotation().T

        before = float(on_screen(anterior)[0])
        canvas.turn(-0.3, 0.0)                       # what a rightward drag does
        self.assertGreater(
            float(on_screen(anterior)[0]), before, "the near side went the wrong way"
        )

        # A downward drag tips the top of the head away and down the screen.
        canvas.reset_view()
        before_y = float(on_screen(superior)[1])
        canvas.turn(0.0, 0.3)
        self.assertGreater(float(on_screen(superior)[1]), before_y)

    def test_the_brain_view_shows_every_mask_in_the_case(self) -> None:
        """One lesion in a head does not answer the question the head is for.

        Lobar against deep is a statement about the pattern -- where this
        reader's findings sit relative to each other -- so all of them are
        drawn, the selected one in its own colour and the rest in the colour
        they already have as neighbours on the slices.
        """

        multi = next(
            (
                item["case_id"]
                for item in self.window.all_cases
                if item["finding_count"] >= 3 and item["file_status"] == "complete"
            ),
            None,
        )
        if multi is None:
            self.skipTest("No complete multi-finding case in this data root.")
        self.window.load_case(multi)
        self.assertTrue(
            self._wait_for(
                lambda: self.window.current_case_id == multi
                and self.window._load_thread is None
                and len(self.window.targets) >= 3
            )
        )
        for index in range(3):
            self.window._select_target_index(index, confirm=False)
            self.window.set_verdict(1)
            self.window.auto_segment()
            self._wait_for(lambda: False, timeout_ms=120)
        self.window._select_target_index(0, confirm=False)
        self._wait_for(lambda: False, timeout_ms=120)

        self.window.open_lesion_3d()
        dialog = self.window._lesion_dialog
        dialog.brain_cb.setChecked(True)
        self._wait_for(lambda: False, timeout_ms=400)

        tints = np.unique(dialog.canvas._tints, axis=0)
        self.assertEqual(len(tints), 2, f"expected the selected mask and the rest: {tints}")
        self.assertIn("other mask", dialog.measurements.text())

        from PySide6.QtGui import QColor

        selected = QColor(self.desktop_app.COLORS["roi"])
        mine = np.array([selected.red(), selected.green(), selected.blue()], dtype=float)
        own = int((dialog.canvas._tints == mine).all(axis=1).sum())
        self.assertGreater(own, 0, "the selected mask lost its own colour")
        self.assertLess(own, len(dialog.canvas._tints), "the others were not drawn")

        # Turning the brain off leaves only the selected mask again.
        dialog.brain_cb.setChecked(False)
        self._wait_for(lambda: False, timeout_ms=200)
        self.assertEqual(len(dialog.canvas._quads), own)

    def test_the_head_can_be_drawn_from_another_sequence(self) -> None:
        """QSM and SWI say different things about a microbleed's surroundings.

        SWI shows the parenchyma; QSM shows where the susceptibility is, which
        is the deep nuclei and the veins.  Which reads better is a property of
        the case, so the choice belongs to the reader rather than to this
        window.
        """

        self.window.set_verdict(1)
        self.window.auto_segment()
        self.assertTrue(self._wait_for(lambda: self.window.roi_volume_mm3() > 0))
        self.window.open_lesion_3d()
        dialog = self.window._lesion_dialog
        dialog.brain_cb.setChecked(True)
        self._wait_for(lambda: False, timeout_ms=400)

        offered = [
            dialog.brain_source.itemData(index)
            for index in range(dialog.brain_source.count())
        ]
        available = [
            key
            for key in self.desktop_app.MODALITY_BUTTON_ORDER
            if self.window.volumes.get(key) is not None
        ]
        self.assertEqual(offered, available, "the list is not what the case has")
        self.assertTrue(dialog.brain_source.isEnabled())

        if "qsm" not in offered or "swi" not in offered:
            self.skipTest("This case does not have both sequences.")

        def head_after(key: str):
            dialog.brain_source.setCurrentIndex(dialog.brain_source.findData(key))
            self._wait_for(lambda: False, timeout_ms=400)
            return (
                np.array(dialog.canvas._cube_fine, copy=True),
                np.array(dialog.canvas._lesion_centre_mm, copy=True),
                dialog.measurements.text(),
            )

        swi_cube, swi_place, swi_text = head_after("swi")
        qsm_cube, qsm_place, qsm_text = head_after("qsm")

        self.assertFalse(np.allclose(swi_cube, qsm_cube), "the head did not change")
        # Same lesion, same place: only the backdrop changed.
        self.assertTrue(np.allclose(swi_place, qsm_place, atol=1e-6), "the lesion moved")
        self.assertIn("SWI centre", swi_text)
        self.assertIn("QSM centre", qsm_text)

        # And the picker only matters while the head is on.
        dialog.brain_cb.setChecked(False)
        self._wait_for(lambda: False, timeout_ms=200)
        self.assertFalse(dialog.brain_source.isEnabled())

    def test_a_head_on_a_different_grid_does_not_move_the_lesion(self) -> None:
        """The mask lives on the label reference's grid.

        Measuring it from the head's centre by index would put it wherever the
        two grids disagree.  The three sequences here share a grid exactly, so
        the correction is zero on real data -- this builds a shifted grid so
        the arithmetic is actually exercised.
        """

        from imaging import Volume

        reference = self.window._label_reference()
        self.assertIsNotNone(reference)
        spacing = np.asarray(reference.voxel_sizes[:3], dtype=float)

        same = self.window._placement_centre(reference, reference)
        self.assertTrue(
            np.allclose(same, np.asarray(reference.shape[:3]) * spacing / 2.0),
            "the plain case stopped being the plain case",
        )

        # A head whose world origin sits 10 mm away along each axis.
        shifted_affine = np.array(reference.affine, dtype=float, copy=True)
        shifted_affine[:3, 3] += 10.0
        shifted = Volume(
            path=str(reference.path) + "#shifted",
            data=reference.data,
            affine=shifted_affine,
            shape=reference.shape,
            voxel_sizes=reference.voxel_sizes,
            orientation=reference.orientation,
        )
        moved = self.window._placement_centre(reference, shifted)
        self.assertFalse(np.allclose(moved, same), "the shift was ignored")
        # A 10 mm world shift is 10 mm of correction, whatever the axis signs.
        self.assertAlmostEqual(float(np.abs(moved - same).max()), 10.0, places=3)

    def test_the_head_is_not_drawn_as_visible_squares(self) -> None:
        """The projection is 96-128 px and lands on about 400.

        That last stretch was where the blockiness came from, not the volume
        sampling: interpolating the rotation costs 24.6 ms a frame against
        156.2, and building the cube trilinear costs 8 ms against 218, for a
        projection that differs by 1.4% of its range -- a mean through a
        hundred samples has already done the anti-aliasing.  A smooth stretch
        costs 0.18 ms.

        Checked by what reaches the screen rather than by reading the render
        hint back.  A nearest-neighbour upscale stamps each source pixel four
        wide, so most pixels simply repeat their neighbour: measured on this
        projection, 77.6% of adjacent pairs are identical against 20.9% with
        a smooth stretch.
        """

        from PySide6.QtGui import QImage

        self.window.set_verdict(1)
        self.window.auto_segment()
        self.assertTrue(self._wait_for(lambda: self.window.roi_volume_mm3() > 0))
        self.window.open_lesion_3d()
        dialog = self.window._lesion_dialog
        dialog.brain_cb.setChecked(True)
        self._wait_for(lambda: False, timeout_ms=400)

        canvas = dialog.canvas
        canvas.resize(400, 400)
        self._wait_for(lambda: False, timeout_ms=120)
        frame = QImage(400, 400, QImage.Format.Format_RGB32)
        canvas.render(frame)

        # How often a pixel simply repeats the one to its left.  Empty
        # background repeats too and says nothing about the stretch, so only
        # pixels with tissue in them are counted.
        same = total = 0
        for y in range(120, 280, 4):
            previous = None
            for x in range(80, 320):
                grey = (frame.pixel(x, y) >> 16) & 255
                if grey < 24:
                    previous = None
                    continue
                if previous is not None:
                    total += 1
                    same += grey == previous
                previous = grey
        self.assertGreater(total, 500, "the head is not on screen to measure")
        repeated = same / total
        self.assertLess(
            repeated,
            0.5,
            f"{repeated:.0%} of pixels repeat their neighbour -- the stretch is not smooth",
        )

    def test_the_brush_keeps_up_with_the_mouse(self) -> None:
        """The readout used to run on every reported mouse position.

        It costs 19 ms with the 3D window shut and 50 ms with the head in it,
        so the paint loop could absorb 53 positions a second, or 20, against
        the 125 a mouse sends.  The brush lagged the cursor and the positions
        that were dropped widened the gaps a stroke had to bridge.  None of
        what the readout shows -- a volume, a diameter, a slice count -- is
        needed within a frame.
        """

        import time

        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        self.window.set_verdict(1)
        self.window.set_tool("brush")
        canvas = self.window.view_panels["axial"].canvas

        def event(kind, position, button):
            return QMouseEvent(
                kind,
                position,
                canvas.mapToGlobal(position),
                button,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )

        start = QPointF(canvas.width() / 2 - 40, canvas.height() / 2)
        canvas.mousePressEvent(
            event(QMouseEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
        )
        moves = 30
        began = time.perf_counter()
        for step in range(moves):
            canvas.mouseMoveEvent(
                event(
                    QMouseEvent.Type.MouseMove,
                    QPointF(start.x() + step * 0.9, start.y() + (step % 4)),
                    Qt.MouseButton.NoButton,
                )
            )
        each = (time.perf_counter() - began) / moves * 1000
        canvas.mouseReleaseEvent(
            event(QMouseEvent.Type.MouseButtonRelease, start, Qt.MouseButton.LeftButton)
        )
        self.assertLess(each, 5.0, f"{each:.1f} ms per mouse-move while painting")

        # The reading still arrives, just once the burst is over.
        self.assertTrue(self._wait_for(lambda: self.window.roi_volume_mm3() > 0))
        self._wait_for(lambda: False, timeout_ms=250)
        self.assertIn("mm³", self.window.roi_label.text())

    def test_a_stroke_is_a_line_however_fast_the_hand_moves(self) -> None:
        """The mouse reports positions, not the path between them.

        With 24 px between reports a stroke came out as six separate blobs,
        and the slower the event loop the wider the gaps -- which made the
        painted result depend on how busy the machine was.  For something that
        ends up in a results table that is not acceptable.
        """

        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        self.window.set_verdict(1)
        self.window.set_tool("brush")
        self.window.brush_spin.setValue(1.0)
        self.window.brush_3d_cb.setChecked(False)
        canvas = self.window.view_panels["axial"].canvas

        def paint(step_px: float) -> tuple[int, int]:
            self.window.clear_roi()
            self._wait_for(lambda: False, timeout_ms=60)

            def event(kind, position, button):
                return QMouseEvent(
                    kind,
                    position,
                    canvas.mapToGlobal(position),
                    button,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )

            start = QPointF(canvas.width() / 2 - 60, canvas.height() / 2)
            canvas.mousePressEvent(
                event(QMouseEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
            )
            x = start.x()
            while x < start.x() + 120:
                x += step_px
                canvas.mouseMoveEvent(
                    event(QMouseEvent.Type.MouseMove, QPointF(x, start.y()), Qt.MouseButton.NoButton)
                )
            canvas.mouseReleaseEvent(
                event(QMouseEvent.Type.MouseButtonRelease, QPointF(x, start.y()), Qt.MouseButton.LeftButton)
            )
            self._wait_for(lambda: False, timeout_ms=60)
            mask = self.window._selected_label_mask()
            coords = np.argwhere(mask)
            if not len(coords):
                return 0, 0
            present = np.unique(coords[:, 0])
            return len(coords), int((np.diff(present) > 1).sum()) + 1

        slowly, slow_pieces = paint(2.0)
        quickly, quick_pieces = paint(24.0)
        self.assertEqual(slow_pieces, 1, "even a slow stroke came out broken")
        self.assertEqual(quick_pieces, 1, f"a quick stroke broke into {quick_pieces} pieces")
        self.assertEqual(
            slowly,
            quickly,
            "the painted mask depends on how fast the hand moved",
        )

    def test_the_dataset_dialog_can_describe_a_different_study(self) -> None:
        """A stranger should not have to edit JSON to open their own data.

        The dialog they already use to pick the workbook is where the shape of
        it belongs: the sheet name, what each sequence is called, the filename
        ending that identifies it, and whether a case may be read without it.
        """

        import dataset_config

        dialog = self.desktop_app.DatasetDialog(
            self.settings, self.window.dataset, config=dataset_config.validate({})
        )
        try:
            # The paths have to be ones that exist, or _accept stops on those
            # before it ever looks at the format.
            dialog.workbook_edit.setText(str(self.source))
            dialog.data_edit.setText(str(self.data_root))
            dialog.db_edit.setText(str(self.db_path))
            self.assertFalse(
                dialog.format_section.is_expanded(),
                "a study that matches the defaults should not be asked eight questions",
            )
            self.assertEqual(dialog.sheet_edit.text(), "MCH-microhemorrage")
            self.assertEqual(set(dialog.sequence_rows), set(self.desktop_app.MODALITY_ORDER))
            _label, _suffix, required = dialog.sequence_rows["mip"]
            self.assertFalse(required.isChecked(), "the projection defaults to optional")

            # Describe somebody else's study.
            dialog.sheet_edit.setText("Findings")
            name, suffix, needed = dialog.sequence_rows["mip"]
            name.setText("T2*")
            suffix.setCurrentText("_T2star.nii.gz")
            needed.setChecked(True)
            collected = dataset_config.validate(dialog._collected_config())
            self.assertEqual(collected["workbook"]["sheet"], "Findings")
            self.assertEqual(collected["sequences"]["mip"]["label"], "T2*")
            self.assertEqual(collected["sequences"]["mip"]["suffix"], "_T2star.nii.gz")
            self.assertTrue(collected["sequences"]["mip"]["required"])

            # And a description that cannot work is refused in place, with the
            # section opened so the reader can see what is wrong.
            dialog.sheet_edit.setText("")
            dialog.format_section.set_expanded(False)
            dialog._accept()
            self.assertIsNone(dialog.dataset, "a broken format was accepted")
            self.assertIn("sheet", dialog.problem_label.text().lower())
            self.assertTrue(dialog.format_section.is_expanded())
        finally:
            dialog.deleteLater()

    def test_the_dialog_reads_the_filename_endings_off_the_data(self) -> None:
        """Typing _chi_nSFCR+0_Avg_AffineRestored.nii.gz from memory is a poor
        welcome; the data folder already knows."""

        import dataset_config

        dialog = self.desktop_app.DatasetDialog(
            self.settings, self.window.dataset, config=dataset_config.validate({})
        )
        try:
            dialog.data_edit.setText(str(self.data_root))
            dialog._detect_suffixes()
            _name, suffix, _required = dialog.sequence_rows["swi"]
            offered = [suffix.itemText(index) for index in range(suffix.count())]
            self.assertTrue(offered, "nothing was detected in the data root")
            self.assertTrue(
                all(item.endswith((".nii", ".nii.gz")) for item in offered), offered
            )
            self.assertIn("Found", dialog.detect_label.text())
            # What was already configured survives if the data agrees with it.
            self.assertTrue(
                suffix.currentText().endswith(".nii.gz"), suffix.currentText()
            )

            dialog.data_edit.setText(str(self.temp_dir / "not-a-folder"))
            dialog._detect_suffixes()
            self.assertIn("No NIfTI files", dialog.detect_label.text())
        finally:
            dialog.deleteLater()

    def test_a_sequence_the_study_cannot_segment_says_so_by_name(self) -> None:
        """The rule was written as "if modality == mip"; it is a property now.

        A study whose projection slot holds something paintable should be able
        to paint on it, and one that renames the projection should see its own
        name in the refusal.
        """

        self.assertFalse(self.desktop_app.can_segment("mip"))
        self.assertTrue(self.desktop_app.can_segment("swi"))
        self.assertTrue(self.desktop_app.can_segment("qsm"))

        self.window.set_verdict(1)
        self.window.set_modality("mip")
        self._wait_for(lambda: self.window.current_modality == "mip", timeout_ms=4000)
        self.window.set_tool("brush")
        self._wait_for(lambda: False, timeout_ms=120)
        self.assertIn("cannot be segmented", self.window._status_label.text())
        self.assertIn(
            self.desktop_app.MODALITY_LABELS["mip"], self.window._status_label.text()
        )

    def test_picking_a_paint_tool_opens_the_tab_with_its_controls(self) -> None:
        self.window.show_panel_tab("review")
        self.window.set_tool("brush")
        self.assertEqual(self.window.current_panel_tab(), "segment")
        self.window.show_panel_tab("review")
        self.window.set_tool("eraser")
        self.assertEqual(self.window.current_panel_tab(), "segment")
        # The Point tool is navigation, not drawing; it leaves the tab alone.
        self.window.show_panel_tab("review")
        self.window.set_tool("point")
        self.assertEqual(self.window.current_panel_tab(), "review")

    def test_save_and_segment_saves_then_hands_over_the_brush(self) -> None:
        """Decide, then draw: the tab with Generate on it, and a brush up.

        Arming the brush while leaving Generate two panels away was the whole
        problem -- having pressed "save and start drawing" there was nowhere
        obvious to draw from.
        """

        self.window.set_verdict(1)
        self.assertTrue(self.window.save_and_segment())
        self.assertEqual(self.window.current_panel_tab(), "segment")
        self.assertEqual(self.window.active_tool, "brush")
        self.assertTrue(self.window.auto_roi_btn.isVisible())
        self.assertFalse(self.window._review_dirty)
        stored = self.review_store.list_targets(
            self.db_path, self.window.current_case_id, "Desktop Test Reader", 1
        )
        self.assertEqual(stored[0]["reader_verify"], 1)
        self.assertEqual(stored[0]["target_id"], self.window.selected_target["target_id"])

    def test_a_mask_without_a_verdict_is_pointed_out_when_leaving(self) -> None:
        """Drawing without deciding is work that will not reach the results.

        Doing nothing at all is not worth a word: moving through cases to look
        at them is a normal thing to do.
        """

        self.window.set_modality("swi")
        # Nothing done: silent.
        self.assertEqual(self.window.unconfirmed_segmentations(), [])

        self.window.auto_segment()
        self.window.set_verdict(None)
        self.assertTrue(self.window.save_current_review(advance=False))
        pending = self.window.unconfirmed_segmentations()
        self.assertEqual(
            [item["target_id"] for item in pending],
            [self.window.selected_target["target_id"]],
        )

        # A verdict on the same finding clears it.
        self.window.set_verdict(1)
        self.assertTrue(self.window.save_current_review(advance=False))
        self.assertEqual(self.window.unconfirmed_segmentations(), [])

    def test_long_help_text_wraps_instead_of_widening_the_panel(self) -> None:
        from PySide6.QtWidgets import QLabel

        areas = list(self.window.panel_pages.values()) + [self.window.details_panel]
        for scroll in areas:
            for label in scroll.widget().findChildren(QLabel):
                if len(label.text()) > 60:
                    with self.subTest(text=label.text()[:40]):
                        self.assertTrue(label.wordWrap(), "long help text must wrap")

    def test_the_toolbar_keeps_the_reading_controls_readable(self) -> None:
        for width, height in ((1540, 960), (1920, 1080)):
            self.window.resize(width, height)
            self._wait_for(lambda: False, timeout_ms=250)
            controls = {
                "SWI": self.window.modality_segments.button("swi"),
                "QSM": self.window.modality_segments.button("qsm"),
                "MIP": self.window.modality_segments.button("mip"),
                "Point": self.window.tool_buttons["point"],
                "Brush": self.window.tool_buttons["brush"],
                "Eraser": self.window.tool_buttons["eraser"],
                "Lesion": self.window.lesion_zoom_btn,
                "Contrast": self.window.contrast_btn,
            }
            squeezed = [
                name
                for name, widget in controls.items()
                if widget.width() < widget.minimumSizeHint().width() - 1
            ]
            with self.subTest(size=(width, height)):
                self.assertEqual(squeezed, [], "toolbar labels are being elided")

    def test_the_window_carries_the_application_icon(self) -> None:
        """A window built directly must not come up blank.

        Relying on the application icon alone would leave the viewer unmarked
        whenever it is constructed without going through ``main``.
        """

        from config import icon_file

        self.assertIsNotNone(icon_file(), "the icon file is missing from the viewer folder")
        self.assertFalse(
            self.window.windowIcon().isNull(),
            "the main window has no icon",
        )

    def test_the_icon_has_the_sizes_a_taskbar_needs(self) -> None:
        """One large bitmap would be downscaled into mush at 16px."""

        from PySide6.QtGui import QIcon

        from config import icon_file

        path = icon_file()
        assert path is not None
        icon = QIcon(str(path))
        self.assertFalse(icon.isNull(), f"Qt could not read {path.name}")
        widths = {size.width() for size in icon.availableSizes()}
        self.assertTrue(any(width <= 32 for width in widths), f"no small size in {sorted(widths)}")
        self.assertTrue(any(width >= 128 for width in widths), f"no large size in {sorted(widths)}")

    def test_the_application_identity_is_set_for_the_windows_taskbar(self) -> None:
        """Without an AppUserModelID the taskbar shows the Python icon."""

        icon = self.desktop_app.apply_application_identity(self.app)
        self.assertIsNotNone(icon)
        self.assertFalse(self.app.windowIcon().isNull())
        if sys.platform != "win32":
            self.skipTest("The AppUserModelID only exists on Windows.")
        import ctypes
        from ctypes import wintypes

        shell = ctypes.windll.shell32
        shell.GetCurrentProcessExplicitAppUserModelID.argtypes = [ctypes.POINTER(wintypes.LPWSTR)]
        buffer = wintypes.LPWSTR()
        self.assertEqual(shell.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(buffer)), 0)
        self.assertEqual(buffer.value, self.desktop_app.APP_MODEL_ID)

    def test_the_case_title_is_not_shortened_when_the_toolbar_has_room(self) -> None:
        """The toolbar now spans the middle and right columns only.

        With the case queue pinned open beside it the toolbar is some 360px
        narrower than the window, so at 1540 the title is the thing that
        gives -- folding the queue hands the width straight back.
        """

        self.window.resize(1920, 1080)
        self._wait_for(lambda: False, timeout_ms=250)
        self.assertEqual(self.window.case_title.text(), self.window.case_title._full_text)

        self.window.resize(1540, 960)
        self.window.set_queue_pinned(False)
        self._wait_for(lambda: False, timeout_ms=250)
        self.assertEqual(
            self.window.case_title.text(),
            self.window.case_title._full_text,
            "folding the queue did not give the toolbar its width back",
        )
        self.window.set_queue_pinned(True)

    def test_every_round_is_offered_and_the_last_one_is_remembered(self) -> None:
        from PySide6.QtCore import Qt

        target_id = str(self.window.selected_target["target_id"])
        self.review_store.save_review(
            self.db_path, target_id=target_id, case_id=self.case_id,
            reader_id="Rounds Reader", review_round=1, verify=1, comment="round one",
        )
        self.review_store.start_new_session(self.db_path, "Rounds Reader")
        # ...and a second round started by accident.
        self.review_store.start_new_session(self.db_path, "Rounds Reader")

        rounds = self.review_store.list_reader_rounds(self.db_path, "Rounds Reader")
        self.assertEqual([int(item["review_round"]) for item in rounds], [2, 1])
        self.assertEqual(rounds[1]["reviewed_count"], 1)
        self.assertEqual(rounds[0]["reviewed_count"], 0)

        # Without a remembered choice the newest round is selected.
        dialog = self.desktop_app.RoundDialog("Rounds Reader", rounds, preferred_round=None)
        try:
            self.assertEqual(dialog.list.count(), 3)  # two rounds plus "new"
            self.assertEqual(dialog.list.currentItem().data(Qt.ItemDataRole.UserRole), 2)
            self.assertIn("nothing recorded yet", dialog.list.item(0).text())

            # Going back to round 1 resumes that round, not the newest.
            row = next(
                index
                for index in range(dialog.list.count())
                if dialog.list.item(index).data(Qt.ItemDataRole.UserRole) == 1
            )
            dialog.list.setCurrentRow(row)
            dialog._accept()
            self.assertEqual(dialog.chosen_round, 1)
            self.assertFalse(dialog.start_new)
            session = self.review_store.resume_session(self.db_path, dialog.chosen_session_id)
            self.assertEqual(int(session["review_round"]), 1)
        finally:
            dialog.deleteLater()

        # The choice is remembered per database and reader, and becomes the
        # default next time.
        self.settings.set_last_round(self.db_path, "Rounds Reader", 1)
        self.assertEqual(self.settings.last_round(self.db_path, "Rounds Reader"), 1)
        self.assertIsNone(self.settings.last_round(self.db_path, "Someone Else"))
        again = self.desktop_app.RoundDialog(
            "Rounds Reader", rounds, preferred_round=self.settings.last_round(self.db_path, "Rounds Reader")
        )
        try:
            self.assertEqual(again.list.currentItem().data(Qt.ItemDataRole.UserRole), 1)
        finally:
            again.deleteLater()

    def test_starting_a_new_round_is_still_offered(self) -> None:
        from PySide6.QtCore import Qt

        self.review_store.start_new_session(self.db_path, "New Round Reader")
        rounds = self.review_store.list_reader_rounds(self.db_path, "New Round Reader")
        dialog = self.desktop_app.RoundDialog("New Round Reader", rounds)
        try:
            last = dialog.list.item(dialog.list.count() - 1)
            self.assertEqual(last.data(Qt.ItemDataRole.UserRole), "new")
            self.assertIn("round 2", last.text())
            dialog.list.setCurrentRow(dialog.list.count() - 1)
            dialog._accept()
            self.assertTrue(dialog.start_new)
            self.assertIsNone(dialog.chosen_round)
        finally:
            dialog.deleteLater()

    # ----------------------------------------------------------------- tools --
    def test_clicking_a_view_does_nothing_until_the_point_tool_is_on(self) -> None:
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        canvas = self.window.view_panels["axial"].canvas
        # An unjudged finding now opens with the Point tool up, which is the
        # point of that behaviour; this test is about the state with no tool,
        # so it puts the tool down first.
        self.window.set_tool(None)
        self.assertIsNone(self.window.active_tool)
        self.assertFalse(canvas._pick_enabled)

        clicks: list = []
        canvas.targetClicked.connect(lambda plane, voxel: clicks.append(plane))
        position = QPointF(canvas.width() / 2 + 10, canvas.height() / 2 + 10)

        def click() -> None:
            canvas.mousePressEvent(
                QMouseEvent(
                    QMouseEvent.Type.MouseButtonPress,
                    position,
                    canvas.mapToGlobal(position),
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )
            )

        click()
        self.assertEqual(clicks, [], "a click with no tool must not move the crosshair")

        self.window.set_tool("point")
        self.assertTrue(canvas._pick_enabled)
        self.assertTrue(self.window.tool_buttons["point"].isChecked())
        click()
        self.assertEqual(clicks, ["axial"])

        # Pressing the tool again turns it off.
        self.window.toggle_point_tool()
        self.assertIsNone(self.window.active_tool)
        self.assertFalse(canvas._pick_enabled)
        click()
        self.assertEqual(clicks, ["axial"])

    def test_right_click_cancels_a_pick_but_a_right_drag_does_not(self) -> None:
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        canvas = self.window.view_panels["axial"].canvas
        self.window.set_tool("point")
        marker = tuple(self.window.marker_ras)

        def right(kind, position):
            canvas.__getattribute__(
                "mousePressEvent" if kind == "press" else "mouseReleaseEvent"
            )(
                QMouseEvent(
                    QMouseEvent.Type.MouseButtonPress
                    if kind == "press"
                    else QMouseEvent.Type.MouseButtonRelease,
                    position,
                    canvas.mapToGlobal(position),
                    Qt.MouseButton.RightButton,
                    Qt.MouseButton.RightButton if kind == "press" else Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                )
            )

        # Pick somewhere away from the finding.
        self.window._on_canvas_target_clicked("axial", np.asarray([20.0, 30.0, 40.0]))
        self.assertNotEqual(tuple(self.window.target_ras), marker)

        # A right *drag* adjusts window/level and leaves the pick alone.
        start = QPointF(120.0, 120.0)
        right("press", start)
        canvas.mouseMoveEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseMove,
                QPointF(220.0, 60.0),
                canvas.mapToGlobal(QPointF(220.0, 60.0)),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.RightButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        right("release", QPointF(220.0, 60.0))
        self.assertNotEqual(tuple(self.window.target_ras), marker, "a drag must not cancel")
        self.assertNotEqual(canvas.window_limits, canvas._auto_window)

        # A right *click* cancels it.
        right("press", start)
        right("release", start)
        self.assertEqual(tuple(self.window.target_ras), marker)

    def test_escape_also_cancels_a_pick(self) -> None:
        marker = tuple(self.window.marker_ras)
        self.window._on_canvas_target_clicked("axial", np.asarray([15.0, 25.0, 35.0]))
        self.assertNotEqual(tuple(self.window.target_ras), marker)
        self.window.clear_picked_position()
        self.assertEqual(tuple(self.window.target_ras), marker)
        self.assertEqual(self.settings.shortcut("cancel_pick"), "Esc")

    # --------------------------------------------------------- segmentation --
    def test_generating_a_mask_uses_the_sequence_on_screen(self) -> None:
        self.assertIsNotNone(self.window.label_volume)
        self.assertFalse(self.window.label_volume.any())
        target_id = str(self.window.selected_target["target_id"])

        self.window.set_modality("swi")
        self.window.auto_segment()
        from_swi = self.window.roi_volume_mm3()
        self.assertGreater(from_swi, 0.0)
        self.assertEqual(self.window.label_sources[target_id], "swi")
        self.assertTrue(self.window._roi_dirty)
        self.assertIn("ROI", self.window.dirty_label.text())

        self.window.set_modality("qsm")
        self.window.auto_segment()
        self.assertEqual(self.window.label_sources[target_id], "qsm")
        self.assertGreater(self.window.roi_volume_mm3(), 0.0)

    def test_the_mip_cannot_generate_but_can_display(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        self.window.set_modality("swi")
        self.window.auto_segment()
        before = self.window.roi_volume_mm3()
        self.window.set_modality("mip")
        original = QMessageBox.information
        QMessageBox.information = staticmethod(lambda *args, **kwargs: None)
        try:
            self.window.auto_segment()
        finally:
            QMessageBox.information = original
        # A projection smears the lesion, so growing there would be wrong.
        self.assertEqual(self.window.roi_volume_mm3(), before)
        # Showing it is fine: every sequence shares the grid.
        self.assertIs(
            self.window.view_panels["axial"].canvas._label_volume, self.window.label_volume
        )

    def test_brush_paints_erases_and_undoes(self) -> None:
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        self.window.set_modality("swi")
        self.window.auto_segment()
        self.window.set_tool("brush")
        canvas = self.window.view_panels["axial"].canvas
        voxel = np.asarray(
            np.unravel_index(np.argmax(self.window.label_volume), self.window.label_volume.shape)
        )
        canvas.set_slice(int(voxel[2]))
        image = canvas._display_array()
        rect, (scale_x, scale_y) = canvas._draw_geometry(int(image.shape[1]), int(image.shape[0]))
        position = QPointF(
            rect.left() + (voxel[0] + 6) * scale_x, rect.top() + (voxel[1] + 6) * scale_y
        )

        def stroke(button, modifiers=Qt.KeyboardModifier.NoModifier):
            canvas.mousePressEvent(
                QMouseEvent(
                    QMouseEvent.Type.MouseButtonPress, position, canvas.mapToGlobal(position),
                    button, button, modifiers,
                )
            )
            canvas.mouseReleaseEvent(
                QMouseEvent(
                    QMouseEvent.Type.MouseButtonRelease, position, canvas.mapToGlobal(position),
                    button, Qt.MouseButton.NoButton, modifiers,
                )
            )

        generated = self.window.roi_volume_mm3()
        stroke(Qt.MouseButton.LeftButton)
        painted = self.window.roi_volume_mm3()
        self.assertGreater(painted, generated)

        # The right button never draws: it means contrast in every tool, so the
        # gesture does not change meaning depending on what is selected.
        window_before = canvas.window_limits
        stroke(Qt.MouseButton.RightButton)
        self.assertAlmostEqual(self.window.roi_volume_mm3(), painted, places=6)

        # Erasing is its own tool, still on the left button.
        self.window.set_tool("eraser")
        stroke(Qt.MouseButton.LeftButton)
        self.assertLess(self.window.roi_volume_mm3(), painted)

        self.window.undo_roi()
        self.assertAlmostEqual(self.window.roi_volume_mm3(), painted, places=6)

        # Alt+left stays window/level with either tool up.
        self.window.set_tool("brush")
        before = self.window.roi_volume_mm3()
        stroke(Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier)
        self.assertAlmostEqual(self.window.roi_volume_mm3(), before, places=6)
        self.assertEqual(window_before, canvas.window_limits)

    def test_move_here_settles_onto_the_focus_and_records_how_far(self) -> None:
        """The recorded coordinate should be the lesion's, not the mouse's."""

        import review_store

        self.window.set_modality("swi")
        target = self.window.selected_target
        source = tuple(float(v) for v in target["ras"])
        volume = self.window.volumes["swi"]

        # Click a voxel off the finding, the way a hand misses by one.
        source_voxel = self.desktop_app.ras_to_voxel(volume.affine, source)
        voxel = source_voxel + np.array([1.0, -1.0, 0.0])
        clicked = tuple(float(v) for v in self.desktop_app.voxel_to_ras(volume.affine, voxel))
        self.window.target_ras = clicked
        self.window.move_finding_here()

        settled = self.window.pending_ras
        self.assertIsNotNone(settled)

        def darkness(ras) -> float:
            index = tuple(
                int(round(v)) for v in self.desktop_app.ras_to_voxel(volume.affine, ras)
            )
            return float(volume.data[index])

        # The recorded point is at least as dark as the click, and snapping it
        # again does not move it: it is a local minimum, which is the property
        # that makes it reproducible between readers.
        self.assertLessEqual(darkness(settled), darkness(clicked) + 1e-6)
        again = self.window._snapped_ras(settled)
        np.testing.assert_allclose(again, settled, atol=1e-6)
        # Refining a click must not become walking to a different lesion: the
        # converged walk is capped at twice the search radius from the click.
        self.assertLessEqual(
            self.review_store.distance_mm(settled, clicked),
            2.0 * self.window.settings.snap_radius_mm + 1e-6,
        )

        self.window.set_verdict(1)
        self.assertTrue(self.window.save_current_review(advance=False))
        rows = review_store.list_targets(
            self.db_path, self.window.current_case_id, "Desktop Test Reader", 1
        )
        report = next(
            r for r in rows[0]["reader_reports"] if r["reader_id"] == "Desktop Test Reader"
        )
        self.assertIsNotNone(report["snap_mm"], "the distance to the focus was not recorded")
        self.assertLess(float(report["snap_mm"]), 2.01)

    def test_snapping_can_be_turned_off(self) -> None:
        self.window.set_modality("swi")
        self.window.settings.update(
            auto_zoom=self.window.settings.auto_zoom,
            lesion_fov_mm=self.window.settings.lesion_fov_mm,
            save_advances=self.window.settings.save_advances,
            default_modality=self.window.settings.default_modality,
            snap_to_lesion=False,
        )
        volume = self.window.volumes["swi"]
        source = tuple(float(v) for v in self.window.selected_target["ras"])
        voxel = self.desktop_app.ras_to_voxel(volume.affine, source) + np.array([3.0, -2.0, 0.0])
        clicked = tuple(float(v) for v in self.desktop_app.voxel_to_ras(volume.affine, voxel))
        self.window.target_ras = clicked
        self.window.move_finding_here()
        np.testing.assert_allclose(self.window.pending_ras, clicked, atol=1e-9)

    def test_scrolling_can_carry_the_cursor_into_the_other_views(self) -> None:
        """Off by default; on is the ITK-SNAP habit."""

        self.window.set_modality("swi")
        before = tuple(self.window.target_ras)
        self.window._on_slice_request("axial", self.window.view_panels["axial"].canvas.slice_index + 4)
        self.assertEqual(tuple(self.window.target_ras), before, "scrolling moved the cursor")

        self.window.settings.update(
            auto_zoom=self.window.settings.auto_zoom,
            lesion_fov_mm=self.window.settings.lesion_fov_mm,
            save_advances=self.window.settings.save_advances,
            default_modality=self.window.settings.default_modality,
            scroll_moves_cursor=True,
        )
        axial = self.window.view_panels["axial"].canvas
        self.window._on_slice_request("axial", axial.slice_index + 4)
        after = tuple(self.window.target_ras)
        self.assertNotEqual(after, before)
        # Only the axis that plane scrolls moved.
        volume = self.window.volumes["swi"]
        moved = self.desktop_app.ras_to_voxel(volume.affine, after)
        stayed = self.desktop_app.ras_to_voxel(volume.affine, before)
        self.assertAlmostEqual(float(moved[0]), float(stayed[0]), places=6)
        self.assertAlmostEqual(float(moved[1]), float(stayed[1]), places=6)
        self.assertAlmostEqual(float(moved[2]), float(axial.slice_index), places=6)
        # And the finding marker stayed where it is.
        self.assertEqual(tuple(self.window.marker_ras), before)

    def test_the_image_can_be_smoothed_when_magnified(self) -> None:
        """Off by default, and the switch has to reach the pixels.

        Nearest neighbour is the honest default: magnified, the squares are
        the voxels, and at lesion zoom a microbleed is a handful of them, so
        interpolation invents edges that are not in the data.  Some readers
        judge roundness better with it on, which is why the preference exists.

        Checked on what is drawn rather than on the flag -- the flag was all
        this test used to look at, and a flag that never reaches the painter
        would have passed.  Measured at 400%: 78.8% of adjacent pixels are
        identical with it off, 21.1% with it on.
        """

        from PySide6.QtGui import QImage

        canvas = self.window.view_panels["axial"].canvas
        self.assertFalse(canvas._smooth_zoom, "nearest neighbour is the honest default")
        self.assertFalse(self.settings.smooth_zoom, "the preference defaults to on")
        canvas.set_zoom_mode("400%")
        self._wait_for(lambda: False, timeout_ms=150)

        def repeated() -> float:
            frame = QImage(canvas.width(), canvas.height(), QImage.Format.Format_RGB32)
            canvas.render(frame)
            same = total = 0
            for y in range(40, canvas.height() - 40, 5):
                previous = None
                for x in range(40, canvas.width() - 40):
                    grey = (frame.pixel(x, y) >> 16) & 255
                    if grey < 12:
                        previous = None
                        continue
                    if previous is not None:
                        total += 1
                        same += grey == previous
                    previous = grey
            self.assertGreater(total, 2000, "no image on screen to measure")
            return same / total

        blocky = repeated()
        self.assertGreater(blocky, 0.5, f"only {blocky:.0%} repeat -- this is already smoothed")
        canvas.set_smooth_zoom(True)
        self.assertTrue(canvas._smooth_zoom)
        self._wait_for(lambda: False, timeout_ms=150)
        smoothed = repeated()
        self.assertLess(
            smoothed,
            0.5,
            f"{smoothed:.0%} of pixels still repeat -- the preference does not reach the paint",
        )

    def test_a_finding_outside_the_volume_is_reported_not_clamped(self) -> None:
        """A coordinate in the wrong space used to be pinned to the edge.

        ``set_target_voxel`` clamps out-of-range voxels and stored the verdict
        in a flag nothing read, so the crosshair sat on the skull with nothing
        to say it was not where the sheet claimed.  This project corrects NIfTI
        origins in a separate notebook, so it is a live failure mode.
        """

        canvas = self.window.view_panels["axial"].canvas
        shape = canvas.volume.shape
        canvas.set_marker_voxel(np.asarray([shape[0] // 2, shape[1] // 2, shape[2] // 2]))
        self.assertTrue(canvas.marker_in_bounds)
        canvas.set_marker_voxel(np.asarray([shape[0] + 40, shape[1] // 2, shape[2] // 2]))
        self.assertFalse(canvas.marker_in_bounds)

        # And the window says so, rather than leaving the reader to notice.
        self.assertFalse(self.window.marker_is_outside_the_volume())
        self.window.marker_ras = (9999.0, 9999.0, 9999.0)
        self.window._apply_target_to_views(recenter=False)
        self.assertTrue(self.window.marker_is_outside_the_volume())
        self.window._report_marker_placement()
        self.assertIn("outside", self.window._status_label.text().lower())

    def test_saving_no_decision_moves_on_instead_of_standing_still(self) -> None:
        """"Not set" is a decision to defer, and the queue has to act on it.

        The advance step looked for findings with no verdict and no comment,
        which is exactly what the finding just saved as "Not set" still is, so
        it re-selected the same one and announced that it had moved on.
        """

        self.assertEqual(len(self.window.targets), 1)
        first_case = self.window.current_case_id
        target_id = self.window.selected_target["target_id"]

        self.window.set_verdict(None)
        self.assertTrue(self.window.save_current_review())
        self.assertTrue(self._wait_for(lambda: self.window.current_case_id != first_case))
        self.assertNotEqual(self.window.current_case_id, first_case)

        # Deferring must not quietly mark the case done: it stays on the list
        # of cases this reader has not finished.
        case = next(
            item
            for item in self.review_store.list_cases(self.db_path, "Desktop Test Reader", 1)
            if item["case_id"] == first_case
        )
        self.assertEqual(case["reader_review_status"], "Unreviewed")
        stored = self.review_store.list_targets(
            self.db_path, first_case, "Desktop Test Reader", 1
        )
        self.assertIsNone(next(t for t in stored if t["target_id"] == target_id)["reader_verify"])

    def test_undo_history_stays_small_and_still_restores(self) -> None:
        """Undo used to copy the whole label volume once per stroke.

        A real case here is 256x256x176, so a uint16 copy is 23 MB and the
        twenty kept steps were 461 MB -- on top of the cached image volumes.
        A stroke only ever touches a handful of voxels, so recording those is
        both smaller by three orders of magnitude and exactly as reversible.
        """

        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        self.window.set_modality("swi")
        self.window.set_tool("brush")
        canvas = self.window.view_panels["axial"].canvas
        image = canvas._display_array()
        rect, (scale_x, scale_y) = canvas._draw_geometry(int(image.shape[1]), int(image.shape[0]))

        def stroke(column: int, row: int) -> None:
            position = QPointF(
                rect.left() + (column + 0.5) * scale_x, rect.top() + (row + 0.5) * scale_y
            )

            def event(kind, buttons):
                return QMouseEvent(
                    kind, position, canvas.mapToGlobal(position),
                    Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier,
                )

            canvas.mousePressEvent(event(QMouseEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton))
            canvas.mouseReleaseEvent(event(QMouseEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton))

        states = []
        for step in range(12):
            states.append(self.window.label_volume.copy())
            stroke(60 + step * 3, 60)

        volume_bytes = self.window.label_volume.nbytes
        self.assertGreater(volume_bytes, 1_000_000, "the test case should be a realistic size")
        history_bytes = self.window._roi_undo_bytes()
        self.assertLess(
            history_bytes,
            volume_bytes,
            f"the whole undo history ({history_bytes} bytes) must cost less than one "
            f"copy of the label volume ({volume_bytes} bytes)",
        )

        # And it still undoes, stroke by stroke, right back to the start.
        for previous in reversed(states):
            self.window.undo_roi()
            np.testing.assert_array_equal(self.window.label_volume, previous)

    def test_generating_reports_what_it_produced(self) -> None:
        """Region growing can stop for the wrong reason, and has to say so.

        A mask that ran into the safety cap is the cap's answer rather than a
        measurement, and one that collapsed to the seed means the reader
        pressed Generate and got nothing.  Both used to look like a success.
        """

        self.window.set_modality("swi")
        self.window.auto_segment()
        details = self.window.last_segmentation
        self.assertIsNotNone(details)
        self.assertGreater(details["voxel_count"], 1)
        for key in ("volume_mm3", "diameter_mm", "longest_mm", "reached_cap", "suspect"):
            self.assertIn(key, details)

        # Squeeze the cap until growth has to run into it, and check it says so.
        self.window.roi_radius_spin.setValue(1.0)
        self.window.auto_segment()
        self.assertTrue(self.window.last_segmentation["reached_cap"])
        self.assertTrue(self.window.last_segmentation["suspect"])
        self.assertIn("cap", self.window._status_label.text().lower())

    def test_a_segmentation_carries_its_settings_into_the_database(self) -> None:
        import review_store

        self.window.set_modality("swi")
        self.window.sensitivity_spin.setValue(3.5)
        self.window.roi_radius_spin.setValue(7.0)
        self.window.auto_segment()
        self.window.set_verdict(1)
        self.assertTrue(self.window.save_current_review(advance=False))

        stored = review_store.list_rois(
            self.db_path, self.window.current_case_id, "Desktop Test Reader", 1
        )
        row = stored[str(self.window.selected_target["target_id"])]
        self.assertEqual(row["method"], "grow")
        self.assertAlmostEqual(float(row["sensitivity"]), 3.5)
        self.assertAlmostEqual(float(row["radius_mm"]), 7.0)
        self.assertEqual(row["generated_from"], "swi")

    def test_editing_a_generated_mask_by_hand_is_recorded_as_such(self) -> None:
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        import review_store

        self.window.set_modality("swi")
        self.window.auto_segment()
        self.window.set_tool("brush")
        canvas = self.window.view_panels["axial"].canvas
        voxel = np.asarray(
            np.unravel_index(np.argmax(self.window.label_volume), self.window.label_volume.shape)
        )
        canvas.set_slice(int(voxel[2]))
        image = canvas._display_array()
        rect, (sx, sy) = canvas._draw_geometry(int(image.shape[1]), int(image.shape[0]))
        pos = QPointF(rect.left() + (voxel[0] + 4.5) * sx, rect.top() + (voxel[1] + 0.5) * sy)
        canvas.mousePressEvent(QMouseEvent(
            QMouseEvent.Type.MouseButtonPress, pos, canvas.mapToGlobal(pos),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease, pos, canvas.mapToGlobal(pos),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))

        self.window.set_verdict(1)
        self.assertTrue(self.window.save_current_review(advance=False))
        row = review_store.list_rois(
            self.db_path, self.window.current_case_id, "Desktop Test Reader", 1
        )[str(self.window.selected_target["target_id"])]
        self.assertEqual(row["method"], "grow+brush")

    def test_the_label_grid_comes_from_the_sequence_readers_segment_on(self) -> None:
        """One label volume per case only works if the sequences share a grid.

        All 125 complete cases in this dataset do, but nothing checked it, and
        the grid was taken from whichever sequence came first in MODALITY_ORDER
        -- QSM -- rather than the one being drawn on.
        """

        self.assertIs(self.window._label_reference(), self.window.volumes["swi"])
        self.assertIsNone(self.window._grid_mismatch())

        # The volume mm3 on screen has to be the one written to the database.
        self.window.set_modality("swi")
        self.window.auto_segment()
        shown = self.window.roi_volume_mm3()
        reference = self.window._label_reference()
        voxels = int(self.window._selected_label_mask().sum())
        self.assertAlmostEqual(shown, voxels * float(np.prod(reference.voxel_sizes)), places=6)

    def test_a_case_whose_sequences_disagree_refuses_to_be_segmented(self) -> None:
        from imaging import Volume

        swi = self.window.volumes["swi"]
        odd = Volume(
            path="synthetic",
            data=np.zeros((8, 8, 8), dtype=np.float32),
            affine=np.eye(4, dtype=np.float64),
            shape=(8, 8, 8),
            voxel_sizes=(1.0, 1.0, 1.0),
            window=(0.0, 1.0),
        )
        self.window.volumes = {"swi": swi, "qsm": odd, "mip": None}
        message = self.window._grid_mismatch()
        self.assertIsNotNone(message)
        self.assertIn("QSM", message)

        self.window._load_label_volume()
        self.assertIsNone(self.window.label_volume, "no shared grid means no label volume")
        self.window.set_tool("brush")
        self.assertIn("grid", self.window._status_label.text().lower())
        self.assertIsNone(self.window.active_tool, "the brush must not arm on a mismatched case")

    def test_saving_does_not_write_a_row_for_every_finding_it_looked_at(self) -> None:
        """Selecting a finding assigns it a label value; that is not a
        segmentation, and it should not cost a database round trip on save."""

        import review_store

        calls: list[str] = []
        original = self.desktop_app.save_roi

        def spy(*args, **kwargs):
            calls.append(str(kwargs.get("target_id")))
            return original(*args, **kwargs)

        self.window.set_modality("swi")
        # Look at every finding of the case, then segment exactly one of them.
        for index in range(len(self.window.targets)):
            self.window._select_target_index(index, confirm=False)
        self.window._select_target_index(0, confirm=False)
        self.window.auto_segment()

        self.desktop_app.save_roi = spy
        try:
            written = self.window._write_label_volume()
        finally:
            self.desktop_app.save_roi = original
        self.assertEqual(
            calls,
            [str(self.window.targets[0]["target_id"])],
            "only the finding that actually has voxels needs a row",
        )
        self.assertEqual(list(written), calls)
        stored = review_store.list_rois(
            self.db_path, self.window.current_case_id, "Desktop Test Reader", 1
        )
        self.assertEqual(len(stored), 1)

    def test_undo_reverses_clearing_and_generating_too(self) -> None:
        self.window.set_modality("swi")
        self.window.auto_segment()
        grown = self.window.label_volume.copy()
        self.assertTrue(grown.any())

        self.window.clear_roi()
        self.assertFalse(self.window._selected_label_mask().any())
        self.window.undo_roi()
        np.testing.assert_array_equal(self.window.label_volume, grown)

        empty = np.zeros_like(grown)
        self.window.label_volume[...] = 0
        self.window.auto_segment()
        self.assertTrue(self.window.label_volume.any())
        self.window.undo_roi()
        np.testing.assert_array_equal(self.window.label_volume, empty)

    def test_right_drag_is_contrast_while_the_brush_is_up(self) -> None:
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        self.window.set_modality("swi")
        self.window.auto_segment()
        self.window.set_tool("brush")
        canvas = self.window.view_panels["axial"].canvas
        before_volume = self.window.roi_volume_mm3()
        before_window = canvas.window_limits

        start = QPointF(canvas.width() / 2, canvas.height() / 2)
        end = QPointF(start.x() + 90, start.y() - 50)
        canvas.mousePressEvent(
            QMouseEvent(QMouseEvent.Type.MouseButtonPress, start, canvas.mapToGlobal(start),
                        Qt.MouseButton.RightButton, Qt.MouseButton.RightButton,
                        Qt.KeyboardModifier.NoModifier)
        )
        canvas.mouseMoveEvent(
            QMouseEvent(QMouseEvent.Type.MouseMove, end, canvas.mapToGlobal(end),
                        Qt.MouseButton.NoButton, Qt.MouseButton.RightButton,
                        Qt.KeyboardModifier.NoModifier)
        )
        canvas.mouseReleaseEvent(
            QMouseEvent(QMouseEvent.Type.MouseButtonRelease, end, canvas.mapToGlobal(end),
                        Qt.MouseButton.RightButton, Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier)
        )
        self.assertNotEqual(canvas.window_limits, before_window, "right-drag must adjust contrast")
        self.assertAlmostEqual(self.window.roi_volume_mm3(), before_volume, places=6)

    def test_growing_from_a_painted_stroke(self) -> None:
        self.window.set_modality("swi")
        volume = self.window.volumes["swi"]
        from imaging import ras_to_voxel

        seed = np.rint(ras_to_voxel(volume.affine, self.window.marker_ras)).astype(int)
        # A short scribble through the lesion, as if painted by hand.
        self.window._on_roi_stroke_started()
        value = self.window._label_value_for(str(self.window.selected_target["target_id"]))
        self.window.label_volume[seed[0] - 1:seed[0] + 2, seed[1], seed[2]] = value
        scribble = int((self.window.label_volume == value).sum())

        self.window.grow_from_stroke()
        grown = int((self.window.label_volume == value).sum())
        self.assertGreater(grown, scribble, "growing should expand the stroke")
        self.assertGreater(self.window.roi_volume_mm3(), 0.0)

        # With nothing painted it says so rather than growing from nowhere.
        self.window.clear_roi()
        self.window.grow_from_stroke()
        self.assertEqual(self.window.roi_volume_mm3(), 0.0)

    def test_the_mask_is_saved_with_the_review_and_comes_back(self) -> None:
        from imaging import load_volume

        self.window.set_modality("swi")
        self.window.auto_segment()
        volume_mm3 = self.window.roi_volume_mm3()
        target_id = str(self.window.selected_target["target_id"])
        self.window.set_verdict(1)
        self.assertTrue(self.window.save_current_review(advance=False))
        self.assertFalse(self.window._roi_dirty)

        path = self.review_store.label_path(self.db_path, self.case_id, "Desktop Test Reader", 1)
        self.assertTrue(path.exists())
        stored = load_volume(path, self.settings.axcodes)
        # The mask must line up with the images it was drawn on.
        self.assertEqual(stored.shape, self.window.volumes["swi"].shape)
        np.testing.assert_allclose(stored.affine, self.window.volumes["swi"].affine, atol=1e-4)

        rows = self.review_store.list_rois(self.db_path, self.case_id, "Desktop Test Reader", 1)
        self.assertIn(target_id, rows)
        self.assertAlmostEqual(rows[target_id]["volume_mm3"], volume_mm3, places=3)

        self.window.load_case(self.case_id, force=True)
        self.assertTrue(self._wait_for(lambda: self.window.label_volume is not None))
        self.assertAlmostEqual(self.window.roi_volume_mm3(), volume_mm3, places=6)

    def test_erasing_leaves_a_neighbouring_finding_alone(self) -> None:
        multi = next(
            (
                case["case_id"]
                for case in self.window.all_cases
                if case["source_count"] >= 2 and case["file_status"] == "complete"
            ),
            None,
        )
        if multi is None:
            self.skipTest("No complete multi-finding case in this data root.")
        self.window.load_case(multi)
        self.assertTrue(self._wait_for(lambda: self.window.current_case_id == multi))
        self.window.set_modality("swi")

        self.window._select_target_index(0, confirm=False)
        self.window.auto_segment()
        first_id = str(self.window.selected_target["target_id"])
        first_value = self.window._label_value_for(first_id)
        first_voxels = int((self.window.label_volume == first_value).sum())

        self.window._select_target_index(1, confirm=False)
        self.window.auto_segment()
        second_id = str(self.window.selected_target["target_id"])
        self.assertNotEqual(self.window._label_value_for(second_id), first_value)

        # Clearing the second finding must not touch the first.
        self.window.clear_roi()
        self.assertEqual(int((self.window.label_volume == first_value).sum()), first_voxels)

    # --------------------------------------------------------- loading state --
    def test_a_sequence_that_exists_says_loading_not_unavailable(self) -> None:
        case = self.review_store.get_case(self.db_path, self.case_id)
        paths = {modality: case[f"{modality}_path"] for modality in ("qsm", "swi", "mip")}
        self.assertTrue(all(paths.values()))
        self.window._set_loading_placeholders(paths)
        for plane, panel in self.window.view_panels.items():
            self.assertEqual(panel.canvas._empty_state, "loading", plane)

        # A case that really has nothing keeps saying so.
        self.window._set_loading_placeholders({modality: None for modality in ("qsm", "swi", "mip")})
        for plane, panel in self.window.view_panels.items():
            self.assertEqual(panel.canvas._empty_state, "missing", plane)

    def test_missing_sequences_are_struck_through(self) -> None:
        self.window._mark_missing_modalities({"swi": "a.nii.gz", "qsm": None, "mip": "c.nii.gz"})
        buttons = self.window.modality_segments
        self.assertFalse(buttons.button("swi").font().strikeOut())
        self.assertTrue(buttons.button("qsm").font().strikeOut())
        self.assertFalse(buttons.button("mip").font().strikeOut())
        self.assertFalse(buttons.button("qsm").isEnabled())
        self.assertTrue(buttons.button("swi").isEnabled())
        # And it clears again for a complete case.
        self.window._mark_missing_modalities({"swi": "a", "qsm": "b", "mip": "c"})
        self.assertFalse(buttons.button("qsm").font().strikeOut())

    # --------------------------------------------------- position correction --
    def _move_selected_finding(self, offset=(3.0, 2.0, 1.0)) -> tuple:
        """Move the finding to an exact coordinate, for the position tests.

        Snapping is turned off here on purpose: these tests are about the
        bookkeeping around a correction -- that the source row is untouched,
        that variants appear, that selecting Source releases it -- and letting
        the coordinate settle onto real anatomy would only make them assert a
        number that depends on the image.  Snapping has its own tests.
        """

        self.window.settings.store.setValue("reading/snap_to_lesion", False)
        source = tuple(self.window.selected_target["source_ras"])
        moved = tuple(a + b for a, b in zip(source, offset))
        self.window.target_ras = moved
        self.window.move_finding_here()
        return source, moved

    def test_a_correction_is_saved_without_touching_the_source(self) -> None:
        target_id = self.window.selected_target["target_id"]
        source, moved = self._move_selected_finding()
        self.assertEqual(self.window.selected_variant, "Desktop Test Reader")
        self.assertTrue(self.window._review_dirty)
        self.window.set_verdict(1)
        self.assertTrue(self.window.save_current_review(advance=False))

        stored = self.review_store.list_targets(
            self.db_path, self.case_id, "Desktop Test Reader", 1
        )
        finding = next(item for item in stored if item["target_id"] == target_id)
        # The source coordinate is untouched, and the reader's own version sits
        # alongside it.
        self.assertEqual(tuple(finding["source_ras"]), source)
        for saved, expected in zip(finding["reader_ras"], moved):
            self.assertAlmostEqual(saved, expected, places=6)
        self.assertAlmostEqual(finding["reader_moved_mm"], 14.0 ** 0.5, places=6)
        self.assertEqual(tuple(finding["effective_ras"]), tuple(finding["reader_ras"]))
        self.assertEqual(len(finding["position_variants"]), 2)

    def test_selecting_source_again_releases_the_correction(self) -> None:
        target_id = self.window.selected_target["target_id"]
        self._move_selected_finding()
        self.window.set_verdict(1)
        self.assertTrue(self.window.save_current_review(advance=False))
        self.assertIsNotNone(self.window.selected_target["reader_ras"])

        index = self.window.position_combo.findData("source")
        self.assertGreaterEqual(index, 0)
        self.window.position_combo.setCurrentIndex(index)
        self.assertEqual(self.window.selected_variant, "source")
        self.assertTrue(self.window.save_current_review(advance=False))
        stored = self.review_store.list_targets(
            self.db_path, self.case_id, "Desktop Test Reader", 1
        )
        finding = next(item for item in stored if item["target_id"] == target_id)
        self.assertIsNone(finding["reader_ras"])

    def test_saving_while_viewing_another_readers_position_adopts_it(self) -> None:
        target_id = str(self.window.selected_target["target_id"])
        source = tuple(self.window.selected_target["source_ras"])
        other = tuple(value + 4.0 for value in source)
        self.review_store.save_review(
            self.db_path,
            target_id=target_id,
            case_id=self.case_id,
            reader_id="Reader Two",
            review_round=1,
            verify=1,
            comment="it is a little lower",
            corrected_ras=other,
        )
        # Reload the case so the other reader's version appears.
        self.window.load_case(self.case_id, force=True)
        self.assertTrue(self._wait_for(lambda: self.window.selected_target is not None))
        index = self.window.position_combo.findData("Reader Two")
        self.assertGreaterEqual(index, 0, "the other reader's position should be offered")
        self.window.position_combo.setCurrentIndex(index)
        self.assertEqual(self.window.selected_variant, "Reader Two")
        for shown, expected in zip(self.window.marker_ras, other):
            self.assertAlmostEqual(shown, expected, places=6)

        self.window.set_verdict(1)
        self.assertTrue(self.window.save_current_review(advance=False))
        stored = self.review_store.list_targets(
            self.db_path, self.case_id, "Desktop Test Reader", 1
        )
        finding = next(item for item in stored if item["target_id"] == target_id)
        for saved, expected in zip(finding["reader_ras"], other):
            self.assertAlmostEqual(saved, expected, places=6)
        # The other reader's record is still theirs and still visible.
        self.assertEqual(len(finding["position_variants"]), 3)

    def test_looking_at_another_position_does_not_dirty_the_review(self) -> None:
        target_id = str(self.window.selected_target["target_id"])
        source = tuple(self.window.selected_target["source_ras"])
        self.review_store.save_review(
            self.db_path,
            target_id=target_id,
            case_id=self.case_id,
            reader_id="Reader Two",
            review_round=1,
            verify=1,
            comment="a bit lower",
            corrected_ras=tuple(value + 4.0 for value in source),
        )
        self.window.load_case(self.case_id, force=True)
        self.assertTrue(self._wait_for(lambda: self.window.selected_target is not None))
        self.assertFalse(self.window._review_dirty)

        index = self.window.position_combo.findData("Reader Two")
        self.window.position_combo.setCurrentIndex(index)
        # Browsing someone else's position is not an edit: no dirty flag, and
        # no unsaved-changes prompt when moving on.
        self.assertFalse(self.window._review_dirty)
        self.assertEqual(self.window.dirty_label.text(), "")
        # But the reader is told what saving would do.
        self.assertEqual(self.window.position_hint.text(), "saving adopts this")
        self.assertTrue(self.window._confirm_dirty())

        # Back to source: nothing of mine is stored, so nothing to warn about.
        self.window.position_combo.setCurrentIndex(self.window.position_combo.findData("source"))
        self.assertFalse(self.window._review_dirty)
        self.assertEqual(self.window.position_hint.text(), "")

        # Moving it myself *is* an edit.
        self._move_selected_finding()
        self.assertTrue(self.window._review_dirty)
        self.assertEqual(self.window.position_hint.text(), "unsaved move")

    def test_saving_after_only_browsing_still_adopts_what_is_shown(self) -> None:
        target_id = str(self.window.selected_target["target_id"])
        source = tuple(self.window.selected_target["source_ras"])
        other = tuple(value + 4.0 for value in source)
        self.review_store.save_review(
            self.db_path, target_id=target_id, case_id=self.case_id, reader_id="Reader Two",
            review_round=1, verify=1, comment="lower", corrected_ras=other,
        )
        self.window.load_case(self.case_id, force=True)
        self.assertTrue(self._wait_for(lambda: self.window.selected_target is not None))
        self.window.position_combo.setCurrentIndex(self.window.position_combo.findData("Reader Two"))
        self.assertFalse(self.window._review_dirty)
        self.assertTrue(self.window.save_current_review(advance=False))
        finding = next(
            item
            for item in self.review_store.list_targets(self.db_path, self.case_id, "Desktop Test Reader", 1)
            if item["target_id"] == target_id
        )
        for saved, expected in zip(finding["reader_ras"], other):
            self.assertAlmostEqual(saved, expected, places=6)

    def test_the_source_position_is_ghosted_while_a_correction_is_shown(self) -> None:
        canvas = self.window.view_panels["axial"].canvas
        self.assertIsNone(canvas._ghost_voxel)
        self._move_selected_finding()
        self.assertIsNotNone(canvas._ghost_voxel)
        index = self.window.position_combo.findData("source")
        self.window.position_combo.setCurrentIndex(index)
        self.assertIsNone(canvas._ghost_voxel)

    def test_holding_the_peek_key_shows_every_position(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent

        canvas = self.window.view_panels["axial"].canvas
        self._move_selected_finding()
        self.assertFalse(canvas._show_variants)

        press = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_V, Qt.KeyboardModifier.NoModifier)
        self.window.keyPressEvent(press)
        self.assertTrue(canvas._show_variants)
        self.assertEqual(len(canvas._variant_markers), 2)
        colours = {marker[1] for marker in canvas._variant_markers}
        self.assertEqual(len(colours), 2, "each position needs its own colour")

        release = QKeyEvent(QKeyEvent.Type.KeyRelease, Qt.Key.Key_V, Qt.KeyboardModifier.NoModifier)
        self.window.keyReleaseEvent(release)
        self.assertFalse(canvas._show_variants)
        # And it never sticks: losing focus turns it off too.
        self.window.keyPressEvent(press)
        self.assertTrue(canvas._show_variants)
        self.window.set_variant_peek(False)
        self.assertFalse(canvas._show_variants)

    def test_moving_a_finding_does_not_move_its_neighbours(self) -> None:
        multi = next(
            (
                case["case_id"]
                for case in self.window.all_cases
                if case["source_count"] >= 2 and case["file_status"] == "complete"
            ),
            None,
        )
        if multi is None:
            self.skipTest("No complete multi-finding case in this data root.")
        self.window.load_case(multi)
        self.assertTrue(self._wait_for(lambda: self.window.current_case_id == multi))
        canvas = self.window.view_panels["axial"].canvas
        before = [voxel.copy() for voxel in canvas._secondary_voxels]
        self._move_selected_finding()
        after = canvas._secondary_voxels
        self.assertEqual(len(before), len(after))
        for first, second in zip(before, after):
            np.testing.assert_allclose(first, second)

    # ------------------------------------------------------------- contrast --
    def test_window_level_starts_automatic_and_applies_to_all_views(self) -> None:
        canvases = [panel.canvas for panel in self.window.view_panels.values()]
        for canvas in canvases:
            self.assertEqual(canvas.window_limits, canvas._auto_window)
        level, window = canvases[0].window_level
        self.window.set_window_level(level + window * 0.25, window * 0.5)
        first = canvases[0].window_limits
        for canvas in canvases[1:]:
            self.assertEqual(canvas.window_limits, first)
            self.assertNotEqual(canvas.window_limits, canvas._auto_window)
        self.window.reset_window_level()
        for canvas in canvases:
            self.assertEqual(canvas.window_limits, canvas._auto_window)

    def test_window_level_is_remembered_per_sequence(self) -> None:
        canvas = self.window.view_panels["axial"].canvas
        level, window = canvas.window_level
        self.window.set_window_level(level, window * 0.4)
        swi_limits = canvas.window_limits

        self.window.set_modality("qsm")
        # QSM has an unrelated intensity range, so it starts from its own
        # automatic window rather than inheriting the SWI one.
        self.assertEqual(canvas.window_limits, canvas._auto_window)
        qsm_level, qsm_window = canvas.window_level
        self.window.set_window_level(qsm_level, qsm_window * 2.0)
        qsm_limits = canvas.window_limits

        self.window.set_modality("swi")
        self.assertEqual(canvas.window_limits, swi_limits)
        self.window.set_modality("qsm")
        self.assertEqual(canvas.window_limits, qsm_limits)

    def test_drag_brightens_and_widens_the_window(self) -> None:
        from PySide6.QtCore import QPointF

        canvas = self.window.view_panels["axial"].canvas
        start_level, start_window = canvas.window_level
        canvas._wl_origin = QPointF(120.0, 120.0)
        canvas._wl_start = canvas.window_limits
        canvas._wl_dragging = True
        # Dragging up brightens the image, which lowers the level.
        canvas._adjust_window_by_drag(QPointF(120.0, 40.0))
        level_after, window_after = canvas.window_level
        self.assertLess(level_after, start_level)
        self.assertAlmostEqual(window_after, start_window, places=6)
        # Dragging right widens the window, which lowers the contrast.
        canvas._wl_start = canvas.window_limits
        canvas._wl_origin = QPointF(120.0, 40.0)
        canvas._adjust_window_by_drag(QPointF(300.0, 40.0))
        self.assertGreater(canvas.window_level[1], window_after)
        canvas._wl_dragging = False
        # The other views followed along.
        self.assertEqual(
            self.window.view_panels["coronal"].canvas.window_limits, canvas.window_limits
        )

    def test_contrast_returns_to_automatic_on_the_next_case_unless_sticky(self) -> None:
        canvas = self.window.view_panels["axial"].canvas
        level, window = canvas.window_level
        self.window.set_window_level(level, window * 0.3)
        manual = canvas.window_limits
        other = next(
            case["case_id"]
            for case in self.window.visible_cases
            if case["case_id"] != self.window.current_case_id
        )
        self.window.load_case(other)
        self.assertTrue(self._wait_for(lambda: self.window.current_case_id == other))
        self.assertEqual(canvas.window_limits, canvas._auto_window)

        self.settings.set_sticky_window(True)
        self.window.set_window_level(*self.window.view_panels["axial"].canvas.window_level)
        self.window.set_window_level(level, window * 0.3)
        back = self.window.current_case_id
        self.window.load_case(
            next(
                case["case_id"]
                for case in self.window.visible_cases
                if case["case_id"] != back
            )
        )
        self.assertTrue(self._wait_for(lambda: self.window.current_case_id != back))
        self.assertEqual(canvas.window_limits, manual)
        self.settings.set_sticky_window(False)

    # ------------------------------------------------------------- prefetch --
    def test_the_next_case_is_prefetched_in_the_background(self) -> None:
        current = self.window.current_case_id
        index = next(
            position
            for position, case in enumerate(self.window.visible_cases)
            if case["case_id"] == current
        )
        following = self.window.visible_cases[index + 1]["case_id"]
        expected = self.review_store.get_case(self.db_path, following)
        path = expected.get(f"{self.window.current_modality}_path")
        if not path:
            self.skipTest(f"{following} has no {self.window.current_modality} volume.")

        self.window._prefetch_timer.stop()
        self.window._start_prefetch()
        self.assertTrue(
            self._wait_for(
                lambda: self.window.volume_cache.get(path, self.settings.axcodes) is not None,
                timeout_ms=20000,
            )
        )
        # And the prefetched volume is the one the next load actually uses.
        cached = self.window.volume_cache.get(path, self.settings.axcodes)
        self.window.load_case(following)
        self.assertTrue(self._wait_for(lambda: self.window.current_case_id == following))
        self.assertIs(self.window.volumes[self.window.current_modality], cached)

    def test_prefetch_covers_all_three_sequences_of_the_next_case(self) -> None:
        current = self.window.current_case_id
        index = next(
            position
            for position, case in enumerate(self.window.visible_cases)
            if case["case_id"] == current
        )
        following = next(
            (
                case
                for case in self.window.visible_cases[index + 1 :]
                if case["file_status"] == "complete"
            ),
            None,
        )
        if following is None:
            self.skipTest("No complete case after the current one.")
        # Put it directly after the current case so the prefetcher targets it.
        self.window.visible_cases = [
            self.window.visible_cases[index],
            following,
        ] + [case for case in self.window.visible_cases if case["case_id"] not in {current, following["case_id"]}]
        record = self.review_store.get_case(self.db_path, following["case_id"])
        expected = [record[f"{modality}_path"] for modality in ("qsm", "swi", "mip")]
        self.assertTrue(all(expected))

        self.window._prefetch_timer.stop()
        self.window._start_prefetch()
        self.assertTrue(
            self._wait_for(lambda: self.window._prefetch_thread is None, timeout_ms=30000)
        )
        axcodes = self.settings.axcodes
        for path in expected:
            self.assertIsNotNone(
                self.window.volume_cache.get(path, axcodes), f"{path} was not prefetched"
            )
        # And the case on screen is still cached, not evicted by the prefetch.
        for modality in ("qsm", "swi", "mip"):
            volume = self.window.volumes[modality]
            self.assertIsNotNone(self.window.volume_cache.get(volume.path, axcodes), modality)

    def test_a_prefetch_thread_is_never_released_while_running(self) -> None:
        """Qt aborts the process if a running QThread is destroyed.

        That is a hard crash with no Python traceback, so the invariant is
        asserted directly rather than waiting for the race to show itself.
        """

        released_while_running: list[str] = []
        original = self.desktop_app.MicrobleedViewer._release_prefetch

        def guarded(window, thread, worker):
            if thread.isRunning():
                released_while_running.append(str(thread))
            return original(window, thread, worker)

        self.desktop_app.MicrobleedViewer._release_prefetch = guarded
        try:
            for _ in range(3):
                self.window._prefetch_timer.stop()
                self.window._start_prefetch()
                self.assertTrue(
                    self._wait_for(
                        lambda: self.window._prefetch_thread is None, timeout_ms=30000
                    ),
                    "the prefetch thread was never released",
                )
        finally:
            self.desktop_app.MicrobleedViewer._release_prefetch = original
        self.assertEqual(released_while_running, [])

    def test_stopping_a_prefetch_waits_for_the_thread(self) -> None:
        self.window._prefetch_timer.stop()
        self.window._start_prefetch()
        thread = self.window._prefetch_thread
        if thread is None:
            self.skipTest("Nothing left to prefetch after this case.")
        self.window._stop_prefetch()
        # Synchronous: the thread has actually exited before we let go of it.
        self.assertFalse(thread.isRunning())
        self.assertIsNone(self.window._prefetch_thread)
        self.assertIsNone(self.window._prefetch_worker)

    def test_prefetch_can_be_switched_off(self) -> None:
        self.settings.update(
            auto_zoom=True,
            lesion_fov_mm=60.0,
            save_advances=True,
            default_modality="swi",
            prefetch=False,
        )
        self.assertFalse(self.settings.prefetch_enabled)
        self.window._prefetch_timer.stop()
        self.window._schedule_prefetch()
        self.assertFalse(self.window._prefetch_timer.isActive())
        self.settings.update(
            auto_zoom=True,
            lesion_fov_mm=60.0,
            save_advances=True,
            default_modality="swi",
            prefetch=True,
        )

    # -------------------------------------------------------- deferred writes --
    def test_log_writes_do_not_block_the_gui_thread(self) -> None:
        """A slow database must never freeze the window.

        The review store often lives in a synchronised folder, where the sync
        client can hold the file for seconds; SQLite waits for the lock, so any
        write on the GUI thread stops the window from repainting.
        """

        import time

        released = []
        original = self.desktop_app.log_event

        def slow_log(*args, **kwargs):
            time.sleep(0.4)
            released.append(args[1] if len(args) > 1 else None)
            return original(*args, **kwargs)

        self.desktop_app.log_event = slow_log
        try:
            start = time.perf_counter()
            for _ in range(5):
                self.window._log_event("stress", case_id=self.window.current_case_id)
            blocked = time.perf_counter() - start
            # Five writes of 0.4 s each would be two seconds of frozen window.
            self.assertLess(blocked, 0.1, f"the GUI thread waited {blocked:.2f}s")
            self.assertTrue(
                self._wait_for(lambda: len(released) == 5, timeout_ms=15000),
                "queued writes were dropped",
            )
        finally:
            self.desktop_app.log_event = original

    def test_queued_writes_are_flushed_before_the_window_closes(self) -> None:
        case_id = self.window.current_case_id
        for index in range(8):
            self.window._log_event("closing_stress", case_id=case_id, details={"n": index})
        self.window._review_dirty = False
        self.window.close()
        self.app.processEvents()
        entries = self.review_store.recent_case_log(self.db_path, case_id, limit=500)
        stressed = [item for item in entries if item["event_type"] == "closing_stress"]
        self.assertEqual(len(stressed), 8)

    def test_a_synced_review_database_is_recognised(self) -> None:
        detect = self.desktop_app.synced_folder_name
        self.assertEqual(detect(Path(r"D:\somebody\OneDrive - Example University\x\review.sqlite")), "onedrive")
        self.assertEqual(detect(Path(r"C:\Users\me\Dropbox\study\review.sqlite")), "dropbox")
        self.assertIsNone(detect(self.temp_dir / "review.sqlite"))

    def test_volume_cache_evicts_and_respects_orientation(self) -> None:
        from imaging import preset_axcodes

        cache = self.desktop_app.VolumeCache(limit=2)
        volume = self.window.volumes[self.window.current_modality]
        cache.put("a.nii.gz", ("L", "P", "I"), volume)
        cache.put("b.nii.gz", ("L", "P", "I"), volume)
        self.assertIsNotNone(cache.get("a.nii.gz", ("L", "P", "I")))
        cache.put("c.nii.gz", ("L", "P", "I"), volume)
        # "a" was used most recently of the first two, so "b" is the one to go.
        self.assertIsNone(cache.get("b.nii.gz", ("L", "P", "I")))
        self.assertEqual(len(cache), 2)
        # A different display preset is a different entry, never a stale hit.
        self.assertIsNone(cache.get("a.nii.gz", preset_axcodes("neurological")))

    # ------------------------------------------------------------ shortcuts --
    def test_shortcuts_can_be_rebound(self) -> None:
        from PySide6.QtGui import QKeySequence

        self.assertEqual(self.settings.shortcut("verdict_no"), "N")
        self.settings.set_shortcuts({"verdict_no": "V", "next_case": "Ctrl+Right"})
        self.window._bind_shortcuts()
        self.assertEqual(
            self.window._shortcuts["verdict_no"].key(), QKeySequence("V")
        )
        self.assertEqual(
            self.window._shortcuts["next_case"].key(), QKeySequence("Ctrl+Right")
        )
        # The sidebar legend is generated from the same table.
        self.assertIn("V", self.window.shortcut_legend.text())
        # The keys sit next to the section title, not inside the buttons, so
        # three verdict buttons still fit a narrow panel.
        self.assertIn("V", self.window.verdict_keys_label.text())
        self.assertIn(self.window.verdict_segments.button("no").text(), ("No",))
        self.assertIn("V", self.window.verdict_segments.button("no").toolTip())

    def test_unbound_shortcut_is_simply_not_registered(self) -> None:
        self.settings.set_shortcuts({"maximize_view": ""})
        self.window._bind_shortcuts()
        self.assertNotIn("maximize_view", self.window._shortcuts)

    def test_duplicate_shortcuts_are_refused(self) -> None:
        from PySide6.QtGui import QKeySequence

        dialog = self.desktop_app.SettingsDialog(self.settings, self.window)
        try:
            dialog.shortcut_edits["verdict_yes"].setKeySequence(QKeySequence("Q"))
            dialog.shortcut_edits["verdict_no"].setKeySequence(QKeySequence("Q"))
            self.assertIsNone(dialog._collect_shortcuts())
            self.assertIn("Q", dialog.shortcut_warning.text())
            dialog.shortcut_edits["verdict_no"].setKeySequence(QKeySequence("W"))
            bindings = dialog._collect_shortcuts()
            self.assertIsNotNone(bindings)
            self.assertEqual(bindings["verdict_yes"], "Q")
            self.assertEqual(bindings["verdict_no"], "W")
        finally:
            dialog.deleteLater()

    # -------------------------------------------------------------- dataset --
    def test_switching_dataset_moves_the_reviews_to_the_new_store(self) -> None:
        from config import Dataset

        self.window.set_verdict(1)
        self.assertTrue(self.window.save_current_review(advance=False))
        first_db = self.window.db_path

        other_db = self.temp_dir / "second_review.sqlite"
        other = Dataset.create(self.source, self.data_root, other_db)
        self.review_store.initialize_store(other.workbook, other.data_root, other.review_db)
        session = self.review_store.start_new_session(other.review_db, "Second Reader")

        self.assertTrue(self.window.switch_dataset(other, session=session))
        self.assertEqual(self.window.db_path, other_db)
        self.assertEqual(self.window.reader_id, "Second Reader")
        self.assertEqual(self.window.dataset, other)
        self.assertTrue(self.window.all_cases)
        self.assertTrue(
            self._wait_for(lambda: self.window.current_case_id is not None)
        )
        # The review written before the switch stays in the first database and
        # is not visible in the new one.
        original = self.review_store.list_targets(first_db, self.case_id, "Desktop Test Reader", 1)
        self.assertEqual(original[0]["reader_verify"], 1)
        moved = self.review_store.list_targets(other_db, self.case_id, "Desktop Test Reader", 1)
        self.assertIsNone(moved[0]["reader_verify"])
        # The previous session was closed rather than left open.
        with self.review_store.connect(first_db) as connection:
            row = connection.execute(
                "SELECT status FROM reader_sessions WHERE session_id = ?",
                (self.session["session_id"],),
            ).fetchone()
        self.assertEqual(row["status"], "closed")

    def test_a_broken_dataset_is_reported_and_changes_nothing(self) -> None:
        from config import Dataset

        before_db = self.window.db_path
        broken = Dataset.create(self.temp_dir / "missing.xlsx", self.data_root, self.temp_dir / "x.sqlite")
        self.assertTrue(broken.problems())
        self.assertEqual(self.window.db_path, before_db)

    def test_recent_datasets_are_remembered(self) -> None:
        from config import Dataset

        first = Dataset.create(self.source, self.data_root, self.temp_dir / "a.sqlite")
        second = Dataset.create(self.source, self.data_root, self.temp_dir / "b.sqlite")
        self.settings.remember_dataset(first)
        self.settings.remember_dataset(second)
        recent = self.settings.recent_datasets()
        self.assertEqual(recent[0], second)
        self.assertIn(first, recent)
        self.settings.remember_dataset(first)
        self.assertEqual(self.settings.recent_datasets()[0], first)
        self.assertEqual(len(self.settings.recent_datasets()), 2)

    def test_settings_round_trip(self) -> None:
        self.settings.update(
            auto_zoom=False, lesion_fov_mm=35.0, save_advances=False, default_modality="mip"
        )
        self.assertFalse(self.settings.auto_zoom)
        self.assertEqual(self.settings.lesion_fov_mm, 35.0)
        self.assertFalse(self.settings.save_advances)
        self.assertEqual(self.settings.default_modality, "mip")
        reloaded = self.desktop_app.ViewerSettings(self.settings.store)
        self.assertFalse(reloaded.auto_zoom)
        self.assertEqual(reloaded.lesion_fov_mm, 35.0)


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class CanvasGeometryTests(unittest.TestCase):
    """Where a voxel is drawn, and which voxel a click lands on.

    A pixmap pixel occupies the half-open interval ``[j, j+1)`` on screen, so
    the voxel it stands for is centred at ``j + 0.5``.  Getting that half pixel
    wrong is invisible in a round trip -- ``voxel -> ras -> voxel`` still
    matches -- but it biases every coordinate a reader records by half a voxel,
    which on this data is 0.43 mm in plane.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        import desktop_app

        cls.desktop_app = desktop_app

    def test_the_background_writer_shuts_down_without_being_killed(self) -> None:
        """``QThread::terminate`` in the middle of a commit is the one thing
        both Qt and SQLite tell you not to do, and the whole reason this thread
        exists is that a synced folder can hold the file for seconds.  A slow
        write has to be waited out and the rest abandoned, never killed."""

        import threading

        writer = self.desktop_app.DatabaseWriter()
        killed: list[int] = []
        writer.terminate = lambda: killed.append(1)  # type: ignore[method-assign]
        started = threading.Event()
        done: list[str] = []
        writer.start()
        try:
            def slow() -> None:
                started.set()
                threading.Event().wait(0.6)
                done.append("slow")

            writer.submit("slow", slow)
            self.assertTrue(started.wait(5.0))
            for index in range(50):
                writer.submit(f"queued {index}", lambda: done.append("queued"))
            # Deliberately shorter than the write in flight.
            writer.stop(timeout_ms=50)
        finally:
            writer.stop(timeout_ms=5000)
        self.assertEqual(killed, [], "the writer thread was killed instead of waited out")
        self.assertFalse(writer.isRunning())
        self.assertIn("slow", done)

    def _canvas(self, plane: str = "axial", spacing=(0.859, 0.859, 1.0)):
        from imaging import Volume

        affine = np.diag([-spacing[0], -spacing[1], -spacing[2], 1.0])
        affine[:3, 3] = [80.0, 90.0, 70.0]
        data = np.zeros((64, 64, 40), dtype=np.float32)
        volume = Volume(
            path="synthetic",
            data=data,
            affine=affine,
            shape=data.shape,
            voxel_sizes=spacing,
            window=(0.0, 1.0),
            source_affine=affine,
            source_shape=data.shape,
        )
        canvas = self.desktop_app.SliceCanvas(plane)
        canvas.resize(600, 600)
        canvas.set_volume(volume)
        canvas.set_slice(20)
        return canvas

    @staticmethod
    def _geometry(canvas):
        from PySide6.QtCore import QPointF

        image = canvas._display_array()
        rect, scales = canvas._draw_geometry(int(image.shape[1]), int(image.shape[0]))

        def centre_of(column: float, row: float) -> QPointF:
            """The widget point at the visual middle of that pixmap pixel."""

            return QPointF(
                rect.left() + (column + 0.5) * scales[0],
                rect.top() + (row + 0.5) * scales[1],
            )

        return rect, scales, centre_of

    def test_a_marker_is_drawn_over_the_voxel_it_names(self) -> None:
        canvas = self._canvas()
        rect, scales, centre_of = self._geometry(canvas)
        point = canvas._image_point_of(np.asarray([32.0, 30.0, 20.0]))
        drawn_x = rect.left() + point[0] * scales[0]
        drawn_y = rect.top() + point[1] * scales[1]
        expected = centre_of(32, 30)
        self.assertAlmostEqual(drawn_x, expected.x(), places=6)
        self.assertAlmostEqual(drawn_y, expected.y(), places=6)

    def test_clicking_the_middle_of_a_voxel_selects_that_voxel(self) -> None:
        for plane, expected in (
            ("axial", (32.0, 30.0, 20.0)),
            ("coronal", (32.0, 20.0, 30.0)),
            ("sagittal", (20.0, 32.0, 30.0)),
        ):
            with self.subTest(plane=plane):
                canvas = self._canvas(plane)
                _rect, _scales, centre_of = self._geometry(canvas)
                image_point = canvas._image_point_at(centre_of(32, 30))
                voxel = canvas._voxel_from_image_point(*image_point)
                np.testing.assert_allclose(voxel, expected, atol=1e-9)

    def test_a_click_and_the_marker_it_produces_agree(self) -> None:
        """Click a voxel, feed the result back as a marker, land on the click."""

        from PySide6.QtCore import QPointF

        canvas = self._canvas()
        rect, scales, centre_of = self._geometry(canvas)
        clicked = centre_of(41, 17)
        voxel = canvas._voxel_from_image_point(*canvas._image_point_at(clicked))
        point = canvas._image_point_of(voxel)
        redrawn = QPointF(rect.left() + point[0] * scales[0], rect.top() + point[1] * scales[1])
        self.assertAlmostEqual(redrawn.x(), clicked.x(), places=6)
        self.assertAlmostEqual(redrawn.y(), clicked.y(), places=6)

    def test_the_brush_can_be_a_sphere_instead_of_a_disc(self) -> None:
        """A microbleed is 2-12 voxels across in three dimensions.

        A brush that only ever touches the slice on screen makes painting one
        lesion five separate strokes on five slices, which is why ITK-SNAP and
        3D Slicer both offer a spherical brush for small lesions.
        """

        canvas = self._canvas()
        _rect, _scales, centre_of = self._geometry(canvas)
        labels = np.zeros((64, 64, 40), dtype=np.uint16)
        canvas.set_label_volume(labels, 7)
        canvas.set_paint_mode("paint")
        canvas.set_brush_radius(2.0)

        canvas.set_brush_3d(False)
        canvas._paint_at(centre_of(32, 30), erase=False)
        flat = np.argwhere(labels == 7)
        self.assertEqual(set(flat[:, 2].tolist()), {20}, "the flat brush left its slice")

        labels[:] = 0
        canvas.set_brush_3d(True)
        canvas._paint_at(centre_of(32, 30), erase=False)
        ball = np.argwhere(labels == 7)
        slices = sorted({int(v) for v in ball[:, 2]})
        self.assertEqual(slices, [18, 19, 20, 21, 22], f"a 2 mm ball on a 1 mm grid: {slices}")
        # Round in millimetres, so it stays round on anisotropic voxels.
        spacing = np.asarray(canvas.volume.voxel_sizes)
        offsets = (ball - np.asarray([32, 30, 20])) * spacing
        self.assertLessEqual(float(np.abs(offsets).max()), 2.0 + 1e-6)
        self.assertGreater(len(ball), len(flat))

    def test_the_eraser_is_spherical_too(self) -> None:
        canvas = self._canvas()
        _rect, _scales, centre_of = self._geometry(canvas)
        labels = np.zeros((64, 64, 40), dtype=np.uint16)
        labels[28:36, 26:34, 17:24] = 7
        canvas.set_label_volume(labels, 7)
        canvas.set_brush_radius(2.0)
        canvas.set_brush_3d(True)
        canvas.set_paint_mode("erase")
        canvas._paint_at(centre_of(32, 30), erase=True)
        self.assertEqual(labels[32, 30, 20], 0)
        self.assertEqual(labels[32, 30, 18], 0, "the eraser did not reach off-slice")
        self.assertEqual(labels[28, 26, 17], 7, "the eraser reached too far")

    def test_the_overlay_can_be_drawn_as_an_outline(self) -> None:
        """A 40% fill hides the voxels the reader is judging.

        At lesion zoom a 3 mm microbleed is most of the viewport, so a filled
        overlay covers the signal that decides whether the mask is right.
        """

        canvas = self._canvas()
        labels = np.zeros((64, 64, 40), dtype=np.uint16)
        labels[28:37, 26:35, 20] = 7          # a solid 9x9 square on this slice
        canvas.set_label_volume(labels, 7)
        canvas.set_slice(20)

        canvas.set_label_outline(False)
        filled = canvas._label_overlay_mask()
        canvas.set_label_outline(True)
        outline = canvas._label_overlay_mask()
        self.assertEqual(int(filled.sum()), 81)
        self.assertEqual(int(outline.sum()), 81 - 49, "the interior was not hollowed out")
        self.assertTrue(bool(outline.any()), "a one-voxel mask must still be visible")

        # A single voxel has no interior, so it stays drawn.
        labels[:] = 0
        labels[32, 30, 20] = 7
        self.assertEqual(int(canvas._label_overlay_mask().sum()), 1)

    def test_the_smallest_brush_still_paints_the_voxel_under_the_cursor(self) -> None:
        """0.3 mm is offered in the panel, and is under half a voxel here."""

        canvas = self._canvas()
        _rect, _scales, centre_of = self._geometry(canvas)
        labels = np.zeros((64, 64, 40), dtype=np.uint16)
        canvas.set_label_volume(labels, 7)
        canvas.set_paint_mode("paint")
        canvas.set_brush_radius(0.3)
        self.assertTrue(canvas._paint_at(centre_of(32, 30), erase=False))
        np.testing.assert_array_equal(np.argwhere(labels == 7), [[32, 30, 20]])

    def test_the_brush_is_centred_on_the_voxel_under_the_cursor(self) -> None:
        canvas = self._canvas()
        _rect, _scales, centre_of = self._geometry(canvas)
        labels = np.zeros((64, 64, 40), dtype=np.uint16)
        canvas.set_label_volume(labels, 7)
        canvas.set_paint_mode("paint")
        for spherical in (False, True):
            canvas.set_brush_3d(spherical)
            for radius in (1.0, 1.5, 2.0, 3.0):
                with self.subTest(radius=radius, spherical=spherical):
                    labels[:] = 0
                    canvas.set_brush_radius(radius)
                    canvas._paint_at(centre_of(32, 30), erase=False)
                    painted = np.argwhere(labels == 7)
                    self.assertGreater(len(painted), 0)
                    for axis, clicked in ((0, 32), (1, 30), (2, 20)):
                        values = painted[:, axis]
                        self.assertEqual(
                            clicked - int(values.min()), int(values.max()) - clicked,
                            f"the stamp is not centred on axis {axis} at r={radius}",
                        )
                    if not spherical:
                        # The flat brush stays on the slice on screen.
                        self.assertEqual(set(painted[:, 2].tolist()), {20})


class ClassicLayoutTests(unittest.TestCase):
    """The preserved classic layout must keep opening a case.

    ``desktop_app_classic.py`` is frozen for side-by-side comparison, so it is
    not developed further, but it shares ``imaging`` and ``review_store`` and
    has to keep working when those change.
    """

    @classmethod
    def setUpClass(cls) -> None:
        source = os.environ.get("TEST_SOURCE_XLSX")
        if not source:
            raise unittest.SkipTest("Set TEST_SOURCE_XLSX to a readable copy of the source workbook.")
        if not (VIEWER_DIR / "desktop_app_classic.py").exists():
            # The frozen layout is kept for local comparison and is not part
            # of a published checkout.
            raise unittest.SkipTest("desktop_app_classic.py is not in this checkout.")
        cls.app = QApplication.instance() or QApplication([])
        cls.source = Path(source)
        cls.data_root = Path(os.environ.get("TEST_DATA_ROOT", PROJECT_DIR / "Data"))

    def test_classic_layout_still_loads_a_case(self) -> None:
        from PySide6.QtCore import QElapsedTimer

        import desktop_app_classic
        import review_store

        temp_dir = Path(tempfile.mkdtemp(prefix="microbleed_classic_test_"))
        try:
            db_path = temp_dir / "review.sqlite"
            review_store.initialize_store(self.source, self.data_root, db_path)
            # Skips rather than fails when the dataset carries no images.
            _sample_case(db_path)
            session = review_store.start_new_session(db_path, "Classic Test Reader")
            window = desktop_app_classic.MicrobleedViewer(db_path, self.data_root, session)
            window.show()
            timer = QElapsedTimer()
            timer.start()
            while window.current_case_id is None and timer.elapsed() < 15000:
                self.app.processEvents()
            self.assertIsNotNone(window.current_case_id)
            self.assertTrue(any(volume is not None for volume in window.volumes.values()))
            self.assertTrue(window.view_panels["axial"].canvas.volume is not None)
            window._review_dirty = False
            window.close()
            self.app.processEvents()
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
