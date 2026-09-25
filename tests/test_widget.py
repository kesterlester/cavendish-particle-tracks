from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile as tf
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialogButtonBox, QMessageBox

from cavendish_particle_tracks._main_widget import (
    IMAGE_LAYER_NAME,
    MEASUREMENTS_LAYER_NAME,
    ParticleTracksWidget,
)

from .conftest import get_dialog


@pytest.mark.parametrize("docking_area", ["left", "bottom"])
def test_open_widget(make_napari_viewer, docking_area):
    """The widget opens in both layouts (they build their buttons differently), and only
    "Load images" is usable until an image layer exists."""
    widget = ParticleTracksWidget(napari_viewer=make_napari_viewer(), docking_area=docking_area)
    assert widget.isVisible() is False
    widget.show()
    assert widget.isVisible() is True

    assert widget.load_button.isEnabled() is True
    assert widget.particle_decays_menu.isEnabled() is False
    assert widget.load_data_button.isEnabled() is False
    assert widget.delete_process.isEnabled() is False
    assert widget.decay_angles_nav_button.isEnabled() is False

    # a tiny in-memory stand-in for loaded images - no files involved
    widget.viewer.add_image(np.random.random((100, 100)), name=IMAGE_LAYER_NAME)

    assert widget.load_button.isEnabled() is False
    assert widget.particle_decays_menu.isEnabled() is True
    assert widget.load_data_button.isEnabled() is True
    assert widget.delete_process.isEnabled() is False
    assert widget.decay_angles_nav_button.isEnabled() is False


def test_add_new_particle_ui(cpt_widget: ParticleTracksWidget):
    assert cpt_widget.table.rowCount() == 0

    cpt_widget.particle_decays_menu.setCurrentIndex(1)

    assert cpt_widget.table.rowCount() == 1
    assert len(cpt_widget.data) == 1


@pytest.mark.parametrize(
    "answer, expected_rows", [(QMessageBox.Yes, 0), (QMessageBox.Cancel, 1)]
)
def test_delete_particle_ui(cpt_widget: ParticleTracksWidget, monkeypatch, answer, expected_rows):
    """Deleting a process removes it from the table and data only if the confirmation box is
    answered Yes (its real buttons are Yes/Cancel - see _on_click_delete_process). The answer is
    patched in just for the delete call: conftest's autouse fixture otherwise makes every message
    box answer Discard, and leaving Cancel in place would also make napari's own teardown refuse
    to close the viewer at the unsaved-data prompt.
    """
    cpt_widget.particle_decays_menu.setCurrentIndex(1)

    assert cpt_widget.table.rowCount() == 1
    assert len(cpt_widget.data) == 1

    with monkeypatch.context() as m:
        m.setattr(QMessageBox, "exec", lambda self: answer)
        cpt_widget._on_click_delete_process()

    assert cpt_widget.table.rowCount() == expected_rows
    assert len(cpt_widget.data) == expected_rows


@pytest.mark.parametrize(
    "data_subdirs, image_count, expect_data_loaded, reload",
    [
        (["my_view1", "my_view2", "my_view3"], [5, 5, 5], True, False),
        (["my_view1", "my_view2", "my_view3"], [2, 2, 2], True, True),
        (["my_view1", "my_view2"], [2, 2], False, False),
        (["my_view1", "my_view2", "my_view3"], [1, 2, 2], False, False),
        (["my_view1", "my_view2", "no_view"], [2, 2, 2], False, False),
    ],
)
def test_load_data(
    cpt_widget: ParticleTracksWidget,
    tmp_path: Path,
    qtbot: QtBot,
    data_subdirs: list[str],
    image_count: list[int],
    expect_data_loaded: bool,
    reload: bool,
):
    """Test loading of images in a folder as 4D image layer with width, height, event, view dimensions."""

    data_layer_index = 0
    if reload:
        cpt_widget.layer_measurements = cpt_widget.viewer.add_points(
            name=MEASUREMENTS_LAYER_NAME
        )
        data_layer_index = 1

    resolution = 8400 if expect_data_loaded else 10
    for subdir, n in zip(data_subdirs, image_count):
        p = tmp_path / subdir
        p.mkdir()
        for i in range(n):
            data = np.random.randint(0, 255, (resolution, resolution, 3), "uint8")
            tf.imwrite(p / f"temp{i}.tif", data)

    def set_directory_and_close(dialog):
        qtbot.addWidget(dialog)
        dialog.setDirectory(str(tmp_path))
        buttonbox = dialog.findChild(QDialogButtonBox, "buttonBox")
        openbutton = buttonbox.children()[1]
        qtbot.mouseClick(openbutton, Qt.LeftButton, delay=1)

    # Open and retrieve file dialog
    get_dialog(
        dialog_trigger=cpt_widget._on_click_load_data,
        dialog_action=set_directory_and_close,
        time_out=5,
    )

    if expect_data_loaded:
        assert len(cpt_widget.viewer.layers) == 2
        assert cpt_widget.viewer.layers[data_layer_index].name == IMAGE_LAYER_NAME
        assert cpt_widget.viewer.layers[data_layer_index].ndim == 4
        assert cpt_widget.viewer.dims.current_step[1] == 0

        # Add a new particle and check the event_number is recorded correctly
        # Move to event 1
        cpt_widget.viewer.dims.set_current_step(1, 1)
        # Add a new particle
        cpt_widget.particle_decays_menu.setCurrentIndex(1)
        assert cpt_widget.table.rowCount() == 1
        assert cpt_widget.data[0].event_number == 1, "The event number should be 1"
        assert (
            cpt_widget.table.item(
                0, cpt_widget._get_table_column_index("event_number")
            ).text()
            == "1"
        )

        # Check that apply_magnification does not show anything in the table
        cpt_widget._on_click_apply_magnification()
        assert not cpt_widget.table.item(
            0, cpt_widget._get_table_column_index("radius_cm")
        ), "The calibrated radius should not be shown in the table"
        assert not cpt_widget.table.item(
            0, cpt_widget._get_table_column_index("decay_length_cm")
        ), "The calibrated radius should not be shown in the table"

    else:
        # def capture_msgbox():
        #    for widget in QApplication.topLevelWidgets():
        #        # top level, all widgets didn't work, active popup didn't
        #        if isinstance(widget, QMessageBox):
        #            return widget
        #    return None

        # qtbot.waitUntil(capture_msgbox, timeout=1000)
        # msgbox = capture_msgbox()
        msgbox = cpt_widget.msg
        # msgbox = QApplication.activeWindow()
        assert isinstance(msgbox, QMessageBox)
        assert msgbox.icon() == QMessageBox.Warning
        assert msgbox.text() == (
            "The data folder must contain three subfolders, one for each view, and each subfolder must contain the same number (>1) of images."
        )


def test_show_hide_buttons(cpt_widget: ParticleTracksWidget):
    """Delete becomes usable once a process is selected; Decay Angles only for the Λ⁰ process
    (the one with decay angles to measure)."""
    cpt_widget.viewer.add_image(np.random.random((100, 100)), name=IMAGE_LAYER_NAME)
    assert cpt_widget.particle_decays_menu.isEnabled() is True
    assert cpt_widget.delete_process.isEnabled() is False
    assert cpt_widget.decay_angles_nav_button.isEnabled() is False

    cpt_widget.particle_decays_menu.setCurrentIndex(1)  # Σ⁺ ⇨ p + π⁰, newly added and selected
    assert cpt_widget.delete_process.isEnabled() is True
    assert cpt_widget.decay_angles_nav_button.isEnabled() is False

    cpt_widget.particle_decays_menu.setCurrentIndex(4)  # Λ⁰ ⇨ p + π⁻, newly added and selected
    assert cpt_widget.delete_process.isEnabled() is True
    assert cpt_widget.decay_angles_nav_button.isEnabled() is True


def _close_napari_window(cpt_widget, monkeypatch, answer):
    """Close napari's main window (what InterceptClose watches), answering any unsaved-data
    prompt with `answer`. Returns (whether the close went ahead, the buttons of each prompt
    shown). The answer is patched in just for this close - see test_delete_particle_ui."""
    prompts = []

    def answer_prompt(self):
        prompts.append(self.standardButtons())
        return answer

    with monkeypatch.context() as m:
        m.setattr(QMessageBox, "exec", answer_prompt)
        closed = cpt_widget.viewer.window._qt_window.close()
    return closed, prompts


def test_close_with_nothing_unsaved_does_not_ask(cpt_widget, monkeypatch):
    closed, prompts = _close_napari_window(cpt_widget, monkeypatch, QMessageBox.Cancel)
    assert closed
    assert prompts == []


@pytest.mark.parametrize(
    "answer, expect_closed", [(QMessageBox.Cancel, False), (QMessageBox.Discard, True)]
)
def test_close_with_unsaved_data_asks_first(cpt_widget, monkeypatch, answer, expect_closed):
    """Unsaved data means closing asks Discard/Cancel first; Cancel keeps the window open.
    Deliberately doesn't check the prompt's wording, only that it's offered and obeyed."""
    cpt_widget.particle_decays_menu.setCurrentIndex(1)  # an unsaved process
    closed, prompts = _close_napari_window(cpt_widget, monkeypatch, answer)
    assert prompts == [QMessageBox.Discard | QMessageBox.Cancel]
    assert closed is expect_closed
