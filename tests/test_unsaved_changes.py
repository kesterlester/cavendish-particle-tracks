"""The unsaved-changes check behind the quit (and load) prompt: dirty_things() names exactly the
parts of the session whose saved form differs from what was last saved or loaded - nothing
spurious, nothing missed, and no crash however many fiducials have been stamped."""
from __future__ import annotations

from pathlib import Path

import pytest
from qtpy.QtWidgets import QFileDialog, QMessageBox

from cavendish_particle_tracks._main_widget import IMAGE_LAYER_NAME, ParticleTracksWidget

from .test_widget import _close_napari_window, _make_image_folder


@pytest.fixture
def session(cpt_widget: ParticleTracksWidget, tmp_path: Path) -> ParticleTracksWidget:
    """A widget with 3 views x 3 events of images loaded - where every session starts."""
    cpt_widget._load_data_from(_make_image_folder(tmp_path, ["view1", "view2", "view3"], [3, 3, 3]))
    assert IMAGE_LAYER_NAME in cpt_widget.viewer.layers
    return cpt_widget


def _save(widget, monkeypatch, path):
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), "CSV files (*.csv)"))
    widget._on_click_save()


def _load(widget, monkeypatch, path):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), "CSV files (*.csv)"))
    widget._on_click_load()


def _go_to(widget, view, event):
    step = list(widget.viewer.dims.current_step)
    step[0], step[1] = view, event
    widget.viewer.dims.current_step = tuple(step)


def _generic_row(widget, view, slot):
    cm = widget.calibration_manager
    layer = cm.generic_calibration_layers()[0]
    return layer, cm._generic_layer_row_index(view, slot, 0, cm._num_events_on_generic_layer(layer))


def _name_fiducial(widget, view=0, slot=0, name="C'", kind="front"):
    """What the right-click menu does to name a generic fiducial."""
    _, row = _generic_row(widget, view, slot)
    widget.calibration_manager.rename_point(row, name, kind)


def _drag_fiducial(widget, view=0, slot=0, dy=5.0):
    layer, row = _generic_row(widget, view, slot)
    data = layer.data.copy()
    data[row][2] += dy
    layer.data = data


def _stamp(widget, event, view=0, slot=0):
    """What the right-click menu does to stamp a (named) generic fiducial into one image."""
    _go_to(widget, view, event)
    layer, row = _generic_row(widget, view, slot)
    widget.calibration_manager.clone_only_this_fid_view_into_event(row, layer.properties["labels"][row], layer)


def test_nothing_done_means_nothing_unsaved(session):
    """Looking around - other views and events, zooming - changes nothing that gets saved."""
    assert session.dirty_things() == []
    _go_to(session, 1, 2)
    _go_to(session, 2, 1)
    session.viewer.camera.zoom *= 2
    assert session.dirty_things() == []


def test_a_new_process_is_unsaved_until_saved(session, monkeypatch, tmp_path):
    session.particle_decays_menu.setCurrentIndex(1)
    assert session.dirty_things() == ["decay table"]
    _save(session, monkeypatch, tmp_path / "s.csv")
    assert session.dirty_things() == []


def test_sorting_the_table_is_not_a_change(session, monkeypatch, tmp_path):
    session.particle_decays_menu.setCurrentIndex(1)
    session.particle_decays_menu.setCurrentIndex(4)
    _save(session, monkeypatch, tmp_path / "s.csv")
    session._on_table_header_clicked(1)
    session._on_table_header_clicked(1)
    assert session.dirty_things() == []


def test_a_moved_fiducial_is_saved_by_saving(session, monkeypatch, tmp_path):
    """The reported bug: once a generic fiducial had moved, the quit prompt claimed unsaved
    calibrations even straight after saving them."""
    _name_fiducial(session)
    _save(session, monkeypatch, tmp_path / "s.csv")
    assert session.dirty_things() == []
    _drag_fiducial(session)
    assert session.dirty_things() == ["calibrations"]
    _save(session, monkeypatch, tmp_path / "s.csv")
    assert session.dirty_things() == []


def test_moving_an_unnamed_fiducial_is_not_a_change(session):
    """Unnamed generic fiducials aren't saved at all, so moving one leaves nothing unsaved."""
    _drag_fiducial(session)
    assert session.dirty_things() == []


def test_naming_a_fiducial_is_a_change(session):
    """The old check compared positions only, so a rename could be lost without a prompt."""
    _name_fiducial(session)
    assert session.dirty_things() == ["calibrations"]


def test_stamps_in_several_events_are_unsaved_until_saved(session, monkeypatch, tmp_path):
    """The old check crashed once stamps changed the per-image layer's size by more than one."""
    _name_fiducial(session)
    _save(session, monkeypatch, tmp_path / "s.csv")
    for event in range(3):
        _stamp(session, event)
    assert session.dirty_things() == ["decay table"]
    _save(session, monkeypatch, tmp_path / "s.csv")
    assert session.dirty_things() == []


def test_a_loaded_session_has_nothing_unsaved(session, monkeypatch, tmp_path):
    _name_fiducial(session)
    _stamp(session, 0)
    _stamp(session, 2)
    session.particle_decays_menu.setCurrentIndex(4)
    _save(session, monkeypatch, tmp_path / "s.csv")
    session.particle_decays_menu.setCurrentIndex(1)  # unsaved, so Load asks first
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.Yes)
    _load(session, monkeypatch, tmp_path / "s.csv")
    assert session.dirty_things() == []
    _go_to(session, 1, 1)
    assert session.dirty_things() == []


def test_quitting_after_unsaved_stamps_asks_first(session, monkeypatch):
    """Where the old crash bit: the quit check itself. It must ask, not crash or skip asking."""
    _name_fiducial(session)
    _stamp(session, 0)
    _stamp(session, 1)
    closed, prompts = _close_napari_window(session, monkeypatch, QMessageBox.Cancel)
    assert prompts == [QMessageBox.Discard | QMessageBox.Cancel]
    assert not closed
