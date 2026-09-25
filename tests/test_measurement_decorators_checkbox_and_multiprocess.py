"""Coverage for the two newest bits of decorator behaviour:

* show_decorators_checkbox ("Show arcs/arrows") - a global on/off for the radius arc and the
  origin-decay arrow, independent of the point-visibility checkboxes.
* _decoratable_process_views - once "Show all processes" is ticked, the arc/arrow decorators
  are drawn for every OTHER process too (not just the selected one), fainter than the selected
  process's own bold decorator, mirroring how _sync_other_processes_layer already dims their
  points.
"""

import numpy as np

from cavendish_particle_tracks._main_widget import (
    ARROW_COLOR,
    ORIGIN_DECAY_ARROW_LAYER_NAME,
    RADIUS_ARC_LAYER_NAME,
    RADIUS_COLOR,
    _faded_color,
)


def _pin_row_to_first_view_and_event(cpt_widget, row):
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    current_view, current_event = cpt_widget.viewer.dims.current_step[:2]
    cpt_widget.data[row].event_number = current_event
    return current_view, current_event


def _add_new_process(cpt_widget) -> int:
    """Selecting any non-header entry in particle_decays_menu APPENDS a brand new process row
    and auto-selects it, then resets the menu back to its header - it is not a "select existing
    process" control. Returns the new row's index."""
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    return len(cpt_widget.data) - 1


def _select_row(cpt_widget, row):
    """Re-select an already-existing row (as opposed to _add_new_process, which always creates a
    new one) via the table, the same way a user re-clicking an existing row would."""
    cpt_widget.table.selectRow(row)


def test_decorators_checkbox_hides_the_radius_arc(cpt_widget):
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_radius_arc_layer()
    _add_new_process(cpt_widget)
    current_view, _ = _pin_row_to_first_view_and_event(cpt_widget, 0)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_track_points([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    cpt_widget._refresh_radius_arc()
    assert len(cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME].data) == 1

    cpt_widget.show_decorators_checkbox.setChecked(False)
    cpt_widget._refresh_radius_arc()
    assert len(cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME].data) == 0

    cpt_widget.show_decorators_checkbox.setChecked(True)
    cpt_widget._refresh_radius_arc()
    assert len(cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME].data) == 1


def test_decorators_checkbox_hides_the_origin_decay_arrow(cpt_widget):
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_origin_decay_arrow_layer()
    _add_new_process(cpt_widget)
    current_view, _ = _pin_row_to_first_view_and_event(cpt_widget, 0)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([0.0, 0.0])
    view_data.set_decay([10.0, 0.0])
    cpt_widget._refresh_origin_decay_arrow()
    assert len(cpt_widget.viewer.layers[ORIGIN_DECAY_ARROW_LAYER_NAME].data) == 2

    cpt_widget.show_decorators_checkbox.setChecked(False)
    cpt_widget._refresh_origin_decay_arrow()
    assert len(cpt_widget.viewer.layers[ORIGIN_DECAY_ARROW_LAYER_NAME].data) == 0

    cpt_widget.show_decorators_checkbox.setChecked(True)
    cpt_widget._refresh_origin_decay_arrow()
    assert len(cpt_widget.viewer.layers[ORIGIN_DECAY_ARROW_LAYER_NAME].data) == 2


def test_other_processes_radius_arcs_are_drawn_faded_when_shown(cpt_widget):
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_radius_arc_layer()

    _add_new_process(cpt_widget)
    current_view, current_event = _pin_row_to_first_view_and_event(cpt_widget, 0)
    cpt_widget.data[0].views[current_view].set_track_points([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])

    # Add a second process with its own, different radius fit at the same (view, event).
    _add_new_process(cpt_widget)
    cpt_widget.data[1].event_number = current_event
    cpt_widget.data[1].views[current_view].set_track_points([[0.0, 2.0], [2.0, 0.0], [0.0, -2.0]])

    # Re-select process 0 so it's the "selected" (bold) one.
    _select_row(cpt_widget, 0)

    # "Show all processes" off: only the selected process's own arc appears.
    cpt_widget.show_all_processes_checkbox.setChecked(False)
    cpt_widget._refresh_radius_arc()
    arc_layer = cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME]
    assert len(arc_layer.data) == 1

    # "Show all processes" on: both arcs appear, selected bold and the other faded.
    cpt_widget.show_all_processes_checkbox.setChecked(True)
    cpt_widget._refresh_radius_arc()
    assert len(arc_layer.data) == 2
    edge_widths = list(arc_layer.edge_width)
    assert sorted(edge_widths) == [1.5, 4]

    bold_index = edge_widths.index(4)
    faded_index = edge_widths.index(1.5)
    assert np.allclose(arc_layer.edge_color[bold_index], np.array(_vispy_rgba(RADIUS_COLOR)), atol=1e-3)
    assert np.allclose(arc_layer.edge_color[faded_index], _faded_color(RADIUS_COLOR), atol=1e-3)


def test_other_processes_origin_decay_arrows_are_drawn_faded_when_shown(cpt_widget):
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_origin_decay_arrow_layer()

    _add_new_process(cpt_widget)
    current_view, current_event = _pin_row_to_first_view_and_event(cpt_widget, 0)
    cpt_widget.data[0].views[current_view].set_origin([0.0, 0.0])
    cpt_widget.data[0].views[current_view].set_decay([10.0, 0.0])

    _add_new_process(cpt_widget)
    cpt_widget.data[1].event_number = current_event
    cpt_widget.data[1].views[current_view].set_origin([0.0, 5.0])
    cpt_widget.data[1].views[current_view].set_decay([10.0, 5.0])

    _select_row(cpt_widget, 0)

    cpt_widget.show_all_processes_checkbox.setChecked(False)
    cpt_widget._refresh_origin_decay_arrow()
    arrow_layer = cpt_widget.viewer.layers[ORIGIN_DECAY_ARROW_LAYER_NAME]
    assert len(arrow_layer.data) == 2  # 1 process x (shaft + head)

    cpt_widget.show_all_processes_checkbox.setChecked(True)
    cpt_widget._refresh_origin_decay_arrow()
    assert len(arrow_layer.data) == 4  # 2 processes x (shaft + head)
    edge_widths = sorted(set(round(w, 3) for w in arrow_layer.edge_width))
    assert edge_widths == [1.5, 3]

    bold_indices = [i for i, w in enumerate(arrow_layer.edge_width) if round(w, 3) == 3]
    faded_indices = [i for i, w in enumerate(arrow_layer.edge_width) if round(w, 3) == 1.5]
    for i in bold_indices:
        assert np.allclose(arrow_layer.edge_color[i], np.array(_vispy_rgba(ARROW_COLOR)), atol=1e-3)
    for i in faded_indices:
        assert np.allclose(arrow_layer.edge_color[i], _faded_color(ARROW_COLOR), atol=1e-3)


def _vispy_rgba(name):
    from vispy.color import Color

    return Color(name).rgba
