"""Regression coverage for _clear_shapes_layer, the helper _do_refresh_radius_arc /
_do_refresh_origin_decay_arrow use to clear their Shapes layer before redrawing it.

Found live via a full traceback: a rectangle-select drag on the (completely separate)
measurement points layer fires a highlight event - and so a radius-arc/arrow refresh - on
every mouse-move tick, fast enough to sometimes race napari's own per-layer slice bookkeeping
for the arc/arrow Shapes layer. When that happens, `Shapes.displayed_index` for that layer is
briefly empty even though the layer still holds shapes, and the napari-recommended
`layer.selected_data = set(range(len(layer.data)))` idiom for "select everything, ready to
remove it" crashes with `ValueError: zero-size array to reduction operation minimum which has
no identity` while computing an on-screen interaction-box overlay we never wanted in the first
place (the layer is editable=False - nothing ever shows that overlay for it).

Reproduced directly here by corrupting a Shapes layer's own `_data_view.slice_key` to a value
that matches none of its shapes - the same "nothing is currently in-slice" state the real race
transiently produces - without needing to simulate an actual mouse drag.
"""

import numpy as np
import pytest

from cavendish_particle_tracks._main_widget import RADIUS_ARC_LAYER_NAME


def _make_desynced_shapes_layer(cpt_widget):
    """A Shapes layer with 2+ shapes whose slice bookkeeping has been forced out of sync with
    its own data - reproducing the race's end state. Needs at least 2 shapes: interaction_box's
    crash-prone create_box() path is only reached for len(index) > 1; a single shape takes a
    different, unaffected code path (see Shapes.interaction_box).
    """
    if not hasattr(cpt_widget, "layer_measurements"):
        cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    layer = cpt_widget._setup_radius_arc_layer()
    shape_a = np.array([[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 10, 10]])
    shape_b = np.array([[0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 30, 10]])
    layer.add(shape_a, shape_type="path")
    layer.add(shape_b, shape_type="path")
    # Desync: no shape's own (view, event) matches this slice_key, so displayed_index is empty.
    layer._data_view.slice_key = np.array([999, 999])
    assert len(layer._data_view.displayed_index) == 0
    return layer


def test_napari_selected_data_setter_really_does_crash_when_desynced(cpt_widget):
    """Documents the underlying napari bug this helper works around - if napari ever fixes
    this, this test (not the production code) is what should start failing."""
    layer = _make_desynced_shapes_layer(cpt_widget)
    with pytest.raises(ValueError, match="zero-size array"):
        layer.selected_data = set(range(len(layer.data)))


def test_clear_shapes_layer_survives_a_desynced_slice(cpt_widget):
    layer = _make_desynced_shapes_layer(cpt_widget)
    cpt_widget._clear_shapes_layer(layer)
    assert len(layer.data) == 0


def test_clear_shapes_layer_is_a_no_op_on_an_already_empty_layer(cpt_widget):
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    layer = cpt_widget._setup_radius_arc_layer()
    assert len(layer.data) == 0
    cpt_widget._clear_shapes_layer(layer)  # must not raise
    assert len(layer.data) == 0


def test_refresh_radius_arc_survives_a_desynced_slice(cpt_widget):
    """End-to-end: the real refresh path (not just the helper in isolation) also survives."""
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    layer = _make_desynced_shapes_layer(cpt_widget)
    assert layer is cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME]

    cpt_widget._refresh_radius_arc()  # must not raise


def test_refresh_radius_arc_with_no_selected_process_survives_a_desynced_slice(cpt_widget):
    """The actual state the crash was hit in live: after changing event (or on a fresh load),
    NO process is selected in the table - there is nothing to auto-select it to - while several
    processes still exist and have full radius fits for the currently displayed event. With
    "Show all processes" ticked, _decoratable_process_views must draw all of them (none marked
    as the bold "selected" one - see its own docstring), which is exactly the >= 2-shape
    situation the desynced-slice race hits.
    """
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_radius_arc_layer()
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    current_view, current_event = cpt_widget.viewer.dims.current_step[:2]

    for _ in range(3):
        cpt_widget.particle_decays_menu.setCurrentIndex(1)
    for i in range(3):
        cpt_widget.data[i].event_number = current_event
        base = i * 100.0
        cpt_widget.data[i].views[current_view].set_track_points(
            [[base, 1.0], [base + 1.0, 0.0], [base, -1.0]]
        )
    cpt_widget.table.clearSelection()
    assert cpt_widget._get_selected_measurement_row() is None

    cpt_widget.show_all_processes_checkbox.setChecked(True)
    layer = cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME]
    layer._data_view.slice_key = np.array([999, 999])

    cpt_widget._refresh_radius_arc()  # must not raise
    assert len(layer.data) == 3
