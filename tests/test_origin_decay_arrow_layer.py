"""Coverage for the origin->decay arrow overlay: a green arrow showing the two points are
linked as one length measurement, updated live wherever the points themselves get restyled.
"""

import numpy as np

from cavendish_particle_tracks._calculate import origin_decay_arrow
from cavendish_particle_tracks._main_widget import ORIGIN_DECAY_ARROW_LAYER_NAME


def _pin_to_first_view_and_event(cpt_widget):
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    current_view = cpt_widget.viewer.dims.current_step[0]
    if cpt_widget.data:
        cpt_widget.data[0].event_number = 0
    return current_view


def test_no_arrow_when_nothing_selected(cpt_widget):
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_origin_decay_arrow_layer()
    cpt_widget._refresh_origin_decay_arrow()
    assert len(cpt_widget.viewer.layers[ORIGIN_DECAY_ARROW_LAYER_NAME].data) == 0


def test_no_arrow_with_only_an_origin_recorded(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_origin_decay_arrow_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([1.0, 2.0])
    cpt_widget._refresh_origin_decay_arrow()

    assert len(cpt_widget.viewer.layers[ORIGIN_DECAY_ARROW_LAYER_NAME].data) == 0


def test_arrow_appears_once_both_origin_and_decay_are_recorded(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_origin_decay_arrow_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([0.0, 0.0])
    view_data.set_decay([10.0, 0.0])
    cpt_widget._refresh_origin_decay_arrow()

    arrow_layer = cpt_widget.viewer.layers[ORIGIN_DECAY_ARROW_LAYER_NAME]
    assert len(arrow_layer.data) == 2  # shaft + arrowhead
    assert list(arrow_layer.shape_type) == ["line", "polygon"]

    expected_shaft, expected_head = origin_decay_arrow([0.0, 0.0], [10.0, 0.0])
    got_shaft = [list(p[2:]) for p in arrow_layer.data[0]]
    got_head = [list(p[2:]) for p in arrow_layer.data[1]]
    assert np.allclose(got_shaft, expected_shaft, atol=1e-3)
    assert np.allclose(got_head, expected_head, atol=1e-3)


def test_arrow_clears_when_decay_is_cleared(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_origin_decay_arrow_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([0.0, 0.0])
    view_data.set_decay([10.0, 0.0])
    cpt_widget._refresh_origin_decay_arrow()
    assert len(cpt_widget.viewer.layers[ORIGIN_DECAY_ARROW_LAYER_NAME].data) == 2

    view_data.set_decay(None)
    cpt_widget._refresh_origin_decay_arrow()
    assert len(cpt_widget.viewer.layers[ORIGIN_DECAY_ARROW_LAYER_NAME].data) == 0

    # Clearing again (already empty) must not raise.
    cpt_widget._refresh_origin_decay_arrow()
    assert len(cpt_widget.viewer.layers[ORIGIN_DECAY_ARROW_LAYER_NAME].data) == 0


def test_restyle_measurement_points_keeps_the_arrow_in_sync(cpt_widget):
    """_restyle_measurement_points is the method that actually runs on every relevant event
    (selection, drag, Record/Clear) - this checks the arrow really is wired into it, not just
    directly callable on its own."""
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_origin_decay_arrow_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([0.0, 0.0])
    view_data.set_decay([10.0, 0.0])
    cpt_widget._sync_measurement_layer_to_selected_process()
    cpt_widget._restyle_measurement_points()

    assert len(cpt_widget.viewer.layers[ORIGIN_DECAY_ARROW_LAYER_NAME].data) == 2


def test_sync_alone_does_not_duplicate_the_arrow_via_reentrant_restyle(cpt_widget):
    """Regression test for a real bug found live via a debug trace: mutating the arc/arrow
    Shapes layers' own .data (remove_selected()/add()) from within _restyle_measurement_points
    could trigger a callback chain reentering that same method before the outer call finished,
    running a second, interleaved clear-then-add cycle on top of the first - a plain
    _sync_measurement_layer_to_selected_process() call alone (no explicit restyle call needed)
    was enough to leave 4 arrow shapes instead of 2.
    """
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_origin_decay_arrow_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([0.0, 0.0])
    view_data.set_decay([10.0, 0.0])
    cpt_widget._sync_measurement_layer_to_selected_process()

    assert len(cpt_widget.viewer.layers[ORIGIN_DECAY_ARROW_LAYER_NAME].data) == 2
