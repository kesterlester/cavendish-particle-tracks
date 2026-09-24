"""Coverage for the radius-arc overlay: a visual line through a radius fit's 3 points, updated
live wherever the points themselves get restyled.
"""

import numpy as np

from cavendish_particle_tracks._calculate import radius_arc_points
from cavendish_particle_tracks._main_widget import RADIUS_ARC_LAYER_NAME


def _pin_to_first_view_and_event(cpt_widget):
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    current_view, current_event = cpt_widget.viewer.dims.current_step[:2]
    # A process row created via particle_decays_menu (not real image loading) defaults
    # to event_number=-1 - _restyle_measurement_points/_refresh_radius_arc only show
    # anything for a row whose OWN event matches what's on screen, so pin it to match.
    if cpt_widget.data:
        cpt_widget.data[0].event_number = current_event
    return current_view


def test_no_arc_when_nothing_selected(cpt_widget):
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_radius_arc_layer()
    cpt_widget._refresh_radius_arc()
    assert len(cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME].data) == 0


def test_no_arc_with_fewer_than_three_track_points(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_radius_arc_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_track_points([[0.0, 1.0], [1.0, 0.0]])
    cpt_widget._refresh_radius_arc()

    assert len(cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME].data) == 0


def test_arc_matches_radius_arc_points_for_the_selected_row(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_radius_arc_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    track_points = [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]]
    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_track_points(track_points)
    cpt_widget._refresh_radius_arc()

    arc_layer = cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME]
    assert len(arc_layer.data) == 1
    expected_2d = radius_arc_points(*track_points)
    got_2d = [list(p[2:]) for p in arc_layer.data[0]]
    assert np.allclose(got_2d, expected_2d, atol=1e-3)  # shapes layer stores as float32
    # view/event coordinates carried through correctly too
    assert all(p[0] == current_view and p[1] == 0 for p in arc_layer.data[0])
    # Regression check: a Shapes layer's `.data = [...]` setter silently defaults a new shape's
    # type to "polygon" (closed - draws an extra edge straight back to the start point) rather
    # than "path" (open), regardless of what shape_type the layer was created with. Caught live:
    # an unwanted straight line appeared connecting the arc's two endpoints.
    assert arc_layer.shape_type == ["path"]


def test_arc_updates_when_track_points_change(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_radius_arc_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_track_points([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    cpt_widget._refresh_radius_arc()
    first_arc = cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME].data[0].copy()

    view_data.set_track_points([[0.0, 2.0], [2.0, 0.0], [0.0, -2.0]])  # bigger circle
    cpt_widget._refresh_radius_arc()
    second_arc = cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME].data[0]

    assert not np.allclose(first_arc, second_arc)


def test_arc_clears_when_radius_is_cleared(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_radius_arc_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_track_points([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    cpt_widget._refresh_radius_arc()
    assert len(cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME].data) == 1

    view_data.set_track_points([])
    cpt_widget._refresh_radius_arc()
    assert len(cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME].data) == 0

    # And clearing again (already empty) must not raise.
    cpt_widget._refresh_radius_arc()
    assert len(cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME].data) == 0


def test_restyle_measurement_points_keeps_the_arc_in_sync(cpt_widget):
    """_restyle_measurement_points is the method that actually runs on every relevant event
    (selection, drag, Record/Clear) - this checks the arc really is wired into it, not just
    directly callable on its own.
    """
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_radius_arc_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_track_points([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    # _restyle_measurement_points colours points already on the canvas - it bails out early if
    # there are none, so render them first, same as the real Record-radius flow does.
    cpt_widget._sync_measurement_layer_to_selected_process()
    cpt_widget._restyle_measurement_points()

    assert len(cpt_widget.viewer.layers[RADIUS_ARC_LAYER_NAME].data) == 1
