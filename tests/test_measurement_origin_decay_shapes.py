"""Coverage for the origin/decay vertex shape distinction (_role_symbol): shape must be able to
identify an origin vertex on its own (a process with no decay recorded yet still needs to read as
"this is the origin"), distinguish it from a decay vertex, and stay out of the way of the
existing colour scheme for length/radius membership.
"""

from cavendish_particle_tracks._main_widget import (
    OTHER_PROCESSES_LAYER_NAME,
    _role_symbol,
)


def _pin_to_first_view_and_event(cpt_widget):
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    current_view = cpt_widget.viewer.dims.current_step[0]
    if cpt_widget.data:
        cpt_widget.data[0].event_number = 0
    return current_view


def test_role_symbol_mapping():
    assert _role_symbol(is_origin=True, is_decay=False) == "ring"
    assert _role_symbol(is_origin=False, is_decay=True) == "triangle_up"
    assert _role_symbol(is_origin=True, is_decay=True) == "diamond"
    assert _role_symbol(is_origin=False, is_decay=False) == "disc"


def test_origin_alone_has_its_own_shape_with_no_decay_recorded(cpt_widget):
    """The whole point of shape over colour: an origin with no decay yet still has to read as
    "the origin", not just "an uncategorised point" (colour alone can't do this - LENGTH_COLOR
    only applies once you know it's a pair)."""
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([1.0, 2.0])
    cpt_widget._sync_measurement_layer_to_selected_process()
    cpt_widget._restyle_measurement_points()

    assert list(cpt_widget.layer_measurements.symbol) == ["ring"]


def test_origin_and_decay_get_distinct_shapes(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([1.0, 2.0])
    view_data.set_decay([3.0, 4.0])
    cpt_widget._sync_measurement_layer_to_selected_process()
    cpt_widget._restyle_measurement_points()

    positions = [tuple(p[2:]) for p in cpt_widget.layer_measurements.data]
    symbols = list(cpt_widget.layer_measurements.symbol)
    by_position = dict(zip(positions, symbols))
    assert by_position[(1.0, 2.0)] == "ring"
    assert by_position[(3.0, 4.0)] == "triangle_up"


def test_a_bare_orphan_and_a_radius_only_point_stay_the_default_disc(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_track_points([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    cpt_widget.layer_measurements.add([[current_view, 0, 50.0, 60.0]])  # bare orphan
    cpt_widget._sync_measurement_layer_to_selected_process()
    cpt_widget._restyle_measurement_points()

    assert all(s == "disc" for s in cpt_widget.layer_measurements.symbol)


def test_other_processes_overlay_also_shows_origin_decay_shapes(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget._setup_other_processes_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([1.0, 2.0])
    view_data.set_decay([3.0, 4.0])

    cpt_widget.show_all_processes_checkbox.setChecked(True)
    # This process is the only one and is selected, so it's excluded from the OTHER-processes
    # overlay by design - add a second process to actually exercise the overlay.
    cpt_widget.particle_decays_menu.setCurrentIndex(2)
    cpt_widget.data[1].event_number = 0
    cpt_widget._sync_other_processes_layer()

    overlay = cpt_widget.viewer.layers[OTHER_PROCESSES_LAYER_NAME]
    positions = [tuple(p[2:]) for p in overlay.data]
    by_position = dict(zip(positions, overlay.symbol))
    assert by_position[(1.0, 2.0)] == "ring"
    assert by_position[(3.0, 4.0)] == "triangle_up"
