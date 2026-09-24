"""Coverage for the origin/decay vertex visual distinction (_role_symbol/_role_face_color): a
process must be able to identify an origin vertex on its own (a process with no decay recorded
yet still needs to read as "this is the origin"), distinguish it from a decay vertex, and stay
out of the way of the existing colour scheme for length/radius membership.

Both origin and decay share the "ring" shape (a filled triangle was tried for decay first, but
found live to visually clash with the O-D arrow's own triangular tip - see _role_symbol's
docstring) - they're told apart by FILL colour instead (yellow for origin, blue for decay, via
_role_face_color).
"""

from cavendish_particle_tracks._main_widget import (
    OTHER_PROCESSES_LAYER_NAME,
    _role_face_color,
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
    assert _role_symbol(is_origin=False, is_decay=True) == "ring"
    assert _role_symbol(is_origin=True, is_decay=True) == "diamond"
    assert _role_symbol(is_origin=False, is_decay=False) == "disc"


def test_role_face_color_mapping():
    assert _role_face_color(is_origin=True, is_decay=False) == "yellow"
    assert _role_face_color(is_origin=False, is_decay=True) == "blue"
    assert _role_face_color(is_origin=True, is_decay=True) == "yellow"
    assert _role_face_color(is_origin=False, is_decay=False) == "white"


def test_origin_alone_has_its_own_shape_with_no_decay_recorded(cpt_widget):
    """The whole point of shape/colour over colour alone: an origin with no decay yet still has
    to read as "the origin", not just "an uncategorised point" (LENGTH_COLOR only applies once
    you know it's a pair)."""
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([1.0, 2.0])
    cpt_widget._sync_measurement_layer_to_selected_process()
    cpt_widget._restyle_measurement_points()

    assert list(cpt_widget.layer_measurements.symbol) == ["ring"]
    assert _face_color_name(cpt_widget.layer_measurements.face_color[0]) == "yellow"


def test_origin_and_decay_get_distinct_face_colors_but_the_same_shape(cpt_widget):
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
    face_colors = list(cpt_widget.layer_measurements.face_color)
    by_position_symbol = dict(zip(positions, symbols))
    by_position_color = dict(zip(positions, face_colors))

    assert by_position_symbol[(1.0, 2.0)] == "ring"
    assert by_position_symbol[(3.0, 4.0)] == "ring"
    assert _face_color_name(by_position_color[(1.0, 2.0)]) == "yellow"
    assert _face_color_name(by_position_color[(3.0, 4.0)]) == "blue"


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


def test_other_processes_overlay_also_shows_origin_decay_shapes_and_colors(cpt_widget):
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
    by_position_symbol = dict(zip(positions, overlay.symbol))
    by_position_color = dict(zip(positions, overlay.face_color))
    assert by_position_symbol[(1.0, 2.0)] == "ring"
    assert by_position_symbol[(3.0, 4.0)] == "ring"
    assert _face_color_name(by_position_color[(1.0, 2.0)]) == "yellow"
    assert _face_color_name(by_position_color[(3.0, 4.0)]) == "blue"


def _face_color_name(rgba) -> str:
    """Map an RGBA array back to the plain colour name asserted against below - napari's
    layer.face_color always returns numeric RGBA, never the original string."""
    r, g, b, a = [round(float(c), 3) for c in rgba]
    named = {
        (1.0, 1.0, 0.0, 1.0): "yellow",
        (0.0, 0.0, 1.0, 1.0): "blue",
        (1.0, 1.0, 1.0, 1.0): "white",
    }
    return named[(r, g, b, a)]
