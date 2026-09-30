"""Coverage for the Record/Clear measurement actions that replaced the old "2 selected points
means a length, 3 means a radius" auto-trigger: selection and action are now separate steps, and
which action applies is chosen explicitly (via the right-click menu or an O/D/R shortcut), not
inferred from how many points happen to be selected.
"""


def _pin_to_first_view_and_event(cpt_widget):
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    return cpt_widget.viewer.dims.current_step[0]


def _make_row_with_points(cpt_widget, points):
    """Adds a process row, a real measurement layer, and `points` (each a plain [y, x] pair) on
    the current view/event slice - mirroring how points actually land on the canvas."""
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)
    cpt_widget.layer_measurements.add([[current_view, 0] + list(p) for p in points])
    view_data = cpt_widget.data[0].views[current_view]
    return view_data


def test_record_origin_vertex_needs_exactly_one_point_selected(cpt_widget):
    view_data = _make_row_with_points(cpt_widget, [[1.0, 2.0], [3.0, 4.0]])

    cpt_widget.layer_measurements.selected_data = {0, 1}  # two points - wrong count
    cpt_widget._record_origin_vertex()
    assert view_data.origin is None

    cpt_widget.layer_measurements.selected_data = {0}
    cpt_widget._record_origin_vertex()
    assert view_data.origin == [1.0, 2.0]


def test_record_decay_vertex(cpt_widget):
    view_data = _make_row_with_points(cpt_widget, [[5.0, 6.0]])

    cpt_widget.layer_measurements.selected_data = {0}
    cpt_widget._record_decay_vertex()
    assert view_data.decay == [5.0, 6.0]


def test_record_radius_needs_exactly_three_points_selected(cpt_widget):
    view_data = _make_row_with_points(cpt_widget, [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])

    cpt_widget.layer_measurements.selected_data = {0, 1}  # only two - wrong count
    cpt_widget._record_radius()
    assert view_data.track_points == []

    cpt_widget.layer_measurements.selected_data = {0, 1, 2}
    cpt_widget._record_radius()
    assert sorted(map(tuple, view_data.track_points)) == [(0.0, -1.0), (0.0, 1.0), (1.0, 0.0)]
    assert view_data.radius_px is not None


def test_no_row_selected_leaves_state_untouched(cpt_widget):
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    _pin_to_first_view_and_event(cpt_widget)
    # No particle_decays_menu selection made - nothing in the process table.
    cpt_widget._record_origin_vertex()  # must not raise
    cpt_widget._clear_origin_vertex()  # must not raise
    assert cpt_widget.data == []


def test_clearing_a_vertex_does_not_delete_the_canvas_point(cpt_widget):
    """Clearing a role only un-labels the point - the point itself stays on the canvas,
    reusable for another role later. This is the whole point of separating selection from
    action: Clear must never behave like the 'x' deletion tool.
    """
    view_data = _make_row_with_points(cpt_widget, [[1.0, 2.0]])
    cpt_widget.layer_measurements.selected_data = {0}
    cpt_widget._record_origin_vertex()
    assert view_data.origin == [1.0, 2.0]

    cpt_widget._clear_origin_vertex()
    assert view_data.origin is None
    assert len(cpt_widget.layer_measurements.data) == 1  # point still there, just un-labelled


def test_clearing_an_unset_vertex_is_a_harmless_no_op(cpt_widget):
    view_data = _make_row_with_points(cpt_widget, [[1.0, 2.0]])
    cpt_widget._clear_origin_vertex()  # nothing recorded yet - must not raise
    assert view_data.origin is None


def test_reusing_the_decay_vertex_as_a_radius_point_collapses_to_one_canvas_point(cpt_widget):
    """The scenario the whole redesign is about: record a decay vertex, then reuse that same
    canvas point as one of the three radius points. Must render as a single shared point, not
    two coincident ones (see also test_measurement_role_dedup.py).
    """
    view_data = _make_row_with_points(
        cpt_widget, [[3.0, 4.0], [7.0, 8.0], [9.0, 10.0]]
    )
    cpt_widget.layer_measurements.selected_data = {0}
    cpt_widget._record_decay_vertex()
    assert view_data.decay == [3.0, 4.0]

    cpt_widget.layer_measurements.selected_data = {0, 1, 2}
    cpt_widget._record_radius()
    assert view_data.track_points[0] == [3.0, 4.0]

    # decay + track0 share a position, so only 3 canvas points should exist, not 3 + 1 extra.
    assert len(cpt_widget.layer_measurements.data) == 3
    assert sorted(cpt_widget._measurement_role_index_map[
        next(idx for idx, roles in cpt_widget._measurement_role_index_map.items()
             if sorted(roles) == ["decay", "track0"])
    ]) == ["decay", "track0"]


def _table_cell(cpt_widget, column):
    item = cpt_widget.table.item(0, cpt_widget._get_table_column_index(column))
    return "" if item is None else item.text()


def test_record_and_clear_update_the_table_cells(cpt_widget):
    """The numbers shown in the process table follow Record/Clear, not just the stored data."""
    # three points on a circle of radius 5, then an origin/decay pair 5 apart (a 3-4-5 triangle)
    _make_row_with_points(
        cpt_widget, [[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0], [10.0, 10.0], [13.0, 14.0]]
    )

    cpt_widget.layer_measurements.selected_data = {0, 1, 2}
    cpt_widget._record_radius()
    assert _table_cell(cpt_widget, "radius_px") == "5.0"

    cpt_widget.layer_measurements.selected_data = {3}
    cpt_widget._record_origin_vertex()
    assert _table_cell(cpt_widget, "decay_length_px") == ""  # no decay vertex yet

    cpt_widget.layer_measurements.selected_data = {4}
    cpt_widget._record_decay_vertex()
    assert _table_cell(cpt_widget, "decay_length_px") == "5.0"

    cpt_widget._clear_radius()
    cpt_widget._clear_decay_vertex()
    assert _table_cell(cpt_widget, "radius_px") == ""
    assert _table_cell(cpt_widget, "decay_length_px") == ""
