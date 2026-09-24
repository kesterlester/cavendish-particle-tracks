"""Regression coverage for a real, serious bug found live via the debug trace log: a delete
followed by an unrelated add elsewhere could restore the measurement layer's point COUNT to its
pre-delete value while every index underneath had shifted - defeating a count-based "has anything
changed" check and letting _propagate_measurement_point_drag apply a stale index->role mapping.
Live, this merged two different roles into one identical coordinate, corrupting a radius fit into
a degenerate (2 identical points) triangle and crashing the whole app with a numpy
singular-matrix error when it tried to fit a circle through it.
"""


class _FakeDataEvent:
    def __init__(self, action, data_indices=()):
        self.action = action
        self.data_indices = data_indices


def _pin_to_first_view_and_event(cpt_widget):
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    current_view = cpt_widget.viewer.dims.current_step[0]
    if cpt_widget.data:
        cpt_widget.data[0].event_number = 0
    return current_view


def test_delete_then_unrelated_add_does_not_corrupt_a_radius_fit(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([1.0, 2.0])
    view_data.set_decay([3.0, 4.0])
    view_data.set_track_points([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    cpt_widget._sync_measurement_layer_to_selected_process()
    assert len(cpt_widget.layer_measurements.data) == 5
    assert cpt_widget._measurement_role_index_map_valid is True

    # Delete the decay point with napari's own delete tool (mirrors the live repro).
    decay_idx = next(
        i for i, p in enumerate(cpt_widget.layer_measurements.data)
        if list(p[2:]) == [3.0, 4.0]
    )
    cpt_widget.layer_measurements.selected_data = {decay_idx}
    cpt_widget.layer_measurements.remove_selected()

    assert view_data.decay is None  # reconciliation already handled this part correctly
    assert cpt_widget._measurement_role_index_map_valid is False
    assert len(cpt_widget.layer_measurements.data) == 4

    # An unrelated bare point gets added elsewhere, restoring the count to the pre-delete value -
    # this is exactly what defeated the old count-based check.
    cpt_widget.layer_measurements.add([[current_view, 0, 99.0, 99.0]])
    assert len(cpt_widget.layer_measurements.data) == 5
    assert cpt_widget._measurement_role_index_map_valid is False  # must still be invalid

    # Now simulate a drag on whatever sits at the map's stale "track0" index.
    origin_before = list(view_data.origin)
    track_points_before = [list(p) for p in view_data.track_points]
    stale_track0_idx = next(idx for idx, roles in cpt_widget._measurement_role_index_map.items() if "track0" in roles)
    cpt_widget.layer_measurements.data[stale_track0_idx][2] = 123.0
    cpt_widget.layer_measurements.data[stale_track0_idx][3] = 456.0
    cpt_widget._propagate_measurement_point_drag(_FakeDataEvent("changed", [stale_track0_idx]))

    # While invalid, propagate must do NOTHING - not even a partial, individually-plausible-looking
    # mutation using the stale map. Exact equality, not just "no duplicates": a stale map applied
    # to shifted indices can just as easily assign a wrong-but-distinct value as a duplicate one.
    assert view_data.origin == origin_before
    assert view_data.decay is None
    assert [list(p) for p in view_data.track_points] == track_points_before
    assert len(view_data.track_points) == 3
    assert len(set(tuple(p) for p in view_data.track_points)) == 3, "duplicate track points - degenerate triangle"
    assert view_data.radius_px is not None  # would be None or raise if the fit were degenerate

    # A real resync (e.g. the user switches row/view, or the deletion path also usually triggers
    # one) must restore normal operation.
    cpt_widget._sync_measurement_layer_to_selected_process()
    assert cpt_widget._measurement_role_index_map_valid is True
