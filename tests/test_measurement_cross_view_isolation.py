"""Regression coverage for a serious real bug found live: a point recorded in one view could
leak into a completely different view's data after switching the View slider and back. Points
must never be visible - or, worse, silently WRITTEN - into the wrong (view, event) slice, no
matter what triggered the switch (the slider, or selecting a different process row).
"""


def _select_process_and_setup(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.data[0].event_number = 0
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget.viewer.dims.set_current_step(1, 0)


def test_recording_a_vertex_in_one_view_never_leaks_into_another_views_data(cpt_widget):
    """Reproduces the exact live sequence: bare points placed in view 1, then view 2, a vertex
    recorded from one of view 2's points, then switching back to view 1.
    """
    _select_process_and_setup(cpt_widget)

    cpt_widget.viewer.dims.set_current_step(0, 1)
    cpt_widget.layer_measurements.add([[1, 0, 10.0, 10.0], [1, 0, 20.0, 20.0]])
    cpt_widget._sync_measurement_layer_to_selected_process()

    cpt_widget.viewer.dims.set_current_step(0, 2)
    cpt_widget.layer_measurements.add([[2, 0, 30.0, 30.0], [2, 0, 40.0, 40.0]])
    cpt_widget._sync_measurement_layer_to_selected_process()

    idx = next(i for i, p in enumerate(cpt_widget.layer_measurements.data) if p[0] == 2 and p[2] == 30.0)
    cpt_widget.layer_measurements.selected_data = {idx}
    cpt_widget._record_origin_vertex()
    assert cpt_widget.data[0].views[2].origin == [30.0, 30.0]

    # Views 0 and 1 must be completely untouched - this is what actually leaked live.
    assert cpt_widget.data[0].views[0].origin is None
    assert cpt_widget.data[0].views[1].origin is None

    cpt_widget.viewer.dims.set_current_step(0, 1)
    cpt_widget._sync_measurement_layer_to_selected_process()

    assert cpt_widget.data[0].views[1].origin is None
    assert cpt_widget.data[0].views[0].origin is None
    # Only view 1's own 2 orphans should be visible on this slice - not view 2's recorded point.
    on_screen_view1 = [p for p in cpt_widget.layer_measurements.data if p[0] == 1]
    assert sorted(tuple(p[2:]) for p in on_screen_view1) == [(10.0, 10.0), (20.0, 20.0)]


def test_a_stale_dims_calls_propagate_but_it_backs_off(cpt_widget):
    """Directly exercises the guard: calling _propagate_measurement_point_drag while
    _last_synced_dims doesn't match the current dims step (simulating a highlight event firing
    before the dims-triggered rebuild has run) must be a complete no-op, never touching any
    ViewData at all.
    """
    _select_process_and_setup(cpt_widget)
    cpt_widget.viewer.dims.set_current_step(0, 1)
    cpt_widget.layer_measurements.add([[1, 0, 10.0, 10.0]])
    cpt_widget._sync_measurement_layer_to_selected_process()

    view1_data = cpt_widget.data[0].views[1]
    view2_data = cpt_widget.data[0].views[2]
    assert view1_data.origin is None and view2_data.origin is None

    # Simulate "dims has already moved to view 2, but the rebuild for it hasn't run yet": in the
    # test fixture, _sync_measurement_layer_to_selected_process is itself connected to the same
    # dims.events.current_step signal, so it runs synchronously and there's no real gap to land
    # in here - so switch normally, then artificially roll _last_synced_dims back to simulate the
    # race window the live bug actually hit (a DIFFERENT signal, layer.events.highlight, firing
    # before this same-signal rebuild gets its turn).
    cpt_widget.viewer.dims.set_current_step(0, 2)
    assert cpt_widget._last_synced_dims == (2, 0)  # confirms the real rebuild did run
    cpt_widget._last_synced_dims = (1, 0)

    class _FakeEvent:
        action = None  # a highlight event

    cpt_widget._propagate_measurement_point_drag(_FakeEvent())

    assert view1_data.origin is None
    assert view2_data.origin is None
