"""Integration-level coverage for _propagate_measurement_point_drag: dragging a canvas point
that represents more than one role (e.g. a decay vertex reused as a radius-fit point) must move
every role it represents together, not just the one napari happens to report.
"""


class _FakeDataChangedEvent:
    def __init__(self, data_indices, action="changed"):
        self.action = action
        self.data_indices = data_indices


class _FakeHighlightEvent:
    """layer.events.highlight carries no .action attribute at all - _on_measurement_points_changed
    reads it via getattr(event, "action", None), which this mirrors."""


def _index_for_roles(role_index_map, *roles):
    wanted = sorted(roles)
    for idx, roles_here in role_index_map.items():
        if sorted(roles_here) == wanted:
            return idx
    raise AssertionError(f"No canvas index found for roles {wanted} in {role_index_map}")


def test_dragging_a_shared_point_moves_every_role_it_represents(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)  # adds and selects a process row
    assert len(cpt_widget.data) == 1

    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    # A fresh points-only viewer's dims don't default to (0, 0) - pin them explicitly rather
    # than assume, and read current_view back rather than hardcode it.
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    current_view = cpt_widget.viewer.dims.current_step[0]

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([1.0, 2.0])
    view_data.set_decay([3.0, 4.0])
    view_data.set_track_points([[3.0, 4.0], [7.0, 8.0], [9.0, 10.0]])  # track0 reuses decay

    cpt_widget._sync_measurement_layer_to_selected_process()

    # 4 canvas points: origin, {decay+track0} merged, track1, track2 - not 5.
    assert len(cpt_widget.layer_measurements.data) == 4
    shared_idx = _index_for_roles(cpt_widget._measurement_role_index_map, "decay", "track0")

    # Simulate the drag napari would have already applied to .data before firing the event -
    # mutate the live array in place, exactly as a real drag would, without going through the
    # .data setter (which would fire a real event of its own and double-apply the propagation).
    cpt_widget.layer_measurements.data[shared_idx][2] = 30.0
    cpt_widget.layer_measurements.data[shared_idx][3] = 40.0

    cpt_widget._propagate_measurement_point_drag(_FakeDataChangedEvent([shared_idx]))

    assert view_data.decay == [30.0, 40.0]
    assert view_data.track_points[0] == [30.0, 40.0]
    # Unshared roles are untouched.
    assert view_data.origin == [1.0, 2.0]
    assert view_data.track_points[1] == [7.0, 8.0]
    assert view_data.track_points[2] == [9.0, 10.0]


def test_dragging_an_unshared_point_only_moves_its_own_role(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    current_view = cpt_widget.viewer.dims.current_step[0]

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([1.0, 2.0])
    view_data.set_decay([3.0, 4.0])

    cpt_widget._sync_measurement_layer_to_selected_process()
    assert len(cpt_widget.layer_measurements.data) == 2

    origin_idx = _index_for_roles(cpt_widget._measurement_role_index_map, "origin")

    cpt_widget.layer_measurements.data[origin_idx][2] = 50.0
    cpt_widget.layer_measurements.data[origin_idx][3] = 60.0

    cpt_widget._propagate_measurement_point_drag(_FakeDataChangedEvent([origin_idx]))

    assert view_data.origin == [50.0, 60.0]
    assert view_data.decay == [3.0, 4.0]


def test_mid_drag_highlight_event_does_not_wipe_a_shared_role(cpt_widget):
    """Regression test for a real bug: napari fires "changing" (not "changed") on every
    mouse-move frame of a live drag, and separately fires a highlight event on every one of
    those same frames too - which _on_measurement_points_changed reacts to unconditionally, with
    no action guard. If propagation only reacted to the final "changed" event, every mid-drag
    highlight event would see the canvas already at its new (in-progress) position while
    view_data still held the old one, read that as a deletion, and drop the shared radius role -
    well before the user ever released the mouse. Propagation must react to "changing" too, so
    view_data never falls behind the canvas by even one frame.
    """
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    current_view = cpt_widget.viewer.dims.current_step[0]

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([1.0, 2.0])
    view_data.set_decay([3.0, 4.0])
    view_data.set_track_points([[3.0, 4.0], [7.0, 8.0], [9.0, 10.0]])  # track0 reuses decay
    cpt_widget._sync_measurement_layer_to_selected_process()

    shared_idx = _index_for_roles(cpt_widget._measurement_role_index_map, "decay", "track0")

    # Frame 1 of the drag: napari has already moved .data, then fires "changing".
    cpt_widget.layer_measurements.data[shared_idx][2] = 15.0
    cpt_widget.layer_measurements.data[shared_idx][3] = 16.0
    cpt_widget._propagate_measurement_point_drag(_FakeDataChangedEvent([shared_idx], action="changing"))
    # ... immediately followed by a highlight event, same frame.
    cpt_widget._on_measurement_points_changed(_FakeHighlightEvent())

    assert view_data.track_points[0] == [15.0, 16.0]  # not dropped
    assert len(view_data.track_points) == 3
    assert view_data.radius_px is not None

    # Frame 2, further along, then drag-end.
    cpt_widget.layer_measurements.data[shared_idx][2] = 30.0
    cpt_widget.layer_measurements.data[shared_idx][3] = 40.0
    cpt_widget._propagate_measurement_point_drag(_FakeDataChangedEvent([shared_idx], action="changing"))
    cpt_widget._on_measurement_points_changed(_FakeHighlightEvent())
    cpt_widget._propagate_measurement_point_drag(_FakeDataChangedEvent([shared_idx], action="changed"))
    cpt_widget._on_measurement_points_changed(_FakeHighlightEvent())

    assert view_data.decay == [30.0, 40.0]
    assert view_data.track_points[0] == [30.0, 40.0]
    assert len(view_data.track_points) == 3
    assert view_data.radius_px is not None


def test_drag_reflected_via_highlight_events_alone_no_changing_event(cpt_widget):
    """Regression test for a second, distinct bug found via a live trace: real napari drags do
    NOT fire a "changing" data event on every mouse-move frame - only layer.events.highlight
    reliably fires every frame, and it carries no per-point delta at all. A fix that only reacted
    to "changing"/"changed" data events (the previous, insufficient fix) leaves view_data stale
    for many consecutive highlight-only frames; reconcile then sees the canvas already at its new
    position while the stored value is still at the old one, and drops the role. This is exactly
    what happened live: dragging an UNSHARED radius point caused ANOTHER, completely untouched
    radius point's role to be dropped several frames into the same drag.
    """
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    current_view = cpt_widget.viewer.dims.current_step[0]

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([1.0, 2.0])
    view_data.set_decay([3.0, 4.0])
    view_data.set_track_points([[3.0, 4.0], [7.0, 8.0], [9.0, 10.0]])  # track0 reuses decay
    cpt_widget._sync_measurement_layer_to_selected_process()
    assert len(cpt_widget.layer_measurements.data) == 4

    r2_idx = _index_for_roles(cpt_widget._measurement_role_index_map, "track1")

    def fire(event):
        # Mirrors the real connection order on either signal: propagation runs first.
        cpt_widget._propagate_measurement_point_drag(event)
        cpt_widget._on_measurement_points_changed(event)

    # Frame 0: the one "changing" data event napari actually fires for this drag.
    cpt_widget.layer_measurements.data[r2_idx][2] = 7.1
    cpt_widget.layer_measurements.data[r2_idx][3] = 8.1
    fire(_FakeDataChangedEvent([r2_idx], action="changing"))

    # Many intervening frames, highlight only - no further "changing" event at all, matching
    # what the live trace actually showed for this exact scenario.
    for step in range(1, 11):
        cpt_widget.layer_measurements.data[r2_idx][2] = 7.1 + step
        cpt_widget.layer_measurements.data[r2_idx][3] = 8.1 + step
        fire(_FakeHighlightEvent())

        # Must never be dropped, at any point mid-drag - not just at the end.
        assert len(view_data.track_points) == 3, f"track_points shrank at step {step}"
        assert view_data.radius_px is not None, f"radius vanished at step {step}"

    # Frame end: the final "changed" data event.
    cpt_widget.layer_measurements.data[r2_idx][2] = 20.0
    cpt_widget.layer_measurements.data[r2_idx][3] = 25.0
    fire(_FakeDataChangedEvent([r2_idx], action="changed"))

    assert view_data.track_points[1] == [20.0, 25.0]
    assert len(view_data.track_points) == 3
    assert view_data.radius_px is not None
    # The untouched roles must never have been disturbed.
    assert view_data.origin == [1.0, 2.0]
    assert view_data.decay == [3.0, 4.0]
    assert view_data.track_points[0] == [3.0, 4.0]
    assert view_data.track_points[2] == [9.0, 10.0]


def test_drag_still_works_after_visiting_a_second_view(cpt_widget):
    """Regression test for a real bug found live: _measurement_role_index_map_point_count used
    to count points across the WHOLE layer (including other_slices - other views'/events' points,
    carried over unchanged on every rebuild), while _propagate_measurement_point_drag compared it
    against a count of the CURRENT SLICE only. The moment a second view held any points at all,
    that mismatch made the first view's current-slice count permanently look smaller than the
    (now inflated-by-the-other-view) "expected" total, so propagation concluded a deletion had
    happened and permanently backed off for that view - confirmed live via the debug trace log,
    which showed the same "slice point count dropped" skip repeating on every single drag frame.
    """
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()

    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    view0 = cpt_widget.viewer.dims.current_step[0]
    view_data_0 = cpt_widget.data[0].views[view0]
    view_data_0.set_track_points([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    cpt_widget._sync_measurement_layer_to_selected_process()

    # Visit a second view and put a role-bearing point there too.
    cpt_widget.viewer.dims.set_current_step(0, 1)
    view1 = cpt_widget.viewer.dims.current_step[0]
    assert view1 != view0
    view_data_1 = cpt_widget.data[0].views[view1]
    view_data_1.set_origin([5.0, 5.0])
    cpt_widget._sync_measurement_layer_to_selected_process()

    # Back to the first view.
    cpt_widget.viewer.dims.set_current_step(0, view0)
    cpt_widget._sync_measurement_layer_to_selected_process()
    assert cpt_widget.viewer.dims.current_step[0] == view0

    track1_idx = _index_for_roles(cpt_widget._measurement_role_index_map, "track1")
    cpt_widget.layer_measurements.data[track1_idx][2] = 99.0
    cpt_widget.layer_measurements.data[track1_idx][3] = 88.0
    cpt_widget._propagate_measurement_point_drag(_FakeDataChangedEvent([track1_idx], action="changed"))

    assert view_data_0.track_points[1] == [99.0, 88.0]
    assert len(view_data_0.track_points) == 3
    assert view_data_0.radius_px is not None
