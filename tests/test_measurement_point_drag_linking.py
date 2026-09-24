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
