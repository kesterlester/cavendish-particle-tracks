"""Regression coverage for a crash in _restyle_measurement_points, found live via a full
traceback: pressing napari's own native "delete selected points" key/action (not our Clear
actions) while a point on the measurement layer is selected.

napari's `Points.remove_selected()` shrinks `layer.data` and, partway through its own body
(before it reaches its own trailing `self.selected_data = set()` cleanup a few lines later),
triggers a `layer.events.highlight` cascade that reaches our `_restyle_measurement_points`
REENTRANTLY. At that instant, `layer.selected_data` still names the just-deleted point's OLD
index - now out of bounds against the just-shrunk colour/symbol arrays. Setting any of
napari's `current_*` properties (which also apply live to whatever's currently selected, not
just "the next new point") then raises `IndexError` deep inside napari's own ColorManager -
nothing to do with our own colour/symbol/size data, which was already correctly reassigned at
the new (post-deletion) length just beforehand.
"""

def _pin_to_first_view_and_event(cpt_widget):
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    current_view = cpt_widget.viewer.dims.current_step[0]
    if cpt_widget.data:
        cpt_widget.data[0].event_number = 0
    return current_view


def test_native_delete_of_the_last_point_does_not_crash_restyle(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_track_points([[10.0, 10.0], [20.0, 10.0], [15.0, 20.0]])
    cpt_widget._sync_measurement_layer_to_selected_process()

    layer = cpt_widget.layer_measurements
    assert len(layer.data) == 3

    last_index = len(layer.data) - 1
    layer.selected_data = {last_index}
    layer.remove_selected()  # the exact call napari:delete_selected_points makes - must not raise

    assert len(layer.data) == 2


def test_native_delete_of_the_last_point_does_not_crash_restyle_with_a_richer_layer(cpt_widget):
    """Same race as above, but with a bigger, more varied layer (origin/decay/track roles mixed
    in) - makes sure the fix isn't accidentally specific to the simplest 3-point case. The crash
    is specifically an off-the-end index, so this still has to delete the LAST point: deleting
    any point whose index stays in-bounds after the shrink (e.g. a middle one) doesn't reach the
    same out-of-bounds access and wouldn't actually exercise the bug.
    """
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([0.0, 0.0])
    view_data.set_decay([50.0, 0.0])
    view_data.set_track_points([[10.0, 10.0], [20.0, 10.0], [15.0, 20.0]])
    cpt_widget._sync_measurement_layer_to_selected_process()

    layer = cpt_widget.layer_measurements
    assert len(layer.data) == 5

    last_index = len(layer.data) - 1
    layer.selected_data = {last_index}
    layer.remove_selected()  # must not raise

    assert len(layer.data) == 4
