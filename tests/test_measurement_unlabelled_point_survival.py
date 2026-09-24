"""Regression coverage for a bug the redesign introduced and this same session caught: since
selection is now separate from action, a freshly-placed point can legitimately sit on the canvas
for a while with no role yet (before the user right-clicks / presses O, D or R). A routine canvas
rebuild - triggered by nothing more than the View/Event dims slider firing a change event, e.g.
because a new point widened napari's own slider range - must never silently drop such a point.
Only a real delete or a Record/Clear action may make one disappear.
"""


def test_unlabelled_point_survives_a_dims_triggered_rebuild(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    current_view = cpt_widget.viewer.dims.current_step[0]

    # A bare click: a point lands on the canvas with no role in view_data at all yet.
    cpt_widget.layer_measurements.add([[current_view, 0, 1.0, 2.0]])
    assert len(cpt_widget.layer_measurements.data) == 1

    # Something nudges the dims slider - the exact trigger doesn't matter, only that this method
    # runs again before the user has recorded anything for that point.
    cpt_widget._sync_measurement_layer_to_selected_process()

    assert len(cpt_widget.layer_measurements.data) == 1
    assert list(cpt_widget.layer_measurements.data[0][2:]) == [1.0, 2.0]


def test_unlabelled_point_coexists_with_recorded_roles_across_rebuilds(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    current_view = cpt_widget.viewer.dims.current_step[0]

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_origin([5.0, 6.0])

    cpt_widget.layer_measurements.add([[current_view, 0, 1.0, 2.0]])  # bare, unrecorded
    cpt_widget._sync_measurement_layer_to_selected_process()

    positions = sorted(tuple(p[2:]) for p in cpt_widget.layer_measurements.data)
    assert positions == [(1.0, 2.0), (5.0, 6.0)]
