"""Coverage for _points_claimed_by_other_processes: a point already claimed by one process must
not look like a free "orphan" - and so become claimable a second time by accident - while a
different process in the same (view, event) is selected.
"""


def _pin_to_first_view_and_event(cpt_widget):
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    return cpt_widget.viewer.dims.current_step[0]


def _add_process(cpt_widget, current_view):
    """Mirrors clicking 'New process': each selection of a non-header combo entry appends and
    auto-selects a new row. event_number only gets set from real image loading, which these
    tests don't do, so pin it explicitly the way other tests in this suite already do.
    """
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    row = len(cpt_widget.data) - 1
    cpt_widget.data[row].event_number = 0
    return row


def test_a_point_claimed_by_another_process_is_hidden_not_offered_as_an_orphan(cpt_widget):
    current_view = _pin_to_first_view_and_event(cpt_widget)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()

    process1_row = _add_process(cpt_widget, current_view)
    cpt_widget.layer_measurements.add([[current_view, 0, 50.0, 60.0]])
    cpt_widget.layer_measurements.selected_data = {0}
    cpt_widget._record_origin_vertex()
    assert cpt_widget.data[process1_row].views[current_view].origin == [50.0, 60.0]

    process2_row = _add_process(cpt_widget, current_view)
    assert process2_row != process1_row
    cpt_widget._sync_measurement_layer_to_selected_process()

    # Process 1's point must not appear at all while process 2 is selected - not as a role (it
    # has none in process 2), and not as an "unclaimed" orphan either.
    on_screen = [tuple(p[2:]) for p in cpt_widget.layer_measurements.data]
    assert (50.0, 60.0) not in on_screen

    # Switching back to process 1 must still show it correctly, undisturbed.
    cpt_widget.table.selectRow(process1_row)
    cpt_widget._sync_measurement_layer_to_selected_process()
    on_screen_p1 = [tuple(p[2:]) for p in cpt_widget.layer_measurements.data]
    assert (50.0, 60.0) in on_screen_p1
    assert cpt_widget.data[process1_row].views[current_view].origin == [50.0, 60.0]


def test_a_genuinely_unclaimed_point_still_shows_as_an_orphan(cpt_widget):
    """Sanity check alongside the regression test above: the new exclusion must not accidentally
    hide points nobody has claimed at all."""
    current_view = _pin_to_first_view_and_event(cpt_widget)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()

    process1_row = _add_process(cpt_widget, current_view)
    cpt_widget.layer_measurements.add([[current_view, 0, 50.0, 60.0]])
    cpt_widget._sync_measurement_layer_to_selected_process()  # still unclaimed by anyone

    process2_row = _add_process(cpt_widget, current_view)
    cpt_widget._sync_measurement_layer_to_selected_process()

    on_screen = [tuple(p[2:]) for p in cpt_widget.layer_measurements.data]
    assert (50.0, 60.0) in on_screen


def test_a_different_events_point_is_never_treated_as_claimed(cpt_widget):
    """A process recorded for a DIFFERENT event must not suppress an orphan in this one - the
    exclusion is scoped to (view, event), same as everything else on this layer."""
    current_view = _pin_to_first_view_and_event(cpt_widget)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()

    process1_row = _add_process(cpt_widget, current_view)
    cpt_widget.data[process1_row].event_number = 7  # a different event
    view_data = cpt_widget.data[process1_row].views[current_view]
    view_data.set_origin([50.0, 60.0])

    process2_row = _add_process(cpt_widget, current_view)  # event 0, per _add_process
    cpt_widget.layer_measurements.add([[current_view, 0, 50.0, 60.0]])
    cpt_widget._sync_measurement_layer_to_selected_process()

    on_screen = [tuple(p[2:]) for p in cpt_widget.layer_measurements.data]
    assert (50.0, 60.0) in on_screen
