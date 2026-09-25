"""Coverage for making the process table sortable by clicking a column header.

The core risk this guards against: self.data's list order and the table's own row order used
to line up exactly (rows are only ever appended, never reordered), so a raw Qt row position
could be used directly as a self.data index everywhere. Once a header click can reorder the
table, that's no longer true - every process instead carries a stable, hidden "_row_id" (see
_assign_row_id), and _get_selected_row/_table_row_for_data_index translate between "row on
screen" and "entry in self.data" via that id rather than raw position.
"""

from cavendish_particle_tracks._main_widget import (
    ParticleTracksWidget,
    _NumericTableWidgetItem,
)


def _add_process_at_event(cpt_widget, event_number: int) -> int:
    """Create a new process and give it event_number - returns its self.data index.
    particle_decays_menu.setCurrentIndex(1) always APPENDS a new row and auto-selects it (see
    _on_click_new_process), never "selects an existing one".

    Sets event_number directly on the fresh ParticleDecay AND writes the matching table cell by
    hand, rather than driving it via the Event dims slider (viewer.dims.set_current_step(1, ...))
    the way the real UI does: with no real image data loaded, the event axis's range collapses to
    just {0} (nothing in the scene has any extent along it), so the slider silently clamps any
    other value straight back to 0 - not a fixture bug, just not worth fighting for what these
    tests need (an event_number to sort by), since the table is a plain snapshot of self.data
    that nothing keeps live-updated as event_number changes after creation anyway (see
    _add_or_update_table_row's docstring: only ever called once, for a brand new row).
    """
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    data_index = len(cpt_widget.data) - 1
    cpt_widget.data[data_index].event_number = event_number
    table_row = cpt_widget._table_row_for_data_index(data_index)
    cpt_widget.table.setItem(
        table_row,
        cpt_widget._get_table_column_index("event_number"),
        _NumericTableWidgetItem(str(event_number)),
    )
    return data_index


def _event_number_column(cpt_widget) -> int:
    return cpt_widget._get_table_column_index("event_number")


def _table_events_in_visual_order(cpt_widget) -> list[int]:
    col = _event_number_column(cpt_widget)
    return [
        int(cpt_widget.table.item(row, col).text()) for row in range(cpt_widget.table.rowCount())
    ]


def _make_widget(make_napari_viewer) -> ParticleTracksWidget:
    viewer = make_napari_viewer()
    widget = ParticleTracksWidget(napari_viewer=viewer)
    widget.layer_measurements = widget._setup_measurement_layer()
    # Widens the Event dims slider's range past its otherwise-degenerate default of just {0}
    # (nothing else in this empty scene has any extent along that axis) - needed by
    # _select_table_row below, not by _add_process_at_event, which never touches dims at all.
    widget.layer_measurements.add([[0, 100, 0, 0]])
    return widget


def _select_table_row(cpt_widget, table_row: int, event_number: int) -> None:
    """Select a table row the way selecting one in the real app does - moving the Event dims
    slider to match first. Necessary here, not just cosmetic: _sync_measurement_layer_to_selected_process
    reacts to every selection change by checking the newly-selected process's OWN event_number
    against whatever event is currently displayed, and calls self.table.clearSelection() (undoing
    the very selectRow() call below, same tick) whenever they disagree - exactly the mismatch a
    bare selectRow(), without first moving dims, would produce here.
    """
    cpt_widget.viewer.dims.set_current_step(1, event_number)
    cpt_widget.table.selectRow(table_row)


def test_row_ids_are_unique_and_stable_across_creation_order(make_napari_viewer):
    cpt_widget = _make_widget(make_napari_viewer)
    _add_process_at_event(cpt_widget, 2)
    _add_process_at_event(cpt_widget, 4)
    _add_process_at_event(cpt_widget, 3)

    ids = [p._row_id for p in cpt_widget.data]
    assert len(set(ids)) == 3  # all unique
    assert ids == sorted(ids)  # assigned in creation order


def test_table_starts_in_creation_order_not_event_order(make_napari_viewer):
    cpt_widget = _make_widget(make_napari_viewer)
    _add_process_at_event(cpt_widget, 2)
    _add_process_at_event(cpt_widget, 4)
    _add_process_at_event(cpt_widget, 3)

    assert _table_events_in_visual_order(cpt_widget) == [2, 4, 3]


def test_clicking_event_number_header_sorts_numerically_not_lexicographically(make_napari_viewer):
    """Event 10 must sort after event 2 - a plain string sort would put "10" before "2"."""
    cpt_widget = _make_widget(make_napari_viewer)
    _add_process_at_event(cpt_widget, 10)
    _add_process_at_event(cpt_widget, 2)
    _add_process_at_event(cpt_widget, 9)

    col = _event_number_column(cpt_widget)
    cpt_widget._on_table_header_clicked(col)

    assert _table_events_in_visual_order(cpt_widget) == [2, 9, 10]


def test_clicking_the_same_header_twice_toggles_ascending_and_descending(make_napari_viewer):
    cpt_widget = _make_widget(make_napari_viewer)
    _add_process_at_event(cpt_widget, 2)
    _add_process_at_event(cpt_widget, 4)
    _add_process_at_event(cpt_widget, 3)

    col = _event_number_column(cpt_widget)
    cpt_widget._on_table_header_clicked(col)
    assert _table_events_in_visual_order(cpt_widget) == [2, 3, 4]

    cpt_widget._on_table_header_clicked(col)
    assert _table_events_in_visual_order(cpt_widget) == [4, 3, 2]


def test_sorting_groups_by_event_then_preserves_creation_order_within_a_group(make_napari_viewer):
    """The actual motivating request: several processes per event, sorted by event first and by
    creation order within each event - achieved for free by Qt's sort being stable, combined
    with the table's un-sorted default already being creation order.
    """
    cpt_widget = _make_widget(make_napari_viewer)
    a = _add_process_at_event(cpt_widget, 3)  # event 3, created 1st
    b = _add_process_at_event(cpt_widget, 2)  # event 2, created 2nd
    c = _add_process_at_event(cpt_widget, 3)  # event 3, created 3rd
    d = _add_process_at_event(cpt_widget, 2)  # event 2, created 4th

    col = _event_number_column(cpt_widget)
    cpt_widget._on_table_header_clicked(col)

    row_id_col = cpt_widget._get_table_column_index("_row_id")
    visual_order_row_ids = [
        int(cpt_widget.table.item(row, row_id_col).text()) for row in range(cpt_widget.table.rowCount())
    ]
    expected_row_ids = [cpt_widget.data[i]._row_id for i in (b, d, a, c)]
    assert visual_order_row_ids == expected_row_ids


def test_get_selected_row_returns_the_right_data_index_after_sorting(make_napari_viewer):
    cpt_widget = _make_widget(make_napari_viewer)
    _add_process_at_event(cpt_widget, 2)
    index_of_event_4_process = _add_process_at_event(cpt_widget, 4)
    _add_process_at_event(cpt_widget, 3)

    col = _event_number_column(cpt_widget)
    cpt_widget._on_table_header_clicked(col)  # ascending: 2, 3, 4 -> event-4 process is last row

    last_visual_row = cpt_widget.table.rowCount() - 1
    _select_table_row(cpt_widget, last_visual_row, event_number=4)

    assert cpt_widget._get_selected_row() == index_of_event_4_process


def test_deleting_a_process_after_sorting_deletes_the_correct_one(make_napari_viewer):
    cpt_widget = _make_widget(make_napari_viewer)
    index_of_event_2_process = _add_process_at_event(cpt_widget, 2)
    _add_process_at_event(cpt_widget, 4)
    _add_process_at_event(cpt_widget, 3)

    col = _event_number_column(cpt_widget)
    cpt_widget._on_table_header_clicked(col)  # ascending: 2, 3, 4 -> event-2 process is first row

    kept_row_id = {
        p._row_id for i, p in enumerate(cpt_widget.data) if i != index_of_event_2_process
    }
    deleted_row_id = cpt_widget.data[index_of_event_2_process]._row_id

    _select_table_row(cpt_widget, 0, event_number=2)
    del cpt_widget.data[cpt_widget._get_selected_row()]
    # Mirrors _on_click_delete_process's own body exactly (minus the confirmation dialog),
    # including the ordering fix: translate to a table row BEFORE the entry is gone. Recomputed
    # here rather than calling _on_click_delete_process directly, to avoid driving the dialog.
    cpt_widget.table.removeRow(0)

    remaining_ids = {p._row_id for p in cpt_widget.data}
    assert remaining_ids == kept_row_id
    assert deleted_row_id not in remaining_ids


def test_table_row_for_data_index_after_deleting_an_earlier_row(make_napari_viewer):
    """_on_click_delete_process's actual ordering: translate to a table row, THEN delete from
    self.data, THEN remove that table row (see its own comment on why that order matters).
    Reproduced directly here rather than via the confirmation dialog it normally shows first -
    QMessageBox.exec() is globally monkeypatched to auto-answer Discard for every test (see
    conftest.auto_discard_unsaved_changes_popup, aimed at a DIFFERENT dialog), which starves
    this one of its "Yes" click the same way it already does for test_widget.py's own
    test_delete_particle_ui (a pre-existing, unrelated baseline failure) - not something to
    route around in test code without changing that shared fixture's scope.
    """
    cpt_widget = _make_widget(make_napari_viewer)
    first_index = _add_process_at_event(cpt_widget, 2)
    second_index = _add_process_at_event(cpt_widget, 4)

    row_id_to_keep = cpt_widget.data[second_index]._row_id

    _select_table_row(cpt_widget, cpt_widget._table_row_for_data_index(first_index), event_number=2)

    selected_row = cpt_widget._get_selected_row()
    table_row = cpt_widget._table_row_for_data_index(selected_row)
    del cpt_widget.data[selected_row]
    cpt_widget.table.removeRow(table_row)

    assert len(cpt_widget.data) == 1
    assert cpt_widget.table.rowCount() == 1
    assert cpt_widget.data[0]._row_id == row_id_to_keep
    assert int(
        cpt_widget.table.item(0, cpt_widget._get_table_column_index("_row_id")).text()
    ) == row_id_to_keep
