from cavendish_particle_tracks.analysis import CalibrationRow, ParticleDecay


def test_blank_row_defaults():
    c = CalibrationRow()
    assert c.name == "Calibration"
    assert c.index == -1
    assert c.event_number == -1
    assert c.saved_vertices == "_ _ _"


def test_saved_vertices_lists_stamped_names_per_view():
    c = CalibrationRow(event_number=3)
    c.views[0].stamp("A", [1.0, 2.0])
    c.views[0].stamp("B'", [3.0, 4.0])
    c.views[2].stamp("C", [5.0, 6.0])
    assert c.saved_vertices == "A/B' _ C"


def test_views_are_independent_per_row():
    c1 = CalibrationRow()
    c2 = CalibrationRow()
    c1.views[0].stamp("A", [1.0, 1.0])
    assert c2.views[0].get("A") is None


def test_vars_to_save_matches_particle_decay_exactly():
    # required for CSV column alignment - the save code writes the header
    # from only the first row, then trusts every other row to line up
    assert CalibrationRow().vars_to_save() == ParticleDecay().vars_to_save()


def test_to_csv_has_no_raw_commas_in_any_field():
    c = CalibrationRow()
    c.views[0].stamp("A", [1.0, 1.0])
    c.views[0].stamp("A'", [1.0, 1.0])
    c.views[0].stamp("B", [1.0, 1.0])
    # the saved_vertices field itself must not smuggle in a comma, or it
    # would silently misalign every column after it in the CSV
    assert "," not in c.saved_vertices


def test_to_csv_includes_name_event_number_and_saved_vertices():
    c = CalibrationRow(event_number=7)
    c.views[1].stamp("D", [0.0, 0.0])
    line = c.to_csv()
    assert "Calibration" in line
    assert "7" in line
    assert "_ D _" in line