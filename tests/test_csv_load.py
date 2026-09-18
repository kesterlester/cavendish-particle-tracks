from cavendish_particle_tracks.analysis import (
    CSV_COLUMNS,
    CalibrationRow,
    GenericFiducialTemplate,
    ParticleDecay,
    load_csv_session,
    round_px,
)


def _write_session(file_name, data, generic_templates):
    # Mirrors _on_click_load's real .csv save branch exactly, including round_px on the
    # appendix table's coordinates - otherwise a test could pass here while the real
    # save/load round trip (which does round) behaves differently.
    with open(file_name, "w", encoding="UTF8", newline="") as f:
        f.write(",".join(CSV_COLUMNS) + "\n")
        for row_group_id, particle in enumerate(data):
            f.write(particle.to_csv_rows(row_group_id))
        if any(len(t.positions) > 0 for t in generic_templates):
            f.write("\n")
            f.write("view,name,x,y,slot_index\n")
            for view_index, template in enumerate(generic_templates):
                for name, xy in template.positions.items():
                    slot_index = template.slot_indices.get(name, "")
                    f.write(f"{view_index},{name},{round_px(xy[0])},{round_px(xy[1])},{slot_index}\n")


def test_round_trips_a_particle_decay_with_origin_decay_and_track_points(tmp_path):
    p = ParticleDecay(name="Σ⁺ ⇨ p + π⁰", index=1, event_number=3)
    p.views[0].set_origin([100.123, 200.456])
    p.views[0].set_decay([150.789, 250.012])
    p.views[2].set_track_points([[10.0, 20.0], [30.0, 55.0], [50.0, 15.0]])

    file_name = tmp_path / "session.csv"
    _write_session(file_name, [p], [GenericFiducialTemplate(), GenericFiducialTemplate(), GenericFiducialTemplate()])

    data, templates = load_csv_session(file_name)
    loaded = data[0]
    assert loaded.name == "Σ⁺ ⇨ p + π⁰"
    assert loaded.index == 1
    assert loaded.event_number == 3
    assert loaded.views[0].origin == [100.1, 200.5]
    assert loaded.views[0].decay == [150.8, 250.0]
    assert loaded.views[0].length_px is not None  # auto-recomputed, not read from a column
    assert loaded.views[2].track_points == [[10.0, 20.0], [30.0, 55.0], [50.0, 15.0]]
    assert loaded.views[2].radius_px is not None


def test_round_trips_decay_angle_lines(tmp_path):
    p = ParticleDecay(name="Λ⁰ ⇨ p + π⁻", index=4, event_number=1)
    p.views[1].set_decay_angle_lines(
        [[[0.0, 0.0], [-1.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, -1.0]]]
    )

    file_name = tmp_path / "session.csv"
    _write_session(file_name, [p], [GenericFiducialTemplate(), GenericFiducialTemplate(), GenericFiducialTemplate()])

    data, _ = load_csv_session(file_name)
    loaded = data[0]
    assert loaded.views[1].decay_angle_lines == [
        [[0.0, 0.0], [-1.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, -1.0]]
    ]
    assert loaded.views[1].phi_proton is not None  # auto-recomputed
    assert loaded.views[0].decay_angle_lines is None


def test_round_trips_a_calibration_row(tmp_path):
    c = CalibrationRow(event_number=3)
    c.views[0].stamp("A", [500.0, 600.0])
    c.views[0].stamp("B'", [700.5, 800.5])
    c.views[2].stamp("C", [900.0, 1000.0])

    file_name = tmp_path / "session.csv"
    _write_session(file_name, [c], [GenericFiducialTemplate(), GenericFiducialTemplate(), GenericFiducialTemplate()])

    data, _ = load_csv_session(file_name)
    loaded = data[0]
    assert isinstance(loaded, CalibrationRow)
    assert loaded.event_number == 3
    assert loaded.views[0].stamped["A"] == [500.0, 600.0]
    assert loaded.views[0].stamped["B'"] == [700.5, 800.5]
    assert loaded.views[2].stamped["C"] == [900.0, 1000.0]
    assert loaded.views[1].stamped == {}


def test_round_trips_generic_templates(tmp_path):
    templates = [
        GenericFiducialTemplate({"A": [10.0, 20.0], "B'": [30.5, 40.25]}, {"A": 0, "B'": 2}),
        GenericFiducialTemplate({}, {}),
        GenericFiducialTemplate({"C": [50.0, 60.0]}, {"C": 4}),
    ]
    file_name = tmp_path / "session.csv"
    _write_session(file_name, [ParticleDecay(name="test", index=1)], templates)

    _, loaded_templates = load_csv_session(file_name)
    assert loaded_templates[0].positions["A"] == [10.0, 20.0]
    assert loaded_templates[0].slot_indices["A"] == 0
    assert loaded_templates[0].positions["B'"] == [30.5, round_px(40.25)]
    assert loaded_templates[0].slot_indices["B'"] == 2
    assert loaded_templates[1].positions == {}
    assert loaded_templates[2].positions["C"] == [50.0, 60.0]


def test_mixed_particle_decay_and_calibration_rows(tmp_path):
    p = ParticleDecay(name="test", index=1, event_number=0)
    c = CalibrationRow(event_number=5)
    c.views[1].stamp("D", [70.0, 80.0])

    file_name = tmp_path / "session.csv"
    _write_session(file_name, [p, c], [GenericFiducialTemplate(), GenericFiducialTemplate(), GenericFiducialTemplate()])

    data, _ = load_csv_session(file_name)
    assert len(data) == 2
    assert isinstance(data[0], ParticleDecay)
    assert isinstance(data[1], CalibrationRow)
    assert data[1].views[1].stamped["D"] == [70.0, 80.0]


def test_no_appendix_table_when_nothing_was_ever_calibrated(tmp_path):
    p = ParticleDecay(name="test", index=1, event_number=0)
    file_name = tmp_path / "session.csv"
    with open(file_name, "w", encoding="UTF8", newline="") as f:
        f.write(",".join(CSV_COLUMNS) + "\n")
        f.write(p.to_csv_rows(row_group_id=0))

    data, templates = load_csv_session(file_name)
    assert len(data) == 1
    assert all(len(t.positions) == 0 for t in templates)


def test_malformed_header_raises_a_clear_error(tmp_path):
    file_name = tmp_path / "bad.csv"
    with open(file_name, "w") as f:
        f.write("totally,wrong,columns\n1,2,3\n")

    try:
        load_csv_session(file_name)
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "columns don't match" in str(e)


def test_wrong_row_count_in_a_group_raises_a_clear_error(tmp_path):
    p = ParticleDecay(name="test", index=1, event_number=0)
    file_name = tmp_path / "bad.csv"
    with open(file_name, "w", encoding="UTF8", newline="") as f:
        f.write(",".join(CSV_COLUMNS) + "\n")
        lines = p.to_csv_rows(row_group_id=0).strip().split("\n")
        f.write(lines[0] + "\n" + lines[1] + "\n")  # only 2 of the expected 3 rows

    try:
        load_csv_session(file_name)
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "instead of the expected 3" in str(e)