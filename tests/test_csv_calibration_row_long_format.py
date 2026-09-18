import io

import pandas as pd

from cavendish_particle_tracks.analysis import CSV_COLUMNS, CalibrationRow, ParticleDecay


def _rows_as_dicts(csv_output):
    lines = csv_output.strip("\n").split("\n")
    return [dict(zip(CSV_COLUMNS, line.split(","))) for line in lines]


def test_produces_exactly_three_lines_one_per_view():
    c = CalibrationRow()
    output = c.to_csv_rows(row_group_id=0)
    assert output.count("\n") == 3
    rows = _rows_as_dicts(output)
    assert [r["view"] for r in rows] == ["0", "1", "2"]


def test_process_level_fields_repeat_identically_across_all_three_rows():
    c = CalibrationRow(event_number=5)
    rows = _rows_as_dicts(c.to_csv_rows(row_group_id=2))
    for row in rows:
        assert row["row_group_id"] == "2"
        assert row["name"] == "Calibration"
        assert row["index"] == "-1"
        assert row["event_number"] == "5"


def test_stamped_fiducials_land_in_the_correct_view_and_column():
    c = CalibrationRow()
    c.views[0].stamp("A", [100.123, 200.456])
    c.views[0].stamp("B'", [300.789, 400.012])
    c.views[2].stamp("C", [50.0, 60.0])
    rows = _rows_as_dicts(c.to_csv_rows(row_group_id=0))

    assert rows[0]["A_x"] == "100.1"
    assert rows[0]["A_y"] == "200.5"
    assert rows[0]["B'_x"] == "300.8"
    assert rows[1]["A_x"] == ""  # nothing stamped in view 1
    assert rows[2]["C_x"] == "50.0"
    assert rows[2]["C_y"] == "60.0"


def test_particle_decay_only_columns_stay_blank():
    c = CalibrationRow()
    c.views[0].stamp("A", [1.0, 2.0])
    rows = _rows_as_dicts(c.to_csv_rows(row_group_id=0))
    for row in rows:
        assert row["origin_x"] == ""
        assert row["radius_px"] == ""
        assert row["phi_proton"] == ""
        assert row["decay_angle_lines_raw"] == ""


def test_mixed_particle_decay_and_calibration_row_share_one_valid_csv():
    p = ParticleDecay(name="test", index=1)
    p.views[0].set_origin([10.0, 20.0])
    c = CalibrationRow(event_number=5)
    c.views[1].stamp("D", [70.0, 80.0])

    csv_text = (
        ",".join(CSV_COLUMNS) + "\n" + p.to_csv_rows(row_group_id=0) + c.to_csv_rows(row_group_id=1)
    )
    df = pd.read_csv(io.StringIO(csv_text))

    assert len(df) == 6
    assert df["origin_x"].dtype == "float64"
    assert df["D_x"].dtype == "float64"
    assert df.loc[4, "D_x"] == 70.0
    assert pd.isna(df.loc[0, "D_x"])