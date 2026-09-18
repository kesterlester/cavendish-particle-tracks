import io

import pandas as pd

from cavendish_particle_tracks.analysis import CSV_COLUMNS, ParticleDecay


def _rows_as_dicts(csv_output):
    lines = csv_output.strip("\n").split("\n")
    return [dict(zip(CSV_COLUMNS, line.split(","))) for line in lines]


def test_produces_exactly_three_lines_one_per_view():
    p = ParticleDecay(name="test", index=1)
    output = p.to_csv_rows(row_group_id=0)
    assert output.count("\n") == 3
    rows = _rows_as_dicts(output)
    assert [r["view"] for r in rows] == ["0", "1", "2"]


def test_process_level_fields_repeat_identically_across_all_three_rows():
    p = ParticleDecay(name="Σ⁺ ⇨ p + π⁰", index=1, event_number=7)
    rows = _rows_as_dicts(p.to_csv_rows(row_group_id=3))
    for row in rows:
        assert row["row_group_id"] == "3"
        assert row["name"] == "Sigma+_to_p_pi0"
        assert row["index"] == "1"
        assert row["event_number"] == "7"


def test_per_view_fields_are_isolated_to_their_own_row():
    p = ParticleDecay(name="test", index=1)
    p.views[0].set_origin([100.123, 200.456])
    p.views[2].set_track_points([[0, 1], [1, 0], [0, -1]])
    rows = _rows_as_dicts(p.to_csv_rows(row_group_id=0))

    assert rows[0]["origin_x"] == "100.1"
    assert rows[0]["origin_y"] == "200.5"
    assert rows[1]["origin_x"] == ""  # untouched view stays blank
    assert rows[2]["origin_x"] == ""  # origin was only ever set on view 0

    assert rows[2]["track1_x"] == "0.0"
    assert rows[0]["track1_x"] == ""  # track points were only set on view 2


def test_unset_fields_are_blank_not_the_string_none():
    p = ParticleDecay(name="test")
    rows = _rows_as_dicts(p.to_csv_rows(row_group_id=0))
    for row in rows:
        assert row["origin_x"] == ""
        assert "None" not in row.values()


def test_fiducial_columns_always_blank_for_a_process_row():
    p = ParticleDecay(name="test")
    rows = _rows_as_dicts(p.to_csv_rows(row_group_id=0))
    for row in rows:
        assert row["A_x"] == ""
        assert row["F'_y"] == ""


def test_decay_angle_lines_packed_for_the_correct_view_only():
    p = ParticleDecay(name="test", index=4)
    p.views[1].set_decay_angle_lines([[[0, 0], [-1, 0]], [[0, 0], [1, 1]], [[0, 0], [1, -1]]])
    rows = _rows_as_dicts(p.to_csv_rows(row_group_id=0))
    assert rows[0]["decay_angle_lines_raw"] == ""
    assert rows[1]["decay_angle_lines_raw"] == "0.0 0.0|-1.0 0.0;0.0 0.0|1.0 1.0;0.0 0.0|1.0 -1.0"
    assert rows[2]["decay_angle_lines_raw"] == ""
    # commas would break CSV column alignment - confirm none leaked into the packed string
    assert "," not in rows[1]["decay_angle_lines_raw"]


def test_output_is_valid_pandas_readable_csv_with_correct_dtypes():
    p = ParticleDecay(name="test", index=1)
    p.views[0].set_origin([100.0, 200.0])
    p.views[0].set_track_points([[0, 1], [1, 0], [0, -1]])

    csv_text = ",".join(CSV_COLUMNS) + "\n" + p.to_csv_rows(row_group_id=0)
    df = pd.read_csv(io.StringIO(csv_text))

    assert len(df) == 3
    assert df["origin_x"].dtype == "float64"
    assert df["view"].dtype == "int64"
    assert pd.isna(df.loc[1, "origin_x"])  # blank cell becomes real NaN, not a string