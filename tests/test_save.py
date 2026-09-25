import csv
from glob import glob
from os import stat

import pytest
from qtpy.QtWidgets import QFileDialog, QMessageBox

from cavendish_particle_tracks.analysis import CSV_COLUMNS


def _save_as(cpt_widget, monkeypatch, path):
    """Click Save and answer the file dialog with `path`, without showing the dialog. Patches the
    static QFileDialog.getSaveFileName that _on_click_save calls, returning what the real dialog
    would: the chosen path, and the CSV filter the user left selected."""
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), "CSV files (*.csv)")
    )
    cpt_widget._on_click_save()


def test_cant_save_empty(cpt_widget, capsys):
    # click the save button
    cpt_widget._on_click_save()

    # check we see the error info
    captured = capsys.readouterr()
    assert "There is no data in the table to save." in captured.out


@pytest.mark.parametrize(
    "file_name, expect_data_loaded, rejection_reason",
    [
        ("my_file.csv", True, None),
        # .pkl saving is deliberately disabled for now (ENABLE_PICKLE = False)
        ("my_file.pkl", False, "pickle_disabled"),
        ("my_file.pdf", False, "invalid_file_type"),
    ],
)
def test_save_single_particle(
    cpt_widget, tmp_path, monkeypatch, capsys, file_name, expect_data_loaded, rejection_reason
):
    # start napari and the particle widget, add a single particle
    cpt_widget.particle_decays_menu.setCurrentIndex(1)  # select the Σ
    assert len(cpt_widget.data) == 1, "Expecting one particle in the table"

    _save_as(cpt_widget, monkeypatch, tmp_path / file_name)

    if expect_data_loaded:
        expected_file_name = file_name  # Expect the file name to be the one we set
        csv_files = glob(str(tmp_path / "*.csv"))
        pkl_files = glob(str(tmp_path / "*.pkl"))

        expect_a_csv_and_have_one = (
            expected_file_name.endswith(".csv") and len(csv_files) == 1
        )
        expect_a_pkl_and_have_one = (
            expected_file_name.endswith(".pkl") and len(pkl_files) == 1
        )
        assert (
            expect_a_csv_and_have_one or expect_a_pkl_and_have_one
        ), "Unexpected number of data files found"

        # Only one file if we've passed the above XOR check
        saved_file = (csv_files + pkl_files)[0]
        assert saved_file.endswith(
            expected_file_name
        ), f"File name {saved_file} does not match expected name: {expected_file_name}"

        saved_file_is_not_empty = stat(saved_file).st_size != 0
        assert saved_file_is_not_empty, f"File {saved_file} is empty"
    elif rejection_reason == "invalid_file_type":
        msgbox = cpt_widget.msg
        assert isinstance(msgbox, QMessageBox)
        assert msgbox.icon() == QMessageBox.Warning
        assert msgbox.text() == (
            "The file must be a CSV (*.csv) or Pickle (*.pkl) file. Please try again."
        )
    elif rejection_reason == "pickle_disabled":
        captured = capsys.readouterr()
        assert "Saving as .pkl is currently disabled" in captured.out


def test_csv_file_has_correct_columns(cpt_widget, tmp_path, monkeypatch):
    # start napari and the particle widget, add a single particle
    cpt_widget.particle_decays_menu.setCurrentIndex(4)  # select the Λ
    assert len(cpt_widget.data) == 1, "Expecting one particle in the table"

    file_name = "test_saved_file.csv"

    _save_as(cpt_widget, monkeypatch, tmp_path / file_name)

    # Check the file has the correct columns
    csv_files = glob(str(tmp_path / "*.csv"))
    assert len(csv_files) == 1, "Expecting one CSV file to be saved"
    with open(csv_files[0], encoding="utf8") as f:
        myreader = csv.reader(f, delimiter=",")
        rows = list(myreader)

    header, data_rows = rows[0], rows[1:]

    # Checked against the real CSV_COLUMNS constant, not a hardcoded number - the old version of
    # this test hardcoded "18 columns" and silently went stale the moment the format grew past that.
    assert header == CSV_COLUMNS
    for row in data_rows:
        assert len(row) == len(CSV_COLUMNS)

    # One freshly created, unmeasured process -> exactly 3 rows (long format, one per view).
    assert len(data_rows) == 3
    for view_index, row in enumerate(data_rows):
        row_dict = dict(zip(CSV_COLUMNS, row))
        assert row_dict["row_group_id"] == "0"
        assert row_dict["name"] == "Lambda0_to_p_pi-"
        assert row_dict["index"] == "4"
        assert row_dict["view"] == str(view_index)
        # Nothing was actually measured, so every per-view measurement column stays blank.
        assert row_dict["origin_x"] == ""
        assert row_dict["radius_px"] == ""
        assert row_dict["phi_proton"] == ""
