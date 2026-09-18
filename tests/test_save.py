import csv
from glob import glob
from os import stat

import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialogButtonBox, QLineEdit, QMessageBox

from cavendish_particle_tracks.analysis import CSV_COLUMNS

from .conftest import get_dialog


def test_cant_save_empty(cpt_widget, capsys):
    # click the save button
    cpt_widget._on_click_save()

    # check we see the error info
    captured = capsys.readouterr()
    assert "There is no data in the table to save." in captured.out


@pytest.mark.parametrize(
    "file_name, expect_data_loaded",
    [
        ("my_file.csv", True),
        ("my_file.pkl", True),
        ("my_file.pdf", False),
    ],
)
def test_save_single_particle(
    cpt_widget, tmp_path, qtbot: QtBot, file_name, expect_data_loaded
):
    # start napari and the particle widget, add a single particle
    cpt_widget.particle_decays_menu.setCurrentIndex(1)  # select the Σ
    assert len(cpt_widget.data) == 1, "Expecting one particle in the table"

    def set_filename_and_close(dialog):
        qtbot.addWidget(dialog)
        dialog.setDirectory(str(tmp_path))
        dialog.findChild(QLineEdit, "fileNameEdit").setText(file_name)
        buttonbox = dialog.findChild(QDialogButtonBox, "buttonBox")
        openbutton = buttonbox.children()[1]
        qtbot.mouseClick(openbutton, Qt.LeftButton, delay=1)

    # Open and retrieve file dialog
    get_dialog(
        dialog_trigger=cpt_widget._on_click_save,
        dialog_action=set_filename_and_close,
        time_out=5,
    )

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
    else:
        msgbox = cpt_widget.msg
        assert isinstance(msgbox, QMessageBox)
        assert msgbox.icon() == QMessageBox.Warning
        assert msgbox.text() == (
            "The file must be a CSV (*.csv) or Pickle (*.pkl) file. Please try again."
        )


def test_csv_file_has_correct_columns(cpt_widget, tmp_path, qtbot: QtBot):
    # start napari and the particle widget, add a single particle
    cpt_widget.particle_decays_menu.setCurrentIndex(4)  # select the Λ
    assert len(cpt_widget.data) == 1, "Expecting one particle in the table"

    file_name = "test_saved_file.csv"

    def set_filename_and_close(dialog):
        # Function of the signature needed to use as a dialog action.
        # Defined internally so we can access the fixtures without passing.
        qtbot.addWidget(dialog)
        dialog.setDirectory(str(tmp_path))
        dialog.findChild(QLineEdit, "fileNameEdit").setText(file_name)
        buttonbox = dialog.findChild(QDialogButtonBox, "buttonBox")
        openbutton = buttonbox.children()[1]
        qtbot.mouseClick(openbutton, Qt.LeftButton, delay=1)

    # Open and retrieve file dialog
    get_dialog(
        dialog_trigger=cpt_widget._on_click_save,
        dialog_action=set_filename_and_close,
        time_out=5,
    )

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
