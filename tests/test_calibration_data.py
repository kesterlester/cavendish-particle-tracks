import pytest

from cavendish_particle_tracks.analysis import (
    CalibrationData,
    FIDUCIAL_NAMES,
    FiducialViewData,
    GenericFiducialTemplate,
)


def test_generic_template_starts_empty():
    t = GenericFiducialTemplate()
    assert t.get_position("A") is None


def test_generic_template_set_and_get():
    t = GenericFiducialTemplate()
    t.set_position("A", [10.0, 20.0])
    assert t.get_position("A") == [10.0, 20.0]


def test_generic_template_rejects_unknown_names():
    t = GenericFiducialTemplate()
    with pytest.raises(ValueError):
        t.set_position("not_a_real_fiducial", [1.0, 2.0])


def test_fiducial_view_data_stamp_and_unstamp():
    v = FiducialViewData()
    assert v.get("B") is None
    v.stamp("B", [5.0, 6.0])
    assert v.get("B") == [5.0, 6.0]
    v.unstamp("B")
    assert v.get("B") is None


def test_fiducial_view_data_rejects_unknown_names():
    v = FiducialViewData()
    with pytest.raises(ValueError):
        v.stamp("not_a_real_fiducial", [1.0, 2.0])


def test_calibration_data_has_three_independent_generic_templates():
    c = CalibrationData()
    assert len(c.generic_templates) == 3
    c.generic_templates[0].set_position("A", [1.0, 1.0])
    assert c.generic_templates[1].get_position("A") is None
    assert c.generic_templates[2].get_position("A") is None


def test_calibration_data_stamp_is_scoped_to_its_own_event_and_view():
    c = CalibrationData()
    c.stamp(event=0, view=1, name="C", xy=[100.0, 200.0])
    assert c.get_stamp(0, 1, "C") == [100.0, 200.0]
    assert c.get_stamp(1, 1, "C") is None  # different event
    assert c.get_stamp(0, 2, "C") is None  # different view


def test_calibration_data_unstamp():
    c = CalibrationData()
    c.stamp(event=0, view=0, name="A", xy=[1.0, 2.0])
    c.unstamp(0, 0, "A")
    assert c.get_stamp(0, 0, "A") is None


def test_unstamping_a_never_stamped_event_view_does_not_error():
    c = CalibrationData()
    c.unstamp(5, 2, "A")  # should just do nothing, not raise


def test_fiducial_names_cover_both_front_and_back_sets():
    assert "A" in FIDUCIAL_NAMES
    assert "A'" in FIDUCIAL_NAMES
    assert "origin" not in FIDUCIAL_NAMES
    assert "decay" not in FIDUCIAL_NAMES