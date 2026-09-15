import pickle

from cavendish_particle_tracks.analysis import (
    CalibrationRow,
    GenericFiducialTemplate,
    ParticleDecay,
    SavedSession,
)


def test_empty_session_round_trips():
    session = SavedSession()
    restored = pickle.loads(pickle.dumps(session))
    assert restored.data == []
    assert restored.generic_templates == []


def test_particle_decay_round_trips_with_full_fidelity():
    p = ParticleDecay(name="test", index=1)
    p.views[0].set_origin([1.0, 2.0])
    p.views[0].set_decay([3.0, 4.0])
    p.views[1].set_track_points([[0, 1], [1, 0], [0, -1]])

    session = SavedSession(data=[p])
    restored = pickle.loads(pickle.dumps(session))

    assert isinstance(restored.data[0], ParticleDecay)
    assert restored.data[0].views[0].origin == [1.0, 2.0]
    assert restored.data[0].views[0].decay == [3.0, 4.0]
    assert restored.data[0].views[0].length_px is not None
    assert restored.data[0].views[1].radius_px is not None


def test_calibration_row_round_trips_with_full_fidelity():
    c = CalibrationRow(event_number=3)
    c.views[0].stamp("A", [5.0, 6.0])
    c.views[2].stamp("B'", [7.0, 8.0])

    session = SavedSession(data=[c])
    restored = pickle.loads(pickle.dumps(session))

    assert isinstance(restored.data[0], CalibrationRow)
    assert restored.data[0].views[0].get("A") == [5.0, 6.0]
    assert restored.data[0].views[2].get("B'") == [7.0, 8.0]
    assert restored.data[0].views[1].stamped == {}


def test_mixed_particle_decay_and_calibration_row_data():
    p = ParticleDecay(name="test", index=1)
    c = CalibrationRow(event_number=0)
    session = SavedSession(data=[p, c])
    restored = pickle.loads(pickle.dumps(session))

    assert len(restored.data) == 2
    assert isinstance(restored.data[0], ParticleDecay)
    assert isinstance(restored.data[1], CalibrationRow)


def test_generic_templates_round_trip():
    t0 = GenericFiducialTemplate()
    t0.set_position("A", [10.0, 20.0])
    t1 = GenericFiducialTemplate()
    t2 = GenericFiducialTemplate()
    t2.set_position("C", [30.0, 40.0])

    session = SavedSession(generic_templates=[t0, t1, t2])
    restored = pickle.loads(pickle.dumps(session))

    assert len(restored.generic_templates) == 3
    assert restored.generic_templates[0].get_position("A") == [10.0, 20.0]
    assert restored.generic_templates[1].get_position("A") is None
    assert restored.generic_templates[2].get_position("C") == [30.0, 40.0]