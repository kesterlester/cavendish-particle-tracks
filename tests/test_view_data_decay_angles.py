import numpy as np
import pytest

from cavendish_particle_tracks.analysis import ViewData


def test_phi_values_start_none():
    view = ViewData()
    assert view.decay_angle_lines is None
    assert view.phi_proton is None
    assert view.phi_pion is None


def test_setting_lines_computes_angles():
    view = ViewData()
    Lambda_track = [[0, 0], [-1, 0]]
    p_track = [[0, 0], [1, 1]]
    pi_track = [[0, 0], [1, -1]]

    view.set_decay_angle_lines([Lambda_track, p_track, pi_track])

    assert view.phi_proton == pytest.approx(np.pi / 4)
    assert view.phi_pion == pytest.approx(-np.pi / 4)


def test_clearing_lines_clears_angles():
    view = ViewData()
    Lambda_track = [[0, 0], [-1, 0]]
    p_track = [[0, 0], [1, 1]]
    pi_track = [[0, 0], [1, -1]]

    view.set_decay_angle_lines([Lambda_track, p_track, pi_track])
    assert view.phi_proton is not None

    view.set_decay_angle_lines(None)
    assert view.decay_angle_lines is None
    assert view.phi_proton is None
    assert view.phi_pion is None


def test_moving_the_lines_recalculates():
    view = ViewData()
    view.set_decay_angle_lines(
        [[[0, 0], [-1, 0]], [[0, 0], [1, 1]], [[0, 0], [1, -1]]]
    )
    first_phi_proton = view.phi_proton

    # drag the proton line somewhere else - angle should follow
    view.set_decay_angle_lines(
        [[[0, 0], [-1, 0]], [[0, 0], [0, 1]], [[0, 0], [1, -1]]]
    )
    assert view.phi_proton != pytest.approx(first_phi_proton)


def test_independent_of_length_and_radius_on_same_view():
    view = ViewData()
    view.set_origin([0.0, 0.0])
    view.set_decay([3.0, 4.0])
    view.set_track_points([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    view.set_decay_angle_lines(
        [[[0, 0], [-1, 0]], [[0, 0], [1, 1]], [[0, 0], [1, -1]]]
    )

    assert view.length_px == pytest.approx(5.0)
    assert view.radius_px == pytest.approx(1.0, rel=1e-3)
    assert view.phi_proton == pytest.approx(np.pi / 4)