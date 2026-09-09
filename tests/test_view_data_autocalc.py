from math import sqrt

import pytest

from cavendish_particle_tracks.analysis import ViewData


def test_length_is_none_until_both_points_are_set():
    view = ViewData()
    assert view.length_px is None
    view.set_origin([0.0, 1.0])
    assert view.length_px is None  # only one of the two points so far
    view.set_decay([1.0, 0.0])
    assert view.length_px == pytest.approx(sqrt(2))


def test_length_updates_regardless_of_which_point_is_set_first():
    view = ViewData()
    view.set_decay([1.0, 0.0])
    view.set_origin([0.0, 1.0])
    assert view.length_px == pytest.approx(sqrt(2))


def test_clearing_origin_clears_length():
    view = ViewData()
    view.set_origin([0.0, 1.0])
    view.set_decay([1.0, 0.0])
    assert view.length_px is not None
    view.set_origin(None)
    assert view.length_px is None


def test_clearing_decay_clears_length():
    view = ViewData()
    view.set_origin([0.0, 1.0])
    view.set_decay([1.0, 0.0])
    view.set_decay(None)
    assert view.length_px is None


def test_moving_a_point_recalculates_length():
    view = ViewData()
    view.set_origin([-3.0, 3.0])
    view.set_decay([-3.0, 2.0])
    assert view.length_px == pytest.approx(1.0)
    # drag the decay point somewhere else - length should follow
    view.set_decay([1.0, 0.0])
    assert view.length_px == pytest.approx(sqrt((-3.0 - 1.0) ** 2 + 3.0**2))


def test_radius_is_none_with_fewer_than_three_track_points():
    view = ViewData()
    assert view.radius_px is None
    view.add_track_point([0.0, 1.0])
    assert view.radius_px is None
    view.add_track_point([1.0, 0.0])
    assert view.radius_px is None


def test_radius_is_calculated_once_three_track_points_are_placed():
    view = ViewData()
    view.add_track_point([0.0, 1.0])
    view.add_track_point([1.0, 0.0])
    view.add_track_point([0.0, -1.0])
    assert view.radius_px == pytest.approx(1.0, rel=1e-3)


def test_radius_goes_back_to_none_if_a_fourth_point_is_added():
    view = ViewData()
    for point in ([0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [5.0, 5.0]):
        view.add_track_point(point)
    assert view.radius_px is None


def test_clear_track_points_clears_radius():
    view = ViewData()
    for point in ([0.0, 1.0], [1.0, 0.0], [0.0, -1.0]):
        view.add_track_point(point)
    assert view.radius_px is not None
    view.clear_track_points()
    assert view.radius_px is None
    assert view.track_points == []


def test_set_track_points_bulk_assignment():
    view = ViewData()
    view.set_track_points([[-6.0, 3.0], [-3.0, 2.0], [0.0, 3.0]])
    assert view.radius_px == pytest.approx(5.0, rel=1e-3)


def test_length_and_radius_are_independent_of_each_other():
    # setting up a radius fit shouldn't touch length, and vice versa
    view = ViewData()
    view.set_origin([0.0, 0.0])
    view.set_decay([3.0, 4.0])
    view.add_track_point([0.0, 1.0])
    view.add_track_point([1.0, 0.0])
    view.add_track_point([0.0, -1.0])
    assert view.length_px == pytest.approx(5.0)
    assert view.radius_px == pytest.approx(1.0, rel=1e-3)
