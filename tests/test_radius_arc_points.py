import numpy as np
import pytest

from cavendish_particle_tracks._calculate import circle_fit, radius, radius_arc_points


def test_circle_fit_matches_radius():
    a, b, c = (0.0, 1.0), (1.0, 0.0), (0.0, -1.0)  # unit circle at origin
    xc, yc, r = circle_fit(a, b, c)
    assert xc == pytest.approx(0.0, abs=1e-9)
    assert yc == pytest.approx(0.0, abs=1e-9)
    assert r == pytest.approx(1.0)
    assert r == pytest.approx(radius(a, b, c))


def test_arc_starts_and_ends_at_the_extreme_points_and_passes_near_the_middle_one():
    a = (1.0, 0.0)
    b = (np.cos(np.pi / 4), np.sin(np.pi / 4))  # 45 degrees - the "middle" point
    c = (0.0, 1.0)
    arc = radius_arc_points(a, b, c, num_segments=41)

    assert arc[0] == pytest.approx(list(a), abs=1e-9)
    assert arc[-1] == pytest.approx(list(c), abs=1e-9)
    # b sits at the angular midpoint, so it should show up ~halfway along a fine-grained arc.
    midpoint = arc[len(arc) // 2]
    assert midpoint == pytest.approx(list(b), abs=1e-6)


def test_arc_order_is_independent_of_input_order():
    a = (1.0, 0.0)
    b = (np.cos(np.pi / 4), np.sin(np.pi / 4))
    c = (0.0, 1.0)
    arc1 = radius_arc_points(a, b, c, num_segments=5)
    arc2 = radius_arc_points(c, a, b, num_segments=5)  # same 3 points, different order
    assert np.allclose(arc1, arc2, atol=1e-9)


def test_all_points_between_the_two_extremes_stay_on_the_fitted_circle():
    a, b, c = (5.0, 5.0), (8.0, 2.0), (2.0, 2.0)
    xc, yc, r = circle_fit(a, b, c)
    arc = radius_arc_points(a, b, c, num_segments=25)
    for x, y in arc:
        assert np.hypot(x - xc, y - yc) == pytest.approx(r, rel=1e-6)


def test_exactly_collinear_points_fall_back_to_a_straight_line():
    a, b, c = (0.0, 0.0), (1.0, 0.0), (2.0, 0.0)
    arc = radius_arc_points(a, b, c)
    assert arc == [[0.0, 0.0], [2.0, 0.0]]  # the two most distant points


def test_nearly_collinear_points_also_fall_back_to_a_straight_line():
    a, b, c = (0.0, 0.0), (1.0, 0.0001), (2.0, 0.0)
    arc = radius_arc_points(a, b, c)
    assert len(arc) == 2  # not a huge, meaningless arc


def test_coincident_points_do_not_crash():
    a = b = c = (3.0, 4.0)
    arc = radius_arc_points(a, b, c)
    assert arc == [[3.0, 4.0], [3.0, 4.0]]
