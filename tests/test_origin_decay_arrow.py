import numpy as np
import pytest

from cavendish_particle_tracks._calculate import origin_decay_arrow


def test_arrow_along_the_x_axis():
    shaft, head = origin_decay_arrow((0.0, 0.0), (10.0, 0.0))
    assert shaft == [[0.0, 0.0], [8.0, 0.0]]
    tip, base_a, base_b = head
    assert tip == pytest.approx([10.0, 0.0])
    assert base_a == pytest.approx([8.0, 0.6])
    assert base_b == pytest.approx([8.0, -0.6])


def test_zero_length_arrow_returns_none():
    assert origin_decay_arrow((5.0, 5.0), (5.0, 5.0)) == (None, None)


def test_head_length_is_a_fixed_fraction_of_total_length_not_a_fixed_size():
    short_shaft, short_head = origin_decay_arrow((0.0, 0.0), (1.0, 0.0), head_length_fraction=0.2)
    long_shaft, long_head = origin_decay_arrow((0.0, 0.0), (100.0, 0.0), head_length_fraction=0.2)

    short_head_length = np.hypot(*(np.array(short_head[0]) - np.array(short_shaft[1])))
    long_head_length = np.hypot(*(np.array(long_head[0]) - np.array(long_shaft[1])))

    assert short_head_length == pytest.approx(0.2)
    assert long_head_length == pytest.approx(20.0)


def test_tip_is_exactly_at_the_decay_point():
    decay = (7.0, -3.0)
    _, head = origin_decay_arrow((1.0, 1.0), decay)
    assert head[0] == list(decay)


def test_shaft_start_is_exactly_at_the_origin_point():
    origin = (1.0, 1.0)
    shaft, _ = origin_decay_arrow(origin, (7.0, -3.0))
    assert shaft[0] == list(origin)


def test_arrowhead_base_is_perpendicular_to_the_shaft_direction():
    origin, decay = (0.0, 0.0), (3.0, 4.0)
    _, head = origin_decay_arrow(origin, decay)
    _, base_a, base_b = [np.array(p) for p in head]
    shaft_direction = np.array(decay) - np.array(origin)
    base_edge = base_a - base_b
    assert np.dot(shaft_direction, base_edge) == pytest.approx(0.0, abs=1e-9)


def test_default_geometry_gives_a_symmetric_triangle_about_the_shaft_line():
    origin, decay = (2.0, 2.0), (2.0, 12.0)  # straight up
    _, head = origin_decay_arrow(origin, decay)
    tip, base_a, base_b = [np.array(p) for p in head]
    midpoint = (base_a + base_b) / 2
    # the shaft direction (0,1) should pass through both the tip and the base midpoint
    assert midpoint[0] == pytest.approx(tip[0])
    assert np.isclose(base_a[1], base_b[1])
    assert not np.isclose(base_a[0], base_b[0])
