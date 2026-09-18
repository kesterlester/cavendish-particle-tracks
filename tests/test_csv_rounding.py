import numpy as np

from cavendish_particle_tracks.analysis import round_angle, round_px


def test_round_px_rounds_to_one_decimal():
    assert round_px(1063.5499999999) == 1063.5


def test_round_px_passes_none_through():
    assert round_px(None) is None


def test_round_px_handles_numpy_scalars():
    # _calculate.py's radius/length functions return numpy scalars, not plain floats
    assert round_px(np.float64(1063.5499999999)) == 1063.5
    assert isinstance(round_px(np.float64(1063.5499999999)), float)


def test_round_px_handles_plain_ints():
    assert round_px(5) == 5.0


def test_round_angle_rounds_to_four_decimals():
    assert round_angle(-1.25124286752) == -1.2512


def test_round_angle_passes_none_through():
    assert round_angle(None) is None


def test_round_angle_handles_numpy_scalars():
    assert round_angle(np.float64(-1.25124286752)) == -1.2512
    assert isinstance(round_angle(np.float64(-1.25124286752)), float)