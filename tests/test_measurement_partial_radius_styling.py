"""Regression coverage for a real bug found live: deleting one of a radius fit's 3 points with
napari's own delete tool correctly cleared radius_px (via _on_measurement_points_changed's
reconciliation, which keeps the other surviving track points rather than discarding them
outright), but _restyle_measurement_points kept colouring the SURVIVING 1 or 2 points as if they
were still a complete radius fit - it only checked "is this point in track_points", never "are
there actually 3 of them", unlike ViewData._recompute_radius's own len==3 check.
"""

import numpy as np
from napari.utils.colormaps.standardize_color import transform_color

from cavendish_particle_tracks._main_widget import RADIUS_COLOR

_RADIUS_RGBA = transform_color(RADIUS_COLOR)[0]


def _count_radius_colored(layer) -> int:
    return sum(1 for color in layer.border_color if np.allclose(color, _RADIUS_RGBA))


def _pin_to_first_view_and_event(cpt_widget):
    cpt_widget.viewer.dims.set_current_step(0, 0)
    cpt_widget.viewer.dims.set_current_step(1, 0)
    current_view = cpt_widget.viewer.dims.current_step[0]
    if cpt_widget.data:
        cpt_widget.data[0].event_number = 0
    return current_view


def test_deleting_one_of_three_radius_points_stops_the_survivors_being_coloured_as_a_radius(cpt_widget):
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_track_points([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    cpt_widget._sync_measurement_layer_to_selected_process()
    assert len(cpt_widget.layer_measurements.data) == 3
    assert view_data.radius_px is not None

    # Simulate napari's own delete tool removing one of the 3 canvas points.
    cpt_widget.layer_measurements.selected_data = {0}
    cpt_widget.layer_measurements.remove_selected()

    # The reconciliation this already fires via layer.events.data should have trimmed
    # track_points down to the 2 survivors - if it hasn't, the rest of this test is moot, so
    # assert it explicitly rather than silently testing nothing.
    assert len(view_data.track_points) == 2
    assert view_data.radius_px is None

    cpt_widget._restyle_measurement_points()

    assert _count_radius_colored(cpt_widget.layer_measurements) == 0


def test_a_complete_radius_fit_is_still_coloured_normally(cpt_widget):
    """Sanity check alongside the regression test above: the len==3 guard must not accidentally
    suppress colouring for an actually-complete radius fit."""
    cpt_widget.particle_decays_menu.setCurrentIndex(1)
    cpt_widget.layer_measurements = cpt_widget._setup_measurement_layer()
    current_view = _pin_to_first_view_and_event(cpt_widget)

    view_data = cpt_widget.data[0].views[current_view]
    view_data.set_track_points([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    cpt_widget._sync_measurement_layer_to_selected_process()
    cpt_widget._restyle_measurement_points()

    assert _count_radius_colored(cpt_widget.layer_measurements) == 3
