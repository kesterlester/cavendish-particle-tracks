"""Warnings the plugin deliberately silences, and only those."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from cavendish_particle_tracks._main_widget import ParticleTracksWidget

WHERE_WARNING = "'where' used without 'out'"


def _draw_a_path(viewer):
    """What drawing a radius arc, O->D arrow or decay-angle line does inside napari."""
    viewer.add_shapes([np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]])], shape_type="path")


def test_napari_shapes_where_warning_is_silenced(make_napari_viewer):
    """napari 0.5's Shapes line drawing trips a harmless numpy warning, which napari shows as a
    pop-up the first time the plugin draws a line (typically on loading a CSV). The widget
    silences exactly that one."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _draw_a_path(make_napari_viewer())
        if not any(WHERE_WARNING in str(w.message) for w in caught):
            pytest.skip("this napari/numpy doesn't emit the warning, so there's nothing to silence")

        caught.clear()
        widget = ParticleTracksWidget(napari_viewer=make_napari_viewer())
        _draw_a_path(widget.viewer)
    assert not [w for w in caught if WHERE_WARNING in str(w.message)]


def test_other_warnings_from_the_same_napari_file_are_not_silenced(make_napari_viewer):
    """The filter is matched on message and module, so nothing else is hidden."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ParticleTracksWidget(napari_viewer=make_napari_viewer())
        warnings.warn_explicit(
            "some other warning", UserWarning, "_shapes_utils.py", 1,
            module="napari.layers.shapes._shapes_utils",
        )
        warnings.warn_explicit(
            WHERE_WARNING + " from elsewhere", UserWarning, "other.py", 1, module="some.other.module",
        )
    messages = [str(w.message) for w in caught]
    assert "some other warning" in messages
    assert WHERE_WARNING + " from elsewhere" in messages
