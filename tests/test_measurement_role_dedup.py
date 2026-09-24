from cavendish_particle_tracks._main_widget import _as_xy, _measurement_roles
from cavendish_particle_tracks.analysis import ViewData


def test_measurement_roles_skips_unset_fields():
    view = ViewData()
    assert _measurement_roles(view) == []


def test_measurement_roles_lists_origin_decay_and_track_points():
    view = ViewData()
    view.set_origin([1.0, 2.0])
    view.set_decay([3.0, 4.0])
    view.set_track_points([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

    roles = dict(_measurement_roles(view))
    assert roles == {
        "origin": [1.0, 2.0],
        "decay": [3.0, 4.0],
        "track0": [5.0, 6.0],
        "track1": [7.0, 8.0],
        "track2": [9.0, 10.0],
    }


def test_as_xy_rounds_and_is_hashable():
    assert _as_xy([1.0000001, 2.0]) == _as_xy([1.0, 2.0])
    assert {_as_xy([1.0, 2.0])} == {(1.0, 2.0)}


def _group_roles_by_position(view):
    """Mirrors the grouping done in _sync_measurement_layer_to_selected_process: roles that
    currently sit at the exact same spot collapse into a single canvas point.
    """
    groups: dict[tuple[float, float], list[str]] = {}
    for role, point in _measurement_roles(view):
        groups.setdefault(_as_xy(point), []).append(role)
    return groups


def test_roles_at_different_positions_stay_separate():
    view = ViewData()
    view.set_origin([1.0, 2.0])
    view.set_decay([3.0, 4.0])

    groups = _group_roles_by_position(view)
    assert len(groups) == 2
    assert groups[(1.0, 2.0)] == ["origin"]
    assert groups[(3.0, 4.0)] == ["decay"]


def test_decay_vertex_reused_as_a_radius_point_collapses_to_one_canvas_point():
    """The scenario this whole feature is about: a user reuses the already-placed decay
    vertex as one of the three points for a radius fit. It must render as ONE canvas point
    representing both roles, not two coincident points."""
    view = ViewData()
    view.set_origin([1.0, 2.0])
    view.set_decay([3.0, 4.0])
    view.set_track_points([[3.0, 4.0], [7.0, 8.0], [9.0, 10.0]])

    groups = _group_roles_by_position(view)
    assert len(groups) == 4  # origin, (decay+track0 shared), track1, track2
    assert sorted(groups[(3.0, 4.0)]) == ["decay", "track0"]
