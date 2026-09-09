from cavendish_particle_tracks.analysis import ParticleDecay, ViewData


def test_particle_decay_has_three_views_by_default():
    p = ParticleDecay(name="test")
    assert len(p.views) == 3
    for view in p.views:
        assert view.origin is None
        assert view.decay is None
        assert view.track_points == []
        assert view.radius_px is None
        assert view.length_px is None


def test_views_are_independent_objects():
    p = ParticleDecay(name="test")
    p.views[0].origin = [10.0, 20.0]
    assert p.views[1].origin is None
    assert p.views[2].origin is None


def test_two_particle_decays_dont_share_views():
    p1 = ParticleDecay(name="one")
    p2 = ParticleDecay(name="two")
    p1.views[0].origin = [1.0, 1.0]
    assert p2.views[0].origin is None


def test_setting_origin_and_decay_on_a_view():
    view = ViewData()
    view.origin = [100.0, 200.0]
    view.decay = [150.0, 250.0]
    assert view.origin == [100.0, 200.0]
    assert view.decay == [150.0, 250.0]


def test_track_points_accumulate():
    view = ViewData()
    view.track_points.append([1.0, 2.0])
    view.track_points.append([3.0, 4.0])
    view.track_points.append([5.0, 6.0])
    assert len(view.track_points) == 3


def test_old_flat_fields_still_work_untouched():
    p = ParticleDecay(name="test")
    p.origin_v0_x = "123.4"
    p.origin_v0_y = "567.8"
    assert p.origin_v0_x == "123.4"
    assert p.origin_v0_y == "567.8"
    assert p.views[0].origin is None
