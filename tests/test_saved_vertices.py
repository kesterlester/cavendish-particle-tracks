from cavendish_particle_tracks.analysis import ParticleDecay


def test_blank_process_shows_all_unmeasured():
    p = ParticleDecay(name="blank")
    assert p.saved_vertices == "___. ___. ___."


def test_non_lambda_process_shows_origin_decay_and_track():
    p = ParticleDecay(name="sigma", index=1)
    p.views[0].set_origin([1.0, 2.0])
    p.views[0].set_decay([3.0, 4.0])
    p.views[0].set_track_points([[0, 1], [1, 0], [0, -1]])
    assert p.saved_vertices == "ODR. ___. ___."


def test_angle_flag_only_applies_to_the_one_lambda_process():
    p = ParticleDecay(name="lambda_p_pi_minus", index=4)
    p.views[1].set_decay_angle_lines(
        [[[0, 0], [-1, 0]], [[0, 0], [1, 1]], [[0, 0], [1, -1]]]
    )
    assert p.saved_vertices == "____ ___A ____"


def test_angle_flag_shows_not_applicable_for_other_processes():
    for index in (1, 2, 3, 5):
        p = ParticleDecay(name="not lambda p pi-", index=index)
        assert p.saved_vertices == "___. ___. ___."


def test_radius_is_not_restricted_for_the_neutral_lambda_decay():
    # deliberate project decision: radius stays available for every
    # process type, including Lambda0 -> n + pi0 (index 5), even though
    # that decay has no charged daughter - only angles are restricted
    p = ParticleDecay(name="lambda_n_pi0", index=5)
    p.views[2].set_track_points([[0, 1], [1, 0], [0, -1]])
    assert p.saved_vertices == "___. ___. __R."


def test_clearing_a_measurement_clears_its_flag():
    p = ParticleDecay(name="sigma", index=1)
    p.views[0].set_origin([1.0, 2.0])
    p.views[0].set_decay([3.0, 4.0])
    assert p.saved_vertices == "OD_. ___. ___."
    p.views[0].set_origin(None)
    assert p.saved_vertices == "_D_. ___. ___."


def test_saved_vertices_is_included_in_vars_to_save():
    p = ParticleDecay(name="test")
    assert "saved_vertices" in p.vars_to_save()