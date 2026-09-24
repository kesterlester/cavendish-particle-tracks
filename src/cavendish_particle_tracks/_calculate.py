import numpy as np

from .analysis import (
    CHAMBER_DEPTH,
    FIDUCIAL_BACK,
    FIDUCIAL_FRONT,
    Fiducial,
)

Point = tuple[float, float]


def circle_fit(a: Point, b: Point, c: Point) -> tuple[float, float, float]:
    """The center (xc, yc) and radius of the one circle passing through all 3 points."""
    lhs = np.array(
        [
            [2 * a[0], 2 * a[1], 1],
            [2 * b[0], 2 * b[1], 1],
            [2 * c[0], 2 * c[1], 1],
        ]
    )
    rhs = np.array(
        [
            a[0] * a[0] + a[1] * a[1],
            b[0] * b[0] + b[1] * b[1],
            c[0] * c[0] + c[1] * c[1],
        ]
    )
    xc, yc, k = np.linalg.solve(lhs, rhs)
    r = np.sqrt(xc * xc + yc * yc + k)
    return xc, yc, r


def radius(a: Point, b: Point, c: Point) -> float:
    _, _, r = circle_fit(a, b, c)
    return r


def length(a: Point, b: Point) -> float:
    pa = np.array(a)
    pb = np.array(b)
    return np.linalg.norm(pa - pb)


# Above this ratio of fitted radius to the points' own spread, the 3 points are close enough to
# collinear that the circle's center is so far away (or the fit so ill-conditioned) that drawing
# the true arc would be visually meaningless or numerically unstable - a straight line through the
# two most distant points is what that arc would look like anyway at this point.
_RADIUS_ARC_STRAIGHT_LINE_THRESHOLD = 50


def radius_arc_points(a: Point, b: Point, c: Point, num_segments: int = 40) -> list[Point]:
    """The shortest arc of the 3-point circle that visits a, b and c in order along the curve
    (not necessarily input order) - i.e. from whichever of the 3 points is at one extreme angle
    round to the one at the other extreme, passing through the middle one, not the long way round.
    Falls back to a straight line through the two most distant of the 3 points when they're too
    close to collinear for a stable circle fit (see _RADIUS_ARC_STRAIGHT_LINE_THRESHOLD).
    """
    points = [a, b, c]
    pairs = [(a, b), (a, c), (b, c)]
    distances = [length(p, q) for p, q in pairs]
    max_pairwise_distance = max(distances)

    def straight_line_fallback() -> list[Point]:
        p, q = pairs[int(np.argmax(distances))]
        return [list(p), list(q)]

    if max_pairwise_distance == 0:
        return straight_line_fallback()  # degenerate: all 3 points coincide

    try:
        xc, yc, r = circle_fit(a, b, c)
    except np.linalg.LinAlgError:
        return straight_line_fallback()  # exactly collinear - the fit is singular

    if not np.isfinite(r) or r > _RADIUS_ARC_STRAIGHT_LINE_THRESHOLD * max_pairwise_distance:
        return straight_line_fallback()

    # np.arctan2 wraps at +/-pi, so a plain ascending sort is wrong whenever the 3 points'
    # true angular span straddles that branch cut (e.g. angles 160, 180, -170 degrees are only
    # 30 degrees apart going the short way, but a naive sort/min/max would compute a ~330 degree
    # arc the WRONG way round the circle). Instead: find the largest of the 3 circular gaps
    # between the sorted angles (wrapping the last gap back to the first) - that gap is the empty
    # stretch of the circle with no data point in it, so the natural arc is everything EXCEPT it,
    # regardless of where the branch cut happens to fall.
    a0, a1, a2 = sorted(np.arctan2(p[1] - yc, p[0] - xc) for p in points)
    gap_after_a0 = a1 - a0
    gap_after_a1 = a2 - a1
    gap_after_a2 = (a0 + 2 * np.pi) - a2  # wraps back round to a0
    if gap_after_a2 >= gap_after_a0 and gap_after_a2 >= gap_after_a1:
        start_angle, end_angle = a0, a2
    elif gap_after_a0 >= gap_after_a1:
        start_angle, end_angle = a1, a0 + 2 * np.pi
    else:
        start_angle, end_angle = a2, a1 + 2 * np.pi
    return [
        [xc + r * np.cos(t), yc + r * np.sin(t)]
        for t in np.linspace(start_angle, end_angle, num_segments)
    ]


def origin_decay_arrow(
    origin: Point,
    decay: Point,
    head_length_fraction: float = 0.2,
    head_width_fraction: float = 0.6,
) -> tuple[list[Point], list[Point]] | tuple[None, None]:
    """The shaft (a 2-point line) and arrowhead (a 3-point triangle) of an arrow pointing from
    the origin vertex to the decay vertex - showing which vertex is which without relying on
    colour or text alone. The arrowhead's length is a FIXED FRACTION of the arrow's own length
    (not a fixed pixel size), so it stays proportionally visible whether the decay length is
    large or tiny, rather than swamping a short one or vanishing on a long one.

    Returns (None, None) if origin and decay coincide (a zero-length arrow has no direction to
    point in) - the degenerate case of a decay length of exactly zero.
    """
    ox, oy = origin
    dx, dy = decay
    arrow_length = np.hypot(dx - ox, dy - oy)
    if arrow_length == 0:
        return None, None

    unit = np.array([(dx - ox) / arrow_length, (dy - oy) / arrow_length])
    perpendicular = np.array([-unit[1], unit[0]])

    head_length = arrow_length * head_length_fraction
    head_width = head_length * head_width_fraction

    tip = np.array([dx, dy])
    base_center = tip - unit * head_length
    base_a = base_center + perpendicular * (head_width / 2)
    base_b = base_center - perpendicular * (head_width / 2)

    shaft = [list(origin), list(base_center)]
    head = [list(tip), list(base_a), list(base_b)]
    return shaft, head

# def magnification(front_fiducial_1: Fiducial, front_fiducial_2: Fiducial,
#                   back_fiducial_1: Fiducial, back_fiducial_2: Fiducial):
#     """
#     This method calculates parameters "a" and "b" which carry information about how transverse image measurements in
#     pixels relate to real world displacements in cm as function of feature depth within the chamber.
#
#     The parameter "a" has units of "cm/pixel". It is the number of cm per pixel for transverse features seem at the
#     front of the chamber.
#
#     The parameter "b" has units of "/pixel".  The parameter b is defined such that the expression
#
#         cm_per_pixel_at_depth_z = a + b*z
#
#     will supply the number of cm per pixel for transverse features which are z cm from the front of the chamber.
#     For example
#
#         a + b*CHAMBER_DEPTH
#
#     would be the number of cm per pixel for transverse features at the chamber rear
#     if CHAMBER_DEPTH were in cm.
#
#     If the cameras did not move during the data taking, and if the film digitisation process did not introduce
#     scale variations between images, and if all cameras were the same distance from the chamber, then to first order
#     (e.g. ignoring lens curvature distortions) the pair (a,b) should be a property of the experiment, rather than a
#     property of a certain view, or of a certain events in a certain view. Nonetheless, (a,b) could be re-measured over
#     multiple views or multiple events to test these conditions and/or to better constrain them.
#     """
#
#     cm_coords_front_fiducial_1 = np.array(FIDUCIAL_FRONT[front_fiducial_1.name])
#     cm_coords_front_fiducial_2 = np.array(FIDUCIAL_FRONT[front_fiducial_2.name])
#     cm_coords_back_fiducial_1 = np.array(FIDUCIAL_BACK[back_fiducial_1.name])
#     cm_coords_back_fiducial_2 = np.array(FIDUCIAL_BACK[back_fiducial_2.name])
#
#     cm_displacement_between_front_fiducials = \
#         np.linalg.norm(cm_coords_front_fiducial_1 - cm_coords_front_fiducial_2)
#
#     cm_displacement_between_back_fiducials = \
#         np.linalg.norm(cm_coords_back_fiducial_1 - cm_coords_back_fiducial_2)
#
#     pixel_displacement_between_front_fiducials = \
#         np.linalg.norm(front_fiducial_1.xy - front_fiducial_2.xy)
#
#     pixel_displacement_between_back_fiducials = \
#         np.linalg.norm(back_fiducial_1.xy - back_fiducial_2.xy)
#
#     cm_per_pixel_at_chamber_front = cm_displacement_between_front_fiducials / pixel_displacement_between_front_fiducials
#     cm_per_pixel_at_chamber_back = cm_displacement_between_back_fiducials / pixel_displacement_between_back_fiducials
#
#     a = cm_per_pixel_at_chamber_front
#
#     b = (cm_per_pixel_at_chamber_back - cm_per_pixel_at_chamber_front) / CHAMBER_DEPTH
#
#     return a, b


def stereoshift(fa: Point, fb: Point, pa: Point, pb: Point):
    # stereoshift = (Delta p)/(Delta f)
    nfa = np.array(fa)
    nfb = np.array(fb)
    npa = np.array(pa)
    npb = np.array(pb)

    return np.linalg.norm(npa - npb) / np.linalg.norm(nfa - nfb)


def depth(
    fa: Fiducial,
    fb: Fiducial,
    pa: Fiducial,
    pb: Fiducial,
    reverse: bool = False,
):
    if reverse:
        # depth_p = (1 - (Delta p)/(Delta f)) * depth_f
        return (1 - stereoshift(fa.xy, fb.xy, pa.xy, pb.xy)) * CHAMBER_DEPTH
    else:
        # depth_p = (Delta p)/(Delta f) * depth_f
        return stereoshift(fa.xy, fb.xy, pa.xy, pb.xy) * CHAMBER_DEPTH


def track_parameters(line):
    slope = (line[0][1] - line[1][1]) / (line[0][0] - line[1][0])
    intercept = line[0][1] - slope * line[0][0]
    return slope, intercept


def angle(line1: np.array, line2: np.array) -> float:
    v1, v2 = np.diff(line1, axis=0)[0], np.diff(line2, axis=0)[0]
    costheta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # 2D cross product, written out by hand rather than via np.cross which newer numpy no longer
    # supports (used to just treat 2d vectors as 3D with z=0 and hand back the z-component)
    sintheta = (v1[0] * v2[1] - v1[1] * v2[0]) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arctan2(sintheta, costheta)
