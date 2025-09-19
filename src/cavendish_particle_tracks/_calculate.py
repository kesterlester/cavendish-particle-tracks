import numpy as np

from .analysis import (
    CHAMBER_DEPTH,
    FIDUCIAL_BACK,
    FIDUCIAL_FRONT,
    Fiducial,
)

Point = tuple[float, float]


def radius(a: Point, b: Point, c: Point) -> float:
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
    return np.sqrt(xc * xc + yc * yc + k)


def length(a: Point, b: Point) -> float:
    pa = np.array(a)
    pb = np.array(b)
    return np.linalg.norm(pa - pb)


def magnification(front_fiducial_1: Fiducial, front_fiducial_2: Fiducial,
                  back_fiducial_1: Fiducial, back_fiducial_2: Fiducial):
    """
    This method calculates parameters "a" and "b" which carry information about how transverse image measurements in
    pixels relate to real world displacements in cm as function of feature depth within the chamber.

    The parameter "a" has units of "cm/pixel". It is the number of cm per pixel for transverse features seem at the
    front of the chamber.

    The parameter "b" has units of "/pixel".  The parameter b is defined such that the expression

        cm_per_pixel_at_depth_z = a + b*z

    will supply the number of cm per pixel for transverse features which are z cm from the front of the chamber.
    For example

        a + b*CHAMBER_DEPTH

    would be the number of cm per pixel for transverse features at the chamber rear
    if CHAMBER_DEPTH were in cm.

    If  the cameras did not move during the data taking, and if the film digitisation process did not introduce
    scale variations between images, and if all cameras were the same distance from the chamber, then to first order
    (e.g. ignoring lens curvature distortions) the pair (a,b) should be a property of the experiment, rather than a
    property of a certain view, or of a certain events in a certain view. Nonetheless, (a,b) could be re-measured over
    multiple views or multiple events to test these conditions and/or to better constrain them.
    """

    cm_coords_front_fiducial_1 = np.array(FIDUCIAL_FRONT[front_fiducial_1.name])
    cm_coords_front_fiducial_2 = np.array(FIDUCIAL_FRONT[front_fiducial_2.name])
    cm_coords_back_fiducial_1 = np.array(FIDUCIAL_BACK[back_fiducial_1.name])
    cm_coords_back_fiducial_2 = np.array(FIDUCIAL_BACK[back_fiducial_2.name])

    cm_displacement_between_front_fiducials = \
        np.linalg.norm(cm_coords_front_fiducial_1 - cm_coords_front_fiducial_2)

    cm_displacement_between_back_fiducials = \
        np.linalg.norm(cm_coords_back_fiducial_1 - cm_coords_back_fiducial_2)

    pixel_displacement_between_front_fiducials = \
        np.linalg.norm(front_fiducial_1.xy - front_fiducial_2.xy)

    pixel_displacement_between_back_fiducials = \
        np.linalg.norm(back_fiducial_1.xy - back_fiducial_2.xy)

    cm_per_pixel_at_chamber_front = cm_displacement_between_front_fiducials / pixel_displacement_between_front_fiducials
    cm_per_pixel_at_chamber_back = cm_displacement_between_back_fiducials / pixel_displacement_between_back_fiducials

    a = cm_per_pixel_at_chamber_front

    b = (cm_per_pixel_at_chamber_back - cm_per_pixel_at_chamber_front) / CHAMBER_DEPTH

    return a, b


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
    sintheta = np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arctan2(sintheta, costheta)
