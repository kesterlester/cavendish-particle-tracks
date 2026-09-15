from dataclasses import dataclass, field

import numpy as np

CHAMBER_DEPTH = 31.6  # cm

FIDUCIAL_FRONT = {
    "C'": [0.0, 0.0],
    "F'": [14.97, -8.67],
    "B'": [15.00, 8.66],
    "D'": [29.91, -0.07],
    "E'": [np.nan, np.nan], # NaN as unknown pos.
    "A'": [np.nan, np.nan], # NaN as unknown pos.
}  # cm

FIDUCIAL_BACK = {
    "C": [-0.02, 0.01],
    "F": [14.95, -8.63],
    "B": [14.92, 8.67],
    "D": [29.90, 0.02],
    "E": [-14.96, -8.62],
    "A": [-15.00, 8.68],
}  # cm

FIDUCIAL_NAMES = frozenset(FIDUCIAL_FRONT) | frozenset(FIDUCIAL_BACK)


@dataclass
class GenericFiducialTemplate:
    """The reusable template fiducial positions for one camera view - dragged into place
    once during general calibration, shared across every event, used as the starting point
    before cloning a fiducial into a specific photo's own FiducialViewData.
    """

    positions: dict = field(default_factory=dict)

    def set_position(self, name: str, xy: list) -> None:
        if name not in FIDUCIAL_NAMES:
            raise ValueError(f"{name!r} is not a recognised fiducial name")
        self.positions[name] = xy

    def get_position(self, name: str):
        return self.positions.get(name)


@dataclass
class FiducialViewData:
    """Calibration fiducial stamps for one camera view, for one specific event - the counterpart
    to ViewData, but holding an arbitrary named set of fiducials rather than a fixed origin/decay/track
    shape, since a photo can have anywhere from zero to all twelve stamped for it.
    """

    stamped: dict = field(default_factory=dict)

    def stamp(self, name: str, xy: list) -> None:
        if name not in FIDUCIAL_NAMES:
            raise ValueError(f"{name!r} is not a recognised fiducial name")
        self.stamped[name] = xy

    def unstamp(self, name: str) -> None:
        self.stamped.pop(name, None)

    def get(self, name: str):
        return self.stamped.get(name)


@dataclass
class CalibrationData:
    """All calibration fiducial data for one loaded dataset: one generic (event-independent)
    template per camera view, plus one FiducialViewData per (event, view) combination, created
    the first time something is actually stamped there.
    """

    generic_templates: list = field(
        default_factory=lambda: [GenericFiducialTemplate() for _ in range(3)]
    )
    event_views: dict = field(default_factory=dict)

    def stamp(self, event: int, view: int, name: str, xy: list) -> None:
        key = (event, view)
        if key not in self.event_views:
            self.event_views[key] = FiducialViewData()
        self.event_views[key].stamp(name, xy)

    def unstamp(self, event: int, view: int, name: str) -> None:
        key = (event, view)
        if key in self.event_views:
            self.event_views[key].unstamp(name)

    def get_stamp(self, event: int, view: int, name: str):
        key = (event, view)
        if key not in self.event_views:
            return None
        return self.event_views[key].get(name)


VTX_NONE = "_"
VTX_ORIGIN = "O"
VTX_DECAY = "D"
VTX_TRACK = "R"
VTX_ANGLE = "A"
VTX_NOT_APPLICABLE = "."

# These are approximate locations for an interesting point in event XXXX and three pairs of fiducials in each view.
debug_points_view_0_calibration_layer = np.array([[1241.8771528 , 4458.80208973],
       [ 547.27520267, 5571.46964481],
       [ 407.311707  , 5217.56602448],
       [1398.43257104, 4105.67068367],
       [1054.91736458, 4083.25071514],
       [1420.60685555, 7057.13841692],
       [1073.42405181, 6349.22310964]])
debug_points_view_1_calibration_layer = np.array([[1492.08818685, 4420.03076255],
       [ 563.13304728, 5126.15190679],
       [ 769.77696761, 5365.1043757 ],
       [1413.74238316, 3668.63590105],
       [1417.98929182, 4235.67176953],
       [1425.5264684 , 6605.98654768],
       [1427.87846858, 6491.42811567]])
debug_points_view_2_calibration_layer = np.array([[1513.9479341 , 4522.05154178],
       [ 353.51428451, 5631.43486332],
       [ 901.86361306, 5273.77193882],
       [1206.91574631, 4175.58159963],
       [1551.4374658 , 4145.4012863 ],
       [1215.94063266, 7113.32379595],
       [1559.19637273, 6401.92405786]])
debug_point_labels_all_calibration_layers = ['point', "B'", 'B', "C'", 'C', "D'", 'D']

"""
napari.current_viewer().layers[-3]
Out[32]: <Points layer 'View 0 calibration layer' at 0x11eae27e0>

napari.current_viewer().layers[-2]
Out[33]: <Points layer 'View 1 calibration layer' at 0x1202e11c0>

napari.current_viewer().layers[-1]
Out[34]: <Points layer 'View 2 calibration layer' at 0x141a7c1d0>

napari.current_viewer().layers[-3].data
Out[28]:
array([[1241.8771528 , 4458.80208973],
       [ 547.27520267, 5571.46964481],
       [ 407.311707  , 5217.56602448],
       [1398.43257104, 4105.67068367],
       [1054.91736458, 4083.25071514],
       [1420.60685555, 7057.13841692],
       [1073.42405181, 6349.22310964]])

napari.current_viewer().layers[-2].data
Out[29]:
array([[1492.08818685, 4420.03076255],
       [ 563.13304728, 5126.15190679],
       [ 769.77696761, 5365.1043757 ],
       [1413.74238316, 3668.63590105],
       [1417.98929182, 4235.67176953],
       [1425.5264684 , 6605.98654768],
       [1427.87846858, 6491.42811567]])

napari.current_viewer().layers[-1].data
Out[30]:
array([[1513.9479341 , 4522.05154178],
       [ 353.51428451, 5631.43486332],
       [ 901.86361306, 5273.77193882],
       [1206.91574631, 4175.58159963],
       [1551.4374658 , 4145.4012863 ],
       [1215.94063266, 7113.32379595],
       [1559.19637273, 6401.92405786]])
"""

TYPICAL_IMAGE_LONG_SIZE_PIX = 8377 # This is just typical. No guarantee that any particular image has this size!
TYPICAL_IMAGE_SHORT_SIZE_PIX = 2753 # This is just typical. No guarantee that any particular image has this size!


VIEW_NAMES = ["view1", "view2", "view3"]

EXPECTED_PROCESSES_NICE_TO_ASCII = {
    "Add process": "Add process",
    "Σ⁺ ⇨ p + π⁰": "Sigma+_to_p_pi0",
    "Σ⁺ ⇨ n + π⁺": "Sigma+_to_n_pi+",
    "Σ⁻ ⇨ n + π⁻": "Sigma-_to_p_pi-",
    "Λ⁰ ⇨ p + π⁻": "Lambda0_to_p_pi-",
    "Λ⁰ ⇨ n + π⁰": "Lambda0_to_m_pi0",
}
EXPECTED_PROCESSES_NICE = [
    key for key in EXPECTED_PROCESSES_NICE_TO_ASCII
]


@dataclass
class Fiducial:
    name: str = ""
    x: float = -1.0e6
    y: float = -1.0e6

    def __str__(self):
        return f"Fiducial(name={self.name}; x={self.x}; y={self.y})"

    @property
    def xy(self):
        return np.array([self.x, self.y])

    @xy.setter
    def xy(self, point):
        self.x = point[0]
        self.y = point[1]


@dataclass
class StereoshiftInfo:
    name: str = ""
    _sf1: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _sf2: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _sp1: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _sp2: list[float] = field(default_factory=lambda: [0.0, 0.0])
    shift_fiducial: float = 0.0
    shift_point: float = 0.0
    stereoshift: float = -1.0
    depth_cm: float = -1.0

    @property
    def spoints(self):
        return [self._sf1, self._sf2, self._sp1, self._sp2]

    @spoints.setter
    def spoints(self, values):
        for i, point in enumerate(self.spoints):
            point[0] = values[i][0]
            point[1] = values[i][1]

    def __str__(self):
        mystring = f"StereoshiftInfo(name={self.name}; "
        for name, point in zip(
            ["sf1", "sf2", "sp1", "sp2"], [self._sf1, self._sf2, self._sp1, self._sp2]
        ):
            x, y = point
            mystring += f"{name}=[{x} {y}]; "
        mystring += f"shift_fiducial={self.shift_fiducial}; "
        mystring += f"shift_point={self.shift_point}; "
        mystring += f"stereoshift={self.stereoshift}; "
        mystring += f"depth_cm={self.depth_cm})"
        return mystring


@dataclass
class ViewData:
    """Everything we've measured in one camera view for one process.

    origin/decay are [x, y] pixel coordinates once someone has placed them,
    None until then. track_points holds the points used for a radius fit
    (up to 3). radius_px/length_px hold the derived results - go through
    the setter methods below (rather than poking the fields directly) and
    they'll stay in sync automatically as points are placed, moved, or
    cleared.
    """

    origin: list[float] | None = None
    decay: list[float] | None = None
    track_points: list[list[float]] = field(default_factory=list)
    radius_px: float | None = None
    length_px: float | None = None
    decay_angle_lines: list[list[list[float]]] | None = None
    phi_proton: float | None = None
    phi_pion: float | None = None

    def set_origin(self, point: list[float] | None) -> None:
        self.origin = point
        self._recompute_length()

    def set_decay(self, point: list[float] | None) -> None:
        self.decay = point
        self._recompute_length()

    def add_track_point(self, point: list[float]) -> None:
        self.track_points.append(point)
        self._recompute_radius()

    def clear_track_points(self) -> None:
        self.track_points = []
        self._recompute_radius()

    def set_track_points(self, points: list[list[float]]) -> None:
        self.track_points = points
        self._recompute_radius()

    def _recompute_length(self) -> None:
        # imported here rather than at the top of the file - _calculate.py
        # imports from analysis.py, so importing it up top would be circular
        from ._calculate import length

        if self.origin is not None and self.decay is not None:
            self.length_px = length(self.origin, self.decay)
        else:
            self.length_px = None

    def _recompute_radius(self) -> None:
        from ._calculate import radius

        if len(self.track_points) == 3:
            self.radius_px = radius(*self.track_points)
        else:
            self.radius_px = None

    def set_decay_angle_lines(self, lines: list[list[list[float]]] | None) -> None:
        """lines is [lambda_line, proton_line, pion_line], each a 2-point [start, end] pair - the
        same shape the Decay Angles Tool already drags around on screen. Pass None to clear a
        previous measurement.
        """
        self.decay_angle_lines = lines
        self._recompute_decay_angles()

    def _recompute_decay_angles(self) -> None:
        from ._calculate import angle

        if self.decay_angle_lines is None:
            self.phi_proton = None
            self.phi_pion = None
            return

        lambda_line, proton_line, pion_line = self.decay_angle_lines
        # The Lambda travels towards the decay vertex, not away from it, so its line needs reversing
        # before comparing directions - same convention the (retired) decay angles dialog used.
        lambda_line_reversed = list(reversed(lambda_line))
        self.phi_proton = angle(lambda_line_reversed, proton_line)
        self.phi_pion = angle(lambda_line_reversed, pion_line)


# Idea is to save a list of ParticleDecays as we go along, and then pandas.DataFrame(list_of_particles) does all the magic
@dataclass
class ParticleDecay:
    name: str = ""
    index: int = 0
    event_number: int = -1
    view_number: int = -1

    # New home for per-view data (one entry per camera view). Nothing reads
    # or writes this yet - the old fields below still do all the real work
    # until we migrate _main_widget.py and _calibration_manager.py over to
    # this in the following steps.
    views: list[ViewData] = field(
        default_factory=lambda: [ViewData(), ViewData(), ViewData()]
    )

    _r1: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _r2: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _r3: list[float] = field(default_factory=lambda: [0.0, 0.0])
    radius_px: float = -1.0
    # radius_cm: float = -1.0
    _d1: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _d2: list[float] = field(default_factory=lambda: [0.0, 0.0])
    decay_length_px: float = -1.0
    #decay_length_cm: float = -1.0
    #magnification_a: float = -1.0
    #magnification_b: float = 0.0
    origin_vertex_stereoshift_info: StereoshiftInfo = field(
        default_factory=StereoshiftInfo
    )
    decay_vertex_stereoshift_info: StereoshiftInfo = field(
        default_factory=StereoshiftInfo
    )
    phi_proton: float = -100
    phi_pion: float = -100

    origin_v0_x: str = ""
    origin_v0_y: str = ""
    origin_v1_x: str = ""
    origin_v1_y: str = ""
    origin_v2_x: str = ""
    origin_v2_y: str = ""
    decay_v0_x: str = ""
    decay_v0_y: str = ""
    decay_v1_x: str = ""
    decay_v1_y: str = ""
    decay_v2_x: str = ""
    decay_v2_y: str = ""

    @property
    def saved_vertices(self) -> str:
        """A quick-glance summary of what's actually been measured, computed fresh from self.views
        every time rather than tracked separately - so it can never drift out of sync with the real
        data the way a manually-updated field could. One 4-character block per view: O/D/T mark
        whether origin/decay/a 3-point radius fit have been saved in that view; A marks whether decay
        angles have been saved for the process as a whole. A '.' means that particular measurement
        doesn't apply to this process type at all - only Lambda0 -> p + pi- has two charged daughters
        to compare angles between, so every other process type shows '.' there.
        """
        angles_applicable = self.index == 4
        blocks = []
        for view in self.views:
            origin_char = VTX_ORIGIN if view.origin is not None else VTX_NONE
            decay_char = VTX_DECAY if view.decay is not None else VTX_NONE
            track_char = VTX_TRACK if len(view.track_points) == 3 else VTX_NONE
            if angles_applicable:
                angle_char = VTX_ANGLE if view.phi_proton is not None else VTX_NONE
            else:
                angle_char = VTX_NOT_APPLICABLE
            blocks.append(origin_char + decay_char + track_char + angle_char)
        return " ".join(blocks)

    def vars_to_show(self, calibrated=False):
        return [
            "event_number",
            "name",
            "radius_px",
            "decay_length_px",
            #"origin_vertex_depth_cm",
            #"decay_vertex_depth_cm",
            # "magnification",
            "phi_proton",
            "phi_pion",
            "saved_vertices",
        ]

    def vars_to_save(self):
        """Variable to save in the output file, all for the moment"""
        vars_to_save = [var for var in self.__dict__ if var[0] != "_"]
        #vars_to_save += ["origin_vertex_depth_cm", "decay_vertex_depth_cm"]
        vars_to_save += ["rpoints", "dpoints", "saved_vertices"]
        # vars_to_save += ["origin_v0_x"]
        # vars_to_save += ["origin_v0_y"]
        # vars_to_save += ["origin_v1_x"]
        # vars_to_save += ["origin_v1_y"]
        # vars_to_save += ["origin_v2_x"]
        # vars_to_save += ["origin_v2_y"]
        # vars_to_save += ["decay_v0_x"]
        # vars_to_save += ["decay_v0_y"]
        # vars_to_save += ["decay_v1_x"]
        # vars_to_save += ["decay_v1_y"]
        # vars_to_save += ["decay_v2_x"]
        # vars_to_save += ["decay_v2_y"]

        return vars_to_save

    @property
    def rpoints(self):
        return [self._r1, self._r2, self._r3]

    @rpoints.setter
    def rpoints(self, values):
        for i, point in enumerate(self.rpoints):
            point[0] = values[i][0]
            point[1] = values[i][1]

    @property
    def dpoints(self):
        return [self._d1, self._d2]

    @dpoints.setter
    def dpoints(self, values):
        for i, point in enumerate(self.dpoints):
            point[0] = values[i][0]
            point[1] = values[i][1]

    #@property
    #def origin_vertex_depth_cm(self):
    #    return self.origin_vertex_stereoshift_info.depth_cm

    #@property
    #def decay_vertex_depth_cm(self):
    #    return self.decay_vertex_stereoshift_info.depth_cm

    @property
    def average_depth_cm(self):
        return self.origin_vertex_stereoshift_info.depth_cm

    #@property
    #def magnification(self):
    #    return self.magnification_a + self.magnification_b * self.average_depth_cm

    def calibrate(self) -> None: # TODO: Try to remove this method
        pass
        #self.radius_cm = self.magnification * self.radius_px
        #self.decay_length_cm = self.magnification * self.decay_length_px

    def to_csv(self):
        mystring = ""
        for var in self.vars_to_save():
            if var in ["rpoints", "dpoints"]:
                mystring += "["
                for point in getattr(self, var):
                    x, y = point
                    mystring += f"[{x} {y}]; "
                mystring = mystring[0:-2] + "],"
            elif var == "name":
                nice_name = str(getattr(self, var))
                if nice_name in EXPECTED_PROCESSES_NICE_TO_ASCII:
                    ascii_name = EXPECTED_PROCESSES_NICE_TO_ASCII[nice_name]
                    name_to_write = ascii_name
                else:
                    name_to_write = nice_name
                mystring += name_to_write + ","
            else:
                mystring += str(getattr(self, var)) + ","
        return mystring[0:-1] + "\n"

@dataclass
class CalibrationRow:
    """One event's calibration fiducial stamps, represented so it can sit alongside
    ParticleDecay rows in the same process table and the same saved CSV - this is
    calibration data for an event, not a decay measurement. index=-1 marks it as
    "not a real process type" (real EXPECTED_PROCESSES indices are 1-5), which
    existing index-based checks already fail harmlessly against without needing to
    know CalibrationRow exists.
    """

    name: str = "Calibration"
    index: int = -1
    event_number: int = -1
    views: list = field(
        default_factory=lambda: [FiducialViewData(), FiducialViewData(), FiducialViewData()]
    )

    @property
    def saved_vertices(self) -> str:
        """Same visual language as ParticleDecay.saved_vertices - one
        space-separated block per view - but each block lists the actual
        fiducial names stamped in that view (joined with '+', not ',',
        to avoid raw commas breaking the naive CSV writer below) rather
        than O/D/R/A flags, since that's what's meaningful here.
        """
        blocks = []
        for view in self.views:
            names = sorted(view.stamped.keys())
            blocks.append("/".join(names) if names else VTX_NONE)
        return " ".join(blocks)

    def vars_to_save(self):
        # Deliberately mirrors ParticleDecay's own column list exactly, asked for fresh
        # each time rather than duplicated by hand - the save code writes the CSV header
        # from only the first row, then trusts every other row's to_csv() to line up with
        # it, so this has to match exactly or a calibration row would silently misalign
        # every column after it.
        return ParticleDecay().vars_to_save()

    def to_csv(self) -> str:
        values = []
        for column in self.vars_to_save():
            if column == "name":
                values.append(self.name)
            elif column == "event_number":
                values.append(str(self.event_number))
            elif column == "saved_vertices":
                values.append(self.saved_vertices)
            else:
                values.append("")
        return ",".join(values) + "\n"