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


# Idea is to save a list of ParticleDecays as we go along, and then pandas.DataFrame(list_of_particles) does all the magic
@dataclass
class ParticleDecay:
    name: str = ""
    index: int = 0
    event_number: int = -1
    view_number: int = -1
    _r1: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _r2: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _r3: list[float] = field(default_factory=lambda: [0.0, 0.0])
    radius_px: float = -1.0
    radius_cm: float = -1.0
    _d1: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _d2: list[float] = field(default_factory=lambda: [0.0, 0.0])
    decay_length_px: float = -1.0
    decay_length_cm: float = -1.0
    magnification_a: float = -1.0
    magnification_b: float = 0.0
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

    def vars_to_show(self, calibrated=False):
        if calibrated:  # TODO: This is a mess!! Some particles may be calibrated and others not. And why not write out radius_px always in case re-anaalysis is needed later. Just always write out everything.
            return [
                "event_number",
                "name",
                "radius_cm",
                "decay_length_cm",
                "origin_vertex_depth_cm",
                "decay_vertex_depth_cm",
                "magnification",
                "phi_proton",
                "phi_pion",
            ]
        else:
            return [
                "event_number",
                "name",
                "radius_px",
                "decay_length_px",
                "origin_vertex_depth_cm",
                "decay_vertex_depth_cm",
                "magnification",
                "phi_proton",
                "phi_pion",
            ]

    def vars_to_save(self):
        """Variable to save in the output file, all for the moment"""
        vars_to_save = [var for var in self.__dict__ if var[0] != "_"]
        vars_to_save += ["origin_vertex_depth_cm", "decay_vertex_depth_cm"]
        vars_to_save += ["rpoints", "dpoints"]
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

    @property
    def origin_vertex_depth_cm(self):
        return self.origin_vertex_stereoshift_info.depth_cm

    @property
    def decay_vertex_depth_cm(self):
        return self.decay_vertex_stereoshift_info.depth_cm

    @property
    def average_depth_cm(self):
        return self.origin_vertex_stereoshift_info.depth_cm

    @property
    def magnification(self):
        return self.magnification_a + self.magnification_b * self.average_depth_cm

    def calibrate(self) -> None:
        self.radius_cm = self.magnification * self.radius_px
        self.decay_length_cm = self.magnification * self.decay_length_px

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
