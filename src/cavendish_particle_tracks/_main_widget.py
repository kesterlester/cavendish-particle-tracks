"""
This module contains the main Cavendish Particle Tracks widget.

It's the students' home widget, that contains the table of particle decays, and
the buttons to perform all analysis calculations, and to export (save) the data
for further analysis.
"""

import glob
import logging
import os
import pickle
import tempfile
import warnings
from logging.handlers import RotatingFileHandler

import dask.array
import napari
import numpy as np
from dask_image.imread import imread
from vispy.color import Color
from qtpy.QtWidgets import (
    QAbstractItemView,
    QAction,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtCore import Qt, QSettings
from ._settings import get_bypass, get_shuffling_seed
# from ._stereoshift_dialog import StereoshiftDialog
from ._calibration_manager import CalibrationManager, GENERIC_CALIBRATION_LAYER_NAME, PER_IMAGE_CALIBRATION_LAYER_NAME
from .intercept_close import InterceptClose
from .analysis import EXPECTED_PROCESSES_NICE, VIEW_NAMES, ParticleDecay, CalibrationRow, FiducialViewData, SavedSession, CSV_COLUMNS, round_px, round_angle, load_csv_session
from ._calculate import origin_decay_arrow, radius_arc_points

ENABLE_MAG = False
# Session files are CSV-only for now - flip this back to True to restore .pkl support.
# Underlying save/load logic is untouched, just gated behind this flag.
ENABLE_PICKLE = False

MEASUREMENTS_LAYER_NAME = "Radii and Lengths"
OTHER_PROCESSES_LAYER_NAME = "Other Processes (view only)"
ANGLES_LAYER_NAME = "Decay Angles Tool"
IMAGE_LAYER_NAME = "Bubble Chamber Data"
RADIUS_ARC_LAYER_NAME = "Radius Arc"
ORIGIN_DECAY_ARROW_LAYER_NAME = "Origin-Decay Arrow"

# Shared between _restyle_measurement_points, _sync_other_processes_layer and the radius-arc
# layer, so a point's ring colour and the arc drawn through a radius fit always agree.
LENGTH_COLOR = "cornflowerblue"
RADIUS_COLOR = "mediumorchid"
BOTH_COLOR = "slateblue"
ARROW_COLOR = "cyan"

_singleton_instance = None

# Opt-in interaction trace for diagnosing bugs that only show up in a live, real-mouse GUI
# session (event ordering / timing that a headless test can't reproduce) - see _debug_trace().
# Off by default (zero cost: _debug_trace() is a no-op until this is turned on). Enable by
# setting the environment variable below before launching, e.g.:
#     CPT_DEBUG_TRACE=1 ./launch_debug.py
# Writes to DEBUG_LOG_PATH, printed once to the terminal on startup when enabled. Bounded to a
# couple of MB total (a rotating file, not an ever-growing one) so it's safe to just leave on.
DEBUG_TRACE_ENV_VAR = "CPT_DEBUG_TRACE"
DEBUG_LOG_PATH = os.path.join(tempfile.gettempdir(), "cavendish_particle_tracks_debug_trace.log")

_debug_trace_logger = logging.getLogger("cavendish_particle_tracks.debug_trace")
_debug_trace_logger.propagate = False


def _maybe_enable_debug_trace() -> None:
    """Called once, from get_singleton() below. Idempotent, so it's harmless if the widget is
    ever constructed more than once in the same process."""
    if not os.environ.get(DEBUG_TRACE_ENV_VAR) or _debug_trace_logger.handlers:
        return
    handler = RotatingFileHandler(DEBUG_LOG_PATH, maxBytes=1_000_000, backupCount=2)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _debug_trace_logger.addHandler(handler)
    _debug_trace_logger.setLevel(logging.DEBUG)
    print(f"[cavendish-particle-tracks] {DEBUG_TRACE_ENV_VAR} is set - debug trace -> {DEBUG_LOG_PATH}")


def _debug_trace(msg: str) -> None:
    if _debug_trace_logger.handlers:
        _debug_trace_logger.debug(msg)


class _NumericTableWidgetItem(QTableWidgetItem):
    """A process-table cell that sorts by numeric value rather than display text. Plain
    QTableWidgetItem sorts lexicographically (e.g. event 10 sorts before event 2), which is
    wrong for every numeric column in this table (event_number, radius_px, ...). Falls back to
    the default text comparison for anything that doesn't parse as a float - a process with no
    radius fit yet shows a blank cell, and blanks should sort together predictably rather than
    raising or silently comparing as zero.
    """

    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except (ValueError, TypeError):
            return super().__lt__(other)


def _as_xy(point) -> tuple[float, float]:
    """Round a 2D point to a hashable, comparison-stable (x, y) tuple - used wherever canvas
    point positions need to be compared or grouped by "same place", since raw floats coming
    back from napari can differ in the last bit or two from what was originally written.
    """
    return (round(float(point[0]), 6), round(float(point[1]), 6))


def _measurement_roles(view_data) -> list[tuple[str, list[float]]]:
    """Every named role a point on the measurement layer can currently fill for one view,
    paired with its stored coordinate: the origin vertex, the decay vertex, and up to 3
    numbered radius-fit track points. Roles with nothing stored yet are omitted.
    """
    roles = [("origin", view_data.origin), ("decay", view_data.decay)]
    roles += [(f"track{i}", point) for i, point in enumerate(view_data.track_points)]
    return [(role, point) for role, point in roles if point is not None]


def _role_symbol(is_origin: bool, is_decay: bool) -> str:
    """The point SHAPE used to tell an origin vertex from a decay vertex apart at a glance -
    unlike colour or text, unaffected by colourblindness, and (like the point's own size) capped
    rather than growing without bound at high zoom, so it never obscures the underlying image.

    Both origin and decay use "ring" (a donut) - a filled triangle was tried for decay first, but
    found live to visually clash with the O-D arrow's own triangular tip: the two triangles point
    in whatever directions their own geometry dictates, which often isn't the arrow's direction,
    reading as confusing rather than informative right where the arrowhead already sits. See
    _role_face_color for how origin and decay are told apart now that they share a shape.
    "diamond" is the rare degenerate case of a single point being both at once (a decay length of
    exactly zero, so no arrow is ever drawn for it - see origin_decay_arrow - meaning there's no
    arrow tip for a distinct shape to clash with here). Anything else (a bare point, or one that's
    only part of a radius fit) stays the plain default "disc", unchanged.
    """
    if is_origin and is_decay:
        return "diamond"
    if is_origin or is_decay:
        return "ring"
    return "disc"


def _role_face_color(is_origin: bool, is_decay: bool) -> str:
    """The point FILL colour additionally distinguishing origin from decay, now that both use the
    same "ring" shape (see _role_symbol) - yellow for origin, blue for decay (chosen once the
    arrow itself became cyan, so all three read as distinct against the grey film background and
    against each other). The rare "both at once" case reuses origin's yellow, arbitrarily but
    consistently; anything else keeps the plain default white fill.
    """
    if is_origin:
        return "yellow"
    if is_decay:
        return "blue"
    return "white"


def _faded_color(name: str, alpha: float = 0.35) -> np.ndarray:
    """The same named colour, but at reduced opacity - used to draw an OTHER process's radius arc
    / origin-decay arrow noticeably fainter than the selected process's own (bold, full-alpha)
    decorator, the same "theirs vs mine" distinction _sync_other_processes_layer already makes for
    points via whole-layer opacity. Shapes layers don't have a per-shape opacity, only a per-shape
    RGBA edge/face colour, so the fade has to be baked into the colour itself here.
    """
    rgba = np.array(Color(name).rgba)
    rgba[3] = alpha
    return rgba


def get_singleton(viewer=None, docking_area: str = "bottom", data_folder=None):
    """Return the singleton ParticleTracksWidget, creating it if necessary."""
    global _singleton_instance
    _maybe_enable_debug_trace()
    if _singleton_instance is None:
        _singleton_instance = ParticleTracksWidget(
            viewer, docking_area=docking_area, data_folder=data_folder
        )
    return _singleton_instance

@InterceptClose
class ParticleTracksWidget(QWidget):
    """Widget containing a simple table of points and track radii per image."""

    layer_measurements: napari.layers.Points

    def dirty_things(self):
        dirty_things = []
        # try catch as could get a callback before we are ready!
        try:

            print(f"DIRTY_THINGS method of _main_widget finds that")
            print(f"{self.data=}")
            print("and")
            print(f"{self._data_at_last_save=}")
            table_is_dirty = (self.data != self._data_at_last_save)
            if table_is_dirty:
                dirty_things.append("decay table")
        except:
            # Attributes are missing so we are not even constructed yet!
            pass

        dirty_things = dirty_things + self.calibration_manager.dirty_things()

        return dirty_things

    def __init__(
        self,
        napari_viewer: napari.Viewer,
        docking_area: str = "bottom",
        data_folder = None,
    ):
        super().__init__()
        self.viewer: napari.Viewer = napari_viewer

        # In normal operation: the user is forced to load data before they can do anything.
        self.bypass_force_load_data = get_bypass()
        # "Load images" persists across restarts (via QSettings - a plist on macOS, the registry
        # on Windows, a config file on Linux, handled automatically) since every student loads
        # from the same shared data folder, so remembering it is a genuine convenience with no
        # downside. Save/load process table deliberately do NOT use QSettings - they're
        # per-student work, so they use a plain in-memory attribute instead, cleared automatically
        # the moment the app closes.
        self._settings = QSettings("CavendishLab", "ParticleTracks")
        self._last_load_process_table_dir = "./"
        self._last_save_process_table_dir = "./"

        self.docking_area = docking_area

        self.shuffling_seed = get_shuffling_seed(fallback=1)

        # define QtWidgets
        self.load_button = QPushButton("Load images")
        self.particle_decays_menu = QComboBox()
        self.particle_decays_menu.addItems(EXPECTED_PROCESSES_NICE)
        self.particle_decays_menu.setCurrentIndex(0)
        self.particle_decays_menu.currentIndexChanged.connect(self._on_click_new_process)
        self.delete_process = QPushButton("Delete process")
        self.show_track_vertices_checkbox = QCheckBox("Show radius points")
        self.show_track_vertices_checkbox.setChecked(True)
        self.show_origin_decay_checkbox = QCheckBox("Show length points")
        self.show_origin_decay_checkbox.setChecked(True)
        self.show_all_processes_checkbox = QCheckBox("Show all processes")
        self.show_all_processes_checkbox.setChecked(False)
        self.show_decorators_checkbox = QCheckBox("Show arcs/arrows")
        self.show_decorators_checkbox.setChecked(True)
        self.show_fiducials_checkbox = QCheckBox("Show fiducial markers")
        self.show_fiducials_checkbox.setChecked(True)
        self.show_per_view_columns_checkbox = QCheckBox("Show per-view breakdown")
        self.show_per_view_columns_checkbox.setChecked(False)
        self.show_per_view_columns_checkbox.stateChanged.connect(
            lambda _: self._set_table_visible_vars(self.show_per_view_columns_checkbox.isChecked())
        )
        # Created here (early), not down near the rest of the Layers panel setup - this needs to
        # exist before set_UI_image_loaded() runs for the first time, a few lines below, the same
        # constraint the checkbox it replaces was already satisfying by being created this early.
        self.decay_angles_nav_button = QPushButton("Decay Angles (K)")
        self.decay_angles_nav_button.setCheckable(True)
        self.decay_angles_nav_button.setEnabled(False)
        # self.stereoshift_button = QPushButton("Stereoshift")
        self.save_data_button = QPushButton("Save process table")
        self.load_data_button = QPushButton("Load process table")

        # setup particle table
        self.table = self._set_up_table()
        self._set_table_visible_vars(False)
        self.table.selectionModel().selectionChanged.connect(
            self._on_row_selection_changed
        )
        # selectionChanged only fires on an actual change, so it never runs on a same-row
        # re-click - table.clicked fires on every click regardless, closing that gap.
        self.table.clicked.connect(self._return_focus_to_canvas)
        # Apply magnification disabled until the magnification parameters are computed
        #self.apply_magnification_button = QRadioButton("Apply magnification")
        #self.apply_magnification_button.setEnabled(False)

        # connect callbacks
        self.load_button.clicked.connect(self._on_click_load_data)
        self.delete_process.clicked.connect(self._on_click_delete_process)
        self.show_track_vertices_checkbox.stateChanged.connect(lambda _: self._restyle_measurement_points())
        self.show_origin_decay_checkbox.stateChanged.connect(lambda _: self._restyle_measurement_points())
        self.show_track_vertices_checkbox.stateChanged.connect(lambda _: self._sync_other_processes_layer())
        self.show_origin_decay_checkbox.stateChanged.connect(lambda _: self._sync_other_processes_layer())
        self.show_all_processes_checkbox.stateChanged.connect(lambda _: self._sync_other_processes_layer())
        # The decorators (radius arc, O-D arrow) aren't drawn from _restyle_measurement_points'
        # own early-return path when the selected process has no points of its own yet, so - like
        # show_all_processes_checkbox above - these are wired directly to the refresh calls rather
        # than relying solely on _restyle_measurement_points to reach them.
        self.show_decorators_checkbox.stateChanged.connect(lambda _: self._refresh_radius_arc())
        self.show_decorators_checkbox.stateChanged.connect(lambda _: self._refresh_origin_decay_arrow())
        self.show_all_processes_checkbox.stateChanged.connect(lambda _: self._refresh_radius_arc())
        self.show_all_processes_checkbox.stateChanged.connect(lambda _: self._refresh_origin_decay_arrow())
        #self.stereoshift_button.clicked.connect(self._on_click_stereoshift)
        #self.apply_magnification_button.toggled.connect(
        #    self._on_click_apply_magnification
        #)
        self.save_data_button.clicked.connect(self._on_click_save)
        self.load_data_button.clicked.connect(self._on_click_load)

        # TODO: find which of these works
        # https://napari.org/stable/gallery/custom_mouse_functions.html
        # self.viewer.mouse_press.callbacks.connect(self._on_mouse_press)
        # self.viewer.events.mouse_press(self._on_mouse_click)

        if self.docking_area == "bottom":
            self.buttonbox = QGridLayout()
            self.buttonbox.addWidget(self.load_data_button, 0, 0)
            self.buttonbox.addWidget(self.save_data_button, 0, 1)
            self.buttonbox.addWidget(self.particle_decays_menu, 1, 0)
            self.buttonbox.addWidget(self.delete_process, 1, 1)
            self.buttonbox.addWidget(self.load_button, 2, 0, 1, 2)
            self.buttonbox.addWidget(self.show_track_vertices_checkbox, 3, 0)
            self.buttonbox.addWidget(self.show_origin_decay_checkbox, 3, 1)
            self.buttonbox.addWidget(self.show_fiducials_checkbox, 4, 0)
            self.buttonbox.addWidget(self.show_all_processes_checkbox, 4, 1)
            self.buttonbox.addWidget(self.show_decorators_checkbox, 5, 0)
            # self.buttonbox.addWidget(self.stereoshift_button, 5, 0)
            # self.buttonbox.addWidget(self.apply_magnification_button, 4, 1)

            self.buttonbox.setColumnStretch(0, 1)
            self.buttonbox.setColumnStretch(1, 1)

            table_container = QWidget()
            table_container_layout = QVBoxLayout()
            table_container_layout.setContentsMargins(0, 0, 0, 0)
            table_container.setLayout(table_container_layout)
            table_container_layout.addWidget(self.show_per_view_columns_checkbox)
            table_container_layout.addWidget(self.table)

            layout_outer = QHBoxLayout()
            self.setLayout(layout_outer)
            layout_outer.addLayout(self.buttonbox)
            self.layout().addWidget(table_container)
            layout_outer.setStretch(0, 0)  # button panel stays at its natural size
            layout_outer.setStretch(1, 1)  # table absorbs any extra width

        else:
            self.buttonbox = QVBoxLayout()
            self.buttonbox.addWidget(self.load_button)
            self.buttonbox.addWidget(self.particle_decays_menu)
            self.buttonbox.addWidget(self.delete_process)
            self.buttonbox.addWidget(self.show_track_vertices_checkbox)
            self.buttonbox.addWidget(self.show_origin_decay_checkbox)
            self.buttonbox.addWidget(self.show_fiducials_checkbox)
            self.buttonbox.addWidget(self.show_all_processes_checkbox)
            self.buttonbox.addWidget(self.show_decorators_checkbox)
            self.buttonbox.addWidget(self.show_per_view_columns_checkbox)
            self.buttonbox.addWidget(self.table)
            # self.buttonbox.addWidget(self.apply_magnification_button)
            # self.buttonbox.addWidget(self.stereoshift_button)
            self.buttonbox.addWidget(self.save_data_button)
            self.buttonbox.addWidget(self.load_data_button)
            self.setLayout(self.buttonbox)

        # Disable some native napari controls
        # NB: Both of these will break in napari 0.6.0
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            # Disable native napari layer controls - show again on closing this widget (hide).
            self.viewer.window._qt_viewer.layerButtons.hide()
            # Disable viewer buttons, prevents accidental crash due to viewing image stack side on.
            self.viewer.window._qt_viewer.viewerButtons.hide()

        self.set_UI_image_loaded(False, self.bypass_force_load_data)
        # TODO: include self.stsh in the logic, depending on what it actually ends up doing

        # Data analysis
        self.data: list[ParticleDecay] = []
        # Monotonically increasing - see _assign_row_id. Never reused, even across deletes, so a
        # stale id from a just-deleted row can never accidentally match a later, unrelated one.
        self._next_row_id = 0
        # (column, order) of the table's last explicit header-click sort, or None before the
        # first click - see _on_table_header_clicked for why this can't just be read back from
        # the header's own sortIndicatorSection()/sortIndicatorOrder() instead.
        self._table_sort_state: tuple[int, "Qt.SortOrder"] | None = None
        import copy
        self._data_at_last_save = copy.deepcopy(self.data) # Need deepcopy as otherwise changes within ParticleData objects are not spotted!

        # self.mag_a = -1.0
        # self.mag_b = 0.0

        # Dialog pointers to reuse
        # self.stereoshift_dlg: StereoshiftDialog | None = None

        @self.viewer.layers.events.connect
        def _on_layerlist_changed(event):
            """When the layer list changes, update the button availability"""
            self.set_button_availability()

        # calibration_manager must exist before any data folder is loaded - _load_data_from()
        # calls into it (e.g. rebuild_generic_layer_for_event_count()) to size the generic
        # calibration layers once the real per-event image count is known. This ordering used to
        # be harmless (main's _load_data_from never touched calibration_manager), but became load
        # -bearing once calibration awareness was added here - a caller that supplies data_folder=
        # to the constructor (rather than loading via the Load button after the widget is fully
        # built) would otherwise hit an AttributeError.
        self._last_synced_dims = None
        self.calibration_manager = CalibrationManager(self, self.viewer)

        if data_folder is not None:
            self._load_data_from(data_folder)

        self.viewer.dims.events.current_step.connect(self._sync_measurement_layer_to_selected_process)
        # Connected here, not up where the checkbox was created, since calibration_manager doesn't
        # exist yet at that point - connecting any earlier would crash the moment the checkbox's initial
        # checked state gets set.
        self.show_fiducials_checkbox.stateChanged.connect(
            lambda _: self.calibration_manager.set_calibration_layer_visibility_and_focus(
                self.show_fiducials_checkbox.isChecked(), False
            )
        )

        # A dedicated panel for jumping straight to the 3 layers someone actually interacts with
        # while measuring/calibrating, so switching between them doesn't require the native layer
        # list at all (12-3). Built as its own dock widget via napari's public API - name= gives
        # it the same Window-menu recovery property the native layer list already has, so it can
        # always be brought back even if something about this panel goes wrong.
        layers_panel = QWidget()
        layers_panel_layout = QVBoxLayout()
        layers_panel_layout.setSpacing(20)
        layers_panel_layout.setContentsMargins(-1, 20, -1, -1)
        layers_panel_layout.addStretch()
        layers_panel.setLayout(layers_panel_layout)
        layers_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self.radii_lengths_nav_button = QPushButton("Radii/Lengths (G)")
        self.generic_fiducials_nav_button = QPushButton("Generic Fiducials (H)")
        self.saved_fiducials_nav_button = QPushButton("Saved Fiducials (J)")

        layers_panel_layout.addWidget(self.radii_lengths_nav_button)
        layers_panel_layout.addWidget(self.generic_fiducials_nav_button)
        layers_panel_layout.addWidget(self.saved_fiducials_nav_button)

        self.radii_lengths_nav_button.clicked.connect(lambda: self._activate_layer(MEASUREMENTS_LAYER_NAME))
        self.generic_fiducials_nav_button.clicked.connect(lambda: self._activate_layer(GENERIC_CALIBRATION_LAYER_NAME))
        self.saved_fiducials_nav_button.clicked.connect(lambda: self._activate_layer(PER_IMAGE_CALIBRATION_LAYER_NAME))

        # Created eagerly here, not lazily on first use (its original 8c-2 design) - its starting
        # position is irrelevant either way, since _load_decay_angle_diagram_for_selected_process
        # always recomputes it fresh the moment it's actually shown. Eager creation means the
        # button below always has something real to activate, with no first-click special case.
        self._setup_decay_angle_diagram_layer()
        layers_panel_layout.addWidget(self.decay_angles_nav_button)
        self.decay_angles_nav_button.clicked.connect(lambda: self._activate_layer(ANGLES_LAYER_NAME))
        layers_panel_layout.addStretch()

        for button in (
                self.radii_lengths_nav_button,
                self.generic_fiducials_nav_button,
                self.saved_fiducials_nav_button,
                self.decay_angles_nav_button,
        ):
            button.setCheckable(True)

        self.viewer.window.add_dock_widget(layers_panel, name="layer", area="left")

        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)

        # Hidden by default now that the Layers panel above covers everything it was used for -
        # still fully recoverable via napari's own native Window menu.
        try:
            self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        except AttributeError:
            pass

        self.viewer.bind_key('g', lambda viewer: self._activate_layer(MEASUREMENTS_LAYER_NAME), overwrite=True)
        self.viewer.bind_key('h', lambda viewer: self._activate_layer(GENERIC_CALIBRATION_LAYER_NAME), overwrite=True)
        self.viewer.bind_key('j', lambda viewer: self._activate_layer(PER_IMAGE_CALIBRATION_LAYER_NAME), overwrite=True)
        self.viewer.bind_key('k', lambda viewer: self._activate_decay_angles_via_shortcut(), overwrite=True)
        # Keyboard equivalents of the measurement layer's right-click menu (see
        # _on_measurement_layer_right_click) - same actions, same preconditions, for anyone who'd
        # rather not right-click every time.
        self.viewer.bind_key('o', lambda viewer: self._record_origin_vertex(), overwrite=True)
        self.viewer.bind_key('d', lambda viewer: self._record_decay_vertex(), overwrite=True)
        self.viewer.bind_key('r', lambda viewer: self._record_radius(), overwrite=True)

    def _activate_decay_angles_via_shortcut(self) -> None:
        # Unlike a button click, calling _activate_layer directly would bypass the button's own
        # disabled state entirely - Qt only blocks mouse clicks on a disabled button, not a
        # shortcut calling the underlying method some other way. Check the same source of truth
        # the button itself uses, rather than duplicating the "only for the right process" rule.
        if self.decay_angles_nav_button.isEnabled():
            self._activate_layer(ANGLES_LAYER_NAME)

    def _activate_layer(self, layer_name: str) -> None:
        if layer_name in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[layer_name]

    def _on_active_layer_changed(self, event=None) -> None:
        """Single source of truth for two things at once, every time the active layer changes
        for any reason at all (button click, keyboard shortcut, or someone using the native layer
        list directly): which of the 4 nav buttons should look highlighted, and whether the decay
        angle diagram should be shown/hidden/reordered. Both are always recomputed fresh from the
        current active layer, not tracked as a transition - matching the same full-resync approach
        used throughout this codebase's calibration syncing, for the same reliability reasons.

        Guarded against its own side effects: showing/hiding/reordering the angles layer, or
        restoring focus at the end, both change the active layer themselves, which would otherwise
        re-trigger this same method.
        """
        if getattr(self, "_setting_decay_angle_visibility", False):
            return

        active_layer = self.viewer.layers.selection.active
        target_name = active_layer.name if active_layer is not None else None

        self._setting_decay_angle_visibility = True
        try:
            self.radii_lengths_nav_button.setChecked(target_name == MEASUREMENTS_LAYER_NAME)
            self.generic_fiducials_nav_button.setChecked(target_name == GENERIC_CALIBRATION_LAYER_NAME)
            self.saved_fiducials_nav_button.setChecked(target_name == PER_IMAGE_CALIBRATION_LAYER_NAME)
            self.decay_angles_nav_button.setChecked(target_name == ANGLES_LAYER_NAME)

            if ANGLES_LAYER_NAME in self.viewer.layers:
                angles_layer = self.viewer.layers[ANGLES_LAYER_NAME]
                if target_name == ANGLES_LAYER_NAME:
                    if not angles_layer.visible:
                        self._load_decay_angle_diagram_for_selected_process()
                        angles_layer.visible = True
                        self.viewer.layers.move(self.viewer.layers.index(angles_layer), len(self.viewer.layers))
                        # Unlike the fiducial layers, this layer's visibility toggles constantly during
                        # normal use - every process switch. A plain "not visible yet" check would force
                        # select mode back on every single show, breaking the mode-memory behaviour already
                        # confirmed working for the other three layers. This needs a genuinely once-ever
                        # flag instead.
                        if not getattr(self, "_decay_angles_mode_initialized", False):
                            angles_layer.mode = "select"
                            self._decay_angles_mode_initialized = True
                else:
                    if angles_layer.visible:
                        angles_layer.visible = False
                        self.viewer.layers.move(self.viewer.layers.index(angles_layer), 0)

            # Whatever actually triggered this call is what the user (or our own code) actually
            # wanted active - restore it in case anything above moved focus elsewhere.
            if target_name is not None and target_name in self.viewer.layers:
                if self.viewer.layers.selection.active != self.viewer.layers[target_name]:
                    self.viewer.layers.selection.active = self.viewer.layers[target_name]
        finally:
            self._setting_decay_angle_visibility = False

    @property
    def camera_center(self):
        # update for 4d implementation as appropriate.
        return (self.viewer.camera.center[1], self.viewer.camera.center[2])

    def _get_selected_points(self, layer_names = None) -> np.array:
        """Returns array of selected points in the viewer.
        Beware that if the caller supplies layer_names with points of different dimensions in them,
        then this function will be unable to create the np.array() for the output.
        So it is the caller's responsibility to provide only layers which have points
        that are compatible with each other."""

        if layer_names == None:
            layer_names = [MEASUREMENTS_LAYER_NAME] # Always allow this layer as it is fully editable.

            # By default also allow people to include a generic calibration point in the selection.
            # However this could be annoying if fiducials keep getting selected and you don't want them to.
            # TODO: Ask for feedback and disable the next line if people don't like it.
            # Actually, this is broken anyway as MEASUREMENTS_LAYER has 4D poitns and GENERIC_CALIBRATION_LAYERS
            # have 2D points, so I would need to co-erce things carefully. So completely disable for now.
            # TODO: Fix in the long term
            # layer_names = layer_names + self.calibration_manager.generic_calibration_layer_names()

        # Filtering selected layer (layer names are unique)
        points_layers = [
            layer for layer in self.viewer.layers if layer.name in layer_names
        ]
        # Returning selected points in the layer
        list_content =  [ points_layer.data[i] for points_layer in points_layers for i in points_layer.selected_data ]
        #print(f"Saw {list_content=}")
        selected_points = np.array(
           list_content
        )
        return selected_points

    def _assign_row_id(self, particle) -> None:
        """Stamp `particle` with a fresh, stable, never-reused id (self._next_row_id) - the
        anchor _get_selected_row() uses to find the right self.data entry regardless of how the
        table is currently sorted (see _get_selected_row and _on_table_header_clicked). Call
        this exactly once per ParticleDecay/CalibrationRow, at the point it's created (including
        freshly loading each row of a saved session - a loaded object was never assigned one).

        Deliberately a plain runtime attribute, not a dataclass field: it's a
        _main_widget.py-level UI bookkeeping detail, not domain data, so it must never appear in
        vars_to_save() and end up written to a saved CSV (it starts with "_", which
        vars_to_save() already filters out - see ParticleDecay.vars_to_save).
        """
        particle._row_id = self._next_row_id
        self._next_row_id += 1

    def _get_selected_row(self) -> int:
        """Returns the index into self.data for the currently selected table row.

        Deliberately NOT just the table's own visual row position (QModelIndex.row()). Now that
        clicking a column header re-sorts the table (see _on_table_header_clicked),
        self.data[visual_row] is only guaranteed to be right for original, creation-order
        display - the whole point of sorting is to break that. Every row instead carries its
        self.data index in a hidden, never-displayed "_row_id" column (see _assign_row_id) that
        survives any reordering; this looks THAT up and returns the matching self.data index.

        Note: due to our selection mode only one row selection is possible. Raises IndexError
        when nothing is selected, same as before - existing callers already rely on that.
        """
        select = self.table.selectionModel()
        rows = select.selectedRows()
        visual_row = rows[0].row()
        row_id = int(self.table.item(visual_row, self._get_table_column_index("_row_id")).text())
        for data_index, particle in enumerate(self.data):
            if getattr(particle, "_row_id", None) == row_id:
                return data_index
        raise IndexError(
            f"Selected table row names _row_id={row_id}, but no self.data entry has it - "
            "table/data desync (every self.data entry must get one via _assign_row_id)."
        )

    def _table_row_for_data_index(self, data_index: int) -> int:
        """The inverse of _get_selected_row: given an index into self.data, returns the CURRENT
        visual row that process occupies in the table right now. Needed everywhere code writes a
        cell for "the process at self.data[i]" - e.g. refreshing its radius_px after a drag - since
        that's no longer necessarily table row i once the table's been sorted (see
        _on_table_header_clicked). Looks up self.data[data_index]'s own _row_id (see
        _assign_row_id) against every row's hidden _row_id cell, the same stable anchor
        _get_selected_row uses in the other direction.
        """
        target_id = self.data[data_index]._row_id
        id_col = self._get_table_column_index("_row_id")
        for table_row in range(self.table.rowCount()):
            if int(self.table.item(table_row, id_col).text()) == target_id:
                return table_row
        raise IndexError(
            f"self.data[{data_index}]'s _row_id={target_id} isn't in the table - "
            "table/data desync."
        )

    def _get_selected_measurement_row(self):
        """Like _get_selected_row(), but returns None for both "nothing selected" and
        "a calibration row is selected" - a CalibrationRow's views hold FiducialViewData,
        not ViewData, so code built around origin/decay/track measurements should treat
        selecting one the same as selecting nothing, rather than crash on the shape mismatch.
        """
        try:
            selected_row = self._get_selected_row()
        except IndexError:
            return None
        if isinstance(self.data[selected_row], CalibrationRow):
            return None
        return selected_row

    def _set_up_table(self) -> QTableWidget:
        """Initial setup of the QTableWidget with one row and columns for each
        point and the calculated radius.
        """
        np = ParticleDecay()
        self.columns = list(np.vars_to_save())
        self.columns_show_summary = np.vars_to_show(False)
        self.columns_show_per_view = np.vars_to_show(True)
        # A hidden, never-displayed column carrying each row's stable _row_id (see
        # _assign_row_id/_get_selected_row) - deliberately not part of vars_to_save()/either
        # show-list above, so _set_table_visible_vars's hide-everything-then-show-only-the-list
        # pass leaves it hidden automatically on every mode toggle, with no special-casing needed.
        self.columns.append("_row_id")
        out = QTableWidget(0, len(self.columns))
        out.setHorizontalHeaderLabels(self.columns)
        out.setSelectionBehavior(QAbstractItemView.SelectRows)
        out.setSelectionMode(QAbstractItemView.SingleSelection)
        out.setEditTriggers(QAbstractItemView.NoEditTriggers)
        out.setSelectionBehavior(QTableWidget.SelectRows)
        out.setColumnHidden(self.columns.index("_row_id"), True)

        # Sorting is deliberately NOT the continuous setSortingEnabled(True) behaviour, which
        # re-sorts on every single programmatic cell write too - this app writes live measurement
        # cells constantly while the user drags a point (see _refresh_per_view_breakdown_cells and
        # friends), and having the row someone's actively working on jump to a new position mid-
        # drag would be jarring. Instead, a header click explicitly triggers a one-off sort, via
        # _on_table_header_clicked - never anything else.
        out.horizontalHeader().setSortIndicatorShown(True)
        out.horizontalHeader().sectionClicked.connect(self._on_table_header_clicked)

        # Summary-mode columns are never shown at the same time as the breakdown-mode ones, so
        # each group can just get its own fixed width rule up front, with no "which mode is
        # active" logic needed at all - the wrong group's columns are always hidden regardless.
        wide_uniform_columns = ["event_number", "name", "radius_px", "decay_length_px", "phi_proton", "phi_pion"]
        for col in wide_uniform_columns:
            idx = self.columns.index(col)
            out.setColumnWidth(idx, 140)
            out.horizontalHeader().setSectionResizeMode(idx, QHeaderView.Fixed)

        # The 12 per-view breakdown columns are only ever shown all at once, so a narrower,
        # content-fitted width suits them better - their header text (e.g. "v1_decay_length_px")
        # is the actual limiting factor here, not their short, already-rounded numeric content.
        fm = out.fontMetrics()
        for view_number in (1, 2, 3):
            for base_col in ("radius_px", "decay_length_px", "phi_proton", "phi_pion"):
                col = f"v{view_number}_{base_col}"
                idx = self.columns.index(col)
                out.setColumnWidth(idx, fm.horizontalAdvance(col) + 24)
                out.horizontalHeader().setSectionResizeMode(idx, QHeaderView.Fixed)

        # saved_vertices is shared by both modes and can genuinely grow unpredictably in either
        # one - starts at the same wide width as the summary columns, then adapts from there.
        saved_vertices_idx = self.columns.index("saved_vertices")
        out.setColumnWidth(saved_vertices_idx, 140)
        out.horizontalHeader().setSectionResizeMode(saved_vertices_idx, QHeaderView.ResizeToContents)
        return out

    def _on_table_header_clicked(self, logical_index: int) -> None:
        """Sort the table by the clicked column - a one-off "sort now" action, toggling
        ascending/descending on repeat clicks of the same column (standard header-click
        behaviour). See _set_up_table's comment on why this is wired by hand via sectionClicked
        rather than QTableWidget's own continuous setSortingEnabled(True).

        Sorting only ever reorders which table ROW each process appears in - self.data's own
        order, and every process's identity, is untouched (see _get_selected_row, which looks
        rows up by their hidden _row_id rather than position for exactly this reason).

        Deliberately tracks the last sort itself (self._table_sort_state) instead of reading
        header.sortIndicatorSection()/sortIndicatorOrder() back to decide the toggle - found
        live (every click landed on descending, never ascending): setSortIndicatorShown(True)
        alone makes QHeaderView natively move its OWN indicator onto the clicked section (always
        defaulting to ascending) as soon as the click is processed, purely a visual affordance,
        independent of setSortingEnabled and before this handler even runs. Reading the header
        back here would just be reading Qt's own just-applied default, not "what was showing
        before this click" - toggling against that landed on descending every single time,
        regardless of the true previous state.
        """
        if self._table_sort_state is not None and self._table_sort_state[0] == logical_index:
            _, previous_order = self._table_sort_state
            order = Qt.DescendingOrder if previous_order == Qt.AscendingOrder else Qt.AscendingOrder
        else:
            order = Qt.AscendingOrder
        self._table_sort_state = (logical_index, order)
        self.table.horizontalHeader().setSortIndicator(logical_index, order)
        self.table.sortItems(logical_index, order)

    def _set_table_visible_vars(self, show_per_view) -> None:
        for _ in range(len(self.columns)):
            self.table.setColumnHidden(_, True)
        show = (
            self.columns_show_per_view if show_per_view else self.columns_show_summary
        )
        show_index = [i for i, item in enumerate(self.columns) if item in set(show)]
        for _ in show_index:
            self.table.setColumnHidden(_, False)

        # name/event_number are shared between both modes, but their ideal width differs - roomy
        # to match the wide summary columns, snug to match the narrow breakdown columns.
        fm = self.table.fontMetrics()
        name_idx = self.columns.index("name")
        event_number_idx = self.columns.index("event_number")
        if show_per_view:
            widest_name = max(["Calibration"] + EXPECTED_PROCESSES_NICE, key=len)
            self.table.setColumnWidth(name_idx, fm.horizontalAdvance(widest_name) + 24)
            self.table.setColumnWidth(event_number_idx, fm.horizontalAdvance("event_number") + 24)
        else:
            self.table.setColumnWidth(name_idx, 140)
            self.table.setColumnWidth(event_number_idx, 140)

    def _display_value(self, value) -> str:
        """Render a measurement value for a table cell blank instead of text "None"."""
        if value is None:
            return ""
        return str(value)

    def _get_table_column_index(self, columntext: str) -> int:
        """Given a column title, return the column index in the table"""
        for i, item in enumerate(self.columns):
            if item == columntext:
                return i

        print("Column ", columntext, " not in the table")
        return -1

    def _add_or_update_table_row(self, row_index: int, particle) -> None:
        """Populate a brand new table row's index/name/event_number/saved_vertices/_row_id cells
        from a ParticleDecay or CalibrationRow already sitting at self.data[row_index]. Shared
        by process creation, calibration row sync, and loading a saved session, so all three ways
        a row can appear stay in sync automatically instead of drifting apart. Only ever called
        for a row just created with .insertRow() - never to update an existing one - so
        row_index is a genuine, uncontested position, not something that could have already
        drifted from self.data's index by the time this runs.
        """
        self.table.setItem(
            row_index, self._get_table_column_index("index"), _NumericTableWidgetItem(str(particle.index))
        )
        self.table.setItem(row_index, self._get_table_column_index("name"), QTableWidgetItem(particle.name))
        self.table.setItem(
            row_index,
            self._get_table_column_index("event_number"),
            _NumericTableWidgetItem(str(particle.event_number)),
        )
        self.table.setItem(
            row_index,
            self._get_table_column_index("_row_id"),
            _NumericTableWidgetItem(str(particle._row_id)),
        )
        self._refresh_saved_vertices_cell(row_index)

    def _rebuild_table_from_data(self) -> None:
        """Clear and fully repopulate the table from self.data - used after loading a saved session,
        where every row needs creating at once rather than one at a time like normal process creation.
        """
        self.table.setRowCount(0)
        for i, particle in enumerate(self.data):
            self.table.insertRow(i)
            self._add_or_update_table_row(i, particle)
        self._refresh_all_measurement_cells()

    def _refresh_summary_cells_for_row(self, row_index: int) -> None:
        """Populate one row's summary radius_px/decay_length_px/phi_proton/phi_pion cells, from
        whichever view is currently displayed - the same "current view" reading
        _sync_measurement_layer_to_selected_process/_refresh_decay_angle_table_cells already use,
        just directly parameterised by row instead of relying on row selection. No-ops for
        CalibrationRow, which has no equivalent measurements at all.
        """
        particle = self.data[row_index]
        if isinstance(particle, CalibrationRow):
            return
        current_view = self.viewer.dims.current_step[0]
        view_data = particle.views[current_view]
        table_row = self._table_row_for_data_index(row_index)
        self.table.setItem(
            table_row, self._get_table_column_index("decay_length_px"),
            _NumericTableWidgetItem(self._display_value(round_px(view_data.length_px))),
        )
        self.table.setItem(
            table_row, self._get_table_column_index("radius_px"),
            _NumericTableWidgetItem(self._display_value(round_px(view_data.radius_px))),
        )
        self.table.setItem(
            table_row, self._get_table_column_index("phi_proton"),
            _NumericTableWidgetItem(self._display_value(round_angle(view_data.phi_proton))),
        )
        self.table.setItem(
            table_row, self._get_table_column_index("phi_pion"),
            _NumericTableWidgetItem(self._display_value(round_angle(view_data.phi_pion))),
        )

    def _refresh_all_measurement_cells(self) -> None:
        """Populate every row's measurement cells (both the summary columns and the per-view
        breakdown columns) at once - called right after rebuilding the table from a loaded
        session, so values are visible immediately rather than only appearing lazily.
        """
        for row_index in range(len(self.data)):
            self._refresh_summary_cells_for_row(row_index)
            self._refresh_per_view_breakdown_cells(row_index)

    def _refresh_per_view_breakdown_cells(self, selected_row: int) -> None:
        """Refresh the 12 v1_/v2_/v3_ per-view breakdown columns for a row. Separate from the
        existing current-view summary cell updates (left untouched), so both display modes stay
        correct regardless of which one is currently toggled visible. No-ops for CalibrationRow,
        which has no equivalent measurements at all.
        """
        particle = self.data[selected_row]
        if isinstance(particle, CalibrationRow):
            return
        table_row = self._table_row_for_data_index(selected_row)
        for view_number in (1, 2, 3):
            for base_col in ("radius_px", "decay_length_px", "phi_proton", "phi_pion"):
                col_name = f"v{view_number}_{base_col}"
                value = getattr(particle, col_name)
                self.table.setItem(
                    table_row,
                    self._get_table_column_index(col_name),
                    _NumericTableWidgetItem(self._display_value(value)),
                )

    def _refresh_saved_vertices_cell(self, selected_row: int) -> None:
        self.table.setItem(
            self._table_row_for_data_index(selected_row),
            self._get_table_column_index("saved_vertices"),
            QTableWidgetItem(str(self.data[selected_row].saved_vertices)),
        )

    def _on_row_selection_changed(self) -> None:
        """Enable/disable calculation buttons depending on the row selection, jump the
        viewer to the selected process's event, and refresh the canvas cursors to match.
        The canvas refresh must always run, even when nothing is selected - that's how
        the canvas gets correctly cleared when a process is deselected (e.g. by the
        dims-mismatch guard in _sync_measurement_layer_to_selected_process), so it can't be
        skipped early just because there's no event to jump to.
        """
        # Redirect focus away from the decay-angle diagram every time the selected row changes,
        # regardless of the new process's type - the shared active-layer listener then handles
        # actually hiding and reordering it, so nothing else needs to duplicate that logic here.
        if (
                ANGLES_LAYER_NAME in self.viewer.layers
                and self.viewer.layers.selection.active == self.viewer.layers[ANGLES_LAYER_NAME]
                and MEASUREMENTS_LAYER_NAME in self.viewer.layers
        ):
            self.viewer.layers.selection.active = self.viewer.layers[MEASUREMENTS_LAYER_NAME]
        self.set_button_availability()

        try:
            selected_row = self._get_selected_row()
        except IndexError:
            selected_row = None

        if selected_row is not None:
            event_number = self.data[selected_row].event_number
            if event_number >= 0:  # -1 means "never actually set"
                self.viewer.dims.set_current_step(1, event_number)

            self._sync_measurement_layer_to_selected_process()
            self._return_focus_to_canvas()

    def _return_focus_to_canvas(self, *_args) -> None:
        # Accepts and ignores args since table.clicked passes a QModelIndex we don't need -
        # returns keyboard focus to the canvas, otherwise it stays on the table after a row
        # click, and the G/H/J/K shortcuts (which need canvas focus to fire) silently stop working.
        try:
            self.viewer.window.qt_viewer.canvas.native.setFocus()
        except AttributeError:
            pass

    def _restyle_measurement_points(self) -> None:
        """Ring-highlight each point on the measurement layer to show what it currently contributes:
        a length pair (origin/decay), a radius fit (three track points), both at once (rare), or
        neither (a stray leftover / unowned point, left at the default
        style). The dot's white fill never changes - only the border.

        Also gives the origin and decay vertices their own distinct SHAPE (see _role_symbol) -
        independent of, and in addition to, the colour above: colour says "which measurement(s)
        does this point contribute to", shape says "is this specifically the origin or decay
        vertex" - a point can be shape-distinct without being colour-distinct (an origin vertex
        not yet paired with a decay one still needs to read as "the origin", even alone).
        """
        if MEASUREMENTS_LAYER_NAME not in self.viewer.layers:
            return
        if getattr(self, "_restyling", False):
            # Reentrant call: found live that mutating the arc/arrow overlay layers' own .data
            # from within this same method (see the _refresh_radius_arc/_refresh_origin_decay_arrow
            # calls below) can trigger a callback chain that calls straight back into this method
            # before the outer call has finished - without this guard, that ran a second,
            # interleaved copy of the whole recolour-and-redraw cycle on top of the first,
            # doubling up the arc/arrow shapes (colours/symbols are idempotent under repeated
            # assignment so that half was invisible; the Shapes layers' add-based content isn't).
            return

        data = self.layer_measurements.data
        if len(data) == 0:
            return

        DEFAULT_BORDER_COLOR = "dimgrey"
        DEFAULT_BORDER_WIDTH = 7

        length_points = []
        radius_points = []
        # Kept separate from length_points (which only needs "is this part of the length pair at
        # all" for colouring) - symbol assignment below needs to know WHICH of the two a point is.
        origin_point = None
        decay_point = None
        selected_row = self._get_selected_measurement_row()

        if selected_row is not None:
            current_view = self.viewer.dims.current_step[0]
            current_event = self.viewer.dims.current_step[1]
            if self.data[selected_row].event_number == current_event:
                view_data = self.data[selected_row].views[current_view]
                if view_data.origin is not None:
                    origin_point = (current_view, current_event, *view_data.origin)
                    length_points.append(origin_point)
                if view_data.decay is not None:
                    decay_point = (current_view, current_event, *view_data.decay)
                    length_points.append(decay_point)
                # Only a COMPLETE set of 3 is an actual radius fit (matching
                # ViewData._recompute_radius's own len==3 check) - 1 or 2 leftover track points,
                # e.g. right after deleting one of 3 with napari's own delete tool, aren't a
                # radius any more and must stop being coloured as one.
                if len(view_data.track_points) == 3:
                    for point in view_data.track_points:
                        radius_points.append((current_view, current_event, *point))

        def matches(point, candidates):
            return any(
                point[0] == c[0]
                and point[1] == c[1]
                and np.isclose(point[2], c[2])
                and np.isclose(point[3], c[3])
                for c in candidates
            )

        NORMAL_SIZE = 20
        HIDDEN_SIZE = 0
        show_track = self.show_track_vertices_checkbox.isChecked()
        show_origin_decay = self.show_origin_decay_checkbox.isChecked()

        border_colors = []
        sizes = []
        symbols = []
        face_colors = []
        for point in data:
            is_length = matches(point, length_points)
            is_radius = matches(point, radius_points)
            is_origin = origin_point is not None and matches(point, [origin_point])
            is_decay = decay_point is not None and matches(point, [decay_point])
            symbols.append(_role_symbol(is_origin=is_origin, is_decay=is_decay))
            face_colors.append(_role_face_color(is_origin=is_origin, is_decay=is_decay))
            if is_length and is_radius:
                border_colors.append(BOTH_COLOR)
            elif is_length:
                border_colors.append(LENGTH_COLOR)
            elif is_radius:
                border_colors.append(RADIUS_COLOR)
            else:
                border_colors.append(DEFAULT_BORDER_COLOR)

            # A point classified as neither (stray/unowned) always stays visible - the checkboxes only
            # hide known origin/decay/track roles, not everything on screen. A point that's both (rare)
            # only hides once BOTH its categories are toggled off - one checkbox unticking shouldn't
            # hide something the other checkbox still claims should be showing.
            if is_length and is_radius:
                visible = show_origin_decay or show_track
            elif is_length:
                visible = show_origin_decay
            elif is_radius:
                visible = show_track
            else:
                visible = True
            sizes.append(NORMAL_SIZE if visible else HIDDEN_SIZE)

        # Guard against the highlight-refresh feedback loop below, and reset the "next new point"
        # default back to plain grey - napari otherwise keeps whatever style we last painted onto the
        # currently-selected points and quietly applies it to the very next brand-new point too.
        self._restyling = True
        try:
            self.layer_measurements.border_color = border_colors
            self.layer_measurements.border_width = [DEFAULT_BORDER_WIDTH] * len(data)
            self.layer_measurements.size = sizes
            self.layer_measurements.symbol = symbols
            self.layer_measurements.face_color = face_colors
            # The current_* setters below aren't just "next new point" defaults - napari also
            # applies them live to whatever's in layer.selected_data right now, indexing straight
            # into the arrays just replaced above. Found live (full traceback): pressing napari's
            # own native delete-point key/action calls Points.remove_selected(), which shrinks
            # .data and - VIA THE SAME layer.events.highlight cascade that reaches
            # _restyle_measurement_points - reaches here BEFORE remove_selected() has reached its
            # own trailing `self.selected_data = set()` cleanup a few lines later. selected_data
            # at that instant still names the just-deleted point's OLD index, which is now out of
            # bounds against the just-shrunk arrays - an IndexError from napari's own internals,
            # not a sign our own colour/symbol/size data is wrong (those were already correctly
            # reassigned above, at the new length, with no index-based access to trip over).
            # Harmless to skip this one tick: _restyle_measurement_points runs again on essentially
            # every subsequent canvas event, so the "next new point" default self-heals immediately.
            try:
                self.layer_measurements.current_border_color = DEFAULT_BORDER_COLOR
                self.layer_measurements.current_border_width = DEFAULT_BORDER_WIDTH
                self.layer_measurements.current_symbol = "disc"
                self.layer_measurements.current_face_color = "white"
            except IndexError:
                _debug_trace(
                    "restyle: SKIP current_* defaults - napari's own selected_data still named a "
                    "just-deleted point's stale index (see _restyle_measurement_points comment)"
                )
            # Force the repaint explicitly, rather than relying on one of the assignments above
            # to trigger it as a side effect - with border_width now staying at a single uniform
            # value, napari may treat that particular assignment as a no-op and skip its own
            # repaint, leaving the (correct) colour applied in data but not actually drawn until
            # something else forces a real re-slice (e.g. navigating to a different event and back).
            self.layer_measurements.refresh()
            # Kept INSIDE the _restyling-guarded block, not after it: found live (via a debug
            # trace showing a plain sync producing 4 arrow shapes instead of 2) that mutating the
            # arc/arrow Shapes layers' own .data - remove_selected()/add() - can itself trigger a
            # reentrant call back into this same method, and outside this guard nothing stopped
            # that from running a second, interleaved clear-then-add cycle on top of the first.
            self._refresh_radius_arc()
            self._refresh_origin_decay_arrow()
        finally:
            self._restyling = False

    def _run_guarded_against_restyle_reentrancy(self, body) -> None:
        """Run `body` (a zero-argument callable) with the same `_restyling` guard
        _restyle_measurement_points uses - shared so any method whose own layer mutations can
        trigger a reentrant call chain back into _restyle_measurement_points (currently
        _do_refresh_radius_arc and _do_refresh_origin_decay_arrow, called both from inside that
        method's own guarded block AND directly from checkbox toggles) is protected either way,
        without double-acquiring or releasing a guard some enclosing call already holds.

        If the guard is already held (this call is nested inside _restyle_measurement_points'
        own guarded block, or inside another guarded call), just run `body` - the existing
        outermost acquisition already protects it. Otherwise acquire the guard for `body`'s
        duration, exactly as _restyle_measurement_points itself does.
        """
        if getattr(self, "_restyling", False):
            body()
            return
        self._restyling = True
        try:
            body()
        finally:
            self._restyling = False

    def _clear_shapes_layer(self, layer: napari.layers.Shapes) -> None:
        """Remove every shape from a Shapes layer we fully own (the radius-arc / O-D-arrow
        overlays), ready to redraw it from scratch.

        Deliberately does NOT go via the public `layer.selected_data = set(range(len(layer.data)))`
        idiom napari itself suggests for this. That property SETTER also computes an on-screen
        "interaction box" from only the shapes napari currently considers displayed in the active
        dims slice (`Shapes.interaction_box`), and crashes with a numpy "zero-size array to
        reduction operation minimum" ValueError whenever NONE of the shapes being selected are
        considered in-slice at that exact instant - found live via a full traceback: reliably
        triggered by a rectangle-select drag on a completely different layer (the measurement
        points layer), which fires a highlight event - and so a call to this method, via
        _restyle_measurement_points - on every mouse-move tick, fast enough to race napari's own
        per-layer slice bookkeeping for whichever of our Shapes layers is being cleared.

        Setting the underlying `_selected_data` attribute directly - the exact same assignment the
        real setter's own first line makes - skips that fragile side computation entirely, and
        costs nothing: this layer is editable=False, so no interactive selection-box overlay is
        ever meant to be shown for it anyway. `remove_selected()` itself only reads `selected_data`
        (the plain getter) and is not affected.
        """
        if len(layer.data) == 0:
            return
        layer._selected_data = set(range(len(layer.data)))
        layer.remove_selected()

    def _points_claimed_by_other_processes(self, selected_row, current_view, current_event) -> set[tuple[float, float]]:
        """The coordinates every OTHER process (not `selected_row`) has already recorded a role
        for, in this exact (view, event) - origin/decay/track points, from _measurement_roles.

        CROSS-PROCESS ISOLATION INVARIANT (load-bearing - preserve this call across any future
        refactor of how the canvas is rendered, not just an incidental implementation detail):
        a canvas point already claimed by a DIFFERENT process must never be offered to the user
        as if it were free ("an orphan", ripe for the right-click menu / O-D-R shortcuts) just
        because they happen to be looking at a different process right now. Each process's roles
        already live in their own fully independent ViewData - _sync_measurement_layer_to_selected_process
        and _propagate_measurement_point_drag only ever read/write self.data[selected_row], so one
        process's DATA can never be corrupted by editing another's (see feedback_gui_race_conditions
        memory for the broader architecture note this belongs to). But CLAIMING an already-claimed
        point is a USER ACTION, not a data race, and nothing stops a user from doing that by
        accident if the display can't tell them apart from a genuinely free point - found live: a
        point recorded as process 1's origin silently looked, to process 2, exactly like an
        unclaimed point sitting on the same (view, event), and got recorded as process 2's radius
        point too, without the user realising the point was already spoken for.

        The two processes' data staying independent afterwards is fine, arguably even correct
        (see the discussion this invariant came from) - what's not fine is the user not knowing
        they'd just done it. This is the ONE call site responsible for preventing it: any
        rendering path that decides what counts as an "unclaimed" point on the canvas must
        consult this, not just the selected process's own roles.
        """
        claimed: set[tuple[float, float]] = set()
        for i, particle in enumerate(self.data):
            if i == selected_row or isinstance(particle, CalibrationRow):
                continue
            if particle.event_number != current_event:
                continue
            for _, point in _measurement_roles(particle.views[current_view]):
                claimed.add(_as_xy(point))
        return claimed

    def _sync_measurement_layer_to_selected_process(self, event=None) -> None:
        """Make the on-canvas origin/decay/track points reflect whichever process is selected in
        the table, for the view+event currently on screen, and refresh the table's radius/length
        cells to match. Runs on row selection and on every View/Event slider move.

        With a process selected, rebuilds the current slice from its saved data. With nothing
        selected, clears the current slice instead of leaving old points sitting there - safe to do
        now that _on_measurement_points_changed bails out immediately whenever nothing's selected,
        so this can no longer be misread as a real deletion the way it once could.
        """
        if MEASUREMENTS_LAYER_NAME not in self.viewer.layers:
            return

        current_view = self.viewer.dims.current_step[0]
        current_event = self.viewer.dims.current_step[1]

        try:
            selected_row = self._get_selected_row()
        except IndexError:
            selected_row = None

        # A process only ever belongs to the one event it was created in. If the Event slider has
        # moved away from that event - most likely by dragging the bar directly rather than
        # clicking a different process row - selection no longer matches what's on screen. Deselect
        # process, rather than displaying (or worse, writing into) a process's points on a photo
        # it doesn't actually belong to.
        if selected_row is not None:
            process_event = self.data[selected_row].event_number
            if process_event >= 0 and process_event != current_event:
                self.table.clearSelection()
                return  # clearing selection re-triggers this method itself

        existing_data = self.layer_measurements.data
        other_slices = []
        current_slice_existing = []
        for point in existing_data:
            if point[0] == current_view and point[1] == current_event:
                current_slice_existing.append(point)
            else:
                other_slices.append(point)

        new_points = []
        view_data = None
        role_groups: dict[tuple[float, float], list[str]] = {}
        # Calibration rows have nothing to put on this layer - their fiducial stamps live on a
        # completely separate layer, managed by CalibrationManager instead of ParticleDecay.views.
        if selected_row is not None and not isinstance(self.data[selected_row], CalibrationRow):
            view_data = self.data[selected_row].views[current_view]
            for role, point in _measurement_roles(view_data):
                # Two roles that currently sit at the exact same spot (e.g. a radius fit reusing
                # the decay vertex) are one physical point in the user's mind, not two - draw it
                # once. _measurement_role_index_map below records every role each drawn point
                # represents, so dragging it (see _propagate_measurement_point_drag) moves all of
                # them together instead of leaving one behind.
                role_groups.setdefault(_as_xy(point), []).append(role)
            for xy in role_groups:
                new_points.append([current_view, current_event, xy[0], xy[1]])

        # A point on this slice that isn't (yet) tied to any role - freshly clicked, not yet
        # recorded via the right-click menu / O/D/R shortcuts - must survive a routine rebuild.
        # Selection is deliberately separate from action now: an unlabelled point can sit on the
        # canvas indefinitely, and only a real delete (the 'x' tool, reconciled elsewhere) or a
        # Record/Clear action may make one disappear, never just the View/Event slider moving.
        # Excludes points already claimed by a DIFFERENT process in this (view, event) - see
        # _points_claimed_by_other_processes's docstring for why this exclusion is load-bearing,
        # not incidental: without it, another process's own recorded point is indistinguishable
        # from a genuinely free one, and can get silently claimed a second time by accident.
        claimed_by_other_processes = self._points_claimed_by_other_processes(
            selected_row, current_view, current_event
        )
        orphan_points = [
            point for point in current_slice_existing
            if _as_xy(point[2:]) not in role_groups
            and _as_xy(point[2:]) not in claimed_by_other_processes
        ]

        # Rewriting .data here is our own routine canvas rebuild, not a real user action - guard it so
        # _on_measurement_points_changed doesn't treat this rebuild as evidence that points were deleted
        # (that reconciliation logic must only ever react to something the user actually did to the canvas).
        self._syncing = True
        try:
            self.layer_measurements.selected_data = set()
            self.layer_measurements.data = other_slices + new_points + orphan_points
            self.layer_measurements.selected_data = set()
        finally:
            self._syncing = False
        self._last_synced_dims = (current_view, current_event)
        # Indices are into the freshly-assigned .data above: other_slices occupy [0, len(other_slices)),
        # our own role-bearing points follow in the same order role_groups was built in, then the
        # orphans - which intentionally get no entry here, since they carry no role to propagate.
        self._measurement_role_index_map = {
            len(other_slices) + i: roles for i, roles in enumerate(role_groups.values())
        }
        # Marks the map as trustworthy again, having just been rebuilt from the current .data -
        # see _invalidate_measurement_role_index_map for why this can't be inferred from a point
        # count instead.
        self._measurement_role_index_map_valid = True

        if view_data is not None:
            table_row = self._table_row_for_data_index(selected_row)
            self.table.setItem(
                table_row,
                self._get_table_column_index("decay_length_px"),
                _NumericTableWidgetItem(self._display_value(round_px(view_data.length_px))),
            )
            self.table.setItem(
                table_row,
                self._get_table_column_index("radius_px"),
                _NumericTableWidgetItem(self._display_value(round_px(view_data.radius_px))),
            )
        if selected_row is not None:
            self._refresh_per_view_breakdown_cells(selected_row)
            self._refresh_saved_vertices_cell(selected_row)

        self._restyle_measurement_points()
        self._sync_other_processes_layer()
        self._refresh_decay_angle_table_cells()
        if ANGLES_LAYER_NAME in self.viewer.layers and self.viewer.layers[ANGLES_LAYER_NAME].visible:
            self._load_decay_angle_diagram_for_selected_process()

    def set_button_availability(self) -> None:
        images_imported = False
        for layer in self.viewer.layers:
            if layer.name == IMAGE_LAYER_NAME:
                images_imported = True
                break
        self.set_UI_image_loaded(images_imported, self.bypass_force_load_data)

        # Save has to be reachable even with an empty table - a student who's only calibrated
        # generic templates so far (no process rows, nothing to select) still has real data
        # worth saving. Base this on "is there anything to save" instead of row selection.
        # calibration_manager doesn't exist yet the first few times this runs - it triggers
        # layer-added events (via its own __init__) before self.calibration_manager is assigned.
        calibration_manager = getattr(self, "calibration_manager", None)
        generic_templates = calibration_manager.calibration_data.generic_templates if calibration_manager else []
        has_anything_to_save = len(self.data) > 0 or any(len(t.positions) > 0 for t in generic_templates)
        self.save_data_button.setEnabled(has_anything_to_save)

        try:
            selected_row = self._get_selected_row()
            self.delete_process.setEnabled(True)
            # self.stereoshift_button.setEnabled(True)
            if self.data[selected_row].index == 4:
                self.decay_angles_nav_button.setEnabled(True)
            else:
                self.decay_angles_nav_button.setEnabled(False)
            return
        except IndexError:
            self.delete_process.setEnabled(False)
            self.decay_angles_nav_button.setEnabled(False)
            # self.apply_magnification_button.setEnabled(False)
            # self.stereoshift_button.setEnabled(False)
            # self.magnification_button.setEnabled(False)

    def set_UI_image_loaded(self, loaded: bool, bypass_load_screen: bool) -> None:
        if bypass_load_screen:
            return
        if loaded:
            self.load_button.setEnabled(False)
            self.particle_decays_menu.setEnabled(True)
            self.load_data_button.setEnabled(True)
        else:
            self.load_button.setEnabled(True)
            self.particle_decays_menu.setEnabled(False)
            self.load_data_button.setEnabled(False)
            self.delete_process.setEnabled(False)
            self.decay_angles_nav_button.setEnabled(False)
            # self.stereoshift_button.setEnabled(False)
            # if ENABLE_MAG:
            # self.apply_magnification_button.setEnabled(False)

    def _selected_points_are_on_current_slice(self, selected_points) -> bool:
        """Check that the selected points are in the current slice of the viewer"""
        for slice_index, data_slice in enumerate(["View", "Event"]):
            current_slice = self.viewer.dims.current_step[slice_index]
            all_points_in_current_slice = all(
                current_slice == point[slice_index] for point in selected_points
            )
            if not all_points_in_current_slice:
                napari.utils.notifications.show_error(
                    f"Measurement points not in current {data_slice}. Measurement not completed."
                )
                return False
        return True

    # _on_click_radius() and _on_click_length() used to live here - now removed that selecting

    def _default_decay_angle_lines(self) -> list:
        """The same starting position the diagram has always used, pulled out so both first-time
        layer creation and later resets (switching to a process/view with no saved diagram yet) use
        one copy of these numbers rather than two that could drift apart.
        """
        origin_x = self.camera_center[0]
        origin_y = self.camera_center[1]
        zoom_factor = self.viewer.camera.zoom

        lambda_line = np.array([
            [origin_x + 100 / zoom_factor, origin_y + 200 / zoom_factor],
            [origin_x + -100 / zoom_factor, origin_y + -100 / zoom_factor],
        ])
        proton_line = np.array([
            [origin_x + 100 / zoom_factor, origin_y + 200 / zoom_factor],
            [origin_x + 200 / zoom_factor, origin_y + 300 / zoom_factor],
        ])
        pion_line = np.array([
            [origin_x + 100 / zoom_factor, origin_y + 200 / zoom_factor],
            [origin_x + 110 / zoom_factor, origin_y + 300 / zoom_factor],
        ])
        return [lambda_line, proton_line, pion_line]

    def _refresh_decay_angle_table_cells(self) -> None:
        try:
            selected_row = self._get_selected_row()
        except IndexError:
            return
        if self.data[selected_row].index != 4:
            return
        current_view = self.viewer.dims.current_step[0]
        view_data = self.data[selected_row].views[current_view]
        table_row = self._table_row_for_data_index(selected_row)
        self.table.setItem(
            table_row,
            self._get_table_column_index("phi_proton"),
            _NumericTableWidgetItem(self._display_value(round_angle(view_data.phi_proton))),
        )
        self.table.setItem(
            table_row,
            self._get_table_column_index("phi_pion"),
            _NumericTableWidgetItem(self._display_value(round_angle(view_data.phi_pion))),
        )
        self._refresh_per_view_breakdown_cells(selected_row)
        self._refresh_saved_vertices_cell(selected_row)

    def _load_decay_angle_diagram_for_selected_process(self) -> None:
        """Populate the diagram with whichever lines the selected process has saved for the current view,
        or reset to the default starting position if it has none yet. Called right before the diagram
        becomes visible (so checking the box always shows the right process's own diagram) and whenever
        the view changes while the diagram is already showing.
        """
        if ANGLES_LAYER_NAME not in self.viewer.layers:
            return
        try:
            selected_row = self._get_selected_row()
        except IndexError:
            return
        if self.data[selected_row].index != 4:
            return

        current_view = self.viewer.dims.current_step[0]
        view_data = self.data[selected_row].views[current_view]
        layer = self.viewer.layers[ANGLES_LAYER_NAME]

        self._loading_decay_angle_diagram = True
        try:
            if view_data.decay_angle_lines is not None:
                layer.data = [np.array(line) for line in view_data.decay_angle_lines]
            else:
                layer.data = self._default_decay_angle_lines()
        finally:
            self._loading_decay_angle_diagram = False

        # Always refresh the table to whatever this view already has stored, independent of
        # the guard above - that guard exists to stop a fresh load from being mistaken for
        # a live edit, not to stop a genuine, already-saved value from being displayed.
        self._refresh_decay_angle_table_cells()

    def _setup_decay_angle_diagram_layer(self):
        """Create the Lambda/p/pi decay-angle diagram directly on the main canvas - reusing the exact
        same layer (and default starting position) the Decay Angles popup already creates, so whichever
        path someone uses, they're looking at the same shapes. Only gets the diagram onto the canvas and
        toggleable; making it view/event-aware and remembering its position per process is separate.
        """
        if ANGLES_LAYER_NAME in self.viewer.layers:
            return self.viewer.layers[ANGLES_LAYER_NAME]

        lines = self._default_decay_angle_lines()
        colors = ["green", "red", "blue"]
        text = {
            "string": ["Λ", "p", "π"],
            "size": 20,
            "color": colors,
            "translation": np.array([-30, 0]),
        }

        shapes_layer = self.viewer.add_shapes(
            lines,
            name=ANGLES_LAYER_NAME,
            shape_type=["line"] * 3,
            edge_width=5,
            edge_color=colors,
            face_color=colors,
            text=text,
            ndim=2,
            visible=False,
        )
        shapes_layer.events.data.connect(self._enforce_decay_angle_lines_coincident)
        shapes_layer.events.data.connect(self._on_decay_angle_diagram_changed)
        return shapes_layer

    def _on_decay_angle_diagram_changed(self, event=None) -> None:
        """Live auto-calculation for the decay angle diagram - mirrors
        _on_measurement_points_changed's role for radius/length. Fires
        alongside _enforce_decay_angle_lines_coincident on the same
        events.data signal; if that correction moves a line, this method
        naturally re-fires against the corrected data right after, so the
        stored result always reflects the final, corrected positions.
        """
        if event is None or event.action != "changed":
            return
        if getattr(self, "_loading_decay_angle_diagram", False):
            return  # this is us loading/resetting the diagram, not a real drag

        try:
            selected_row = self._get_selected_row()
        except IndexError:
            return

        # Only the one process type this diagram is even for - and only
        # while the viewer is actually showing that process's own event,
        # not some other event napari's internal handling might briefly
        # report mid-navigation (the same race that once caused real data
        # loss for radius/length - guarding against it here from the start).
        if self.data[selected_row].index != 4:
            return
        current_event = self.viewer.dims.current_step[1]
        if self.data[selected_row].event_number != current_event:
            return

        layer = self.viewer.layers[ANGLES_LAYER_NAME]
        if len(layer.data) != 3:
            return

        lambda_line, proton_line, pion_line = layer.data
        current_view = self.viewer.dims.current_step[0]
        view_data = self.data[selected_row].views[current_view]
        view_data.set_decay_angle_lines([
            [list(map(float, lambda_line[0])), list(map(float, lambda_line[1]))],
            [list(map(float, proton_line[0])), list(map(float, proton_line[1]))],
            [list(map(float, pion_line[0])), list(map(float, pion_line[1]))],
        ])

        table_row = self._table_row_for_data_index(selected_row)
        self.table.setItem(
            table_row,
            self._get_table_column_index("phi_proton"),
            _NumericTableWidgetItem(self._display_value(round_angle(view_data.phi_proton))),
        )
        self.table.setItem(
            table_row,
            self._get_table_column_index("phi_pion"),
            _NumericTableWidgetItem(self._display_value(round_angle(view_data.phi_pion))),
        )
        self._refresh_per_view_breakdown_cells(selected_row)
        self._refresh_saved_vertices_cell(selected_row)

    def _enforce_decay_angle_lines_coincident(self, event=None) -> None:
        """Keep the Lambda/p/pi lines meeting at a shared decay vertex -
        the same correction the Decay Angles popup already applies, wired
        up here too so it works even if someone only ever uses the
        checkbox and never opens the popup.
        """
        if event is None or event.action != "changed":
            return

        layer = self.viewer.layers[ANGLES_LAYER_NAME]
        shapes_modified = event.data_indices
        data = layer.data
        if len(shapes_modified) in (1, 2):
            for i in range(3):
                if (i not in shapes_modified) and (
                        layer.data[i][0] != layer.data[shapes_modified[0]][0]
                ).any():
                    data[i][0] = layer.data[shapes_modified[0]][0]
                    layer.data = data

    # def _on_click_stereoshift(self) -> StereoshiftDialog:
    #     """When the 'Calculate stereoshift' button is clicked, open stereoshift dialog."""
    #     # Different behaviour to the Magnification dialog, waiting for the definition of the stereoshift layer structure
    #     if self.stereoshift_dlg is not None:
    #         self.stereoshift_dlg.show()
    #         self.stereoshift_dlg.raise_()
    #         return self.stereoshift_dlg
    #     self.stereoshift_dlg = StereoshiftDialog(self)
    #     self.stereoshift_dlg.show()
    #     self.stereoshift_dlg.raise_()
    #     return self.stereoshift_dlg

    def _on_click_load(self) -> None:
        """Restore a previously saved session (process rows, calibration rows, and generic fiducial
        templates) from a .pkl or .csv file, replacing whatever's currently in the table. CSV
        loading reconstructs the same (data, generic_templates) shape .pkl loading already
        produces (see load_csv_session), so everything below this point is fully format-agnostic.
        """
        if IMAGE_LAYER_NAME not in self.viewer.layers:
            napari.utils.notifications.show_error(
                "Load images first - the event/view slots a saved session refers to don't exist until then."
            )
            return
        if self.dirty_things():
            confirmation_dialog = QMessageBox()
            confirmation_dialog.setText("Loading will discard the current, unsaved session.")
            confirmation_dialog.setInformativeText("Do you want to continue?")
            confirmation_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            confirmation_dialog.setDefaultButton(QMessageBox.Cancel)
            if confirmation_dialog.exec() != QMessageBox.Yes:
                return
        file_name, selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load file",
            self._last_load_process_table_dir,
            "Pickle files (*.pkl);;CSV files (*.csv)" if ENABLE_PICKLE else "CSV files (*.csv)",
            "",
            QFileDialog.DontUseNativeDialog,
        )
        if file_name in {"", None}:
            return
        self._last_load_process_table_dir = os.path.dirname(file_name)
        if os.path.splitext(file_name)[1] == "":
            file_name += ".csv" if "csv" in selected_filter.lower() else ".pkl"
        try:
            if file_name.endswith(".csv"):
                data, generic_templates = load_csv_session(file_name)
            else:
                if not ENABLE_PICKLE:
                    napari.utils.notifications.show_error(
                        "Loading .pkl files is currently disabled - please load a .csv file instead."
                    )
                    return
                with open(file_name, "rb") as handle:
                    session = pickle.load(handle)
                data, generic_templates = session.data, session.generic_templates
        except Exception as e:
            napari.utils.notifications.show_error(f"Could not load {file_name}: {e}")
            return
        self.table.clearSelection()
        self.data = data
        # A freshly loaded row has never been through _on_click_new_process/the calibration-sync
        # branch, so it has no _row_id yet - assign one now, same as either of those would have.
        for particle in self.data:
            self._assign_row_id(particle)
        self._rebuild_table_from_data()
        # Restore calibration: generic templates come straight from the loaded session;
        # event_views is rebuilt from the just-restored CalibrationRow entries rather than also
        # loaded separately, so there's only ever one copy of per-event stamp data to keep
        # consistent.
        self.calibration_manager.calibration_data.generic_templates = generic_templates
        self.calibration_manager.calibration_data.event_views = {}
        for particle in self.data:
            if isinstance(particle, CalibrationRow):
                for view_index, view_data in enumerate(particle.views):
                    if view_data.stamped:
                        self.calibration_manager.calibration_data.event_views[
                            (particle.event_number, view_index)] = view_data
        self.calibration_manager._restore_generic_calibration_layers(generic_templates)
        self.calibration_manager._restore_event_calibration_layer()
        # Same reasoning as the __init__-time fix: without re-marking clean here, the calibration
        # dirty-check baseline stays stale relative to what was just loaded.
        self.calibration_manager.mark_clean()
        # refresh_symbol_sizes() is otherwise only triggered by a zoom event.
        self.calibration_manager.refresh_symbol_sizes()
        import copy
        self._data_at_last_save = copy.deepcopy(self.data)  # a freshly loaded session isn't "dirty"

        self.set_button_availability()
        napari.utils.notifications.show_info("Loaded " + file_name)

    def _on_click_load_data(self) -> None:
        """When the 'Load data' button is clicked, a dialog opens to select the folder containing the data.
        The folder should contain three subfolders named as variations of 'view1', 'view2' and 'view3', and each subfolder should contain the same number of images.
        The images in each folder are loaded as a stack, and the stack is named according to the subfolder name.
        """
        # setup UI
        test_file_dialog = QFileDialog(self)
        test_file_dialog.setFileMode(QFileDialog.Directory)
        # retrieve image folder
        folder_name = test_file_dialog.getExistingDirectory(
            self,
            "Choose folder",
            self._settings.value("last_load_images_dir", "./"),
            QFileDialog.DontUseNativeDialog
            | QFileDialog.DontResolveSymlinks
            | QFileDialog.ShowDirsOnly
            | QFileDialog.HideNameFilterDetails,
        )
        if folder_name not in {"", None}:
            self._settings.setValue("last_load_images_dir", folder_name)
        self._load_data_from(folder_name)

    def _load_data_from(self, folder_name):

        self.msg = QMessageBox()
        self.msg.setIcon(QMessageBox.Warning)
        self.msg.setWindowTitle(f"Invalid folder name.")
        self.msg.setStandardButtons(QMessageBox.Ok)
        self.msg.setText(
            f"The Cavendish Particle Tracks plug in was asked to load bubble chamber photos from a folder named:\n\n   '{folder_name}'\n\nThis folder name seems to be invalid. The data folder must exist, must contain three subfolders (one for each view), and each subfolder must contain the same number (>1) of images. View folders are identified by case insensitive partial matches against the strings in {VIEW_NAMES}."
        )

        if folder_name in {"", None}:
            self.msg.show()
            return

        folder_subdirs = glob.glob(folder_name + "/*/")
        print(f"folder_subdirs was {repr(folder_subdirs)}")   
        cannot_continue = len(folder_subdirs)==0
        # Excludes cases that either the folder did not exist, or the folder existed but was empty of subdirs.
            
        if cannot_continue: 
            self.msg.show()
            return

        folder_subdirs.sort() # Get View2, View1, View3 into order. No reason not to!
        
        # Checks whether the image folder contains a subdirectory for each view.
        three_subdirectories = len(folder_subdirs) == 3
        # Checks that these subdirectories correspond to event views.
        subdir_names_contain_views = all(
            any(view in name.lower() for name in folder_subdirs) for view in VIEW_NAMES
        )
        # Checks that each subdirectory contains the same number of images.
        image_count_first = len(glob.glob(folder_subdirs[0] + "/*"))
        more_than_one_image = image_count_first > 1
        same_image_count = all(
            len(glob.glob(subdir + "/*")) == image_count_first
            for subdir in folder_subdirs
        )
        # If all checks are passed, load the images where the event number is a
        # new spatial dimension (stack) and the views are layers.
        if not (
            three_subdirectories
            and subdir_names_contain_views
            and same_image_count
            and more_than_one_image
        ):
            self.msg.show()
            return

        def crop(array):
            # Crops view 1 and 2 to same size as view 3 by removing whitespace
            # on left, as images align on the right.
            # this number is the width of image 3.
            magic_number_smallest_view_pixels = -8377
            return array[:, :, magic_number_smallest_view_pixels:, :]

        def rotate(array):
            # Not sure what dimensions 0 and 3 are, but below
            # the ::-1 in dimension 1 reverses the short (i.e. the y) direction in the images, and
            # the ::-1 in dimension 2 reverses the long (i.e. the x) direction in the images.
            return array[:, ::-1, ::-1, :]

        # Shuffle the images to avoid bias in the order of the events
        shuffling_indices = np.random.RandomState(self.shuffling_seed).permutation(
            image_count_first
        )

        stacks = []
        for subdir in folder_subdirs:
            stack: dask.array.Array = imread(subdir + "/*")
            stack = crop(stack)
            stack = rotate(stack)
            # Shuffle each view stack in the same way
            stack = stack[shuffling_indices]
            stacks.append(stack)

        # Concatenate stacks along new spatial dimension such that we have a view, and event slider
        concatenated_stack = dask.array.stack(stacks, axis=0)
        self.viewer.add_image(concatenated_stack,
                              name=IMAGE_LAYER_NAME,
                              )
        bubble_chamber_layer = self.viewer.layers[IMAGE_LAYER_NAME]
        self.viewer.dims.axis_labels = ("View", "Event", "Y", "X")

        # The generic calibration layer was built with a placeholder single-event count at
        # CalibrationManager construction time, since the real count isn't knowable until now.
        self.calibration_manager.rebuild_generic_layer_for_event_count(image_count_first)

        # Move to the first event in the series
        self.viewer.dims.set_current_step(1, 0)

        # Create measurements layer if not already there
        self.layer_measurements = self._setup_measurement_layer()
        self._setup_other_processes_layer()
        self._setup_radius_arc_layer()
        self._setup_origin_decay_arrow_layer()

        # Move bubble chamber layer to the bottom
        self.viewer.layers.move(self.viewer.layers.index(bubble_chamber_layer), 0)

        # Disable the load button after loading the data (interim solution until we can move to bottom-docked UI)
        self.load_button.setEnabled(False)

        # Now that the image and its layers have fully settled, safe to make the calibration cursors visible
        self.calibration_manager.set_calibration_layer_visibility_and_focus(True, False)

    def _setup_measurement_layer(self):
        """Create a Points layer for the measurement of the radii and lengths."""

        if MEASUREMENTS_LAYER_NAME in self.viewer.layers:
            return self.viewer.layers[MEASUREMENTS_LAYER_NAME]
        else:
            layer = self.viewer.add_points(
                name=MEASUREMENTS_LAYER_NAME,
                ndim=4,
                size=20,
                border_width=7,
                border_width_is_relative=False,
            )
            # Order matters: invalidation must run before propagation, on every event, so that an
            # add/remove disables propagation from this same event onwards - see
            # _invalidate_measurement_role_index_map's docstring. Propagation itself must run
            # ahead of the deletion-reconciliation below so every role sharing a dragged point is
            # already updated before that method inspects them - see
            # _propagate_measurement_point_drag's docstring for why it's connected to highlight
            # too, not just data.
            layer.events.data.connect(self._invalidate_measurement_role_index_map)
            layer.events.data.connect(self._propagate_measurement_point_drag)
            layer.events.data.connect(self._on_measurement_points_changed)
            layer.events.highlight.connect(self._propagate_measurement_point_drag)
            layer.events.highlight.connect(self._on_measurement_points_changed)
            layer.mouse_drag_callbacks.append(self._on_measurement_layer_right_click)
            return layer

    def _setup_other_processes_layer(self):
        """A second, non-interactive points layer showing every OTHER process's saved points for the
        current view/event, for visual comparison. Deliberately one-way (data -> canvas only) - nothing
        ever reads this layer's own contents back to infer anything. editable=False should make it unclickable.
        """
        if OTHER_PROCESSES_LAYER_NAME in self.viewer.layers:
            return self.viewer.layers[OTHER_PROCESSES_LAYER_NAME]
        layer = self.viewer.add_points(
            name=OTHER_PROCESSES_LAYER_NAME,
            ndim=4,
            size=20,
            border_width=7,
            border_width_is_relative=False,
        )
        layer.editable = False
        layer.opacity = 0.4
        # Adding a new layer makes napari activate it automatically, silently stealing "active layer"
        # status. Hand it straight back, or every point placed afterwards lands on this layer instead.
        self.viewer.layers.selection.active = self.layer_measurements
        return layer

    def _decoratable_process_views(self, current_view: int, current_event: int):
        """Yield (view_data, is_selected) for every process whose radius arc / origin-decay arrow
        should currently be drawn: the selected process first (is_selected=True), then - only when
        "Show all processes" is ticked - every OTHER process at this (view, event) too
        (is_selected=False), the same set _sync_other_processes_layer already shows as dimmed
        points, so the decorators stay visually consistent with the points they connect. A single
        shared generator so _refresh_radius_arc and _refresh_origin_decay_arrow (and any future
        process-scoped decorator) all respect the checkbox identically, rather than each growing
        its own bespoke all-processes handling.
        """
        selected_row = self._get_selected_measurement_row()
        if selected_row is not None:
            particle = self.data[selected_row]
            if particle.event_number == current_event:
                yield particle.views[current_view], True

        if not self.show_all_processes_checkbox.isChecked():
            return
        for i, particle in enumerate(self.data):
            if i == selected_row:
                continue
            if isinstance(particle, CalibrationRow):
                continue
            if particle.event_number != current_event:
                continue
            yield particle.views[current_view], False

    def _setup_radius_arc_layer(self):
        """A single-shape-per-process overlay tracing each visible process's radius fit as an arc
        through its 3 points (see _calculate.radius_arc_points), in the same colour as their
        highlight ring - the visual counterpart of that colour, showing the 3 points are linked
        as one radius measurement rather than 3 coincidentally same-coloured ones. The selected
        process's arc is drawn bold; when "Show all processes" is ticked, every other process's
        arc is drawn too, faded (see _faded_color) - see _decoratable_process_views. Refreshed by
        _refresh_radius_arc, called from _restyle_measurement_points so it updates live on every
        trigger that already recolours the points (selection, drag, Record/Clear, ...), and
        directly from the show_decorators_checkbox/show_all_processes_checkbox toggles.
        """
        if RADIUS_ARC_LAYER_NAME in self.viewer.layers:
            return self.viewer.layers[RADIUS_ARC_LAYER_NAME]
        layer = self.viewer.add_shapes(
            name=RADIUS_ARC_LAYER_NAME,
            ndim=4,
            shape_type="path",
            edge_color=RADIUS_COLOR,
            edge_width=4,
            face_color="transparent",
        )
        layer.editable = False
        self.viewer.layers.selection.active = self.layer_measurements
        return layer

    def _refresh_radius_arc(self) -> None:
        if RADIUS_ARC_LAYER_NAME not in self.viewer.layers:
            return
        # Mutating a Shapes layer's own .data (remove_selected()/add(), below) can trigger a
        # callback chain that reenters _restyle_measurement_points, which calls straight back into
        # this same method before the outer call has finished (see _restyle_measurement_points'
        # own reentrancy comment for the original occurrence of this). That method's _restyling
        # guard only protects calls made from INSIDE its own guarded block - it does nothing for
        # this method being called directly (show_decorators_checkbox/show_all_processes_checkbox
        # now do exactly that), so this method has to hold the same guard itself for the direct-call
        # case. _run_guarded_against_restyle_reentrancy makes that the same guard either way: a
        # direct call acquires it for the duration; a call already made from inside
        # _restyle_measurement_points's own guarded block (which already holds it) just runs.
        self._run_guarded_against_restyle_reentrancy(self._do_refresh_radius_arc)

    def _do_refresh_radius_arc(self) -> None:
        layer = self.viewer.layers[RADIUS_ARC_LAYER_NAME]

        # Assigning `.data = [...]` directly onto a Shapes layer - whether to clear it (`= []`)
        # or to replace its content - crashes in this napari version (a slicing internals bug,
        # not specific to this layer): selecting everything and removing it is the reliable way
        # to clear one. `.data = [...]` also silently defaults a new shape's type to "polygon"
        # (closed - draws an unwanted edge straight back to the start point) regardless of the
        # shape_type this layer was created with, rather than "path" (open) - `.add(...,
        # shape_type="path")` is the only reliable way found to set it correctly, so this always
        # clears first and re-adds, never reassigns `.data` in place. See _clear_shapes_layer for
        # why that clearing goes via a direct attribute assignment, not `layer.selected_data = ...`.
        self._clear_shapes_layer(layer)

        if not self.show_decorators_checkbox.isChecked():
            return

        current_view = self.viewer.dims.current_step[0]
        current_event = self.viewer.dims.current_step[1]
        for view_data, is_selected in self._decoratable_process_views(current_view, current_event):
            if len(view_data.track_points) != 3:
                continue
            arc_2d = radius_arc_points(*view_data.track_points)
            arc_shape = np.array([[current_view, current_event, *point] for point in arc_2d])
            if is_selected:
                layer.add(arc_shape, shape_type="path", edge_color=RADIUS_COLOR, edge_width=4)
            else:
                layer.add(
                    arc_shape, shape_type="path", edge_color=_faded_color(RADIUS_COLOR), edge_width=1.5
                )

    def _setup_origin_decay_arrow_layer(self):
        """A two-shape-per-process overlay (a line shaft plus a filled triangle arrowhead - see
        _calculate.origin_decay_arrow) pointing from each visible process's origin vertex to its
        decay vertex - the length equivalent of the radius arc, showing the two points are linked
        as one length measurement rather than two independently recorded ones. The selected
        process's arrow is drawn bold (ARROW_COLOR); when "Show all processes" is ticked, every
        other process's arrow is drawn too, faded - see _decoratable_process_views. Refreshed by
        _refresh_origin_decay_arrow, called from _restyle_measurement_points alongside the radius
        arc so both update on every trigger that already recolours the points (selection, drag,
        Record/Clear, ...), and directly from the show_decorators_checkbox/show_all_processes_checkbox
        toggles.
        """
        if ORIGIN_DECAY_ARROW_LAYER_NAME in self.viewer.layers:
            return self.viewer.layers[ORIGIN_DECAY_ARROW_LAYER_NAME]
        layer = self.viewer.add_shapes(
            name=ORIGIN_DECAY_ARROW_LAYER_NAME,
            ndim=4,
            edge_color=ARROW_COLOR,
            face_color=ARROW_COLOR,
            edge_width=3,
        )
        layer.editable = False
        self.viewer.layers.selection.active = self.layer_measurements
        return layer

    def _refresh_origin_decay_arrow(self) -> None:
        if ORIGIN_DECAY_ARROW_LAYER_NAME not in self.viewer.layers:
            return
        # See _refresh_radius_arc's comment on why this needs the same reentrancy guard.
        self._run_guarded_against_restyle_reentrancy(self._do_refresh_origin_decay_arrow)

    def _do_refresh_origin_decay_arrow(self) -> None:
        layer = self.viewer.layers[ORIGIN_DECAY_ARROW_LAYER_NAME]

        # Same napari Shapes-layer quirks as the radius arc - see _refresh_radius_arc's comment
        # and _clear_shapes_layer: always clear via that helper (never `.data = [...]`), always
        # add via .add(..., shape_type=...) (never rely on `.data =` for the shape type either).
        self._clear_shapes_layer(layer)

        if not self.show_decorators_checkbox.isChecked():
            return

        current_view = self.viewer.dims.current_step[0]
        current_event = self.viewer.dims.current_step[1]
        for view_data, is_selected in self._decoratable_process_views(current_view, current_event):
            if view_data.origin is None or view_data.decay is None:
                continue
            shaft_2d, head_2d = origin_decay_arrow(view_data.origin, view_data.decay)
            if shaft_2d is None:
                continue
            shaft_shape = np.array([[current_view, current_event, *point] for point in shaft_2d])
            head_shape = np.array([[current_view, current_event, *point] for point in head_2d])
            edge_color = ARROW_COLOR if is_selected else _faded_color(ARROW_COLOR)
            edge_width = 3 if is_selected else 1.5
            layer.add(shaft_shape, shape_type="line", edge_color=edge_color, edge_width=edge_width)
            layer.add(
                head_shape,
                shape_type="polygon",
                edge_color=edge_color,
                face_color=edge_color,
                edge_width=edge_width,
            )

    def _sync_other_processes_layer(self) -> None:
        """Populate the read-only 'other processes' layer for the current
        view/event, excluding whichever process is selected (its points
        already live on the interactive layer - no need to duplicate them
        here). Uses the same white-dot/coloured-ring/shaped look as the
        interactive layer (colour, symbol - see _role_symbol); the layer's
        own opacity is the only thing that distinguishes 'theirs' from 'mine'.

        Per-POINT styling only - the arc/arrow decorators for other processes are drawn
        separately, by _refresh_radius_arc/_refresh_origin_decay_arrow via
        _decoratable_process_views, which this checkbox also gates.
        """
        if OTHER_PROCESSES_LAYER_NAME not in self.viewer.layers:
            return

        layer = self.viewer.layers[OTHER_PROCESSES_LAYER_NAME]

        if not self.show_all_processes_checkbox.isChecked():
            layer.data = []
            return

        current_view = self.viewer.dims.current_step[0]
        current_event = self.viewer.dims.current_step[1]
        try:
            selected_row = self._get_selected_row()
        except IndexError:
            selected_row = None

        show_track = self.show_track_vertices_checkbox.isChecked()
        show_origin_decay = self.show_origin_decay_checkbox.isChecked()

        NORMAL_SIZE = 20
        HIDDEN_SIZE = 0

        points = []
        border_colors = []
        sizes = []
        symbols = []
        face_colors = []
        for i, particle in enumerate(self.data):
            if i == selected_row:
                continue
            if isinstance(particle, CalibrationRow):
                continue
            if particle.event_number != current_event:
                continue
            view_data = particle.views[current_view]

            origin_xy = _as_xy(view_data.origin) if view_data.origin is not None else None
            decay_xy = _as_xy(view_data.decay) if view_data.decay is not None else None
            length_xy = {xy for xy in (origin_xy, decay_xy) if xy is not None}
            # Same "complete set of 3 only" rule as _restyle_measurement_points - see its comment.
            radius_xy = {_as_xy(p) for p in view_data.track_points} if len(view_data.track_points) == 3 else set()

            for xy in length_xy | radius_xy:
                is_length = xy in length_xy
                is_radius = xy in radius_xy
                if is_length and is_radius:
                    color = BOTH_COLOR
                    visible = show_origin_decay or show_track
                elif is_length:
                    color = LENGTH_COLOR
                    visible = show_origin_decay
                else:
                    color = RADIUS_COLOR
                    visible = show_track

                points.append([current_view, current_event, xy[0], xy[1]])
                border_colors.append(color)
                sizes.append(NORMAL_SIZE if visible else HIDDEN_SIZE)
                # Same shape/fill distinction as _restyle_measurement_points - see _role_symbol
                # and _role_face_color.
                is_origin = xy == origin_xy
                is_decay = xy == decay_xy
                symbols.append(_role_symbol(is_origin=is_origin, is_decay=is_decay))
                face_colors.append(_role_face_color(is_origin=is_origin, is_decay=is_decay))

        layer.data = points
        if points:
            layer.face_color = face_colors
            layer.border_color = border_colors
            layer.border_width = [7] * len(points)
            layer.size = sizes
            layer.symbol = symbols

    def _invalidate_measurement_role_index_map(self, event=None) -> None:
        """A structural change to the measurement layer - a point added or removed, as opposed
        to "changing"/"changed" which only move an EXISTING point's value at a fixed index - can
        shift every later index up or down, making _measurement_role_index_map's cached
        index->role mapping point at entirely the wrong point. Connected ahead of
        _propagate_measurement_point_drag on layer.events.data, so that method sees the
        just-invalidated map on this very same event too, not just from the next one onwards.

        _measurement_role_index_map_valid only becomes True again once
        _sync_measurement_layer_to_selected_process has actually rebuilt the map from the
        current .data - not on any timer or count check. A point count can coincidentally
        recover after a delete-then-add (found live: it silently let a stale map merge two
        different roles into one identical coordinate, corrupting a radius fit into a
        degenerate triangle and crashing numpy with a singular-matrix error) - only knowing
        specifically that a rebuild happened is actually sound.
        """
        if event is not None and getattr(event, "action", None) in ("adding", "added", "removing", "removed"):
            self._measurement_role_index_map_valid = False

    def _propagate_measurement_point_drag(self, event=None) -> None:
        """Keep every role in _measurement_role_index_map synced to its point's live canvas
        position - both so a canvas point representing more than one role (e.g. a decay vertex
        reused as one of the three radius-fit points) moves all of them together, and so a
        single-role point's drag is reflected at all.

        Connected ahead of _on_measurement_points_changed on both layer.events.data and
        layer.events.highlight, and must run first: once every role is synced here, stored
        coordinates already match the canvas, so that method's own deletion-reconciliation
        (which compares stored vs on-canvas positions) correctly sees nothing missing, rather
        than mistaking an in-progress drag for a deletion.

        Deliberately event-shape-agnostic rather than trying to track exactly which point moved
        on which specific event: real napari drags turned out (found via a live trace, not
        documented behaviour) not to fire a "changing" data event on every mouse-move frame -
        only highlight does, and it carries no per-point delta at all. So instead of reading
        event.data_indices, this just re-reads every KNOWN role's live position by index on every
        opportunity it gets. That's only safe if no point has actually been added or removed since
        _measurement_role_index_map was last built - either can shift every later index up or
        down, so trusting stale indices then could silently relabel the wrong point.

        Guarded via _measurement_role_index_map_valid, set False by
        _invalidate_measurement_role_index_map on any structural change and only set True again
        once _sync_measurement_layer_to_selected_process has actually rebuilt the map. An earlier
        version of this guard compared point COUNTS instead ("has it dropped since the map was
        built") - found live to be unsound: a delete followed by an unrelated add elsewhere can
        restore the old count while every index underneath has shifted, and count-based
        comparison can't tell the difference. That let a stale map corrupt a radius fit into two
        identical track points (crashing numpy with a singular-matrix error) well after the
        actual deletion had happened, not right after it. Validity has to be tracked directly,
        not inferred from a number that can coincidentally recover.

        Also guarded against a real cross-view data corruption bug found live: switching the
        View/Event slider fires layer.events.highlight (napari's own internal re-slicing) before
        _sync_measurement_layer_to_selected_process - connected to the same dims.events.current_step
        signal - has run to rebuild _measurement_role_index_map for the new slice. Without the
        _last_synced_dims check below, this method would read current_view/view_data fresh (so
        already pointing at the NEW view's ViewData) but role_index_map and self.layer_measurements.data
        still describing the OLD view - copying one view's point straight into another view's
        data. _on_measurement_points_changed already carried this exact guard; this method never
        had it.
        """
        if event is None:
            return
        if getattr(event, "action", None) not in (None, "changing", "changed"):
            return  # explicitly not for add/remove-related actions - see docstring
        if getattr(self, "_restyling", False) or getattr(self, "_syncing", False):
            return

        current_view = self.viewer.dims.current_step[0]
        current_event = self.viewer.dims.current_step[1]
        current_dims_now = (current_view, current_event)
        if current_dims_now != getattr(self, "_last_synced_dims", None):
            # The View/Event slider has already moved on, but our own dims-triggered rebuild for
            # the new slice hasn't run yet - role_index_map and self.layer_measurements.data still
            # describe the OLD slice. Bail out - the sync callback about to run will fix this
            # properly (matching _on_measurement_points_changed's identical guard, for the
            # identical reason).
            return

        if not getattr(self, "_measurement_role_index_map_valid", False):
            _debug_trace("propagate: SKIP - role_index_map invalidated by a structural change "
                         "since it was last rebuilt")
            return

        role_index_map = getattr(self, "_measurement_role_index_map", {})
        if not role_index_map:
            return

        selected_row = self._get_selected_measurement_row()
        if selected_row is None:
            return

        if self.data[selected_row].event_number not in (-1, current_event):
            return  # viewer has wandered to a different event mid-navigation - not meaningful yet

        data = self.layer_measurements.data
        view_data = self.data[selected_row].views[current_view]

        _debug_trace(f"propagate: event_action={getattr(event, 'action', None)!r} "
                     f"role_index_map={role_index_map} "
                     f"BEFORE track_points={view_data.track_points} decay={view_data.decay} origin={view_data.origin}")

        for idx, roles in role_index_map.items():
            if idx >= len(data):
                continue
            new_xy = [float(data[idx][2]), float(data[idx][3])]
            for role in roles:
                if role == "origin":
                    view_data.set_origin(new_xy)
                elif role == "decay":
                    view_data.set_decay(new_xy)
                elif role.startswith("track"):
                    track_index = int(role[len("track"):])
                    if track_index < len(view_data.track_points):
                        updated_track_points = list(view_data.track_points)
                        updated_track_points[track_index] = new_xy
                        view_data.set_track_points(updated_track_points)

        _debug_trace(f"propagate: AFTER track_points={view_data.track_points} "
                     f"decay={view_data.decay} origin={view_data.origin} radius_px={view_data.radius_px}")

    def _selected_measurement_target(self):
        """The (row, ViewData) a measurement action should act on, or (None, None) if there's
        nothing valid selected in the table right now - no process row selected, or a
        CalibrationRow selected (which has no origin/decay/track measurements at all)."""
        selected_row = self._get_selected_measurement_row()
        if selected_row is None:
            return None, None
        current_view = self.viewer.dims.current_step[0]
        return selected_row, self.data[selected_row].views[current_view]

    def _selected_measurement_point_count(self) -> int:
        """How many currently-selected canvas points are on the view/event on screen right now -
        0 if the selection is empty or spans more than one slice. Used to decide which Record
        actions the right-click menu should currently offer; unlike
        _selected_points_are_on_current_slice, this never shows an error - it's queried on every
        right-click just to build the menu, not only when the user has committed to an action.
        """
        selected_points = self._get_selected_points()
        if len(selected_points) == 0:
            return 0
        current_step = self.viewer.dims.current_step
        if not all(current_step[i] == point[i] for point in selected_points for i in (0, 1)):
            return 0
        return len(selected_points)

    def _finish_measurement_action(self) -> None:
        """Shared tail for every Record/Clear measurement action below: rebuild the canvas -
        which also refreshes _measurement_role_index_map, stale otherwise and liable to break the
        next drag on a point whose role just changed - and its table cells, then recolour to
        match the new role assignment."""
        self._sync_measurement_layer_to_selected_process()
        self._restyle_measurement_points()

    def _record_origin_vertex(self) -> None:
        selected_row, view_data = self._selected_measurement_target()
        if selected_row is None:
            napari.utils.notifications.show_error("Select a process row before recording a vertex.")
            return
        selected_points = self._get_selected_points()
        if len(selected_points) != 1:
            napari.utils.notifications.show_error("Select exactly one point to record an origin vertex.")
            return
        if not self._selected_points_are_on_current_slice(selected_points):
            return  # already showed its own error
        view_data.set_origin(list(selected_points[0][2:]))
        self._finish_measurement_action()

    def _record_decay_vertex(self) -> None:
        selected_row, view_data = self._selected_measurement_target()
        if selected_row is None:
            napari.utils.notifications.show_error("Select a process row before recording a vertex.")
            return
        selected_points = self._get_selected_points()
        if len(selected_points) != 1:
            napari.utils.notifications.show_error("Select exactly one point to record a decay vertex.")
            return
        if not self._selected_points_are_on_current_slice(selected_points):
            return
        view_data.set_decay(list(selected_points[0][2:]))
        self._finish_measurement_action()

    def _record_radius(self) -> None:
        selected_row, view_data = self._selected_measurement_target()
        if selected_row is None:
            napari.utils.notifications.show_error("Select a process row before recording a radius.")
            return
        selected_points = self._get_selected_points()
        if len(selected_points) != 3:
            napari.utils.notifications.show_error("Select exactly three points to record a radius.")
            return
        if not self._selected_points_are_on_current_slice(selected_points):
            return
        view_data.set_track_points([list(p[2:]) for p in selected_points])
        self._finish_measurement_action()

    def _clear_origin_vertex(self) -> None:
        selected_row, view_data = self._selected_measurement_target()
        if selected_row is None:
            napari.utils.notifications.show_error("Select a process row before clearing a vertex.")
            return
        if view_data.origin is None:
            napari.utils.notifications.show_error("No origin vertex is currently recorded for this view.")
            return
        view_data.set_origin(None)
        self._finish_measurement_action()

    def _clear_decay_vertex(self) -> None:
        selected_row, view_data = self._selected_measurement_target()
        if selected_row is None:
            napari.utils.notifications.show_error("Select a process row before clearing a vertex.")
            return
        if view_data.decay is None:
            napari.utils.notifications.show_error("No decay vertex is currently recorded for this view.")
            return
        view_data.set_decay(None)
        self._finish_measurement_action()

    def _clear_radius(self) -> None:
        selected_row, view_data = self._selected_measurement_target()
        if selected_row is None:
            napari.utils.notifications.show_error("Select a process row before clearing a radius.")
            return
        if not view_data.track_points:
            napari.utils.notifications.show_error("No radius is currently recorded for this view.")
            return
        view_data.set_track_points([])
        self._finish_measurement_action()

    def _on_measurement_layer_right_click(self, layer, event):
        """Right-click drop-down menu for the measurement layer: lets the user choose what a
        selection of points means (origin vertex, decay vertex, or radius fit) instead of that
        being decided implicitly by how many points happen to be selected. That old behaviour
        punished anyone who could only reach a 3-point selection by ctrl-clicking one point at a
        time - the moment their selection passed through 2 points on its way to 3, it would
        already have been recorded as a length they never asked for.

        Only triggers when the right-click lands on one of the CURRENTLY SELECTED points, the
        usual "act on the selection" convention (file managers, word processors, ...) - selection
        and action are deliberately separate operations here, exactly as in those.
        """
        our_sort_of_event = event.type == "mouse_press" and event.button == 2
        if not our_sort_of_event:
            return

        coords = layer.world_to_data(event.position)

        while event.type != "mouse_release":
            yield

        index_under_cursor = layer.get_value(coords, world=True)
        if index_under_cursor is None or index_under_cursor not in layer.selected_data:
            return

        menu = QMenu(self.viewer.window._qt_window)

        header = QAction("Measurement actions:", menu)
        header.setEnabled(False)
        font = header.font()
        font.setBold(True)
        header.setFont(font)
        menu.addAction(header)
        menu.addSeparator()

        selected_row, view_data = self._selected_measurement_target()
        point_count = self._selected_measurement_point_count()
        have_target = selected_row is not None

        def add_action(label: str, enabled: bool, callback) -> None:
            action = QAction(label, menu)
            action.setEnabled(enabled)
            if enabled:
                action.triggered.connect(lambda _, cb=callback: cb())
            menu.addAction(action)

        add_action("Record origin vertex   (O)", have_target and point_count == 1, self._record_origin_vertex)
        add_action("Record decay vertex   (D)", have_target and point_count == 1, self._record_decay_vertex)
        add_action("Record radius   (R)", have_target and point_count == 3, self._record_radius)
        menu.addSeparator()
        add_action("Clear origin vertex", have_target and view_data.origin is not None, self._clear_origin_vertex)
        add_action("Clear decay vertex", have_target and view_data.decay is not None, self._clear_decay_vertex)
        add_action("Clear radius", have_target and bool(view_data.track_points), self._clear_radius)

        menu.exec_(event.native.globalPos())
        event.handled = True

    def _on_measurement_points_changed(self, event=None) -> None:
        """Live auto-calculation and cleanup - fires whenever a point on the measurement
        layer is placed, dragged, removed, or (re)selected.

        Reconciles deletions: if a point that used to be part of this view's saved
        origin/decay/track data is no longer on the canvas (e.g. selected and deleted with
        the 'x' tool), clears it from the stored data too.

        Deciding what a *new* selection means (an origin vertex, a decay vertex, a radius fit)
        is handled explicitly instead, via the right-click menu / keyboard shortcuts - see
        _on_measurement_layer_right_click and the Record/Clear methods above it. This method no
        longer reacts to selection counts at all.
        """
        if getattr(self, "_restyling", False) or getattr(self, "_syncing", False):
            return

        current_dims_now = (self.viewer.dims.current_step[0], self.viewer.dims.current_step[1])
        if current_dims_now != getattr(self, "_last_synced_dims", None):
            # The View/Event slider has already moved on, but our own dims-triggered rebuild for the
            # new slice hasn't run yet - napari can fire its own highlight event the instant the slider
            # moves, ahead of our own listener on that same event. The canvas data at this exact moment
            # still reflects the OLD slice, so comparing it against anything would be comparing against
            # a stale, about-to-be-replaced snapshot. Bail out - our own sync callback is about to run
            # and fix this properly.
            return

        selected_row = self._get_selected_measurement_row()
        if selected_row is None:
            return

        current_view = self.viewer.dims.current_step[0]
        current_event = self.viewer.dims.current_step[1]

        # If the viewer has already wandered to a different event than the one this process
        # actually belongs to - e.g. mid-drag of the Event slider, before the row has actually
        # been deselected - nothing below is meaningful: napari's own slice-change handling
        # can prune point selection and fire a highlight event right at this moment, and comparing
        # the process's real (correctly untouched) points against this wrong event's canvas slice
        # would misread "not on this slice" as "deleted". Bail out rather than trust it.
        if self.data[selected_row].event_number != current_event:
            return

        view_data = self.data[selected_row].views[current_view]

        current_xy_on_canvas = {
            _as_xy(p[2:])
            for p in self.layer_measurements.data
            if p[0] == current_view and p[1] == current_event
        }
        _debug_trace(f"reconcile: event_action={getattr(event, 'action', None)!r} "
              f"current_xy_on_canvas={current_xy_on_canvas} "
              f"origin={view_data.origin} decay={view_data.decay} track_points={view_data.track_points}")

        if view_data.origin is not None and _as_xy(view_data.origin) not in current_xy_on_canvas:
            _debug_trace(f"reconcile: CLEARING origin {view_data.origin}")
            view_data.set_origin(None)
        if view_data.decay is not None and _as_xy(view_data.decay) not in current_xy_on_canvas:
            _debug_trace(f"reconcile: CLEARING decay {view_data.decay}")
            view_data.set_decay(None)
        remaining_track_points = [p for p in view_data.track_points if _as_xy(p) in current_xy_on_canvas]
        if len(remaining_track_points) != len(view_data.track_points):
            _debug_trace(f"reconcile: CLEARING track_points {view_data.track_points} -> {remaining_track_points}")
            view_data.set_track_points(remaining_track_points)

        table_row = self._table_row_for_data_index(selected_row)
        self.table.setItem(
            table_row,
            self._get_table_column_index("decay_length_px"),
            _NumericTableWidgetItem(self._display_value(round_px(view_data.length_px))),
        )
        self.table.setItem(
            table_row,
            self._get_table_column_index("radius_px"),
            _NumericTableWidgetItem(self._display_value(round_px(view_data.radius_px))),
        )
        self._refresh_per_view_breakdown_cells(selected_row)
        self._refresh_saved_vertices_cell(selected_row)
        self._restyle_measurement_points()

    def _sync_calibration_rows_into_table(self) -> None:
        """Create or update a CalibrationRow (and its table row) for every event that has
        at least one fiducial stamped, mirroring calibration_manager.calibration_data.event_views.
        Called by CalibrationManager itself whenever a stamp is added or removed. New rows are not
        auto-selected the way a new process is - there's nothing else that needs to happen beyond
        the row simply appearing.
        """
        event_views = self.calibration_manager.calibration_data.event_views
        events_with_stamps = sorted({event for (event, view) in event_views})

        # An event that already has a row but has since had every stamp deleted no longer appears in
        # events_with_stamps at all (that dict is fully rebuilt from scratch each sync) - without this,
        # its row would just be silently skipped forever instead of being reset to show it's now empty.
        existing_calibration_events = {
            row.event_number for row in self.data if isinstance(row, CalibrationRow)
        }
        events_to_refresh = sorted(existing_calibration_events | set(events_with_stamps))

        for event_number in events_to_refresh:
            views = [FiducialViewData(), FiducialViewData(), FiducialViewData()]
            for view_index in range(3):
                stamped_view = event_views.get((event_number, view_index))
                if stamped_view is not None:
                    views[view_index] = stamped_view
            existing_row_index = None
            for i, row in enumerate(self.data):
                if isinstance(row, CalibrationRow) and row.event_number == event_number:
                    existing_row_index = i
                    break

            is_now_empty = all(len(v.stamped) == 0 for v in views)
            if is_now_empty:
                # No stamps left for this event at all - remove the row entirely rather than
                # leaving a permanent, un-removable "_ _ _" placeholder behind. "Delete process" is
                # deliberately blocked for calibration rows (10d-3), so deleting every stamp is the
                # only way a student has to remove one - this makes that action actually complete.
                if existing_row_index is not None:
                    # Must translate BEFORE deleting - _table_row_for_data_index reads
                    # self.data[existing_row_index]._row_id, gone once the entry's deleted.
                    table_row = self._table_row_for_data_index(existing_row_index)
                    del self.data[existing_row_index]
                    self.table.removeRow(table_row)
                continue

            if existing_row_index is not None:
                self.data[existing_row_index].views = views
                self._refresh_saved_vertices_cell(existing_row_index)
            else:
                new_row = CalibrationRow(event_number=event_number, views=views)
                self._assign_row_id(new_row)
                self.data.append(new_row)
                self.table.insertRow(self.table.rowCount())
                row_index = self.table.rowCount() - 1
                self._add_or_update_table_row(row_index, new_row)

    def _on_click_new_process(self) -> None:
        """When the 'New process' button is clicked, append a new blank row to
        the table and select the first cell ready to receive the first point.
        """
        if self.particle_decays_menu.currentIndex() < 1: # Not the header!
            return

        # add a new particle to data
        new_particle = ParticleDecay()
        nice_name = self.particle_decays_menu.currentText()
        # ascii_name = EXPECTED_PROCESSES_NICE_TO_ASCII[nice_name]
        new_particle.name = nice_name
        new_particle.index = self.particle_decays_menu.currentIndex()
        #new_particle.magnification_a = self.mag_a
        #new_particle.magnification_b = self.mag_b

        # Record the event and view number if the data has been loaded
        # Potentially this could be used to check the measurements are done in the right event
        data_has_been_loaded = IMAGE_LAYER_NAME in self.viewer.layers
        if data_has_been_loaded:
            new_particle.event_number = self.viewer.dims.current_step[1]
            new_particle.view_number = self.viewer.dims.current_step[0]

        self._assign_row_id(new_particle)
        self.data += [new_particle]

        # add particle (== new row) to the table and select it. _add_or_update_table_row must run
        # BEFORE selectRow, not after: selectRow synchronously fires _on_row_selection_changed,
        # which calls _get_selected_row(), which reads this row's _row_id cell - selecting the
        # row before that cell (or any other) has actually been written would crash.
        self.table.insertRow(self.table.rowCount())
        row_index = self.table.rowCount() - 1
        self._add_or_update_table_row(row_index, new_particle)
        self.table.selectRow(row_index)

        print(self.data[-1])
        self.particle_decays_menu.setCurrentIndex(0)

        # Adding a process is a strong signal you're about to place radius/length points - hand
        # focus back to that layer, in case you were last working in a calibration layer.
        if MEASUREMENTS_LAYER_NAME in self.viewer.layers:
            self.viewer.layers.selection.active = self.layer_measurements

    def _on_click_delete_process(self) -> None:
        """Delete particle from table and data"""
        try:
            selected_row = self._get_selected_row()
        except IndexError:
            napari.utils.notifications.show_error("The table of processes is empty so no process can be deleted.")
            return
        else:
            if isinstance(self.data[selected_row], CalibrationRow):
                napari.utils.notifications.show_error(
                    "Calibration rows can't be deleted here - delete the individual fiducial stamps on the per-image layer instead."
                )
                return
            confirmation_dialog = QMessageBox()
            confirmation_dialog.setText("Deleting selected particle")
            confirmation_dialog.setInformativeText("Do you want to continue?")
            confirmation_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            confirmation_dialog.setDefaultButton(QMessageBox.Cancel)
            return_code = confirmation_dialog.exec()

            if return_code == QMessageBox.Yes:
                # Must translate BEFORE deleting - _table_row_for_data_index reads
                # self.data[selected_row]._row_id, which no longer exists once the entry's gone.
                table_row = self._table_row_for_data_index(selected_row)
                del self.data[selected_row]
                self.table.removeRow(table_row)

    # def _propagate_magnification(self, a: float, b: float) -> None:
    #     """Assigns a and b to the class magnification parameters and to each of the particles in data"""
    #     self.mag_a = a
    #     self.mag_b = b
    #     for particle in self.data:
    #         particle.magnification_a = a
    #         particle.magnification_b = b

    # def _on_click_apply_magnification(self) -> None:
    #     """Changes the visualisation of the table to show calibrated values for radius and decay_length"""
    #     if self.apply_magnification_button.isChecked():
    #         self._apply_magnification()
    #     self._set_table_visible_vars(self.apply_magnification_button.isChecked())

    # def _apply_magnification(self) -> None:
    #     """Calculates magnification and calibrated radius and length for each particle in data"""
    #
    #     for i in range(len(self.data)):
    #         self.data[i].calibrate()
    #         self.table.setItem(
    #             i,
    #             self._get_table_column_index("magnification"),
    #             QTableWidgetItem(str(self.data[i].magnification)),
    #         )
    #         # if the radius has been computed before, show the calibrated value
    #         if self.table.item(i, self._get_table_column_index("radius_px")) is not None:
    #             self.table.setItem(
    #                 i,
    #                 self._get_table_column_index("radius_cm"),
    #                 QTableWidgetItem(str(self.data[i].radius_cm)),
    #             )
    #         if (
    #             self.table.item(i, self._get_table_column_index("decay_length_px"))
    #             is not None
    #         ):
    #             self.table.setItem(
    #                 i,
    #                 self._get_table_column_index("decay_length_cm"),
    #                 QTableWidgetItem(str(self.data[i].decay_length_cm)),
    #             )

    def _on_click_save(self) -> None:
        """Save list of particles to csv file. When the 'Save' button is clicked, the data
        is saved to a csv file with the current date and time as the filename.
        """

        generic_templates = self.calibration_manager.calibration_data.generic_templates
        has_generic_calibration = any(len(t.positions) > 0 for t in generic_templates)

        # TODO: CSV format still can't represent an empty process table - see the check inside
        # the .csv branch below.
        if not len(self.data) and not has_generic_calibration:
            napari.utils.notifications.show_error(
                "There is no data in the table to save."
            )
            print("There is no data in the table to save.")
            return

        # getSaveFileName is a static Qt method - calling it on an instance (the old code's
        # file_dialog.getSaveFileName(...)) still just invokes the static version, silently
        # ignoring any setNameFilter/setDefaultSuffix/setAcceptMode set on that instance. That's
        # the actual reason neither extension was ever being auto-appended. Calling it properly
        # as a static method and handling the extension explicitly, using the filter the user
        # actually picked (the second return value, previously discarded).
        file_name, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save file",
            self._last_save_process_table_dir,
            "Pickle files (*.pkl);;CSV files (*.csv)" if ENABLE_PICKLE else "CSV files (*.csv)",
            "Pickle files (*.pkl)" if ENABLE_PICKLE else "CSV files (*.csv)",
            QFileDialog.DontUseNativeDialog,
        )
        if file_name in {"", None}:
            return
        self._last_save_process_table_dir = os.path.dirname(file_name)

        # Only fill in a missing extension - an explicitly wrong one (e.g. someone typing "myfile.pdf")
        # should still fall through to the "invalid file type" branch below, not get silently coerced
        # into a valid extension.
        if os.path.splitext(file_name)[1] == "":
            file_name += ".csv" if "csv" in selected_filter.lower() else ".pkl"

        # Save as pickle if file_name ends with .pkl
        if file_name.endswith(".pkl"):
            if not ENABLE_PICKLE:
                napari.utils.notifications.show_error(
                    "Saving as .pkl is currently disabled - please save as .csv instead."
                )
                return
            session = SavedSession(data=self.data, generic_templates=generic_templates)
            with open(file_name, "wb") as handle:
                pickle.dump(session, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save as .csv if file_name ends with .csv
        elif file_name.endswith(".csv"):
            with open(file_name, "w", encoding="UTF8", newline="") as f:
                # Main table: every process and calibration entry, long format (one CSV column set shared
                # by both row types, 3 rows each - see CSV_COLUMNS/to_csv_rows for the full design). Unlike
                # the old single-row format, an empty self.data is fully representable here - the table just
                # has a header and zero data rows - so the old "CSV can't represent calibration-only data"
                # restriction no longer applies.
                f.write(",".join(CSV_COLUMNS) + "\n")
                for row_group_id, particle in enumerate(self.data):
                    f.write(particle.to_csv_rows(row_group_id))

                # Generic fiducial templates are workspace-level, not tied to any event, so they don't fit
                # the main table's row shape at all - a small second table, separated by a blank line, only
                # written if there's actually something in it.
                has_generic_calibration_data = any(len(t.positions) > 0 for t in generic_templates)
                if has_generic_calibration_data:
                    f.write("\n")
                    f.write("view,name,x,y,slot_index\n")
                    for view_index, template in enumerate(generic_templates):
                        for fiducial_name, xy in template.positions.items():
                            slot_index = template.slot_indices.get(fiducial_name, "")
                            f.write(f"{view_index},{fiducial_name},{round_px(xy[0])},{round_px(xy[1])},{slot_index}\n")

        else:
            self.msg = QMessageBox()
            self.msg.setIcon(QMessageBox.Warning)
            self.msg.setWindowTitle("Invalid file type")
            self.msg.setStandardButtons(QMessageBox.Ok)
            self.msg.setText(
                "The file must be a CSV (*.csv) or Pickle (*.pkl) file. Please try again."
            )
            self.msg.show()
            return

        #print(f"SSSSSAAAVING BEFORE {self._data_at_last_save=}")
        import copy
        self._data_at_last_save = copy.deepcopy(self.data) # mark as clean!  Need deepcopy as otherwise changes within ParticleData objects are not spotted!
        #print(f"SSSSSAAAVING AFTER {self._data_at_last_save=}")
        napari.utils.notifications.show_info("Data saved to " + file_name)
