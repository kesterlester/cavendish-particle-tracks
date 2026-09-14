"""
This module contains the main Cavendish Particle Tracks widget.

It's the students' home widget, that contains the table of particle decays, and
the buttons to perform all analysis calculations, and to export (save) the data
for further analysis.
"""

import glob
import pickle
import warnings

import dask.array
import napari
import numpy as np
from dask_image.imread import imread
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ._decay_angles_dialog import DecayAnglesDialog
from ._image_calibration_dialog import ImageCalibrationDialog
from ._settings import get_bypass, get_shuffling_seed
# from ._stereoshift_dialog import StereoshiftDialog
from ._calibration_manager import CalibrationManager
from .intercept_close import InterceptClose
from .analysis import EXPECTED_PROCESSES_NICE, VIEW_NAMES, ParticleDecay, VTX_ORIGIN, VTX_DECAY, VTX_NONE

ENABLE_MAG = False

MEASUREMENTS_LAYER_NAME = "Radii and Lengths"
OTHER_PROCESSES_LAYER_NAME = "Other Processes (view only)"
IMAGE_LAYER_NAME = "Bubble Chamber Data"

_singleton_instance = None

def get_singleton(viewer=None, docking_area: str = "bottom", data_folder=None):
    """Return the singleton ParticleTracksWidget, creating it if necessary."""
    global _singleton_instance
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

        self.docking_area = docking_area

        self.shuffling_seed = get_shuffling_seed(fallback=1)

        # define QtWidgets
        self.load_button = QPushButton("Load images")
        self.particle_decays_menu = QComboBox()
        self.particle_decays_menu.addItems(EXPECTED_PROCESSES_NICE)
        self.particle_decays_menu.setCurrentIndex(0)
        self.particle_decays_menu.currentIndexChanged.connect(self._on_click_new_process)
        self.delete_process = QPushButton("Delete process")
        self.decay_angles_button = QPushButton("Calculate decay angles")
        self.show_track_vertices_checkbox = QCheckBox("Show radius points")
        self.show_track_vertices_checkbox.setChecked(True)
        self.show_origin_decay_checkbox = QCheckBox("Show length points")
        self.show_origin_decay_checkbox.setChecked(True)
        self.show_all_processes_checkbox = QCheckBox("Show all processes")
        self.show_all_processes_checkbox.setChecked(False)
        # self.stereoshift_button = QPushButton("Stereoshift")
        self.image_calibration_button = QPushButton("Image Calibration")
        self.save_data_button = QPushButton("Save process table")

        # setup particle table
        self.table = self._set_up_table()
        self._set_table_visible_vars(False)
        self.table.selectionModel().selectionChanged.connect(
            self._on_row_selection_changed
        )
        # Apply magnification disabled until the magnification parameters are computed
        #self.apply_magnification_button = QRadioButton("Apply magnification")
        #self.apply_magnification_button.setEnabled(False)

        # connect callbacks
        self.load_button.clicked.connect(self._on_click_load_data)
        self.delete_process.clicked.connect(self._on_click_delete_process)
        self.decay_angles_button.clicked.connect(self._on_click_decay_angles)
        self.show_track_vertices_checkbox.stateChanged.connect(lambda _: self._restyle_measurement_points())
        self.show_origin_decay_checkbox.stateChanged.connect(lambda _: self._restyle_measurement_points())
        self.show_track_vertices_checkbox.stateChanged.connect(lambda _: self._sync_other_processes_layer())
        self.show_origin_decay_checkbox.stateChanged.connect(lambda _: self._sync_other_processes_layer())
        self.show_all_processes_checkbox.stateChanged.connect(lambda _: self._sync_other_processes_layer())
        #self.stereoshift_button.clicked.connect(self._on_click_stereoshift)
        #self.apply_magnification_button.toggled.connect(
        #    self._on_click_apply_magnification
        #)
        self.save_data_button.clicked.connect(self._on_click_save)

        self.image_calibration_button.clicked.connect(self._on_click_calibration)
        # TODO: find which of these works
        # https://napari.org/stable/gallery/custom_mouse_functions.html
        # self.viewer.mouse_press.callbacks.connect(self._on_mouse_press)
        # self.viewer.events.mouse_press(self._on_mouse_click)

        if self.docking_area == "bottom":
            self.buttonbox = QGridLayout()
            self.buttonbox.addWidget(self.load_button, 0, 0)
            self.buttonbox.addWidget(self.particle_decays_menu, 1, 0)
            self.buttonbox.addWidget(self.delete_process, 1, 1)
            self.buttonbox.addWidget(self.decay_angles_button, 2, 0)
            self.buttonbox.addWidget(self.save_data_button, 2, 1)
            self.buttonbox.addWidget(self.show_track_vertices_checkbox, 3, 0)
            self.buttonbox.addWidget(self.show_origin_decay_checkbox, 3, 1)
            self.buttonbox.addWidget(self.show_all_processes_checkbox, 4, 0)
            #self.buttonbox.addWidget(self.stereoshift_button, 5, 0)
            self.buttonbox.addWidget(self.image_calibration_button, 0, 1)
            #self.buttonbox.addWidget(self.apply_magnification_button, 4, 1)


            layout_outer = QHBoxLayout()
            self.setLayout(layout_outer)
            layout_outer.addLayout(self.buttonbox)
            self.layout().addWidget(self.table)

        else:
            self.buttonbox = QVBoxLayout()
            self.buttonbox.addWidget(self.load_button)
            self.buttonbox.addWidget(self.particle_decays_menu)
            self.buttonbox.addWidget(self.delete_process)
            self.buttonbox.addWidget(self.decay_angles_button)
            self.buttonbox.addWidget(self.show_track_vertices_checkbox)
            self.buttonbox.addWidget(self.show_origin_decay_checkbox)
            self.buttonbox.addWidget(self.show_all_processes_checkbox)
            self.buttonbox.addWidget(self.table)
            #self.buttonbox.addWidget(self.apply_magnification_button)
            #self.buttonbox.addWidget(self.stereoshift_button)
            self.buttonbox.addWidget(self.image_calibration_button)
            self.buttonbox.addWidget(self.save_data_button)
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
        import copy
        self._data_at_last_save = copy.deepcopy(self.data) # Need deepcopy as otherwise changes within ParticleData objects are not spotted!

        # self.mag_a = -1.0
        # self.mag_b = 0.0

        # Dialog pointers to reuse
        self.mag_dlg: ImageCalibrationDialog | None = None
        #self.stereoshift_dlg: StereoshiftDialog | None = None
        self.decay_angles_dlg: DecayAnglesDialog | None = None

        @self.viewer.layers.events.connect
        def _on_layerlist_changed(event):
            """When the layer list changes, update the button availability"""
            self.set_button_availability()

        if data_folder is not None:
            self._load_data_from(data_folder)

        self._last_synced_dims = None
        self.calibration_manager = CalibrationManager(self, self.viewer)
        self.viewer.dims.events.current_step.connect(self._sync_measurement_layer_to_selected_process)

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

    def _get_selected_row(self) -> np.array:
        """Returns the selected row in the table.

        Note: due to our selection mode only one row selection is possible.
        """
        select = self.table.selectionModel()
        rows = select.selectedRows()
        return rows[0].row()

    def _set_up_table(self) -> QTableWidget:
        """Initial setup of the QTableWidget with one row and columns for each
        point and the calculated radius.
        """
        np = ParticleDecay()
        self.columns = list(np.vars_to_save())
        # self.columns += ["magnification"]
        self.columns_show_calibrated = np.vars_to_show(True)
        self.columns_show_uncalibrated = np.vars_to_show(False)
        out = QTableWidget(0, len(self.columns))
        out.setHorizontalHeaderLabels(self.columns)
        out.setSelectionBehavior(QAbstractItemView.SelectRows)
        out.setSelectionMode(QAbstractItemView.SingleSelection)
        out.setEditTriggers(QAbstractItemView.NoEditTriggers)
        out.setSelectionBehavior(QTableWidget.SelectRows)
        return out

    def _set_table_visible_vars(self, calibrated) -> None:
        for _ in range(len(self.columns)):
            self.table.setColumnHidden(_, True)
        show = (
            self.columns_show_calibrated if calibrated else self.columns_show_uncalibrated
        )
        show_index = [i for i, item in enumerate(self.columns) if item in set(show)]
        for _ in show_index:
            self.table.setColumnHidden(_, False)

    def _get_table_column_index(self, columntext: str) -> int:
        """Given a column title, return the column index in the table"""
        for i, item in enumerate(self.columns):
            if item == columntext:
                return i

        print("Column ", columntext, " not in the table")
        return -1

    def _on_row_selection_changed(self) -> None:
        """Enable/disable calculation buttons depending on the row selection, jump the
        viewer to the selected process's event, and refresh the canvas cursors to match.
        The canvas refresh must always run, even when nothing is selected - that's how
        the canvas gets correctly cleared when a process is deselected (e.g. by the
        dims-mismatch guard in _sync_measurement_layer_to_selected_process), so it can't be
        skipped early just because there's no event to jump to.
        """
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

    def _restyle_measurement_points(self) -> None:
        """Ring-highlight each point on the measurement layer to show what it currently contributes:
        a length pair (origin/decay), a radius fit (three track points), both at once (rare), or
        neither (a stray leftover / unowned point, left at the default
        style). The dot's white fill never changes - only the border.
        """
        if MEASUREMENTS_LAYER_NAME not in self.viewer.layers:
            return

        data = self.layer_measurements.data
        if len(data) == 0:
            return

        DEFAULT_BORDER_COLOR = "dimgrey"
        DEFAULT_BORDER_WIDTH = 7
        LENGTH_COLOR = "cornflowerblue"
        RADIUS_COLOR = "mediumorchid"
        BOTH_COLOR = "slateblue"

        length_points = []
        radius_points = []
        try:
            selected_row = self._get_selected_row()
        except IndexError:
            selected_row = None

        if selected_row is not None:
            current_view = self.viewer.dims.current_step[0]
            current_event = self.viewer.dims.current_step[1]
            if self.data[selected_row].event_number == current_event:
                view_data = self.data[selected_row].views[current_view]
                if view_data.origin is not None:
                    length_points.append((current_view, current_event, *view_data.origin))
                if view_data.decay is not None:
                    length_points.append((current_view, current_event, *view_data.decay))
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
        for point in data:
            is_length = matches(point, length_points)
            is_radius = matches(point, radius_points)
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
            self.layer_measurements.current_border_color = DEFAULT_BORDER_COLOR
            self.layer_measurements.current_border_width = DEFAULT_BORDER_WIDTH
            # Force the repaint explicitly, rather than relying on one of the assignments above
            # to trigger it as a side effect - with border_width now staying at a single uniform
            # value, napari may treat that particular assignment as a no-op and skip its own
            # repaint, leaving the (correct) colour applied in data but not actually drawn until
            # something else forces a real re-slice (e.g. navigating to a different event and back).
            self.layer_measurements.refresh()
        finally:
            self._restyling = False

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
        other_slices = [
            point
            for point in existing_data
            if not (point[0] == current_view and point[1] == current_event)
        ]

        new_points = []
        view_data = None
        if selected_row is not None:
            view_data = self.data[selected_row].views[current_view]
            for point in (view_data.origin, view_data.decay, *view_data.track_points):
                if point is not None:
                    new_points.append([current_view, current_event, point[0], point[1]])

        # Rewriting .data here is our own routine canvas rebuild, not a real user action - guard it so
        # _on_measurement_points_changed doesn't treat this rebuild as evidence that points were deleted
        # (that reconciliation logic must only ever react to something the user actually did to the canvas).
        self._syncing = True
        try:
            self.layer_measurements.selected_data = set()
            self.layer_measurements.data = other_slices + new_points
            self.layer_measurements.selected_data = set()
        finally:
            self._syncing = False
        self._last_synced_dims = (current_view, current_event)

        if selected_row is not None:
            self.table.setItem(
                selected_row,
                self._get_table_column_index("decay_length_px"),
                QTableWidgetItem(str(view_data.length_px)),
            )
            self.table.setItem(
                selected_row,
                self._get_table_column_index("radius_px"),
                QTableWidgetItem(str(view_data.radius_px)),
            )

        self._restyle_measurement_points()
        self._sync_other_processes_layer()

    def set_button_availability(self) -> None:
        images_imported = False
        for layer in self.viewer.layers:
            if layer.name == IMAGE_LAYER_NAME:
                images_imported = True
                break
        self.set_UI_image_loaded(images_imported, self.bypass_force_load_data)
        try:
            selected_row = self._get_selected_row()
            self.save_data_button.setEnabled(True)
            self.delete_process.setEnabled(True)
            ## think about these two + cal once done.
            self.image_calibration_button.setEnabled(True)
            #self.stereoshift_button.setEnabled(True)
            if self.data[selected_row].index < 4:
                self.decay_angles_button.setEnabled(False)
                return
            elif self.data[selected_row].index == 4:
                self.decay_angles_button.setEnabled(True)
                return
        except IndexError:
            self.delete_process.setEnabled(False)
            self.decay_angles_button.setEnabled(False)
            # self.apply_magnification_button.setEnabled(False)
            #self.stereoshift_button.setEnabled(False)
            # self.magnification_button.setEnabled(False)
            self.save_data_button.setEnabled(False)

    def set_UI_image_loaded(self, loaded: bool, bypass_load_screen: bool) -> None:
        if bypass_load_screen:
            return
        if loaded:
            self.load_button.setEnabled(False)
            self.particle_decays_menu.setEnabled(True)
            self.image_calibration_button.setEnabled(True)
        else:
            self.load_button.setEnabled(True)
            self.particle_decays_menu.setEnabled(False)
            self.delete_process.setEnabled(False)
            self.decay_angles_button.setEnabled(False)
            #self.stereoshift_button.setEnabled(False)
            self.save_data_button.setEnabled(False)
            self.image_calibration_button.setEnabled(False)
            #if ENABLE_MAG:
            #self.apply_magnification_button.setEnabled(False)

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

    def put_xy_and_view_into_table(self,
                                    xy,
                                    view,
                                    is_origin_vertex : bool, # False implies is_decay_vertex
                                    delete = False,
                                    ):
        try:
            selected_row = self._get_selected_row()
        except IndexError:
            napari.utils.notifications.show_error("The table of processes is empty. Create a process first.")
            return
        else:
            if delete:
                napari.utils.notifications.show_info(
                    f"Deleting coords {xy} from row {selected_row+1} of table for view {view}.")
            else:
                napari.utils.notifications.show_info(
                    f"Adding coords {xy} to row {selected_row+1} of table for view {view}.")

            if not delete:
                x, y = xy
                # Next two lines break a numpy link. Just seems sensible to do.
                x = float(x)
                y = float(y)
                marker_char = VTX_ORIGIN if is_origin_vertex else VTX_DECAY
            else:
                x, y = "", ""
                marker_char = VTX_NONE
            marker_char_pos = view + (0 if is_origin_vertex else 4)

            old_saved_vertices = self.data[selected_row].saved_vertices
            # replace char of old_saved_vertices at index marker_char_pos with marker_char:
            new_saved_vertices = old_saved_vertices[:marker_char_pos] + marker_char + old_saved_vertices[marker_char_pos+1:]

            self.data[selected_row].saved_vertices = new_saved_vertices
            self.table.setItem(
                selected_row,
                self._get_table_column_index("saved_vertices"),
                QTableWidgetItem(str(self.data[selected_row].saved_vertices)),
            )

            #print(f"put_xy_and_view_into_table will be using {x=} and {y=} when {xy=} as {delete=}")

            if is_origin_vertex:
                if view == 0:
                    self.data[selected_row].origin_v0_x = x
                    self.data[selected_row].origin_v0_y = y
                if view == 1:
                    self.data[selected_row].origin_v1_x = x
                    self.data[selected_row].origin_v1_y = y
                if view == 2:
                    self.data[selected_row].origin_v2_x = x
                    self.data[selected_row].origin_v2_y = y
            else: # decay vertex
                if view == 0:
                    self.data[selected_row].decay_v0_x = x
                    self.data[selected_row].decay_v0_y = y
                if view == 1:
                    self.data[selected_row].decay_v1_x = x
                    self.data[selected_row].decay_v1_y = y
                if view == 2:
                    self.data[selected_row].decay_v2_x = x
                    self.data[selected_row].decay_v2_y = y

    # _on_click_radius() and _on_click_length() used to live here - now removed that selecting
    # points auto-calculates both live (see _on_measurement_points_changed in _setup_measurement_layer).

    def _on_click_decay_angles(self) -> DecayAnglesDialog:
        """When the 'Calculate decay angles' buttong is clicked, open the decay angles dialog"""
        if self.decay_angles_dlg is not None:
            self.decay_angles_dlg.show()
            self.decay_angles_dlg.raise_()
            self._activate_calibration_layer(self.decay_angles_dlg.cal_layer)
            return self.decay_angles_dlg
        self.decay_angles_dlg = DecayAnglesDialog(self)
        self.decay_angles_dlg.show()
        self.decay_angles_dlg.raise_()
        return self.decay_angles_dlg

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
            "./",
            QFileDialog.DontUseNativeDialog
            | QFileDialog.DontResolveSymlinks
            | QFileDialog.ShowDirsOnly
            | QFileDialog.HideNameFilterDetails,
        )

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

        # Move to the first event in the series
        self.viewer.dims.set_current_step(1, 0)

        # Create measurements layer if not already there
        self.layer_measurements = self._setup_measurement_layer()
        self._setup_other_processes_layer()

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
            layer.events.data.connect(self._on_measurement_points_changed)
            layer.events.highlight.connect(self._on_measurement_points_changed)
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

    def _sync_other_processes_layer(self) -> None:
        """Populate the read-only 'other processes' layer for the current
        view/event, excluding whichever process is selected (its points
        already live on the interactive layer - no need to duplicate them
        here). Uses the same white-dot/coloured-ring look as the
        interactive layer; the layer's own opacity is the only thing that
        distinguishes 'theirs' from 'mine'.
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
        LENGTH_COLOR = "cornflowerblue"
        RADIUS_COLOR = "mediumorchid"

        BOTH_COLOR = "slateblue"

        def as_xy(point):
            return (round(float(point[0]), 6), round(float(point[1]), 6))

        points = []
        border_colors = []
        sizes = []
        for i, particle in enumerate(self.data):
            if i == selected_row:
                continue
            if particle.event_number != current_event:
                continue
            view_data = particle.views[current_view]

            length_xy = {as_xy(p) for p in (view_data.origin, view_data.decay) if p is not None}
            radius_xy = {as_xy(p) for p in view_data.track_points}

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

        layer.data = points
        if points:
            layer.face_color = ["white"] * len(points)
            layer.border_color = border_colors
            layer.border_width = [7] * len(points)
            layer.size = sizes

    def _on_measurement_points_changed(self, event=None) -> None:
        """Live auto-calculation and cleanup - fires whenever a point on the measurement
        layer is placed, dragged, removed, or (re)selected.

        First, reconciles deletions: if a point that used to be part of this view's saved
        origin/decay/track data is no longer on the canvas (e.g. selected and deleted with
        the 'x' tool), clears it from the stored data too.

        Then, if exactly 2 or 3 points are currently selected (and all on the current View/Event
        slice), treats them as a fresh origin/decay pair or radius fit and saves that instead.
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

        try:
            selected_row = self._get_selected_row()
        except IndexError:
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

        def as_xy(point):
            return (round(float(point[0]), 6), round(float(point[1]), 6))

        current_xy_on_canvas = {
            as_xy(p[2:])
            for p in self.layer_measurements.data
            if p[0] == current_view and p[1] == current_event
        }

        if view_data.origin is not None and as_xy(view_data.origin) not in current_xy_on_canvas:
            view_data.set_origin(None)
        if view_data.decay is not None and as_xy(view_data.decay) not in current_xy_on_canvas:
            view_data.set_decay(None)
        remaining_track_points = [p for p in view_data.track_points if as_xy(p) in current_xy_on_canvas]
        if len(remaining_track_points) != len(view_data.track_points):
            view_data.set_track_points(remaining_track_points)

        selected_points = self._get_selected_points()
        if len(selected_points) in (2, 3):
            on_current_slice = all(
                self.viewer.dims.current_step[i] == point[i]
                for point in selected_points
                for i in (0, 1)
            )
            if on_current_slice:
                selected_points_xy = [point[2:] for point in selected_points]
                if len(selected_points_xy) == 2:
                    view_data.set_origin(list(selected_points_xy[0]))
                    view_data.set_decay(list(selected_points_xy[1]))
                else:
                    view_data.set_track_points([list(p) for p in selected_points_xy])

        self.table.setItem(
            selected_row,
            self._get_table_column_index("decay_length_px"),
            QTableWidgetItem(str(view_data.length_px)),
        )
        self.table.setItem(
            selected_row,
            self._get_table_column_index("radius_px"),
            QTableWidgetItem(str(view_data.radius_px)),
        )
        self._restyle_measurement_points()

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

        self.data += [new_particle]

        # add particle (== new row) to the table and select it
        self.table.insertRow(self.table.rowCount())
        self.table.selectRow(self.table.rowCount() - 1)
        self.table.setItem(
            self.table.rowCount() - 1,
            self._get_table_column_index("index"),
            QTableWidgetItem(str(new_particle.index)),
        )
        self.table.setItem(
            self.table.rowCount() - 1,
            self._get_table_column_index("name"),
            QTableWidgetItem(new_particle.name),
        )
        self.table.setItem(
            self.table.rowCount() - 1,
            self._get_table_column_index("event_number"),
            QTableWidgetItem(str(new_particle.event_number)),
        )
        self.table.setItem(
            self.table.rowCount() - 1,
            self._get_table_column_index("saved_vertices"),
            QTableWidgetItem(str(new_particle.saved_vertices)),
        )
        #self.table.setItem(
        #    self.table.rowCount() - 1,
        #    self._get_table_column_index("magnification"),
        #    QTableWidgetItem(str(new_particle.magnification)),
        #)

        print(self.data[-1])
        self.particle_decays_menu.setCurrentIndex(0)

    def _on_click_delete_process(self) -> None:
        """Delete particle from table and data"""
        try:
            selected_row = self._get_selected_row()
        except IndexError:
            napari.utils.notifications.show_error("The table of processes is empty so no process can be deleted.")
            return
        else:
            confirmation_dialog = QMessageBox()
            confirmation_dialog.setText("Deleting selected particle")
            confirmation_dialog.setInformativeText("Do you want to continue?")
            confirmation_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            confirmation_dialog.setDefaultButton(QMessageBox.Cancel)
            return_code = confirmation_dialog.exec()

            if return_code == QMessageBox.Yes:
                del self.data[selected_row]
                self.table.removeRow(selected_row)

    def _on_click_calibration(self) -> ImageCalibrationDialog:
        """When the 'image calibratiob' button is clicked, open the image calibration dialog"""
        if self.mag_dlg is None:
            self.mag_dlg = ImageCalibrationDialog(self)

        self.mag_dlg.show()
        self.mag_dlg.raise_()
        return self.mag_dlg

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
        """Save list of particles to csv file.
        When the 'Save' button is clicked, the data is saved to a csv file with the current date and time as the filename.
        """

        # TODO: This is bad. There should be no reason a user cannot save
        # an empty file. However, it seems to be here as some of the code
        # below needs to access self.data[0] to write headers!!!
        if not len(self.data): # Disabling as no reason not to save
            napari.utils.notifications.show_error(
                "There is no data in the table to save."
            )
            print("There is no data in the table to save.")
            return

        # setup UI
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("CSV files (*.csv); Pickle files (*.pkl)")
        file_dialog.setDefaultSuffix("csv")
        # retrieve image folder
        file_name, _ = file_dialog.getSaveFileName(
            self,
            "Save file",
            "./",
            "CSV files (*.csv);;Pickle files (*.pkl)",
            "CSV files (*.csv)",
            QFileDialog.DontUseNativeDialog,
        )

        if file_name in {"", None}:
            return

        # Save as pickle if file_name ends with .pkl
        if file_name.endswith(".pkl"):
            with open(file_name, "wb") as handle:
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save as .csv if file_name ends with .csv
        elif file_name.endswith(".csv"):
            with open(file_name, "w", encoding="UTF8", newline="") as f:
                # write the header
                f.write(",".join(self.data[0].vars_to_save()) + "\n") # TODO: FIX! Should not access data[0] as this prevents saving empty file.

                # write the data
                f.writelines([particle.to_csv() for particle in self.data])

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

    # Probably no longer needed once mag and angle dialogs work same way as stereo!
    def _activate_calibration_layer(self, layer):
        """Show the calibration layer and move it to the top"""
        layer.visible = True
        # Move the calibration layer to the top
        self.viewer.layers.move(
            self.viewer.layers.index(layer),
            len(self.viewer.layers),
        )
        self.viewer.layers.selection.active = layer

    # Probably no longer needed once mag and angle dialogs work same way as stereo!
    def _deactivate_calibration_layer(self, layer):
        """Hide the calibration layer and move it to the bottom"""
        self.viewer.layers.select_previous()
        layer.visible = False
        # Move the calibration layer to the bottom
        self.viewer.layers.move(self.viewer.layers.index(layer), 0)

