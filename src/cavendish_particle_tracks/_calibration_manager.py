
import numpy as np
import napari

from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QTableWidgetItem,
    QAction,
    QMenu,
    QInputDialog,
)
from qtpy.QtGui import QCursor, QMouseEvent
from qtpy.QtCore import Qt, QEvent, QTimer, QPoint
from .napari_tools import make_move_only, overwrite_layer

"""
Calibration points (locations of fiducials) come in two types:

(1) generic points (also called Workspace points)
(2) points in specific events

The former (Generic or workspace points) are not associated to any individual event, 
but serve as markers for roughtly where they might be, or serve to assist in the placement
of specific points -- i.e. points of the latter type. E.g. the former could be default 
locations for the latter before the latter are committed or tweaked.

One can currently debug the calibration manager from within the napari console after
opening stereoshift_dialog with:

import cavendish_particle_tracks as cpt
cm = cpt.get_singleton().calibration_manager
cm.load_calibration()

import cavendish_particle_tracks as cpt
cm = cpt.get_singleton().calibration_manager
cm.save_calibration()

Note that it will likely move into the main wiget rather than be held by the stereo
dialog ... so the above may change.
"""

view_indices = (0, 1, 2)
GENERIC_CALIBRATION_LAYER_NAMES = [f"Calibration (generic; camera {v + 1})" for v in view_indices]
PER_IMAGE_CALIBRATION_LAYER_NAME = "Calibration (per-image)"


class CalibrationManager:
    """
    This class stores, restores, and manages access to generic
    and specific calibration data.
    """

    def filename_for_generic_calibration_layer(self, view_index):
        return "CPT_calibration_layer_GC_" + str(view_index)

    num_generic_front_back_fid_pairs = 3

    def __init__(self, viewer):

        self.viewer = viewer

        # TODO: Try to avoid re-storing this redundant list of generic calibration layers .... should to live only in viewer?
        self._generic_calibration_layers = self._setup_calibration_layers()  # Returns a list of napari point layers.

        # Make sure we are only shown when commanded!
        # We can only do this once we can call self.generic_calibration_layers()
        assert hasattr(self, "_generic_calibration_layers")
        self.set_calibration_layer_visibility_and_focus(False, False)

        # Lastly, setup callbacks:
        self._setup_callbacks()

    def _setup_callbacks(self):
        for layer in self.generic_calibration_layers():
            # This is the callback to allow right-click on generic fiducials:
            layer.mouse_drag_callbacks.append(self.on_mouse)
            make_move_only(layer)

        # This is the thing that changes which fiducials are visible when the view slider is slid:
        self.viewer.dims.events.current_step.connect(self.callback_calibration_layer_visibility)

        # symbol size management
        self.viewer.camera.events.zoom.connect(self.callback_symbol_size)

    # TODO: could make generic_calibration_layers subservient to generic_calibration_layer_names instead of current way round.
    def generic_calibration_layer_names(self):
        return [layer.name for layer in self.generic_calibration_layers()]

    # TODO: could make generic_calibration_layers subservient to generic_calibration_layer_names instead of current way round.
    def generic_calibration_layers(self):
        return self._generic_calibration_layers

    def all_calibration_layers(self):
        # A simple python list of napari points layers.
        return self.generic_calibration_layers()  # Later might add  [ self.specific_calibration_layer ]

    def load_calibration(self):
        self._setup_calibration_layers(read_from_file=True)
        self._refresh_visibility_and_focus_of_calibration_layers()
        self.refresh_symbol_sizes()

    def save_calibration(self):
        for i, layer in enumerate(self.generic_calibration_layers()):
            layer.save(self.filename_for_generic_calibration_layer(i)) # saves to csv file

    # Callback for when the 'View' slider changes:
    def callback_calibration_layer_visibility(self, event):
        self._refresh_visibility_and_focus_of_calibration_layers()

    def callback_symbol_size(self, event: napari.utils.events.Event):
        screen_pixels_per_data_pixel = event.value # This value (up to zoom changes since call)
        # should be the same as self.viewer.camera.zoom.
        # Since that is what the refresh_symbol_sizes method uses as its default, both of
        # the following should be equal for all practical puropses:

        # Alternative one:
        # self.refresh_symbol_sizes(screen_pixels_per_data_pixel)
        # Alternative two:
        self.refresh_symbol_sizes()

    def refresh_symbol_sizes(self, screen_pixels_per_data_pixel=None):
        # https://napari.org/dev/guides/events_reference.html says that
        # event.value is "Scale from canvas pixels to world pixels." which is
        # not very clear. Experiment seems to clarify that it is "data pixel width" / "screen pixel width"
        # which (be careful here!) has units of "screen_pixels_per_data_pixel".
        # rather than the reciprocal of this.

        # I would like symbols for fiducials to typically be a fixed number of screen pixels in height, so that
        # they remain easy to see even when you zoom out.
        # If you zoom in far enough you are probably trying to place them precisely, so in that case I may wish
        # them to shrink a bit for fine placement, but this might not be necessary.
        if screen_pixels_per_data_pixel == None:
            screen_pixels_per_data_pixel = self.viewer.camera.zoom

        symbol_sizes_as_fractions_of_generic_symbol_size = {
            "front" : 1.0,
            "back" : 0.7,
            "point" : 0.5,
            "" : 1.0, # Generic or no-name
            None : 1.0 # Generic or no-name
        }

        generic_symbol_size_in_screen_pixels = 20

        symbol_sizes_in_screen_pixels = {
            key : val*generic_symbol_size_in_screen_pixels
            for key, val in symbol_sizes_as_fractions_of_generic_symbol_size.items()
        }

        symbol_sizes_in_data_pixels = {
            key : val/screen_pixels_per_data_pixel
            for key, val in symbol_sizes_in_screen_pixels.items()
        }

        def data_pixel_size_for(typ: str):
            if typ in symbol_sizes_in_data_pixels:
                return symbol_sizes_in_data_pixels[typ]
            else:
                return symbol_sizes_in_data_pixels[None] # Generic or no-name

        for layer in self.generic_calibration_layers():
            types = layer.properties["types"]
            orig_symbol_sizes = layer.size.copy()
            new_symbol_sizes = [ data_pixel_size_for(typ)  for typ in types ]
            layer.size = new_symbol_sizes # This updates the symbol size as desired.

    # Show or hide calibration layers
    def set_calibration_layer_visibility_and_focus(self, visbility: bool, focus: bool):
        # visibility=True means that the correct view will be rendered and the others hidden, otherwise none will be shown.
        # focus=True means that when the relevant view is made visible, it will also be given focus.
        self._calibration_layer_visibility = visbility
        self._calibration_layer_focus = focus
        self._refresh_visibility_and_focus_of_calibration_layers()

    # Private method to make all the calibration layers invisible:
    def _hide_calibration_layers(self):
        for layer in self.generic_calibration_layers():
            if layer.visible != False: # Avoid generating unnecessary triggers:
                layer.visible = False

    # Make the correct calibration layers visible/invisible based on the view slider:
    def _show_and_activate_correct_calibration_layer(self):
        current_view = self.viewer.dims.current_step[0]  # axis 0 is 'View', 1 is 'Event', 2 and 3 are x and y

        for i, layer in enumerate(self.generic_calibration_layers()):
            # Make the current view active if so requested:
            if self._calibration_layer_focus and i == current_view:
                if self.viewer.layers.selection.active != layer: # Avoid generating unnecessary triggers
                    self.viewer.layers.selection.active = layer

            # Make the relevant views visible or invisible:
            desired_state =  (i == current_view)
            if layer.visible != desired_state: # Avoid generating unnecessary triggers:
                layer.visible = desired_state

    def _refresh_visibility_and_focus_of_calibration_layers(self):
        if self._calibration_layer_visibility:
            self._show_and_activate_correct_calibration_layer()
        else:
            self._hide_calibration_layers()

    def rename_point(self, idx, name, type):
        # print(f"Renaming point idx={idx} with name={name}")

        other_idx = None  # Default
        other_name = None  # Default
        if type == "front":
            other_idx = idx + 1  # we store front-then-back, so back is +1 on.
            other_name = name[:-1]  # all bar the last character (to remove prime)
        if type == "back":
            other_idx = idx - 1  # we store front-then-back, so front is -1 on.
            other_name = name + "'"

        for layer in self.generic_calibration_layers():
            # print(f"before alteration {layer.text}")
            layer.text.values[idx] = name  # This change is needed for display purposes.
            layer.properties["labels"][idx] = name  # This change is needed for saving purposes.
            # print(f"after  alteration {layer.text}")
            if other_idx is not None and other_name is not None:
                layer.text.values[other_idx] = other_name  # This change is needed for display purposes.
                layer.properties["labels"][other_idx] = other_name  # This change is needed for saving purposes.

            layer.refresh()

    def on_mouse(self, layer, event):
        # This implements a right-click drop-down menu in response to a point in a generic calibration layer.
        # Note that on mac CTRL-left-click is a synonym for vanilla right-click, so don't expect to be able to use
        # CTRL as a modifier for left-click!  Note that add-to-selection in mac is CMD-left-click, so no
        # conflict with that.

        if event.button == 2:  # right-click!  Testing for mouse_release not mouse_press due to napari/qt bug that means that canas gets stuck in drag mode

            print(f"Before {event.type=} {event.button=}")
            if event.type == "mouse_press":
                # Wait until mouse button release
                print("yielding")
                yield
            print(f"After {event.type=} {event.button=}")

            coords = layer.world_to_data(event.position)
            #print(f"coords = {coords}")
            index_of_nearest_point = layer.get_value(coords, world=True)
            #print(f"index_of_nearest_point is {index_of_nearest_point}")
            if index_of_nearest_point is None:
                # This was not a right-click on a point.
                return

            i = index_of_nearest_point  # Just abbreviation shorthand.

            type = layer.properties["types"][i]

            def show_drop_down_menu(type):
                # build popup menu
                menu = QMenu(self.viewer.window._qt_window)

                header = QAction("Set name:", menu)
                header.setEnabled(False)  # makes it unclickable
                font = header.font()
                font.setBold(True)
                header.setFont(font)
                menu.addAction(header)
                menu.addSeparator()

                from .analysis import FIDUCIAL_FRONT, FIDUCIAL_BACK

                if type == "front":
                    fixed_names = list(FIDUCIAL_FRONT.keys())
                elif type == "back":
                    fixed_names = list(FIDUCIAL_BACK.keys())
                else:
                    fixed_names = ["point"]

                # add fixed names
                for fname in fixed_names:
                    act = QAction(fname, menu)
                    act.triggered.connect(lambda _, f=fname: self.rename_point(i, f, type))
                    menu.addAction(act)

                type_is_fiducial = type in ["front", "back"]

                if not type_is_fiducial:
                    # "no name" entry
                    noname = QAction("❌ Delete name", menu)
                    noname.triggered.connect(lambda _: self.rename_point(i, "", type))
                    menu.addAction(noname)

                    # custom name entry
                    def custom_name_calback():
                        text, ok = QInputDialog.getText(
                            self.viewer.window._qt_window,
                            "Custom name",
                            "Enter name:",
                        )
                        if ok and text.strip():
                            self.rename_point(i, text.strip(), type)

                    menu.addSeparator()
                    custom_name_menu_item = QAction("Set custom name ...", menu)
                    custom_name_menu_item.triggered.connect(custom_name_calback)
                    menu.addAction(custom_name_menu_item)

                if type_is_fiducial:
                    def clone_into_current_image_callback():
                        pass

                    menu.addSeparator()
                    clone_into_current_image_menu_item = QAction("Save for this event ...", menu)
                    clone_into_current_image_menu_item.triggered.connect(clone_into_current_image_callback)
                    menu.addAction(clone_into_current_image_menu_item)

                # popup at cursor position
                menu.exec_(pos)  # use captured global cursor pos -- see (*) below

            # capture cursor now
            pos = QCursor.pos()  # (*)

            """
            We can't just call show_drop_down_menu() in the next line as it 
            results in our capuring the right click by hiding the 
            right mouse button RELEASE.  So we do this instead:
            """
            QTimer.singleShot(100, lambda: show_drop_down_menu(type))

    def _default_generic_calibration_layers(self):

        from .analysis import TYPICAL_IMAGE_LONG_SIZE_PIX, TYPICAL_IMAGE_SHORT_SIZE_PIX

        origin_x = 0.5 * TYPICAL_IMAGE_SHORT_SIZE_PIX # actually how far DOWN !!
        spread_x = 0.15 * TYPICAL_IMAGE_SHORT_SIZE_PIX # actually vertical spread !!

        point_origin_y = 0.25 * TYPICAL_IMAGE_LONG_SIZE_PIX # actually how far ACROSS !!
        fid_origin_y = 0.5 * TYPICAL_IMAGE_LONG_SIZE_PIX
        fid_step_y = 0.12 * TYPICAL_IMAGE_LONG_SIZE_PIX

        #sc = 100 # Works OK on CLG's macbook but not on linux
        #sc = 40 # Attempt to find something that works passably on both linux and mac.

        # First position the point being measured:
        labels = [ "point", ]
        symbols = [ "disc", ]
        colours = ["cyan", ]
        types = ["point", ]
        #symbol_sizes = [1 * sc, ]
        points_in_generic_view = [ [origin_x, point_origin_y, ], ]

        # Now position the Front/Back fiducial pairs:
        for i in range(CalibrationManager.num_generic_front_back_fid_pairs):
            labels += [ "", "", ]
            types += ["front", "back", ]
            symbols += ["x", "x",]
            #symbol_sizes += [1*sc, 0.5*sc,]
            points_in_generic_view += [
                [origin_x - spread_x, fid_origin_y + i * fid_step_y, ],
                [origin_x + spread_x, fid_origin_y + i * fid_step_y, ],
            ]
            if i==0:
                colours += [
                    "#55ff00",  # front fiducial (light green)
                    "#00aa00",  # back fiducial (dark green)
                    ]
            elif i==1:
                colours += [
                    "#ff5500",  # front fiducial (light red)
                    "#aa0000",  # back fiducial (dark red)
                ]
            else:
                colours += [
                    "#5500ff",  # front fiducial (light blue)
                    "#0000aa",  # back fiducial (dark blue)
                ]


        # Displace the generic points 100 to the left, or not at all, or 100 to the right, depending on view:
        points_in_view = [
            np.array([ np.array(point)+np.array([(v-1)*100, 0,]) for point in points_in_generic_view ])
            for v in view_indices
        ]

        # If in debug mode can replace the points and labels with ones that are physically interesting.
        # Don't give this option to the students!
        debug_fiducial_mode = True

        # The pre-made points assume 3 pairs of fiducials in each view, so:
        if debug_fiducial_mode and CalibrationManager.num_generic_front_back_fid_pairs == 3:
            from .analysis import debug_points_view_0_calibration_layer
            from .analysis import debug_points_view_1_calibration_layer
            from .analysis import debug_points_view_2_calibration_layer
            from .analysis import debug_point_labels_all_calibration_layers
            points_in_view = [
                debug_points_view_0_calibration_layer,
                debug_points_view_1_calibration_layer,
                debug_points_view_2_calibration_layer,
            ] # overwrites old points_in_view
            labels = debug_point_labels_all_calibration_layers

        for v in view_indices:
            print(f"Point in view {v} are {points_in_view[v]}")

        layers = [
            self._single_generic_configuration_layer
            (v, points_in_view[v], labels, colours, types, symbols, #symbol_sizes
            )
             for v in view_indices
            ]

        return layers

    def _get_generic_calibration_layers_from_file(self):
        layers = []

        for v in view_indices:

            from .io import read_csv_with_constructors

            filename = self.filename_for_generic_calibration_layer(v) + ".csv"

            Point = lambda x, y : np.array([float(x),float(y)])
            constructors = [
                (Point, 'axis-0', 'axis-1'),
                (str, 'labels'),
                (str, 'colours'),
                (str, 'types'),
                (str, 'symbols'),
                # (int, 'symbol_sizes')
                ]

            # points, labels, colours, types, symbols, symbol_sizes = read_csv_with_constructors(filename, constructors)
            points, labels, colours, types, symbols = read_csv_with_constructors(filename, constructors)

            layers.append(self._single_generic_configuration_layer(
                v, points, labels, colours, types, symbols, #symbol_sizes
            ))

        return layers

    def _single_generic_configuration_layer(self, view_index,
                                            points, labels, colours, types, symbols, #symbol_sizes
                                            ):
        props = {
            'labels': labels,
            'colours': colours,
            'types': types,
            'symbols': symbols,
            ### 'symbol_sizes': symbol_sizes,
        }
        layer = napari.layers.Points(
            points,
            name=GENERIC_CALIBRATION_LAYER_NAMES[view_index],
            # size=20,
            ### size=symbol_sizes,
            properties=props,
            border_width=7,
            border_width_is_relative=False,
            border_color=colours,
            face_color=colours,
            symbol=symbols,
            # out_of_slice_display=False,
            visible=False,
        )
        layer.text = {
            'string': labels,
            'color': colours,
            'size': 12,
            'anchor': 'center',
            'translation': np.array([-150, 0]),  # move text 150 (data) pixels up
        }
        return layer

    def _setup_calibration_layers(self, read_from_file=False):

        if read_from_file:
            layers = self._get_generic_calibration_layers_from_file()
        else:
            layers = self._default_generic_calibration_layers()

        # Overwrite data if layer already exists, otherwise make a note of new layers
        new_layers = []
        for layer in layers:
            if layer.name in self.viewer.layers:
                # Existing layer, so callbacks already exist too, so just overwrite old layer data:
                overwrite_layer(self.viewer.layers[layer.name], layer)
                print("\n\n REPLACING DATA \n\n")
            else:
                # New layer!
                new_layers.append(layer)
                print("\n\n NOTING NEW LAYER \n\n")

        # Tell Napari about any new generic calibration layers:
        for new_layer in new_layers:
            self.viewer.add_layer(new_layer)

        return new_layers
