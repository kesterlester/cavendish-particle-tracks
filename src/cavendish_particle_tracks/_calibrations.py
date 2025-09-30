
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
cm._setup_calibration_layers(read_from_file=True)
cm._refresh_visibility_and_focus_of_calibration_layers()

import cavendish_particle_tracks as cpt
cm = cpt.get_singleton().calibration_manager
cm.save_calibration()

Note that it will likely move into the main wiget rather than be held by the stereo
dialog ... so the above may change.
"""

view_indices = (0, 1, 2)
names_of_generic_calibration_layers = [f"Calibration workspace for view {v}" for v in view_indices]


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

    def generic_calibration_layers(self):
        return self._generic_calibration_layers

    def all_calibration_layers(self):
        # A simple python list of napari points layers.
        return self.generic_calibration_layers()  # Later might add  [ self.specific_calibration_layer ]

    def save_calibration(self):
        for i, layer in enumerate(self.generic_calibration_layers()):
            layer.save(self.filename_for_generic_calibration_layer(i)) # saves to csv file

    # Callback for when the 'View' slider changes:
    def callback_calibration_layer_visibility(self, event):
        self._refresh_visibility_and_focus_of_calibration_layers()

    # Show or hide calibration layers
    def set_calibration_layer_visibility_and_focus(self, visbility: bool, focus: bool):
        # visibility=True means that the correct view will be rendered and the others hidden, otherwise none will be shown.
        # focus=True means that when the relevant view is made visible, it will also be given focus.
        self._calibration_layer_visibility = visbility
        self._calibration_layer_focus = focus
        self._refresh_visibility_and_focus_of_calibration_layers()

    # Private method to Make all the calibration layers invisible:
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
            layer.text.values[idx] = name
            # print(f"after  alteration {layer.text}")
            if other_idx is not None and other_name is not None:
                layer.text.values[other_idx] = other_name

            layer.refresh()

    def on_mouse(self, layer, event):
        print(f"Detected mouse event {event} on layer {layer}")
        if event.button == 2:
            coords = layer.world_to_data(event.position)
            print(f"coords = {coords}")
            ind = layer.get_value(coords, world=True)
            print(f"ind is {ind}")
            if ind is None:
                return
            i = ind

            type = layer.properties["types"][i]

            print(f" got 1 and type {type}")

            def show_menu(type):
                # build popup menu
                menu = QMenu(self.viewer.window._qt_window)

                header = QAction("Set name", menu)
                header.setEnabled(False)  # makes it unclickable
                font = header.font()
                font.setBold(True)
                header.setFont(font)
                menu.addAction(header)
                menu.addSeparator()

                # fixed_names = ["Alpha", "Beta", "Gamma"]
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

                print(f" got 2 ")

                # "no name" entry
                noname = QAction("❌ Clear", menu)
                noname.triggered.connect(lambda _: self.rename_point(i, "", type))
                menu.addAction(noname)

                print(f" got 3 ")

                # custom name entry
                def custom_name():
                    print(f"In custom name")
                    text, ok = QInputDialog.getText(
                        self.viewer.window._qt_window,
                        "Custom name",
                        "Enter name:",
                    )
                    if ok and text.strip():
                        self.rename_point(i, text.strip(), type)

                print(f" got 4 ")

                menu.addSeparator()
                custom = QAction("Custom name ...", menu)
                custom.triggered.connect(custom_name)
                menu.addAction(custom)

                print(f" got 5 ")

                # popup at cursor position
                menu.exec_(pos)  # use captured global cursor pos -- see (*) below

            print(" got 6 ")

            # capture cursor now
            pos = QCursor.pos()  # (*)

            """
            We can't just all show_menu() in the next line as it 
            results in our capuring the right click by hiding the 
            right mouse button RELEASE.
            """
            QTimer.singleShot(100, lambda: show_menu(type))

            print(f" got 7 ")





    def _default_generic_calibration_layers(self):

        from .analysis import TYPICAL_IMAGE_LONG_SIZE_PIX, TYPICAL_IMAGE_SHORT_SIZE_PIX

        origin_x = 0.5 * TYPICAL_IMAGE_SHORT_SIZE_PIX # actually how far DOWN !!
        spread_x = 0.15 * TYPICAL_IMAGE_SHORT_SIZE_PIX # actually vertical spread !!

        point_origin_y = 0.25 * TYPICAL_IMAGE_LONG_SIZE_PIX # actually how far ACROSS !!
        fid_origin_y = 0.5 * TYPICAL_IMAGE_LONG_SIZE_PIX
        fid_step_y = 0.12 * TYPICAL_IMAGE_LONG_SIZE_PIX

        sc = 100

        # First position the point being measured:
        labels = [ "point", ]
        symbols = [ "disc", ]
        colours = ["cyan", ]
        types = ["point", ]
        symbol_sizes = [1 * sc, ]
        points_in_generic_view = [ [origin_x, point_origin_y, ], ]

        # Now position the Front/Back fiducial pairs:
        for i in range(CalibrationManager.num_generic_front_back_fid_pairs):
            labels += [ "", "", ]
            types += ["front", "back", ]
            symbols += ["x", "x",]
            symbol_sizes += [4*sc, 1*sc,]
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
            (v, points_in_view[v], labels, colours, types, symbols, symbol_sizes)
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
                (int, 'symbol_sizes')
                ]

            points, labels, colours, types, symbols, symbol_sizes \
                = read_csv_with_constructors(filename, constructors)

            layers.append(self._single_generic_configuration_layer(
                v, points, labels, colours, types, symbols, symbol_sizes))

        return layers

    def _single_generic_configuration_layer(self, view_index,
                                            points, labels, colours, types, symbols, symbol_sizes):
        props = {
            'labels': labels,
            'colours': colours,
            'types': types,
            'symbols': symbols,
            'symbol_sizes': symbol_sizes,
        }
        layer = napari.layers.Points(
            points,
            name=names_of_generic_calibration_layers[view_index],
            # size=20,
            size=symbol_sizes,
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
            'translation': np.array([-150, 0]),  # move text 150 pixels up
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