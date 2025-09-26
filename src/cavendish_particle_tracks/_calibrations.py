
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
"""
Calibration points (locations of fiducials) come in two types:

(1) generic points (also called Workspace points)
(2) points in specific events

The former (Generic or workspace points) are not associated to any individual event, 
but serve as markers for roughtly where they might be, or serve to assist in the placement
of specific points -- i.e. points of the latter type. E.g. the former could be default 
locations for the latter before the latter are committed or tweaked.

"""


class CalibrationManager:
    """
    This class stores, restores, and manages access to generic
    and specific calibration data.
    """

    __DEFAULT_FILENAME = "CTP_calibrations.napiri"
    num_generic_front_back_fid_pairs = 3

    def __init__(self, viewer):
        self.viewer = viewer
        self.generic_calibration_layers = self._setup_stereoshift_layers()

    def _calibration_layers(self):
        return self.generic_calibration_layers  # + [ self.specific_calibration_layer ]

    def save_calibration(self, filename=__DEFAULT_FILENAME):
        # We don't want to save all layers like this:
        # viewer.layers.save("all_layers.napari")

        # See https://napari.org/dev/api/napari.components.LayerList.html
        napari.components.LayerList(self._calibration_layers()).save(filename)

    def load_calibration(self, filename=__DEFAULT_FILENAME):
        self.viewer.open(filename)


    def _deactivate_calibration_layers(self):  # Move to CalibrationManager??
        """On cancel suppress the points_Stereoshift layer"""
        for layer in self.generic_calibration_layers:
            layer.visible = False

    def _activate_calibration_layers(self): # Move to CalibrationManager??
        current_view = self.viewer.dims.current_step[0]  # axis 0 is 'View', 1 is 'Event', 2 and 3 are x and y

        for i, layer in enumerate(self.generic_calibration_layers):
            if i == current_view:
                self.viewer.layers.selection.active = layer
                print("Turn off this line to stop the autolayer change!!")

            layer.visible = (i == current_view)

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

        for layer in self.generic_calibration_layers:
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

            type = layer.types[i]

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

    def _setup_stereoshift_layers(self):

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

        view_indices = (0, 1, 2)

        name_of_view = [ f"Calibration workspace for view {v}" for v in view_indices ]

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

        # create a points layer where the face_color is set by the good_point feature
        # and the edge_color is set via a color map (grayscale) on the confidence
        # feature.
        # points_layer =
        props = {'labels': labels, }

        layers = [
            self.viewer.add_points(
            points_in_view[v],
            name=name_of_view[v],
            #size=20,
            size=symbol_sizes,
            properties=props,
            border_width=7,
            border_width_is_relative=False,
            border_color=colours,
            face_color=colours,
            symbol=symbols,
            #out_of_slice_display=False,
            ) for v in view_indices
            ]

        # Setup other things for each layer:
        for layer in layers:

            layer.text = {
                'string': labels,
                'color': colours,
                'size': 12,
                'anchor': 'center',
                'translation': np.array([-150, 0]), # move text 150 pixels up
            }
            layer.types = types

            layer.mouse_drag_callbacks.append(self.on_mouse)


        return layers