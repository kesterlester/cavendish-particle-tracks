import copy
from typing import TYPE_CHECKING

import napari
import numpy as np
from napari.utils.notifications import show_error
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

from ._calculate import depth, length, stereoshift
from .analysis import Fiducial, StereoshiftInfo

if TYPE_CHECKING:
    from ._main_widget import ParticleTracksWidget


class StereoshiftDialog(QDialog):
    def __init__(self, parent: "ParticleTracksWidget"):
        super().__init__(parent)

        self.parent = parent

        self.setWindowTitle("Stereoshift")

        self.num_front_back_fid_pairs = 3

        # drop-down lists of vertex
        self.vertex_combobox = QComboBox()
        self.vertex_combobox.addItem("Origin vertex")
        self.vertex_combobox.addItem("Decay vertex")
        self.vertex_combobox.currentIndexChanged.connect(self._on_click_vertex)

        # Choose one:
        self.RIGHT_CLICK = "right click"
        self.DOUBLE_CLICK = "double clic"
        #self.point_menu_type = self.DOUBLE_CLICK  # Less intuitive but does not get stuck in pan mode afterwards.
        self.point_menu_type = self.RIGHT_CLICK # More intuitive but gets stuck in pan mode afterwards.

        # text boxes for points
        self.textboxes = [QLabel(self) for _ in range(4)]

        # text boxes for results
        self.tshift_fiducial = QLabel(self)
        self.tshift_point = QLabel(self)
        self.tstereoshift = QLabel(self)
        self.tdepth = QLabel(self)

        self.results = [
            self.tshift_fiducial,
            self.tshift_point,
            self.tstereoshift,
            self.tdepth,
        ]
        for textbox in self.textboxes + self.results:
            textbox.setMinimumWidth(200)

        bss = QPushButton("Calculate")
        bss.clicked.connect(self._on_click_calculate)

        bap = QPushButton("Save to table")
        bap.clicked.connect(self._on_click_save_to_table)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.buttonBox.clicked.connect(self.reject)

        lviewf1 = QLabel("View 1")
        lviewf2 = QLabel("View 2")
        lviewb1 = QLabel("View 1")
        lviewb2 = QLabel("View 2")

        self.label_stereoshift = QLabel(
            "Stereo shift (shift_p/shift_f = depth_p/depth_f)"
        )

        # layout
        self.setLayout(QGridLayout())
        self.layout().addWidget(QLabel("Select Vertex"), 0, 0, 1, 2)
        self.layout().addWidget(self.vertex_combobox, 0, 2)
        self.layout().addWidget(QLabel("Fiducial coordinates"), 2, 0, 1, 2)
        for i, widget in enumerate(
            [lviewf1, self.textboxes[0], lviewf2, self.textboxes[1]]
        ):
            self.layout().addWidget(widget, i // 2 + 3, i % 2 + 1)
        self.layout().addWidget(QLabel("Point coordinates"), 5, 0, 1, 2)
        for i, widget in enumerate(
            [lviewb1, self.textboxes[2], lviewb2, self.textboxes[3]]
        ):
            self.layout().addWidget(widget, i // 2 + 6, i % 2 + 1)

        self.layout().addWidget(bss, 7, 0, 1, 3)

        self.layout().addWidget(
            self.label_stereoshift,
            9,
            0,
            1,
            3,
        )
        # self.layout().addWidget(self.table, 9, 0, 1, 3)
        self.layout().addWidget(QLabel("Fiducial shift"), 10, 1)
        self.layout().addWidget(self.tshift_fiducial, 10, 2)
        self.layout().addWidget(QLabel("Point shift"), 11, 1)
        self.layout().addWidget(self.tshift_point, 11, 2)
        self.layout().addWidget(QLabel("Ratio"), 12, 1)
        self.layout().addWidget(self.tstereoshift, 12, 2)
        self.layout().addWidget(QLabel("Point depth (cm)"), 13, 1)
        self.layout().addWidget(self.tdepth, 13, 2)
        self.layout().addWidget(bap, 14, 0, 1, 3)
        self.layout().addWidget(self.buttonBox, 15, 0, 1, 3)

        # Setup points layer
        self.cal_layers = self._setup_stereoshift_layers()
        self.parent.viewer.dims.events.current_step.connect(self._callback_that_activates_calibration_layers)

        # Stereoshift related parameters
        self.stereoshift_info = StereoshiftInfo()
        self.stereoshift_info.name = "origin_vertex"

    def _setup_stereoshift_layers(self):
        # retrieve current camera position
        from .analysis import TYPICAL_IMAGE_LONG_SIZE_PIX, TYPICAL_IMAGE_SHORT_SIZE_PIX
        #origin_x = self.parent.camera_center[0]
        #origin_y = self.parent.camera_center[1]
        origin_x = 0.5 * TYPICAL_IMAGE_SHORT_SIZE_PIX # actually how far DOWN !!
        spread_x = 0.15 * TYPICAL_IMAGE_SHORT_SIZE_PIX # actually vertical spread !!

        point_origin_y = 0.25 * TYPICAL_IMAGE_LONG_SIZE_PIX # actually how far ACROSS !!
        fid_origin_y = 0.5 * TYPICAL_IMAGE_LONG_SIZE_PIX
        fid_step_y = 0.12 * TYPICAL_IMAGE_LONG_SIZE_PIX

        #zoom_factor = self.parent.viewer.camera.zoom
        zoom_factor = 0.05


        sc = 100

        # First position the point being measured:
        labels = [ "point", ]
        symbols = [ "disc", ]
        colours = ["cyan", ]
        types = ["point", ]
        symbol_sizes = [1 * sc, ]
        points_in_generic_view = [ [origin_x, point_origin_y, ], ]

        # Now position the Front/Back fiducial pairs:
        for i in range(self.num_front_back_fid_pairs):
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

        name_of_view = [ f"View {v} calibration layer" for v in view_indices ]

        # Displace the generic points 100 to the left, or not at all, or 100 to the right, depending on view:
        points_in_view = [
            np.array([ np.array(point)+np.array([(v-1)*100, 0,]) for point in points_in_generic_view ])
            for v in view_indices
        ]

        for v in view_indices:
            print(f"Point in view {v} are {points_in_view[v]}")

        """
        text = {
            "string": labels,
            "size": 14,
            "color": colors,
            "translation": np.array([-30, 0]),
        }
        """

        # create a points layer where the face_color is set by the good_point feature
        # and the edge_color is set via a color map (grayscale) on the confidence
        # feature.
        # points_layer =
        props = {'labels': labels, }

        layers = [
            self.parent.viewer.add_points(
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

        # set the edge_color mode to colormap
        # points_layer.edge_color_mode = 'colormap'
        for layer in layers:

            layer.text = {
                'string': labels,
                'color': colours,
                'size': 12,
                'anchor': 'center',
                'translation': np.array([-75, 0]), # move text 25 pixels up
            }
            layer.types = types

            if self.point_menu_type == self.RIGHT_CLICK:
                layer.mouse_drag_callbacks.append(self.on_mouse)
            if self.point_menu_type == self.DOUBLE_CLICK:
                layer.mouse_double_click_callbacks.clear() # want to disable double click zoom
                layer.mouse_double_click_callbacks.append(self.on_mouse)

        return layers

    def rename_point(self, idx, name, type):
        #print(f"Renaming point idx={idx} with name={name}")

        other_idx = None # Default
        other_name = None # Default
        if type == "front":
            other_idx = idx + 1 # we store front-then-back, so back is +1 on.
            other_name = name[:-1] # all bar the last character (to remove prime)
        if type == "back":
            other_idx = idx - 1 # we store front-then-back, so front is -1 on.
            other_name = name + "'"

        for layer in self.cal_layers:
            # print(f"before alteration {layer.text}")
            layer.text.values[idx] = name
            #print(f"after  alteration {layer.text}")
            if other_idx is not None and other_name is not None:
                layer.text.values[other_idx] = other_name

            layer.refresh()

    def on_mouse(self, layer, event):
        print(f"Detected mouse event {event} on layer {layer}")
        if (
                self.point_menu_type == self.DOUBLE_CLICK or
                self.point_menu_type == self.RIGHT_CLICK and event.button == 2
            ):
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
                menu = QMenu(self.parent.viewer.window._qt_window)

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
                        self.parent.viewer.window._qt_window,
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
                menu.exec_(pos) # use captured global cursor pos -- see (*) below

            print(" got 6 ")

            # capture cursor now
            pos = QCursor.pos() # (*)

            """
            We can't just all show_menu() in the next line as it 
            results in our capuring the right click by hiding the 
            right mouse button RELEASE.
            """
            QTimer.singleShot(100, lambda : show_menu(type) )

            print(f" got 7 ")


    # Callback for when the 'View' slider changes
    def _callback_that_activates_calibration_layers(self, event):
        self._activate_calibration_layers()

    def _on_click_vertex(self) -> None:
        """When vertex is selected, update the name of the vertex"""
        if self.vertex_combobox.currentIndex() == 0:
            self.stereoshift_info.name = "origin_vertex"
        if self.vertex_combobox.currentIndex() == 1:
            self.stereoshift_info.name = "decay_vertex"

    def _on_click_calculate(self) -> None:
        """When 'Calculate' button is clicked, calculate stereoshift and populate table"""

        # Add points coords to corresponding text box
        for i in range(len(self._fiducial_views)):
            (
                self._fiducial_views[i].x,
                self._fiducial_views[i].y,
            ) = (
                77 #self.cal_layer.data[i + 2] - self.cal_layer.data[i % 2]
            )
            self.textboxes[i].setText(str(self._fiducial_views[i].xy))

        # Calculate stereoshift and depth
        self.stereoshift_info.shift_fiducial = length(self.f(1).xy, self.f(2).xy)
        self.stereoshift_info.shift_point = length(self.b(1).xy, self.b(2).xy)
        self.stereoshift_info.stereoshift = stereoshift(
            *[view.xy for view in self._fiducial_views]
        )
        self.stereoshift_info.depth_cm = depth(
            self.f(1),
            self.f(2),
            self.b(1),
            self.b(2),
            reverse=False,
        )
        self.stereoshift_info.spoints = 77 #self.cal_layer.data[2:]

        # Populate the table
        self.tshift_fiducial.setText(str(self.stereoshift_info.shift_fiducial))
        self.tshift_point.setText(str(self.stereoshift_info.shift_point))
        self.tstereoshift.setText(str(self.stereoshift_info.stereoshift))
        self.tdepth.setText(str(self.stereoshift_info.depth_cm))

    def _on_click_save_to_table(self) -> None:
        """When 'Save to table' button is clicked, propagate stereoshift and depth to main table"""
        # Propagate to particle
        try:
            selected_row = self.parent._get_selected_row()
        except IndexError:
            show_error("There are no particles in the table.")
        else:
            # Figure out what vertex to calculate stereoshift for
            what_vertex = self.vertex_combobox.currentIndex()
            # Save the stereoshift info to the particle
            if what_vertex == 0:
                self.parent.data[selected_row].origin_vertex_stereoshift_info = (
                    copy.deepcopy(self.stereoshift_info)
                )
            else:
                self.parent.data[selected_row].decay_vertex_stereoshift_info = (
                    copy.deepcopy(self.stereoshift_info)
                )
            # Update the table
            self.parent.table.setItem(
                selected_row,
                self.parent._get_table_column_index(
                    self.stereoshift_info.name + "_stereoshift_info"
                ),
                QTableWidgetItem(str(self.stereoshift_info)),
            )
            self.parent.table.setItem(
                selected_row,
                self.parent._get_table_column_index(
                    self.stereoshift_info.name + "_depth_cm"
                ),
                QTableWidgetItem(str(self.stereoshift_info.depth_cm)),
            )

            napari.utils.notifications.show_info(
                "Stereoshift of "
                + self.stereoshift_info.name.replace("_", " ")
                + " saved to particle "
                + str(selected_row)
            )

    def _deactivate_calibration_layers(self):
        """On cancel suppress the points_Stereoshift layer"""
        for layer in self.cal_layers:
            layer.visible = False
            # self.parent._deactivate_calibration_layer(layer)

    def _activate_calibration_layers(self):
        current_view = self.parent.viewer.dims.current_step[0]  # axis 0 is 'View', 1 is 'Event', 2 and 3 are x and y

        for i, layer in enumerate(self.cal_layers):
            if i == current_view:
                self.parent.viewer.layers.selection.active = layer
                print("Turn off this line to stop the autolayer change!!")

            layer.visible = (i == current_view)

    def show(self) -> None:
        self._activate_calibration_layers()
        return super().show()

    def reject(self) -> None:
        self._deactivate_calibration_layers()
        return super().reject()
