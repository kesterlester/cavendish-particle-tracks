
import numpy as np
import napari
import tempfile
import pandas as pd
from napari.layers.utils.stack_utils import stack_to_images

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

from .analysis import VIEW_NAMES, CalibrationData, FIDUCIAL_NAMES
from .napari_tools import (
    make_move_only,
    overwrite_layer,
)
from .tools import Accumulator

"""
Calibration points (locations of fiducials) come in two types:

(1) generic points (also called Workspace points)
(2) points in specific events

The former (Generic or workspace points) are not associated to any individual event, 
but serve as markers for roughtly where they might be, or serve to assist in the placement
of specific points -- i.e. points of the latter type. E.g. the former could be default 
locations for the latter before the latter are committed or tweaked.
"""

view_indices = (0, 1, 2)
GENERIC_CALIBRATION_LAYER_NAMES = [f"Calibration (generic; camera {v + 1})" for v in view_indices]
PER_IMAGE_CALIBRATION_LAYER_NAME = "Calibration (per-image)"

class CalibrationManager:
    """
    This class stores, restores, and manages access to generic
    and specific calibration data.
    """

    num_generic_front_back_fid_pairs = 3

    def __init__(self, parent, viewer):

        self.parent = parent
        self.viewer = viewer
        self.calibration_data = CalibrationData()

        self.event_calibration_layer()  # Makes this layer and puts it earlier in the layer list than the next guys:
        # TODO: Try to avoid re-storing this redundant list of generic calibration layers .... should to live only in viewer?
        self._generic_calibration_layers = self._setup_calibration_layers()  # Returns a list of napari point layers.

        # Calibration layers start hidden - nothing to look at before an image is loaded. _load_data_from() in
        # _main_widget.py switches them on once loading has actually finished, to avoid a napari timing issue
        # where flipping visibility mid-image-load crashes the renderer.
        assert hasattr(self, "_generic_calibration_layers")
        self.set_calibration_layer_visibility_and_focus(False, False)

        self.mark_clean()

        # Lastly, setup callbacks:
        self._setup_callbacks()

    def mark_clean(self):
        import copy
        self.last_clean_state = copy.deepcopy(self.state())

    def dirty_things(self):
        calibrations_are_dirty = False

        #print("JJJJJJJJ", len(self.last_clean_state))
        #print("KKKKKKKK", len(self.state()))

        for a, A in zip(self.last_clean_state, self.state()):
            data, meta, _ = a
            DATA, META, _ = A
            if (data != DATA).any():
                calibrations_are_dirty = True
                break
            # TODO: Insert meta comparison to

        if calibrations_are_dirty:
            return [ "calibrations" ]
        else:
            return []

    def state(self):
        ans = tuple(l.as_layer_data_tuple() for l in self.generic_calibration_layers()) + \
                            ( self.event_calibration_layer().as_layer_data_tuple(), ) # Don't forget that comma!
        return ans

    def event_calibration_layer(self) -> napari.layers.Points:
        # TODO: This could break if the user first created a layer with exactly the right name before we construct.
        # Maybe should not reference by NAME but keep a reference.
        if PER_IMAGE_CALIBRATION_LAYER_NAME in self.parent.viewer.layers:
            # Layer already exists, so just return it:
            return self.parent.viewer.layers[PER_IMAGE_CALIBRATION_LAYER_NAME]

        # Layer does not already exist, so construct and return it:
        layer = self.parent.viewer.add_points(name=PER_IMAGE_CALIBRATION_LAYER_NAME, ndim=4, visible=False)
        layer.properties = {"labels": np.array([], dtype=object), }
        print(f"MOO SEE {layer.properties}")
        layer.text = {
            'string': 'labels',
            # This is a key in properties. Somehow it causes the error "Applying the encoding failed. Using the fallback value instead."
            'color': 'white',
            'size': 12,  # text size
            'anchor': 'center',
            'translation': np.array([0, 0, -150, 0]),  # move text 150 (data) pixels up.  4D since 4D
        }
        layer.events.data.connect(self._sync_calibration_data_from_event_layer)
        return layer

    def _sync_calibration_data_from_event_layer(self, event=None) -> None:
        """Rebuild self.calibration_data.event_views entirely from the per-image calibration
        layer's current contents - a full rebuild rather than tracking individual adds/removes,
        same reasoning as the generic-template sync: cheap given the realistic number of stamps,
        and it avoids the per-point diffing bugs the measurement layer needed real work to get
        right elsewhere in this refactor. This also naturally handles deletions, since there's
        no explicit 'remove a stamp' menu action - a student can only delete one via napari's
        own selection tool, which still fires this same event.
        """
        layer = self.event_calibration_layer()
        self.calibration_data.event_views = {}
        labels = layer.properties.get("labels", [])
        for i, point in enumerate(layer.data):
            if i >= len(labels):
                continue
            name = labels[i]
            if name not in FIDUCIAL_NAMES:
                continue
            view = int(point[0])
            evt = int(point[1])
            xy = [float(point[2]), float(point[3])]
            self.calibration_data.stamp(evt, view, name, xy)
        self.parent._sync_calibration_rows_into_table()
        self.parent.set_button_availability()

        print("event_views now:", self.calibration_data.event_views)

    def _sync_generic_template_from_layer(self, view_index) -> None:
        """Mirror one generic calibration layer's current fiducial positions into
        self.calibration_data - a full resync from the layer's own .data/.properties rather
        than surgically patching just the point that changed, since there are at most a dozen
        fiducials per view and a full resync is cheap. Points that aren't real fiducials yet
        (still blank, or the vestigial 'origin'/'decay' calibration points) are simply skipped,
        since FIDUCIAL_NAMES doesn't include them.
        """
        layer = self.generic_calibration_layers()[view_index]
        template = self.calibration_data.generic_templates[view_index]
        # Reset first, not just add/overwrite - otherwise a fiducial that gets relabelled (e.g.
        # fixing a mis-click) leaves its OLD name behind forever, frozen at its old position,
        # looking like a third real fiducial that never actually existed. Confirmed directly: two
        # relabels produced three saved names sharing duplicate coordinates.
        template.positions = {}
        template.slot_indices = {}
        labels = layer.properties.get("labels", [])
        for i, point in enumerate(layer.data):
            if i >= len(labels):
                continue
            name = labels[i]
            if name in FIDUCIAL_NAMES:
                template.set_position(name, [float(point[0]), float(point[1])], slot_index=i)
        self.parent.set_button_availability()

    def _setup_callbacks(self):
        for view_index, layer in enumerate(self.generic_calibration_layers()):
            # This is the callback to allow right-click on generic fiducials:
            layer.mouse_drag_callbacks.append(self.on_mouse)
            make_move_only(layer)
            # Dragging a fiducial is one of the two ways its position can change - the
            # other is being freshly labelled, handled separately at the end of rename_point()
            # below, since labelling alone doesn't fire this event.
            layer.events.data.connect(
                lambda event, v=view_index: self._sync_generic_template_from_layer(v)
            )

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
        return self.generic_calibration_layers() + [ self.event_calibration_layer() ]

    # Callback for when the 'View' slider changes:
    def callback_calibration_layer_visibility(self, event):
        self._refresh_visibility_and_focus_of_all_calibration_layers()

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

        for layer in self.all_calibration_layers():
            orig_symbol_sizes = layer.size.copy()
            if "types" in layer.properties:
                types = layer.properties["types"]
            else:
                types = [None,] * len(layer.data)  # Fallback for when types are not supplied.
            new_symbol_sizes = [ data_pixel_size_for(typ)  for typ in types ]
            layer.size = new_symbol_sizes # This updates the symbol size as desired.

    # Show or hide calibration layers
    def set_calibration_layer_visibility_and_focus(self, visbility: bool, focus: bool):
        # visibility=True means that the correct view will be rendered and the others hidden, otherwise none will be shown.
        # focus=True means that when the relevant view is made visible, it will also be given focus.
        self._calibration_layer_visibility = visbility
        self._calibration_layer_focus = focus
        self._refresh_visibility_and_focus_of_all_calibration_layers()

    # Private method to make all the calibration layers invisible:
    def _hide_generic_calibration_layers(self):
        for layer in self.generic_calibration_layers():
            if layer.visible != False: # Avoid generating unnecessary triggers:
                layer.visible = False

    # Make the correct calibration layers visible/invisible based on the view slider:
    def _show_and_activate_correct_generic_calibration_layer(self):
        current_view = self.viewer.dims.current_step[0]  # axis 0 is 'View', 1 is 'Event', 2 and 3 are image row and col

        for i, layer in enumerate(self.generic_calibration_layers()):
            # Make the current view active if so requested:
            if self._calibration_layer_focus and i == current_view:
                if self.viewer.layers.selection.active != layer: # Avoid generating unnecessary triggers
                    self.viewer.layers.selection.active = layer

            # Make the relevant views visible or invisible:
            desired_state =  (i == current_view)
            if layer.visible != desired_state: # Avoid generating unnecessary triggers:
                layer.visible = desired_state
                layer.mode = "select"

    def _refresh_visibility_and_focus_of_all_calibration_layers(self):
        if self._calibration_layer_visibility:
            self._show_and_activate_correct_generic_calibration_layer()
            self.event_calibration_layer().visible = True
        else:
            self._hide_generic_calibration_layers()
            # Same guard _hide_generic_calibration_layers() already uses on the other layers,
            # and for the same reason - a redundant assignment still fires a real napari event,
            # which can crash if it happens while an image is still mid-insertion (the exact
            # class of bug fixed once already).
            event_layer = self.event_calibration_layer()
            if event_layer.visible != False:
                event_layer.visible = False

    def clone_only_this_fid_view_into_event(self, idx, name, generic_calibration_layer):
        #print(f"About to clone generic fiducial {idx=} with {name=}")
        destination_layer = self.event_calibration_layer()

        # TODO: Either current_event should be passed in (like view) or view should use current_step look up.
        # It makes no sense for one to do one and the other the other!
        current_event = self.viewer.dims.current_step[1]  # axis 0 is 'View', 1 is 'Event', 2 and 3 are image row and col

        # Find view from generic_calibration_layer:
        view = [l.name for l in self.generic_calibration_layers()].index(
            generic_calibration_layer.name)  # names are unique
        #print(f"Clone thinks {view=}.")

        # Don't allow unnamed fid insertion:
        if name == "" or name == None:
            napari.utils.notifications.show_error(f'Fiducial must have a name before it can be cloned.')
            return

        # Check that there is not already a fid with this name, and forbid injection if there is.
        # We do not want to have more than one fid with the same name in the event layer at a given view.
        mask = ((destination_layer.data[:, 0] == view) &
                (destination_layer.data[:, 1] == current_event) &
                (destination_layer.properties["labels"] == name))
        #print(f"Search for {ma,e} Saw {mask=} when \ndestination_layer.properties['labels'] was {destination_layer.properties['labels']} "
        #      f"and \ndestination_layer.data was {destination_layer.data}")

        if mask.any():
            napari.utils.notifications.show_error(f'There is already a fiducial named "{name}" in '
                                                  f"camera {view+1}'s view of event {current_event}.")
            return

        xy = generic_calibration_layer.data[idx]
        label = generic_calibration_layer.properties["labels"][idx]
        #print(f"properties were {generic_calibration_layer.properties["labels"]}")
        #print(f"Found label {label=} in clone_fid_into_event for {view=}") # Correct label is being found, but wrong one stored.

        # Extend xy coords to 4D by adding view and event:
        fiducial_coords_4d_for_this_fiducial_in_view = [view, current_event, xy[0], xy[1]]

        # destination_layer.add() combined with current_properties has a confirmed napari quirk
        # here: adding one new point silently overwrites EVERY existing row's "labels" property
        # too, not just the new row's. Verified directly - printing labels immediately before and
        # after .add() showed already-correct earlier labels get corrupted the moment a second point
        # is added. Sidestepping it entirely: build the full new data/properties arrays ourselves and
        # assign them wholesale, rather than trusting .add() + current_properties to do it incrementally.
        old_data = destination_layer.data
        old_labels = list(destination_layer.properties.get("labels", []))
        new_point = np.array([fiducial_coords_4d_for_this_fiducial_in_view])
        new_data = np.vstack([old_data, new_point]) if len(old_data) else new_point
        new_labels = old_labels + [label]

        destination_layer.data = new_data
        destination_layer.properties = {"labels": np.array(new_labels, dtype=object)}
        destination_layer.current_symbol = "disc"
        # events.data fires the instant .data is assigned above, before the .properties line
        # even runs - so our connected sync callback would see the new point with a still-stale
        # labels array, skip it, and only pick it up on the NEXT stamp. Call it again explicitly,
        # now that both assignments are actually done, to guarantee correctness.
        self._sync_calibration_data_from_event_layer()

        destination_layer.text = destination_layer.text # Needed to get layer.text to become "aware" of property changes
        self.refresh_symbol_sizes()
        destination_layer.refresh() # render

    def clone_all_views_of_this_fid_into_event(self, idx, name):
        #print(f"About to clone generic fiducial {idx=} with {name=} into event.")
        for view, generic_calibration_layer in enumerate(self.generic_calibration_layers()):
            self.clone_only_this_fid_view_into_event(idx, name, generic_calibration_layer)

    def rename_point(self, idx, name, type):
        # print(f"Renaming point idx={idx} to name={name}")

        other_idx = None  # Default
        other_name = None  # Default

        # Could add in the next check, but it is arguably unnecessary.
        # type_is_fiducial = (type == "front" or type == "back")
        # if type_is_fiducial and name != "":
        #    # Check that there is not already a fid with this name, and forbid if there is.
        #    # We do not want multiple pairs of generic fids!
        #    for layer in self.generic_calibration_layers():
        #        if name in layer.properties["labels"]:
        #            napari.utils.notifications.show_error(f'A generic fiducial named "{name}" already exists.')
        #            return

        if type == "front":
            other_idx = idx + 1  # we store front-then-back, so back is +1 on.
            other_name = name[:-1]  # all bar the last character (to remove prime)
        if type == "back":
            other_idx = idx - 1  # we store front-then-back, so front is -1 on.
            other_name = name + "'"

        for layer in self.generic_calibration_layers():
            layer.properties["labels"][idx] = name  # This change is needed for saving purposes.
            if other_idx is not None and other_name is not None:
                layer.properties["labels"][other_idx] = other_name  # This change is needed for saving purposes.
            # Mutating properties in place like this doesn't trigger napari's own redraw. The
            # "reassign text to itself" trick used elsewhere in this file only seems to reliably
            # force an immediate redraw for whichever layer is currently active, not merely
            # visible - explicitly calling refresh() is the more forceful mechanism we already
            # confirmed works reliably elsewhere in this codebase.
            layer.text = layer.text
            layer.refresh()

        # This direct array mutation doesn't fire any napari event on its own, unlike a drag - so
        # a fresh label would otherwise never reach self.calibration_data until/unless the same
        # point later also happens to be dragged. Sync explicitly here instead.
        for view_index in range(len(self.generic_calibration_layers())):
            self._sync_generic_template_from_layer(view_index)

            layer.text = layer.text  # necessary so that layer.text becomes "aware" of the changes we made to layer.properties
            layer.refresh() # render changes to screen

    def on_mouse(self, layer, event):
        # This implements a right-click drop-down menu in response to a point in a generic calibration layer.
        # Note that on mac CTRL-left-click is a synonym for vanilla right-click, so don't expect to be able to use
        # CTRL as a modifier for left-click!  Note that add-to-selection in mac is CMD-left-click, so no
        # conflict with that.

        # Mouse events are defined in here https://github.com/vispy/vispy/blob/main/vispy/app/canvas.py

        """
        A traceback during a mouse event

      (venv) Gorfrog-MacWheird:cavendish-particle-tracks lester$   File "/Users/lester/github/cavendish-particle-tracks/./launch_debug.py", line 25, in <module>
    run()
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_qt/qt_event_loop.py", line 469, in run
    app.exec_()
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/backends/_qt.py", line 626, in event
    out = super(QtBaseCanvasBackend, self).event(ev)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/backends/_qt.py", line 496, in mousePressEvent
    self._vispy_mouse_press(
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/base.py", line 184, in _vispy_mouse_press
    ev = self._vispy_canvas.events.mouse_press(**kwargs)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/util/event.py", line 453, in __call__
    self._invoke_callback(cb, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/util/event.py", line 469, in _invoke_callback
    cb(event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_vispy/canvas.py", line 470, in _on_mouse_press
    self._process_mouse_event(mouse_press_callbacks, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_vispy/canvas.py", line 413, in _process_mouse_event
    mouse_callbacks(self.viewer, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/utils/interactions.py", line 125, in mouse_press_callbacks
    gen = mouse_drag_func(obj, event)
  File "/var/folders/xh/gkx93pyn2xl5jh11l4xwgcgr0000gn/T/ipykernel_68550/2820818860.py", line 5, in store
    traceback.print_stack()





     Here is another traceback during a mouse press handler that YIELDS after processing the press, and then receives a RELEASE:


     (venv) Gorfrog-MacWheird:cavendish-particle-tracks lester$ HELLO i=0
  File "/Users/lester/github/cavendish-particle-tracks/./launch_debug.py", line 25, in <module>
    run()
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_qt/qt_event_loop.py", line 469, in run
    app.exec_()
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/backends/_qt.py", line 626, in event
    out = super(QtBaseCanvasBackend, self).event(ev)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/backends/_qt.py", line 496, in mousePressEvent
    self._vispy_mouse_press(
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/base.py", line 184, in _vispy_mouse_press
    ev = self._vispy_canvas.events.mouse_press(**kwargs)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/util/event.py", line 453, in __call__
    self._invoke_callback(cb, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/util/event.py", line 469, in _invoke_callback
    cb(event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_vispy/canvas.py", line 470, in _on_mouse_press
    self._process_mouse_event(mouse_press_callbacks, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_vispy/canvas.py", line 413, in _process_mouse_event
    mouse_callbacks(self.viewer, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/utils/interactions.py", line 129, in mouse_press_callbacks
    next(gen)
  File "/var/folders/xh/gkx93pyn2xl5jh11l4xwgcgr0000gn/T/ipykernel_68550/740299691.py", line 7, in store
    traceback.print_stack()
Before event.type='mouse_press' event.button=2
yielding
HELLO i=1
  File "/Users/lester/github/cavendish-particle-tracks/./launch_debug.py", line 25, in <module>
    run()
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_qt/qt_event_loop.py", line 469, in run
    app.exec_()
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/backends/_qt.py", line 626, in event
    out = super(QtBaseCanvasBackend, self).event(ev)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/backends/_qt.py", line 506, in mouseReleaseEvent
    self._vispy_mouse_release(
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/base.py", line 224, in _vispy_mouse_release
    ev = self._vispy_canvas.events.mouse_release(**kwargs)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/util/event.py", line 453, in __call__
    self._invoke_callback(cb, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/util/event.py", line 469, in _invoke_callback
    cb(event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_vispy/canvas.py", line 484, in _on_mouse_release
    self._process_mouse_event(mouse_release_callbacks, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_vispy/canvas.py", line 413, in _process_mouse_event
    mouse_callbacks(self.viewer, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/utils/interactions.py", line 216, in mouse_release_callbacks
    next(gen)
  File "/var/folders/xh/gkx93pyn2xl5jh11l4xwgcgr0000gn/T/ipykernel_68550/740299691.py", line 7, in store
    traceback.print_stack()
After event.type='mouse_release' event.button=2

        """
        our_sort_of_event = event.type == "mouse_press" and event.button == 2

        if not our_sort_of_event:
            return

        assert our_sort_of_event

        # Record position at mouse down (not at mouse up which could be different if there was a drag inbetween):
        coords = layer.world_to_data(event.position)

        # Wait for release:
        while event.type != "mouse_release":
            # use of yield explained in https://forum.image.sc/t/custom-mouse-shortcuts-to-help-creating-labels-in-napari/70930/6
            # There is also a sideways reference in https://napari.org/0.4.18/gallery/mouse_drag_callback.html# and another in
            # https://github.com/napari/napari/issues/3246#issuecomment-905803916 which mentions a generator being expected for the callback.
            # Perhaps this is the main documentation by example (but it does not have a mouse release): https://github.com/napari/napari/blob/main/examples/mouse_drag_callback.py
            yield
        assert event.type == "mouse_release"

        # We can now make the menu appear:

        index_of_nearest_point = layer.get_value(coords, world=True)
        if index_of_nearest_point is None:
            # This was not a right-click on a point.
            event.handled = True
            return

        i = index_of_nearest_point  # Just abbreviation shorthand.

        type = layer.properties["types"][i]
        name = layer.properties["labels"][i]

        def show_drop_down_menu(type):
            # build popup menu
            menu = QMenu(self.viewer.window._qt_window)

            # Choose one:
            THING = "name"
            #THING = "label"

            type_is_fiducial = type in ["front", "back"]

            if type_is_fiducial:
                header = QAction("Fiducial actions:", menu)
            else:
                header = QAction("Calibration point actions:", menu)

            header.setEnabled(False)  # makes it unclickable
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            menu.addAction(header)
            menu.addSeparator()

            from .analysis import FIDUCIAL_FRONT, FIDUCIAL_BACK

            if type == "front":
                fixed_names = sorted(FIDUCIAL_FRONT.keys())
            elif type == "back":
                fixed_names = sorted(FIDUCIAL_BACK.keys())
            else:
                fixed_names = ["origin", "decay", ]

            # add fixed names
            for fname in fixed_names:
                act = QAction(f'Set {THING} to "{fname}"', menu)
                act.triggered.connect(lambda _, f=fname: self.rename_point(i, f, type))
                menu.addAction(act)

            if type_is_fiducial:
                menu.addSeparator()
                clone_into_current_image_menu_item = QAction("Insert ONLY THIS VIEW of this fiducial into current event ...", menu)
                clone_into_current_image_menu_item.triggered.connect(
                    lambda _: self.clone_only_this_fid_view_into_event(i, name, layer))
                menu.addAction(clone_into_current_image_menu_item)

                clone_into_current_image_menu_item = QAction("Insert ALL VIEWS OF this fiducial into current event ...", menu)
                clone_into_current_image_menu_item.triggered.connect(lambda _: self.clone_all_views_of_this_fid_into_event(i, name))
                menu.addAction(clone_into_current_image_menu_item)

            # popup at cursor position
            menu.exec_(event.native.globalPos())

        show_drop_down_menu(type)
        event.handled = True

    def _default_generic_layer_arrays(self):
        """The default label/colour/type/symbol layout shared by every generic calibration layer,
        plus each view's own default point positions. Factored out so a brand-new session and restoring
        a saved one (filling in defaults for anything the save file didn't cover) share one copy
        of this layout, instead of risking two copies drifting apart.
        """
        from .analysis import TYPICAL_IMAGE_LONG_SIZE_PIX, TYPICAL_IMAGE_SHORT_SIZE_PIX

        origin_x = 0.5 * TYPICAL_IMAGE_SHORT_SIZE_PIX  # actually how far DOWN !!
        spread_x = 0.15 * TYPICAL_IMAGE_SHORT_SIZE_PIX  # actually vertical spread !!

        fid_step_y = 0.12 * TYPICAL_IMAGE_LONG_SIZE_PIX
        # Centre the whole span of fiducial pairs on the image's middle, rather than starting AT
        # the middle and spreading downward from there - that offset used to leave room above for
        # the now-removed origin/decay points, which no longer applies. Computed from the actual
        # pair count so it stays correct if that number is ever changed.
        fid_origin_y = (
                0.5 * TYPICAL_IMAGE_LONG_SIZE_PIX
                - 0.5 * (CalibrationManager.num_generic_front_back_fid_pairs - 1) * fid_step_y
        )

        # Front/Back fiducial pairs (the old "origin"/"decay" measurement points that used to
        # live here are gone - vestigial since Step 4 moved live origin/decay/track measurement
        # onto the "Radii and Lengths" layer; the menu actions and table-cloning methods that
        # only ever applied to them have been removed too):
        labels = []
        symbols = []
        colours = []
        types = []
        points_in_generic_view = []

        # Now position the Front/Back fiducial pairs:
        for i in range(CalibrationManager.num_generic_front_back_fid_pairs):
            labels += ["", "", ]
            types += ["front", "back", ]
            symbols += ["x", "x", ]
            points_in_generic_view += [
                [origin_x - spread_x, fid_origin_y + i * fid_step_y, ],
                [origin_x + spread_x, fid_origin_y + i * fid_step_y, ],
            ]
            if i == 0:
                colours += [
                    "#55ff00",  # front fiducial (light green)
                    "#00aa00",  # back fiducial (dark green)
                ]
            elif i == 1:
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
            np.array([np.array(point) + np.array([(v - 1) * 100, 0, ]) for point in points_in_generic_view])
            for v in view_indices
        ]

        return points_in_view, labels, colours, types, symbols

    def _default_generic_calibration_layers(self):
        points_in_view, labels, colours, types, symbols = self._default_generic_layer_arrays()

        # If in debug mode can replace the points and labels with ones that are physically interesting.
        # Don't give this option to the students!
        debug_fiducial_mode = False

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
            ]  # overwrites old points_in_view
            labels = debug_point_labels_all_calibration_layers

        layers = [
            self._single_generic_configuration_layer
            (v, points_in_view[v], labels, colours, types, symbols,  # symbol_sizes
             )
            for v in view_indices
        ]

        return layers

    def _restore_generic_calibration_layers(self, generic_templates) -> None:
        """Rebuild the 3 generic calibration layers from a loaded session's saved fiducial positions,
        filling in the standard default position/label for any slot the save file didn't cover.
        These layers are move-only (no add/delete), so a name missing entirely after a load would
        otherwise be permanently unplaceable for the rest of the session.

        Each name is put back in its own original slot (via slot_indices) wherever possible, so colour
        and relative layout next to any still-unlabelled slots both survive a round-trip unchanged.
        Falls back to claiming the first available slot of the matching type only for names with no
        recorded slot (e.g. a .pkl saved before slot tracking existed) or a slot collision.
        """
        from .analysis import FIDUCIAL_FRONT, FIDUCIAL_BACK

        points_in_view, default_labels, colours, types, symbols = self._default_generic_layer_arrays()

        for view_index in range(3):
            template = generic_templates[view_index]
            labels = list(default_labels)
            points = [list(p) for p in points_in_view[view_index]]

            assigned_slots = set()
            unresolved_names = []
            for name, xy in template.positions.items():
                slot_index = template.slot_indices.get(name)
                is_front = name in FIDUCIAL_FRONT
                is_back = name in FIDUCIAL_BACK
                slot_type_matches = slot_index is not None and 0 <= slot_index < len(types) and (
                        (is_front and types[slot_index] == "front")
                        or (is_back and types[slot_index] == "back")
                )
                if slot_type_matches and slot_index not in assigned_slots:
                    labels[slot_index] = name
                    points[slot_index] = xy
                    assigned_slots.add(slot_index)
                else:
                    unresolved_names.append(name)

            if unresolved_names:
                front_slots = [i for i, t in enumerate(types) if t == "front" and i not in assigned_slots]
                back_slots = [i for i, t in enumerate(types) if t == "back" and i not in assigned_slots]
                for name in sorted(unresolved_names):
                    target_slots = front_slots if name in FIDUCIAL_FRONT else back_slots
                    if not target_slots:
                        continue  # nowhere left - only possible with more saved names than slots
                    slot_index = target_slots.pop(0)
                    labels[slot_index] = name
                    points[slot_index] = template.positions[name]

            layer = self.generic_calibration_layers()[view_index]
            layer.data = np.array(points)
            # All four properties keys must be rebuilt together - overwriting only "labels" would
            # silently wipe "types", which the right-click menu depends on to know what a slot is.
            layer.properties = {
                "labels": np.array(labels, dtype=object),
                "colours": colours,
                "types": types,
                "symbols": symbols,
            }
            layer.text = layer.text
            layer.refresh()

        for view_index in range(3):
            self._sync_generic_template_from_layer(view_index)

    def _restore_event_calibration_layer(self) -> None:
        """Rebuild the per-image calibration layer's visible stamps from calibration_data.event_views
        (already restored by the data-side load). This layer holds every event's stamps at once, sliced
        by view/event for display, same as during normal live use.
        """
        layer = self.event_calibration_layer()
        points = []
        labels = []
        for (event, view), fiducial_view_data in self.calibration_data.event_views.items():
            for name, xy in fiducial_view_data.stamped.items():
                points.append([view, event, xy[0], xy[1]])
                labels.append(name)

        layer.data = np.array(points) if points else np.empty((0, 4))
        layer.properties = {"labels": np.array(labels, dtype=object)}
        layer.current_symbol = "disc"
        layer.text = layer.text
        layer.refresh()
        # events.data fires the instant .data is assigned above, before .properties has even run -
        # the same lag bug fixed once already elsewhere in this file.
        self._sync_calibration_data_from_event_layer()

    def _single_generic_configuration_layer(self, view_index,
                                            points, labels, colours, types, symbols, #symbol_sizes
                                            ):
        """
        The point of this function is to provide a single route through which generic config layers are constructed,
        so that even if such layers need internally derived settings, or things not in a csv file, they can be applied
        universally and consistently.
        For example, the layer names are taken from GENERIC_CALIBRATION_LAYER_NAMES[view_index].
        """
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
            'string': 'labels', # This is a key in properties
            'color': colours,
            'size': 12,
            'anchor': 'center',
            'translation': np.array([-150, 0]),  # move text 150 (data) pixels up
        }
        return layer

    def _setup_calibration_layers(self):
        layers = self._default_generic_calibration_layers()

        # Overwrite data if layer already exists, otherwise make a note of new layers
        new_layers = []
        for layer in layers:
            if layer.name in self.viewer.layers:
                # Existing layer, so callbacks already exist too, so just overwrite old layer data:
                overwrite_layer(self.viewer.layers[layer.name], layer)
                #print("\n\n REPLACING DATA \n\n")
            else:
                # New layer!
                new_layers.append(layer)
                #print("\n\n NOTING NEW LAYER \n\n")

        # Tell Napari about any new generic calibration layers:
        for new_layer in new_layers:
            self.viewer.add_layer(new_layer)

        return new_layers
